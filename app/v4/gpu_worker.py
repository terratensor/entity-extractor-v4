#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GPU Worker - воркер 3 конвейера
Выполняет инференс на GPU и извлекает сущности с высоким качеством
Использует проверенную логику из v1 для правильной обработки BIO-тегов
"""

import logging
import time
from queue import Queue, Empty, Full
from typing import List, Dict, Any, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForTokenClassification, AutoTokenizer

from app.v4.shutdown import StoppableThread

logger = logging.getLogger(__name__)


class GPUWorker(StoppableThread):
    """
    Выполняет инференс на GPU и извлекает сущности.
    Использует правильную BIO-разметку из v1 для максимального качества.
    """
    
    # Маппинг всех возможных BIO-тегов в группы
    ENTITY_GROUP_MAP = {
        'B-LOC': 'LOC', 'I-LOC': 'LOC',
        'B-PER': 'PER', 'I-PER': 'PER',
        'B-ORG': 'ORG', 'I-ORG': 'ORG',
        # 'B-DATE': 'DATE', 'I-DATE': 'DATE', # Добавили DATE, чтобы он не потерялся при маппинге
        'B-MISC': 'MISC', 'I-MISC': 'MISC'
    }
    
    def __init__(
        self,
        device_id: int,
        config,
        input_queue: Queue,
        output_queue: Queue,
        stop_event,
        name: Optional[str] = None
    ):
        """
        Args:
            device_id: ID GPU устройства
            config: конфигурация (ModelConfig + GPU параметры)
            input_queue: очередь входных чанков (от tokenizer)
            output_queue: очередь выходных сущностей (для writer)
            stop_event: событие остановки
            name: имя потока
        """
        name = name or f"GPU-{device_id}"
        super().__init__(name=name, stop_event=stop_event)
        
        self.device_id = device_id
        self.config = config
        self.input_queue = input_queue
        self.output_queue = output_queue
        
        # Параметры модели
        self.model_name = config.model.name
        self.max_tokens = config.model.max_tokens
        self.min_confidence = config.model.min_confidence
        self.include_positions = getattr(config.model, 'include_positions', False)
        
        # Параметры GPU
        self.batch_size = config.batch_size
        self.precision = getattr(config, 'precision', 'float16')
        
        # Модель и токенизатор (загружаются в потоке)
        self.model = None
        self.tokenizer = None
        self.device = None
        self.id2label = None
        
        # Статистика
        self.stats = {
            'processed_chunks': 0,
            'processed_docs': set(),  # уникальные документы
            'total_tokens': 0,
            'entities_found': 0,
            'entities_by_type': {'LOC': 0, 'PER': 0, 'ORG': 0, 'MISC': 0},
            'inference_time': 0,
            'batches_processed': 0,
            'start_time': None
        }
        
        logger.info(f"🤖 {self.name} инициализирован (device={device_id}, batch={self.batch_size})")
    
    def run(self) -> None:
        """Основной цикл инференса."""
        logger.info(f"🚀 {self.name} запущен на cuda:{self.device_id}")
        
        # Загружаем модель и токенизатор
        self._load_model()
        
        self.stats['start_time'] = time.time()
        
        while not self.should_stop():
            try:
                # Набираем батч из очереди
                batch = self._collect_batch()
                
                if not batch:
                    # Нет данных, ждём
                    self.safe_sleep(0.1)
                    continue
                
                # Выполняем инференс
                results = self._process_batch(batch)
                
                # Отправляем результаты в выходную очередь
                for result in results:
                    while not self.should_stop():
                        try:
                            self.output_queue.put(result, timeout=0.5)
                            break
                        except Full:
                            logger.debug(f"{self.name}: выходная очередь переполнена, ждём...")
                            if not self.safe_sleep(0.1):
                                break
                
                # Обновляем статистику
                self.stats['batches_processed'] += 1
                self.stats['processed_chunks'] += len(batch)
                
                # Периодическое логирование
                if self.stats['batches_processed'] % 10 == 0:
                    elapsed = time.time() - self.stats['start_time']
                    chunks_per_sec = self.stats['processed_chunks'] / elapsed if elapsed > 0 else 0
                    logger.info(
                        f"{self.name}: обработано {self.stats['processed_chunks']} чанков, "
                        f"{len(self.stats['processed_docs'])} документов, "
                        f"сущностей: {self.stats['entities_found']}, "
                        f"скорость: {chunks_per_sec:.1f} чанков/сек"
                    )
                
            except Exception as e:
                logger.error(f"{self.name}: критическая ошибка: {e}", exc_info=True)
                # При ошибке пропускаем батч и продолжаем
        
        # Завершение работы
        elapsed = time.time() - self.stats['start_time']
        logger.info(
            f"📊 {self.name} завершён: обработано {self.stats['processed_chunks']} чанков, "
            f"{len(self.stats['processed_docs'])} документов, "
            f"найдено {self.stats['entities_found']} сущностей, "
            f"время: {elapsed:.1f} сек"
        )
    
    def _load_model(self) -> None:
        """Загружает модель и токенизатор на указанный GPU."""
        try:
            self.device = torch.device(f'cuda:{self.device_id}')
            
            logger.info(f"{self.name}: загрузка токенизатора {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            logger.info(f"{self.name}: загрузка модели {self.model_name}")
            
            # Загружаем модель с нужной точностью
            dtype = torch.float16 if self.precision == 'float16' else torch.float32
            
            self.model = AutoModelForTokenClassification.from_pretrained(
                self.model_name,
                torch_dtype=dtype
            ).to(self.device)
            
            self.model.eval()
            
            # Сохраняем маппинг id->label
            self.id2label = self.model.config.id2label
            
            logger.info(f"{self.name}: модель успешно загружена на {self.device}")
            
        except Exception as e:
            logger.error(f"{self.name}: ошибка загрузки модели: {e}")
            raise
    
    def _collect_batch(self) -> List[Dict[str, Any]]:
        """
        Набирает батч из входной очереди.
        
        Returns:
            List[Dict]: список чанков для обработки
        """
        batch = []
        
        # Пытаемся набрать batch_size элементов
        for _ in range(self.batch_size):
            try:
                chunk = self.input_queue.get(timeout=0.1)
                if chunk is None:  # сигнал остановки
                    if batch:
                        self.input_queue.put(None)  # возвращаем сигнал для других
                    break
                batch.append(chunk)
            except Empty:
                break
        
        return batch
    
    def _process_batch(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Обрабатывает батч чанков, используя aggregation_strategy=None как в v1
        """
        if not batch:
            return []
        
        batch_start = time.time()
        
        # Подготовка текстов
        texts = [chunk['text'] for chunk in batch]
        
        # Используем пайплайн с aggregation_strategy=None
        if not hasattr(self, 'ner_pipeline'):
            from transformers import pipeline
            
            # Правильный параметр - 'dtype', а не 'torch_dtype'
            model_kwargs = {}
            if self.precision == 'float16':
                model_kwargs['dtype'] = torch.float16
            
            self.ner_pipeline = pipeline(
                "ner",
                model=self.model,
                tokenizer=self.tokenizer,
                device=self.device,
                aggregation_strategy=None,
                model_kwargs=model_kwargs
            )
        
        # Получаем сырые предсказания для всех текстов в батче
        try:
            batch_results = self.ner_pipeline(texts, batch_size=len(texts))
        except Exception as e:
            logger.error(f"Ошибка в пайплайне: {e}")
            # В случае ошибки возвращаем пустые результаты для всех чанков
            batch_results = [[] for _ in texts]
            
            # Если ошибка связана с типами данных, пробуем пересоздать пайплайн с float32
            if "Half but found Float" in str(e) and self.precision == 'float16':
                logger.warning(f"{self.name}: пробуем пересоздать пайплайн с float32")
                self.precision = 'float32'
                delattr(self, 'ner_pipeline')
                # Рекурсивный вызов с новым пайплайном
                return self._process_batch(batch)
        
        results = []
        
        for i, chunk in enumerate(batch):
            token_entities = batch_results[i] if i < len(batch_results) else []
            
            # Фильтруем по confidence
            token_entities = [t for t in token_entities if t['score'] >= self.min_confidence]
            
            # [ВАЖНО] Корректируем позиции с учетом глобального смещения
            global_start = chunk.get('global_start', 0)
            if global_start > 0:
                for t in token_entities:
                    t['start'] += global_start
                    t['end'] += global_start
            
            # Извлекаем сущности
            entities = self._extract_entities_v1(token_entities, chunk['text'])
            
            # Формируем результат
            result = {
                'id': chunk['id'],
                'chunk_id': chunk['chunk_id'],
                'total_chunks': chunk['total_chunks'],
                'text': chunk['text'],
                'global_start': global_start,
                'entities': entities,
                'stats': {
                    'tokens': len(chunk['input_ids']),
                    'entities_count': len(entities)
                }
            }
            
            results.append(result)
            
            # Обновляем статистику
            self.stats['processed_docs'].add(chunk['id'])
            self.stats['total_tokens'] += len(chunk['input_ids'])
            self.stats['entities_found'] += len(entities)
            
            for entity in entities:
                etype = entity['type']
                if etype in self.stats['entities_by_type']:
                    self.stats['entities_by_type'][etype] += 1
        
        self.stats['inference_time'] += time.time() - batch_start
        self.stats['processed_chunks'] += len(batch)
        
        return results

    def _extract_entities_v1(self, token_entities: List[Dict], original_text: str) -> List[Dict]:
        """
        Улучшенная версия с обработкой ошибочных B- токенов
        """
        if not token_entities:
            return []
        
        # Группируем токены в целые сущности
        grouped_entities = []
        i = 0
        
        while i < len(token_entities):
            current = token_entities[i]
            
            # Пропускаем не-сущности
            if current['entity'] == 'O':
                i += 1
                continue
            
            # Начало новой сущности (B- или первая)
            if current['entity'].startswith('B-') or i == 0:
                # Собираем все токены этой сущности
                entity_tokens = [current]
                current_type = self._get_entity_type(current['entity'])
                j = i + 1
                
                while j < len(token_entities):
                    next_token = token_entities[j]
                    next_type = self._get_entity_type(next_token['entity'])
                    
                    # Проверяем, может ли следующий токен быть частью текущей сущности
                    can_be_continuation = False
                    
                    # Случай 1: Правильное продолжение (I- того же типа)
                    if next_token['entity'].startswith('I-') and next_type == current_type:
                        can_be_continuation = True
                    
                    # Случай 2: Ошибочное B- того же типа, но токены идут подряд в тексте
                    elif next_token['entity'].startswith('B-') and next_type == current_type:
                        # Проверяем, что между токенами нет пробела или он минимальный
                        gap = next_token['start'] - entity_tokens[-1]['end']
                        if gap <= 1:  # нет пробела или только один символ
                            can_be_continuation = True
                            # Можно добавить отладку если нужно
                            # logger.debug(f"  ⚠️ Исправление ошибочного B-: {next_token['word']}")
                    
                    if can_be_continuation:
                        entity_tokens.append(next_token)
                        j += 1
                    else:
                        break
                
                # Объединяем токены в сущность
                merged = self._merge_tokens_v1(entity_tokens, original_text)
                if merged:
                    grouped_entities.append(merged)
                
                i = j
            else:
                i += 1
        
        return grouped_entities

    def _merge_tokens_v1(self, token_entities: List[Dict], original_text: str) -> Optional[Dict]:
        """
        ТОЧНАЯ КОПИЯ ИЗ v1 merge_token_entities
        Добавлены позиции для расширения
        """
        if not token_entities:
            return None
        
        # Сортируем токены по позиции
        sorted_tokens = sorted(token_entities, key=lambda x: x['start'])
        
        # Собираем текст с пробелами
        text_parts = []
        last_end = None
        
        for token in sorted_tokens:
            # Убираем символ подчёркивания в начале токена
            word = token['word'].replace('▁', '')
            
            # Добавляем пробел, если это не первый токен и есть разрыв
            if last_end is not None and token['start'] > last_end:
                # Проверяем размер разрыва
                if token['start'] - last_end >= 1:
                    text_parts.append(' ')
            
            text_parts.append(word)
            last_end = token['end']
        
        full_text = ''.join(text_parts)
        
        # Берём минимальный score
        confidence = min(t['score'] for t in sorted_tokens)
        
        # Определяем тип
        first_label = sorted_tokens[0]['entity']
        entity_type = self.ENTITY_GROUP_MAP.get(first_label, 'MISC')
        
        # В v1 нет фильтрации по длине, но можно добавить
        if len(full_text) < 2:
            return None
        
        # Базовый результат
        result = {
            'text': full_text,
            'type': entity_type,
            'confidence': round(confidence, 4)
        }
        
        # [ВАЖНО] Всегда добавляем позиции, если они есть
        if token_entities and 'start' in token_entities[0]:
            result['positions'] = [
                {'start': t['start'], 'end': t['end']} 
                for t in sorted_tokens
            ]
        
        return result
        
    def _get_entity_type(self, label: str) -> str:
        """Извлекает тип сущности из BIO-тега."""
        if label.startswith(('B-', 'I-')):
            return label[2:]
        return label
 
    def get_stats(self) -> Dict[str, Any]:
        """Возвращает статистику работы воркера."""
        stats = self.stats.copy()
        stats['processed_docs_count'] = len(self.stats['processed_docs'])
        stats['processed_docs'] = list(self.stats['processed_docs'])[-100:]  # последние 100 для отладки
        
        if stats['start_time']:
            stats['current_time'] = time.time() - stats['start_time']
        
        return stats