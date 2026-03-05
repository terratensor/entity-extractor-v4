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
    
    # Маппинг BIO-тегов в группы сущностей (из v1)
    ENTITY_GROUP_MAP = {
        'B-LOC': 'LOC', 'I-LOC': 'LOC',
        'B-PER': 'PER', 'I-PER': 'PER',
        'B-ORG': 'ORG', 'I-ORG': 'ORG',
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
        Обрабатывает батч чанков с правильным извлечением сущностей.
        
        Args:
            batch: список чанков от tokenizer
            
        Returns:
            List[Dict]: результаты с извлечёнными сущностями
        """
        if not batch:
            return []
        
        batch_start = time.time()
        
        # Подготовка тензоров
        input_ids_list = [chunk['input_ids'] for chunk in batch]
        attention_mask_list = [chunk['attention_mask'] for chunk in batch]
        
        # Определяем максимальную длину в батче (для паддинга)
        max_len = max(len(ids) for ids in input_ids_list)
        
        # Паддинг до одинаковой длины
        padded_input_ids = []
        padded_attention_mask = []
        
        for ids, mask in zip(input_ids_list, attention_mask_list):
            pad_len = max_len - len(ids)
            padded_input_ids.append(ids + [self.tokenizer.pad_token_id] * pad_len)
            padded_attention_mask.append(mask + [0] * pad_len)
        
        # Преобразуем в тензоры на GPU
        input_ids = torch.tensor(padded_input_ids, device=self.device, dtype=torch.long)
        attention_mask = torch.tensor(padded_attention_mask, device=self.device, dtype=torch.long)
        
        # Инференс
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Получаем логиты и вычисляем предсказания с уверенностью
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        
        # Вычисляем confidence через softmax (как в v1)
        probs = F.softmax(logits, dim=-1)
        confidence = torch.max(probs, dim=-1).values
        
        results = []
        
        for i, chunk in enumerate(batch):
            # Получаем предсказания для этого чанка (обрезаем паддинг)
            chunk_len = len(chunk['input_ids'])
            chunk_preds = predictions[i, :chunk_len].cpu().tolist()
            chunk_conf = confidence[i, :chunk_len].cpu().tolist()
            
            # Получаем оригинальный текст
            original_text = chunk.get('original_text', '')
            
            # Получаем токены
            tokens = self.tokenizer.convert_ids_to_tokens(chunk['input_ids'])
            
            # Получаем offset_mapping для точных позиций
            try:
                inputs = self.tokenizer(
                    original_text,
                    return_offsets_mapping=True,
                    max_length=self.max_tokens,
                    truncation=True
                )
                offsets = inputs['offset_mapping']
            except:
                offsets = [(0, 0)] * len(tokens)
            
            # Создаём список токенов с предсказаниями (как в v1)
            token_entities = []
            for j, (token, pred_id, conf) in enumerate(zip(tokens, chunk_preds, chunk_conf)):
                if j >= len(offsets):
                    break
                    
                # Пропускаем специальные токены
                if token in ['<s>', '</s>', '<pad>']:
                    continue
                
                # Получаем метку
                label = self.model.config.id2label.get(pred_id, 'O')
                
                # Пропускаем если уверенность ниже порога
                if conf < self.min_confidence:
                    continue
                
                token_entities.append({
                    'word': token,
                    'entity': label,
                    'score': conf,
                    'start': offsets[j][0] if offsets[j] else 0,
                    'end': offsets[j][1] if offsets[j] else 0
                })
            
            # Извлекаем сущности используя правильную BIO-логику из v1
            entities = self._extract_entities_from_tokens(token_entities, original_text)
            
            # Формируем результат
            result = {
                'id': chunk['id'],
                'chunk_id': chunk['chunk_id'],
                'total_chunks': chunk['total_chunks'],
                'entities': entities,
                'stats': {
                    'tokens': chunk_len,
                    'entities_count': len(entities)
                }
            }
            
            results.append(result)
            
            # Обновляем статистику
            self.stats['processed_docs'].add(chunk['id'])
            self.stats['total_tokens'] += chunk_len
            self.stats['entities_found'] += len(entities)
            
            for entity in entities:
                etype = entity['type']
                if etype in self.stats['entities_by_type']:
                    self.stats['entities_by_type'][etype] += 1
        
        # Обновляем время инференса
        self.stats['inference_time'] += time.time() - batch_start
        
        return results
    
    def _extract_entities_from_tokens(self, token_entities: List[Dict], original_text: str) -> List[Dict]:
        """
        Извлекает сущности из списка токенов с BIO-разметкой.
        ТОЧНАЯ КОПИЯ ЛОГИКИ ИЗ v1 extractor.py
        
        Args:
            token_entities: список токенов с entity, score, start, end
            original_text: оригинальный текст
            
        Returns:
            List[Dict]: список сущностей
        """
        if not token_entities:
            return []
        
        entities = []
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
                j = i + 1
                
                while j < len(token_entities):
                    next_token = token_entities[j]
                    # Продолжение сущности (I- того же типа)
                    if next_token['entity'].startswith('I-') and \
                       self._get_entity_type(next_token['entity']) == self._get_entity_type(current['entity']):
                        entity_tokens.append(next_token)
                        j += 1
                    else:
                        break
                
                # Объединяем токены в сущность
                merged = self._merge_token_entities(entity_tokens, original_text)
                if merged:
                    entities.append(merged)
                
                i = j
            else:
                i += 1
        
        return entities
    
    def _get_entity_type(self, label: str) -> str:
        """Извлекает тип сущности из BIO-тега."""
        if label.startswith(('B-', 'I-')):
            return label[2:]
        return label
    
    def _merge_token_entities(self, token_entities: List[Dict], original_text: str) -> Optional[Dict]:
        """
        Объединяет токены одной сущности в целое слово с правильными пробелами.
        ИСПРАВЛЕННАЯ ВЕРСИЯ: использует оригинальный текст для сохранения пробелов
        """
        if not token_entities:
            return None
        
        # Сортируем токены по позиции
        sorted_tokens = sorted(token_entities, key=lambda x: x['start'])
        
        # Получаем текст напрямую из оригинального текста по позициям
        start_pos = sorted_tokens[0]['start']
        end_pos = sorted_tokens[-1]['end']
        
        # Вырезаем текст из оригинала (это сохранит все оригинальные пробелы и дефисы)
        full_text = original_text[start_pos:end_pos]
        
        # Нормализуем пробелы (один пробел между словами, но сохраняем дефисы)
        full_text = ' '.join(full_text.split())
        
        # Если текст получился подозрительно коротким, используем запасной вариант
        if len(full_text) < 2:
            # Запасной вариант: склейка токенов
            text_parts = []
            for token in sorted_tokens:
                word = token['word'].replace('▁', '')
                if text_parts and not word.startswith(("'", "-", ".", ",", ")", "(", ":", ";")):
                    text_parts.append(' ')
                text_parts.append(word)
            full_text = ''.join(text_parts)
            full_text = ' '.join(full_text.split())
        
        # Берём минимальный score
        confidence = min(t['score'] for t in sorted_tokens)
        
        # Определяем тип
        first_label = sorted_tokens[0]['entity']
        entity_type = self.ENTITY_GROUP_MAP.get(first_label, 'MISC')
        
        result = {
            'text': full_text,
            'type': entity_type,
            'confidence': round(confidence, 4)
        }
        
        # Опционально добавляем позиции
        if self.include_positions:
            result['positions'] = [
                {'start': t['start'], 'end': t['end']} 
                for t in sorted_tokens
            ]
        
        return result
    
    def _extract_entities_from_tokens(self, token_entities: List[Dict], original_text: str) -> List[Dict]:
        """
        Извлекает сущности из списка токенов с BIO-разметкой.
        ТОЧНАЯ КОПИЯ ЛОГИКИ ИЗ v1 extractor.py
        """
        if not token_entities:
            return []
        
        entities = []
        i = 0
        
        while i < len(token_entities):
            current = token_entities[i]
            
            # Пропускаем не-сущности
            if current['entity'] == 'O':
                i += 1
                continue
            
            # Начало новой сущности (B-)
            if current['entity'].startswith('B-'):
                # Собираем все токены этой сущности
                entity_tokens = [current]
                j = i + 1
                
                while j < len(token_entities):
                    next_token = token_entities[j]
                    # Продолжение сущности (I- того же типа)
                    if next_token['entity'].startswith('I-') and \
                       self._get_entity_type(next_token['entity']) == self._get_entity_type(current['entity']):
                        entity_tokens.append(next_token)
                        j += 1
                    else:
                        break
                
                # Объединяем токены в сущность
                merged = self._merge_token_entities(entity_tokens, original_text)
                if merged and len(merged['text']) >= 2:  # Игнорируем слишком короткие
                    entities.append(merged)
                
                i = j
            else:
                i += 1
        
        return entities
    
    def get_stats(self) -> Dict[str, Any]:
        """Возвращает статистику работы воркера."""
        stats = self.stats.copy()
        stats['processed_docs_count'] = len(self.stats['processed_docs'])
        stats['processed_docs'] = list(self.stats['processed_docs'])[-100:]  # последние 100 для отладки
        
        if stats['start_time']:
            stats['current_time'] = time.time() - stats['start_time']
        
        return stats