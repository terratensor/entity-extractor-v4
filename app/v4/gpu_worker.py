#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GPU Worker - воркер 3 конвейера
Выполняет инференс на GPU и извлекает сущности с высоким качеством
Использует проверенный код из прототипа для извлечения сущностей
"""

import logging
import time
from queue import Queue, Empty, Full
from typing import List, Dict, Any, Optional, Tuple

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from app.v4.shutdown import StoppableThread

logger = logging.getLogger(__name__)


class GPUWorker(StoppableThread):
    """
    Выполняет инференс на GPU и извлекает сущности.
    Полная копия логики из прототипа для максимального качества.
    """
    
    # Маппинг BIO-тегов в группы сущностей (из прототипа)
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
        Обрабатывает батч чанков.
        
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
        
        # Инференс (модель сама приведёт эмбеддинги к своему типу)
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Получаем предсказания
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        
        # Для каждого элемента батча извлекаем сущности
        results = []
        
        for i, chunk in enumerate(batch):
            # Получаем предсказания для этого чанка (обрезаем паддинг)
            chunk_len = len(chunk['input_ids'])
            chunk_preds = predictions[i, :chunk_len].cpu().tolist()
            
            # Извлекаем сущности
            entities = self._extract_entities_from_chunk(
                chunk=chunk,
                predictions=chunk_preds,
                tokenizer=self.tokenizer,
                model=self.model
            )
            
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
    
    def _extract_entities_from_chunk(
        self,
        chunk: Dict[str, Any],
        predictions: List[int],
        tokenizer,
        model
    ) -> List[Dict[str, Any]]:
        """
        Извлекает сущности из чанка с использованием оригинального текста.
        ПОЛНАЯ КОПИЯ ЛОГИКИ ИЗ ПРОТОТИПА.
        """
        original_text = chunk.get('original_text', '')
        input_ids = chunk['input_ids']
        
        # Получаем offset_mapping для точных позиций
        inputs = tokenizer(
            original_text,
            return_offsets_mapping=True,
            max_length=self.max_tokens,
            truncation=True
        )
        offsets = inputs['offset_mapping']
        
        tokens = tokenizer.convert_ids_to_tokens(input_ids)
        entities = []
        
        i = 0
        while i < len(tokens):
            token = tokens[i]
            
            if token in ['<s>', '</s>', '<pad>'] or i >= len(offsets):
                i += 1
                continue
            
            label_id = predictions[i]
            label = model.config.id2label.get(label_id, 'O')
            
            if label == 'O' or label.startswith('I-'):
                i += 1
                continue
            
            if label.startswith('B-'):
                start_idx = i
                entity_tokens = [token]
                j = i + 1
                
                while j < len(tokens) and j < len(offsets):
                    next_token = tokens[j]
                    if next_token in ['<s>', '</s>', '<pad>']:
                        j += 1
                        continue
                    
                    next_label_id = predictions[j]
                    next_label = model.config.id2label.get(next_label_id, 'O')
                    
                    if next_label.startswith('I-') and next_label[2:] == label[2:]:
                        entity_tokens.append(next_token)
                        j += 1
                    else:
                        break
                
                # Извлекаем текст по оригинальным позициям
                if offsets and start_idx < len(offsets) and j-1 < len(offsets):
                    start_pos = offsets[start_idx][0]
                    end_pos = offsets[j-1][1]
                    entity_text = original_text[start_pos:end_pos]
                    
                    # Очистка
                    entity_text = ' '.join(entity_text.split())
                else:
                    # Запасной вариант
                    entity_text = self._merge_tokens(entity_tokens)
                
                if entity_text and len(entity_text) >= 2:
                    entity_type = label[2:] if label.startswith('B-') else label
                    if entity_type not in self.ENTITY_GROUP_MAP.values():
                        entity_type = 'MISC'
                    
                    entities.append({
                        'text': entity_text,
                        'type': entity_type,
                        'confidence': 0.9,  # TODO: вычислять реальную уверенность
                        'chunk_id': chunk['chunk_id'],
                        'start': start_pos if self.include_positions else None,
                        'end': end_pos if self.include_positions else None
                    })
                
                i = j
            else:
                i += 1
        
        return entities
    
    def _merge_tokens(self, tokens: List[str]) -> str:
        """
        Склеивает токены в слово с учётом особенностей XLM-RoBERTa.
        
        Args:
            tokens: список токенов (например, ['▁Моск', 'ва'])
            
        Returns:
            str: склеенное слово
        """
        text_parts = []
        
        for token in tokens:
            # XLM-RoBERTa использует ▁ для обозначения начала слова
            word = token.replace('▁', '')
            
            if text_parts and not word.startswith(("'", "-", ".", ",", ")", "(", ":", ";")):
                # Добавляем пробел между словами, если это не пунктуация
                text_parts.append(' ')
            
            text_parts.append(word)
        
        # Объединяем и чистим лишние пробелы
        full_text = ''.join(text_parts)
        full_text = ' '.join(full_text.split())
        
        return full_text
    
    def get_stats(self) -> Dict[str, Any]:
        """Возвращает статистику работы воркера."""
        stats = self.stats.copy()
        stats['processed_docs_count'] = len(self.stats['processed_docs'])
        stats['processed_docs'] = list(self.stats['processed_docs'])[-100:]  # последние 100 для отладки
        
        if stats['start_time']:
            stats['current_time'] = time.time() - stats['start_time']
        
        return stats