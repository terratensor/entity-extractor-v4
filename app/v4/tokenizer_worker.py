#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tokenizer Worker - воркер 2 конвейера
Многопоточная токенизация и разбивка текстов на чанки
"""

import logging
import time
from queue import Queue, Empty, Full
from typing import List, Dict, Any, Optional

from transformers import AutoTokenizer

from app.v4.shutdown import StoppableThread

logger = logging.getLogger(__name__)


class TokenizerWorker(StoppableThread):
    """
    Выполняет токенизацию документов и разбивку на чанки.
    Работает в несколько потоков для максимальной загрузки CPU.
    """
    
    def __init__(
        self,
        worker_id: int,
        config,
        input_queue: Queue,
        output_queue: Queue,
        stop_event,
        name: Optional[str] = None
    ):
        """
        Args:
            worker_id: идентификатор воркера (для логирования)
            config: конфигурация модели (ModelConfig)
            input_queue: очередь входных документов (от reader)
            output_queue: очередь выходных чанков (для GPU)
            stop_event: событие остановки
            name: имя потока
        """
        name = name or f"Tokenizer-{worker_id}"
        super().__init__(name=name, stop_event=stop_event)
        
        self.worker_id = worker_id
        self.config = config
        
        self.input_queue = input_queue
        self.output_queue = output_queue
        
        # Токенизатор загружается в потоке
        self.tokenizer = None
        
        # Статистика
        self.stats = {
            'processed_docs': 0,
            'total_chunks': 0,
            'total_tokens': 0,
            'long_docs': 0,  # документы, превысившие max_tokens
            'processing_time': 0,
            'start_time': None
        }
        
        logger.info(f"🤖 {self.name} инициализирован")
    
    def run(self) -> None:
        """Основной цикл токенизации."""
        logger.info(f"🚀 {self.name} запущен")
        
        # Загружаем токенизатор (каждый поток загружает свой)
        self._load_tokenizer()
        
        self.stats['start_time'] = time.time()
        docs_processed = 0
        
        while not self.should_stop():
            try:
                # Получаем документ из очереди
                doc = self.input_queue.get(timeout=1.0)
                
                if doc is None:  # сигнал остановки
                    self.input_queue.task_done()
                    break
                
                # Токенизируем и разбиваем на чанки
                chunks = self._tokenize_document(doc)
                
                # Отправляем чанки в выходную очередь
                for chunk in chunks:
                    # Проверяем место в очереди
                    while not self.should_stop():
                        try:
                            self.output_queue.put(chunk, timeout=0.5)
                            break
                        except Full:
                            logger.debug(f"{self.name}: выходная очередь переполнена, ждём... "
                                        f"(размер: {self.output_queue.qsize()})")
                            if not self.safe_sleep(0.1):
                                break
                    
                    if self.should_stop():
                        break
                
                # Обновляем статистику
                docs_processed += 1
                self.input_queue.task_done()
                
                # Периодическое логирование
                if docs_processed % 100 == 0:
                    elapsed = time.time() - self.stats['start_time']
                    logger.info(f"{self.name}: обработано {docs_processed} док, "
                               f"{self.stats['total_chunks']} чанков, "
                               f"скорость: {docs_processed/elapsed:.1f} док/сек")
                
            except Empty:
                # Нет документов в очереди, продолжаем ждать
                continue
            except Exception as e:
                logger.error(f"{self.name}: ошибка обработки: {e}", exc_info=True)
                # Продолжаем работу со следующим документом
        
        # Завершение работы
        self.stats['processing_time'] = time.time() - self.stats['start_time']
        logger.info(f"📊 {self.name} завершён: обработано {docs_processed} док, "
                   f"создано {self.stats['total_chunks']} чанков")
    
    def _load_tokenizer(self) -> None:
        """Загружает токенизатор модели."""
        try:
            logger.info(f"{self.name}: загрузка токенизатора {self.config.name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.name)
            logger.info(f"{self.name}: токенизатор загружен")
        except Exception as e:
            logger.error(f"{self.name}: ошибка загрузки токенизатора: {e}")
            raise
    
    def _tokenize_document(self, doc: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Токенизирует документ и разбивает на чанки.
        
        Args:
            doc: документ вида {'id': id, 'text': text}
            
        Returns:
            List[Dict]: список чанков с полями:
                - id: исходный ID документа
                - chunk_id: номер чанка
                - total_chunks: всего чанков
                - input_ids: список ID токенов
                - attention_mask: маска внимания
                - token_count: количество токенов в чанке
                - text: исходный текст
        """
        doc_id = doc['id']
        text = doc['text']
        
        # Используем return_overflowing_tokens для автоматической разбивки
        stride = 0
        if self.config.overlap_ratio > 0:
            stride = int(self.config.max_tokens * self.config.overlap_ratio)
        
        # ВАЖНО: добавляем return_offsets_mapping=True
        inputs = self.tokenizer(
            text,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            max_length=self.config.max_tokens,
            stride=stride,
            truncation=True,
            return_tensors=None
        )
        
        chunks = []
        chunk_count = len(inputs['input_ids'])
        
        # Статистика по длинным документам
        if chunk_count > 1:
            self.stats['long_docs'] += 1
        
        for i in range(chunk_count):
            input_ids = inputs['input_ids'][i]
            attention_mask = inputs['attention_mask'][i]
            offsets = inputs['offset_mapping'][i]
            
            # Определяем глобальные позиции чанка
            global_start = None
            global_end = None
            
            for offset in offsets:
                if offset and offset[1] > 0:  # есть ненулевая длина
                    if global_start is None:
                        global_start = offset[0]
                    global_end = offset[1]
            
            # Вырезаем текст чанка
            if global_start is not None and global_end is not None:
                chunk_text = text[global_start:global_end]
            else:
                # Запасной вариант: декодируем токены
                chunk_text = self.tokenizer.decode(input_ids, skip_special_tokens=True)
                global_start = 0
                global_end = len(chunk_text)
            
            chunks.append({
                'id': doc_id,
                'chunk_id': i,
                'total_chunks': chunk_count,
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'text': chunk_text,
                'global_start': global_start,  # позиция в оригинале
                'global_end': global_end,      # конечная позиция
                'token_count': len(input_ids)
            })
            
            # Обновляем статистику
            self.stats['total_chunks'] += 1
            self.stats['total_tokens'] += len(input_ids)
        
        self.stats['processed_docs'] += 1
        
        return chunks
    
    def get_stats(self) -> Dict[str, Any]:
        """Возвращает статистику работы воркера."""
        stats = self.stats.copy()
        if stats['start_time']:
            stats['current_time'] = time.time() - stats['start_time']
        return stats