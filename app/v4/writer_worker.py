#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Writer Worker - воркер 4 конвейера
Собирает чанки документов, объединяет сущности и записывает в CSV
"""

import logging
import time
import csv
import os
from queue import Queue, Empty
from collections import defaultdict
from typing import List, Dict, Any, Optional, Set
from pathlib import Path

from app.v4.shutdown import StoppableThread
from app.v4.expansion import WordExpander

logger = logging.getLogger(__name__)


class WriterWorker(StoppableThread):
    """
    Собирает результаты от GPU, группирует по документам и записывает в CSV.
    Поддерживает буферизацию для повышения производительности.
    """
    
    def __init__(
        self,
        config,  # это OutputConfig
        input_queue: Queue,
        checkpoint_manager,
        stop_event,
        verbose: bool = False,
        name: str = "WriterWorker"
    ):
        """
        Args:
            config: конфигурация вывода (OutputConfig)
            input_queue: очередь входных результатов (от GPU)
            checkpoint_manager: менеджер чекпоинтов
            stop_event: событие остановки
            name: имя потока
        """
        super().__init__(name=name, stop_event=stop_event)
        
        self.config = config
        self.input_queue = input_queue
        self.checkpoint = checkpoint_manager
        self.verbose = verbose
        
        # Параметры вывода
        self.output_path = Path(config.path)
        self.delimiter = config.delimiter
        self.include_confidence = config.include_confidence
        self.include_positions = config.include_positions
        self.flush_interval = config.flush_interval
        self.buffer_size = config.buffer_size
        
        # [ЭКСПЕРИМЕНТАЛЬНЫЙ ПАРАМЕТР] Расширение слов
        self.enable_expansion = getattr(config, 'enable_expansion', False)
        self.expander = None
        
        if self.enable_expansion:
            expansion_params = getattr(config, 'expansion_params', {})
            # Добавляем verbose из настроек writer
            expansion_params['verbose'] = self.verbose
            self.expander = WordExpander(expansion_params)
            logger.info(f"🤖 WordExpander инициализирован (включено расширение слов)")
            logger.info(f"   Параметры: {expansion_params}")
        
        # Буферы
        self.buffer = []  # теперь хранит списки значений, а не строки
        self.pending_docs = {}
        self.completed_docs = set()
        
        # Для текстов документов (нужно для расширения)
        self.doc_texts = {}
        self.text_cleanup_threshold = 1000
        self.doc_text_parts = {}  # doc_id -> {chunk_id: text}
        
        # Для отслеживания прогресса
        self.last_flush_time = time.time()
        self.last_checkpoint_save = 0
        self.bytes_written = 0
        
        # Статистика
        self.stats = {
            'processed_chunks': 0,
            'processed_docs': 0,
            'completed_docs': 0,
            'total_entities': 0,
            'entities_by_type': defaultdict(int),
            'write_time': 0,
            'buffer_flushes': 0,
            'start_time': None
        }
        
        # Создаём директорию для выходного файла
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"🤖 {self.name} инициализирован, выходной файл: {self.output_path}")
        logger.info(f"   include_positions: {self.include_positions}, enable_expansion: {self.enable_expansion}")
    
    def run(self) -> None:
        """Основной цикл обработки результатов."""
        logger.info(f"🚀 {self.name} запущен")
        
        self.stats['start_time'] = time.time()
        
        # Открываем файл для записи
        file_handle = self._open_file()
        
        try:
            while not self.should_stop() or not self.input_queue.empty():
                try:
                    # Получаем результат от GPU
                    result = self.input_queue.get(timeout=1.0)
                    
                    # Обрабатываем результат
                    self._process_result(result)
                    
                    # Проверяем необходимость сброса буфера
                    self._check_flush(file_handle)
                    
                    # Проверяем необходимость сохранения чекпоинта
                    self._check_checkpoint()
                    
                except Empty:
                    # Нет данных, проверяем таймаут для сброса
                    self._check_flush(file_handle)
                    continue
                except Exception as e:
                    logger.error(f"{self.name}: ошибка обработки: {e}", exc_info=True)
            
            # Финальный сброс буфера
            self._flush_buffer(file_handle)
            
            # Сохраняем финальный чекпоинт
            if self.completed_docs:
                last_id = max(self.completed_docs)
                self.checkpoint.save(
                    last_id=last_id,
                    processed=len(self.completed_docs),
                    stats=self._get_checkpoint_stats()
                )
            
        finally:
            file_handle.close()
            logger.info(f"{self.name}: файл закрыт")
        
        # Итоговая статистика
        elapsed = time.time() - self.stats['start_time']
        logger.info(
            f"📊 {self.name} завершён: обработано {self.stats['processed_docs']} док, "
            f"{self.stats['completed_docs']} завершено, "
            f"{self.stats['total_entities']} сущностей, "
            f"записей в буфере: {len(self.buffer)}, "
            f"время: {elapsed:.1f} сек"
        )
    
    def _open_file(self):
        """
        Открывает файл для записи, создает csv.writer и добавляет заголовок если файл новый.
        
        Returns:
            file object: открытый файл
        """
        file_exists = self.output_path.exists()
        
        # Открываем в режиме добавления
        f = open(self.output_path, 'a', encoding='utf-8', newline='')
        
        # Если файл новый, пишем заголовок через csv.writer
        if not file_exists:
            writer = csv.writer(f, delimiter=self.delimiter,
                               quoting=csv.QUOTE_MINIMAL,
                               doublequote=True)
            header = ['doc_id', 'entity_type', 'entity_text']
            if self.include_confidence:
                header.append('confidence')
            if self.include_positions:
                header.extend(['start_pos', 'end_pos'])
            
            writer.writerow(header)
            f.flush()
            logger.info(f"{self.name}: создан новый файл с заголовком")
        
        return f
    
    def _process_result(self, result: Dict[str, Any]) -> None:
        """
        Обрабатывает один результат от GPU.
        
        Args:
            result: результат от GPUWorker с полями:
                - id: ID документа
                - chunk_id: номер чанка
                - total_chunks: всего чанков
                - entities: список сущностей в чанке
        """
        doc_id = result['id']
        chunk_id = result['chunk_id']
        total_chunks = result['total_chunks']
        entities = result.get('entities', [])
        
        # Сохраняем текст каждого чанка
        if 'text' in result:
            if doc_id not in self.doc_text_parts:
                self.doc_text_parts[doc_id] = {}
            self.doc_text_parts[doc_id][chunk_id] = result['text']
            
            if self.verbose:
                logger.warning(f"🔥 Сохранен чанк {chunk_id}/{total_chunks} для doc {doc_id}, длина: {len(result['text'])}")
            
            # Если получили все чанки - собираем полный текст
            if len(self.doc_text_parts[doc_id]) == total_chunks:
                # Собираем в порядке чанков
                full_text = ''
                for i in range(total_chunks):
                    if i in self.doc_text_parts[doc_id]:
                        full_text += self.doc_text_parts[doc_id][i]
                self._store_doc_text(doc_id, full_text)
                
                if self.verbose:
                    logger.warning(f"🔥 Собран ПОЛНЫЙ текст для doc {doc_id}, длина: {len(full_text)}")
                
                # Очищаем части, они больше не нужны
                del self.doc_text_parts[doc_id]
        
        # Если документ состоит из одного чанка - сразу готов к записи
        if total_chunks == 1:
            self._write_entities(doc_id, entities)
            self.completed_docs.add(doc_id)
            self.stats['completed_docs'] += 1
            return
        
        # Многочанковый документ - сохраняем в pending
        if doc_id not in self.pending_docs:
            self.pending_docs[doc_id] = {}
        
        self.pending_docs[doc_id][chunk_id] = entities
        
        # Проверяем, все ли чанки получены
        if len(self.pending_docs[doc_id]) == total_chunks:
            # Собираем все сущности в порядке чанков
            all_entities = []
            for i in range(total_chunks):
                if i in self.pending_docs[doc_id]:
                    all_entities.extend(self.pending_docs[doc_id][i])
            
            # Записываем
            self._write_entities(doc_id, all_entities)
            
            # Очищаем pending
            del self.pending_docs[doc_id]
            self.completed_docs.add(doc_id)
            self.stats['completed_docs'] += 1
    
    def _write_entities(self, doc_id: int, entities: List[Dict]) -> None:
        """
        Добавляет сущности в буфер для записи с опциональным расширением.
        В буфер добавляются списки значений, а не готовая строка.
        """
        # Получаем полный текст документа
        doc_text = self.doc_texts.get(doc_id, "")
        
        if self.verbose:
            logger.warning(f"🔥 _write_entities для doc {doc_id}, entities: {len(entities)}, текст длина: {len(doc_text)}")
        
        for i, entity in enumerate(entities):
            if self.verbose:
                logger.warning(f"🔥 entity {i}: type={entity.get('type')}, text='{entity.get('text')}'")
                logger.warning(f"🔥   has positions: {'positions' in entity}")
            
            processed_entity = entity
            
            # [ЭКСПЕРИМЕНТАЛЬНЫЙ КОД] Расширение слов
            if self.enable_expansion and self.expander and 'positions' in entity:
                if self.verbose:
                    logger.warning(f"🔥   ПОПЫТКА РАСШИРЕНИЯ для '{entity.get('text')}'")
                processed_entity = self.expander.expand_entity(entity, doc_text)
                
                if processed_entity != entity and self.verbose:
                    logger.warning(f"🔥   РАСШИРЕНО: '{entity.get('text')}' -> '{processed_entity.get('text')}'")
            
            # ======================================================================
            # ФОРМИРОВАНИЕ СПИСКА ЗНАЧЕНИЙ (НЕ СТРОКИ!)
            # ======================================================================
            row = [
                str(doc_id),
                processed_entity.get('type', 'MISC'),
                processed_entity.get('text', '')
            ]
            
            if self.include_confidence:
                row.append(str(round(processed_entity.get('confidence', 0.5), 4)))
            
            if self.include_positions:
                if 'positions' in processed_entity and processed_entity['positions']:
                    first_pos = processed_entity['positions'][0]
                    last_pos = processed_entity['positions'][-1]
                    row.append(str(first_pos.get('start', 0)))
                    row.append(str(last_pos.get('end', 0)))
                else:
                    row.extend(['0', '0'])
            
            # Добавляем СПИСОК в буфер, csv.writer сам обработает экранирование
            self.buffer.append(row)
            self.stats['total_entities'] += 1
        
        # Периодический вывод статистики расширений
        if (self.enable_expansion and self.expander and 
            self.stats['completed_docs'] % 100 == 0 and self.stats['completed_docs'] > 0):
            exp_stats = self.expander.get_stats()
            expanded = exp_stats.get('expanded_total', 0)
            attempts = exp_stats.get('attempts', 0)
            percent = exp_stats.get('expand_percent', 0)
            
            logger.info(f"📊 Статистика расширений: попыток={attempts}, "
                       f"расширено={expanded} ({percent}%)")
    
    def _check_flush(self, file_handle) -> None:
        """Проверяет необходимость сброса буфера на диск."""
        should_flush = (
            len(self.buffer) >= self.buffer_size or
            time.time() - self.last_flush_time >= self.flush_interval
        )
        
        if should_flush and self.buffer:
            self._flush_buffer(file_handle)
    
    def _store_doc_text(self, doc_id: int, text: str):
        """Сохраняет текст документа для последующего расширения."""
        self.doc_texts[doc_id] = text
        
        # Очистка старых записей (чтобы не разрасталась память)
        if len(self.doc_texts) > self.text_cleanup_threshold:
            # Удаляем записи с id меньше текущего - 1000
            min_keep = doc_id - 1000
            self.doc_texts = {k: v for k, v in self.doc_texts.items() if k > min_keep}

    def _check_checkpoint(self) -> None:
        """Проверяет необходимость сохранения чекпоинта."""
        if not self.completed_docs:
            return
        
        current_completed = len(self.completed_docs)
        
        # Сохраняем каждые N документов или по времени
        if (current_completed - self.last_checkpoint_save >= 
            getattr(self.checkpoint, 'save_interval', 1000)):
            
            last_id = max(self.completed_docs)
            self.checkpoint.save(
                last_id=last_id,
                processed=current_completed,
                stats=self._get_checkpoint_stats()
            )
            self.last_checkpoint_save = current_completed
    
    def _get_checkpoint_stats(self) -> Dict:
        """Возвращает статистику для сохранения в чекпоинт."""
        return {
            'processed': self.stats['completed_docs'],
            'total_entities': self.stats['total_entities'],
            'entities_by_type': dict(self.stats['entities_by_type']),
            'bytes_written': self.bytes_written
        }
    
    def _flush_buffer(self, file_handle) -> None:
        """Принудительный сброс буфера в указанный файл с использованием csv.writer."""
        if not self.buffer:
            return
        
        # Создаем writer для этого файла с правильными параметрами экранирования
        writer = csv.writer(file_handle, delimiter=self.delimiter,
                           quoting=csv.QUOTE_MINIMAL,
                           doublequote=True,
                           escapechar=None)
        
        # Записываем все строки из буфера
        writer.writerows(self.buffer)
        file_handle.flush()
        os.fsync(file_handle.fileno())
        
        # Обновляем статистику
        current_pos = file_handle.tell()
        self.bytes_written = current_pos
        self.buffer.clear()
        self.last_flush_time = time.time()
        self.stats['buffer_flushes'] += 1
        
        logger.debug(f"{self.name}: буфер сброшен, записей: {len(self.buffer)}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Возвращает статистику работы воркера."""
        stats = self.stats.copy()
        stats['entities_by_type'] = dict(self.stats['entities_by_type'])
        stats['pending_docs'] = len(self.pending_docs)
        stats['buffer_size'] = len(self.buffer)
        
        # Явно добавляем total_entities для совместимости
        stats['total_entities'] = self.stats.get('total_entities', 0)
        
        if stats['start_time']:
            stats['current_time'] = time.time() - stats['start_time']
        
        return stats