#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Тест WriterWorker
"""

import os
import sys
import time
import logging
import threading
import tempfile
from queue import Queue
from pathlib import Path

# Добавляем путь к проекту
sys.path.insert(0, str(Path(__file__).parent))

from app.v4.config import OutputConfig
from app.v4.checkpoint import CheckpointManager
from app.v4.writer_worker import WriterWorker

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_writer():
    """Тестируем WriterWorker"""
    print("\n" + "=" * 60)
    print("Тест WriterWorker")
    print("=" * 60)
    
    # Создаём временный файл для вывода
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
        output_file = tmp.name
    
    try:
        # Конфиг вывода
        config = OutputConfig(
            format='csv',
            path=output_file,
            delimiter='|',
            include_confidence=True,
            include_positions=False,
            flush_interval=5,
            buffer_size=10
        )
        
        # Очереди
        input_queue = Queue(maxsize=100)
        stop_event = threading.Event()
        
        # Чекпоинт (временный)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as cp_tmp:
            checkpoint_file = cp_tmp.name
        
        checkpoint = CheckpointManager(checkpoint_file, save_interval=3)
        
        # Создаём writer
        writer = WriterWorker(
            config=config,
            input_queue=input_queue,
            checkpoint_manager=checkpoint,
            stop_event=stop_event,
            name="TestWriter"
        )
        
        writer.start()
        
        # Тестовые данные
        test_results = [
            # Документ из одного чанка
            {
                'id': 1001,
                'chunk_id': 0,
                'total_chunks': 1,
                'entities': [
                    {'type': 'LOC', 'text': 'Москва', 'confidence': 0.99},
                    {'type': 'LOC', 'text': 'Россия', 'confidence': 0.98}
                ]
            },
            # Документ из двух чанков
            {
                'id': 1002,
                'chunk_id': 0,
                'total_chunks': 2,
                'entities': [
                    {'type': 'LOC', 'text': 'Санкт-Петербург', 'confidence': 0.97}
                ]
            },
            {
                'id': 1002,
                'chunk_id': 1,
                'total_chunks': 2,
                'entities': [
                    {'type': 'LOC', 'text': 'Нева', 'confidence': 0.95},
                    {'type': 'PER', 'text': 'Петр I', 'confidence': 0.89}
                ]
            },
            # Документ из трёх чанков
            {
                'id': 1003,
                'chunk_id': 0,
                'total_chunks': 3,
                'entities': [
                    {'type': 'LOC', 'text': 'АБРАУ-ДЮРСО', 'confidence': 0.99}
                ]
            },
            {
                'id': 1003,
                'chunk_id': 1,
                'total_chunks': 3,
                'entities': [
                    {'type': 'LOC', 'text': 'Краснодарский край', 'confidence': 0.98}
                ]
            },
            {
                'id': 1003,
                'chunk_id': 2,
                'total_chunks': 3,
                'entities': [
                    {'type': 'LOC', 'text': 'Новороссийск', 'confidence': 0.97},
                    {'type': 'LOC', 'text': 'Абрау', 'confidence': 0.96}
                ]
            }
        ]
        
        # Отправляем результаты
        for result in test_results:
            input_queue.put(result)
            time.sleep(0.1)  # небольшая задержка для имитации реальности
        
        # Даём время на обработку
        logger.info("Обработка...")
        time.sleep(2)
        
        # Останавливаем
        stop_event.set()
        writer.join(timeout=5)
        
        # Проверяем выходной файл
        logger.info(f"Проверка выходного файла: {output_file}")
        
        with open(output_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        logger.info(f"Всего записей: {len(lines) - 1}")  # минус заголовок
        
        # Выводим первые несколько строк
        for i, line in enumerate(lines[:10]):
            logger.info(f"  {i}: {line.strip()}")
        
        # Проверяем статистику
        stats = writer.get_stats()
        logger.info(f"Статистика writer: {stats}")
        
        # Проверяем чекпоинт
        checkpoint.load()
        logger.info(f"Чекпоинт: last_id={checkpoint.last_id}, processed={checkpoint.processed}")
        
        # Проверяем корректность
        assert stats['completed_docs'] == 3, "Должно быть 3 завершённых документа"
        assert stats['total_entities'] == 9, "Должно быть 9 сущностей"
        
        print("\n✅ Тест WriterWorker пройден!")
        
    finally:
        # Очистка
        if os.path.exists(output_file):
            os.unlink(output_file)
        if os.path.exists(checkpoint_file):
            os.unlink(checkpoint_file)


if __name__ == "__main__":
    test_writer()