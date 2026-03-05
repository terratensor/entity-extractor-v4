#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Тест ManticoreReader
"""

import sys
import time
import logging
import threading
from queue import Queue
from pathlib import Path

# Добавляем путь к проекту
sys.path.insert(0, str(Path(__file__).parent))

from app.v4.config import SourceConfig
from app.v4.reader import ManticoreReader

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_reader():
    """Тестируем ManticoreReader с реальной БД"""
    print("\n" + "=" * 60)
    print("Тест ManticoreReader")
    print("=" * 60)
    
    # Создаём конфиг для теста
    config = SourceConfig(
        host='localhost',
        port=9306,
        table='library2026',
        batch_size=100
    )
    
    # Создаём очередь и событие остановки
    output_queue = Queue(maxsize=1000)
    stop_event = threading.Event()
    
    # Создаём reader
    reader = ManticoreReader(
        config=config,
        checkpoint_last_id=0,
        output_queue=output_queue,
        stop_event=stop_event,
        name="TestReader"
    )
    
    # Запускаем в отдельном потоке
    reader.start()
    
    # Даём поработать 5 секунд
    logger.info("Reader работает 5 секунд...")
    
    # Собираем результаты
    docs_received = 0
    start_time = time.time()
    
    try:
        while time.time() - start_time < 5:
            try:
                doc = output_queue.get(timeout=0.5)
                docs_received += 1
                if docs_received <= 5:
                    logger.info(f"Пример документа: ID={doc['id']}, "
                               f"текст: {doc['text'][:50]}...")
                output_queue.task_done()
            except:
                pass
        
        logger.info(f"Получено документов: {docs_received}")
        
    finally:
        # Останавливаем reader
        stop_event.set()
        reader.join(timeout=5)
    
    # Выводим статистику
    stats = reader.get_stats()
    logger.info(f"Статистика reader: {stats}")
    
    if docs_received > 0:
        print("\n✅ Тест пройден: документы получены")
        return True
    else:
        print("\n❌ Тест не пройден: не получено ни одного документа")
        return False


if __name__ == "__main__":
    test_reader()