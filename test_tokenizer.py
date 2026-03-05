#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Тест TokenizerWorker
"""

import sys
import time
import logging
import threading
from queue import Queue
from pathlib import Path

# Добавляем путь к проекту
sys.path.insert(0, str(Path(__file__).parent))

from app.v4.config import ModelConfig
from app.v4.tokenizer_worker import TokenizerWorker

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_tokenizer():
    """Тестируем TokenizerWorker"""
    print("\n" + "=" * 60)
    print("Тест TokenizerWorker")
    print("=" * 60)
    
    # Конфиг модели
    config = ModelConfig(
        name="Davlan/xlm-roberta-large-ner-hrl",
        max_tokens=512,
        overlap_ratio=0.0,
        min_confidence=0.5
    )
    
    # Очереди
    input_queue = Queue(maxsize=100)
    output_queue = Queue(maxsize=1000)
    stop_event = threading.Event()
    
    # Тестовые документы
    test_docs = [
        {"id": 1, "text": "Короткий текст"},
        {"id": 2, "text": "А" * 1000},  # 1000 символов
        {"id": 3, "text": "X" * 10000},  # 10000 символов
        {"id": 4, "text": "Тест с дефисом и пробелами. Москва-река, Санкт-Петербург"},
        {"id": 5, "text": """АБРАУ-ДЮРСО, посёлок городского типа (с 1948) в России, 
                           в Краснодарском крае, в 14 км к 3. от Новороссийска, 
                           на берегу озера Абрау."""}
    ]
    
    # Заполняем входную очередь
    for doc in test_docs:
        input_queue.put(doc)
    
    # Создаём и запускаем воркер
    worker = TokenizerWorker(
        worker_id=0,
        config=config,
        input_queue=input_queue,
        output_queue=output_queue,
        stop_event=stop_event,
        name="TestTokenizer"
    )
    
    worker.start()
    
    # Ждём обработки
    worker.join(timeout=30)
    
    # Собираем результаты
    chunks = []
    while not output_queue.empty():
        chunks.append(output_queue.get())
    
    # Выводим статистику
    logger.info(f"Всего чанков: {len(chunks)}")
    
    # Группируем по документам
    docs_chunks = {}
    for chunk in chunks:
        doc_id = chunk['id']
        if doc_id not in docs_chunks:
            docs_chunks[doc_id] = []
        docs_chunks[doc_id].append(chunk)
    
    # Проверяем результаты
    for doc_id, doc_chunks in docs_chunks.items():
        logger.info(f"Документ {doc_id}: {len(doc_chunks)} чанков")
        
        # Проверяем, что чанки идут по порядку
        chunk_ids = [c['chunk_id'] for c in doc_chunks]
        assert chunk_ids == list(range(len(doc_chunks))), f"Неправильный порядок чанков: {chunk_ids}"
        
        # Проверяем total_chunks
        for chunk in doc_chunks:
            assert chunk['total_chunks'] == len(doc_chunks), "total_chunks не совпадает"
        
        # Выводим информацию о первом чанке
        if doc_chunks:
            first = doc_chunks[0]
            logger.info(f"  Первый чанк: {first['token_count']} токенов")
            logger.info(f"  input_ids[:10]: {first['input_ids'][:10]}")
    
    # Выводим статистику воркера
    stats = worker.get_stats()
    logger.info(f"Статистика воркера: {stats}")
    
    # Проверяем, что все документы обработаны
    assert stats['processed_docs'] == len(test_docs), "Не все документы обработаны"
    
    print("\n✅ Тест TokenizerWorker пройден!")


if __name__ == "__main__":
    test_tokenizer()