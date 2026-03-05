#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Тест GPUWorker с реальным инференсом
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
from app.v4.gpu_worker import GPUWorker

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SimpleGPUConfig:
    """Простая конфигурация для GPU"""
    def __init__(self, model, device_id, batch_size, precision):
        self.model = model
        self.device_id = device_id
        self.batch_size = batch_size
        self.precision = precision


def test_gpu_worker():
    """Тестируем GPUWorker на реальных данных"""
    print("\n" + "=" * 60)
    print("Тест GPUWorker")
    print("=" * 60)
    
    # Конфиг модели
    model_config = ModelConfig(
        name="Davlan/xlm-roberta-large-ner-hrl",
        max_tokens=512,
        overlap_ratio=0.0,
        min_confidence=0.5,
        include_positions=False
    )
    
    # Очереди
    tokenizer_queue = Queue(maxsize=100)
    gpu_queue = Queue(maxsize=100)
    writer_queue = Queue(maxsize=100)
    stop_event = threading.Event()
    
    # Тестовые документы (взяты из реального примера)
    test_docs = [
        {"id": 1001, "text": "Москва - столица России. Санкт-Петербург основан Петром I."},
        {"id": 1002, "text": "АБРАУ-ДЮРСО, посёлок городского типа в Краснодарском крае."},
        {"id": 1003, "text": "Париж, Лондон, Берлин - крупные европейские столицы."},
        {"id": 1004, "text": "Река Волга впадает в Каспийское море."},
        {"id": 1005, "text": """АБРАУ-ДЮРСО, посёлок городского типа (с 1948) в России, 
                               в Краснодарском крае, в 14 км к 3. от Новороссийска, 
                               на берегу озера Абрау. Нас. 2,8 т. ч. Виноградники, 
                               производство высококачественных шампанских и столовых вин."""}
    ]
    
    # Заполняем очередь для токенизатора
    for doc in test_docs:
        tokenizer_queue.put(doc)
    
    # Запускаем токенизатор
    tokenizer = TokenizerWorker(
        worker_id=0,
        config=model_config,
        input_queue=tokenizer_queue,
        output_queue=gpu_queue,
        stop_event=stop_event,
        name="TestTokenizer"
    )
    tokenizer.start()
    
    # Запускаем GPUWorker
    gpu_config = SimpleGPUConfig(
        model=model_config,
        device_id=0,
        batch_size=4,
        precision="float16"
    )
    
    gpu = GPUWorker(
        device_id=0,
        config=gpu_config,
        input_queue=gpu_queue,
        output_queue=writer_queue,
        stop_event=stop_event,
        name="TestGPU"
    )
    gpu.start()
    
    # Даем время на обработку
    logger.info("Обработка...")
    time.sleep(10)
    
    # Останавливаем
    stop_event.set()
    tokenizer.join(timeout=5)
    gpu.join(timeout=5)
    
    # Собираем результаты
    results = []
    while not writer_queue.empty():
        results.append(writer_queue.get())
    
    logger.info(f"Получено результатов: {len(results)}")
    
    # Группируем по документам
    docs_entities = {}
    for result in results:
        doc_id = result['id']
        if doc_id not in docs_entities:
            docs_entities[doc_id] = []
        docs_entities[doc_id].extend(result['entities'])
    
    # Выводим найденные сущности
    for doc_id, entities in docs_entities.items():
        logger.info(f"\nДокумент {doc_id}:")
        for entity in entities:
            logger.info(f"  {entity['type']}: {entity['text']}")
    
    # Статистика GPU
    gpu_stats = gpu.get_stats()
    logger.info(f"\nСтатистика GPU: {gpu_stats}")
    
    # Проверяем, что нашли хотя бы какие-то сущности
    total_entities = sum(len(entities) for entities in docs_entities.values())
    assert total_entities > 0, "Не найдено ни одной сущности!"
    
    print("\n✅ Тест GPUWorker пройден!")


if __name__ == "__main__":
    test_gpu_worker()