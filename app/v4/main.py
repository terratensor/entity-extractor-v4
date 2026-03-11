#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Главный файл конвейерной обработки v4
Запускает все воркеры и управляет их взаимодействием
"""

import os
import sys
import time
import logging
import argparse
import signal
from queue import Queue
from pathlib import Path
from typing import List, Dict, Any, Optional

# Добавляем путь к проекту
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.v4.config import load_config, Config, GPUDeviceConfig
from app.v4.reader import ManticoreReader
from app.v4.tokenizer_worker import TokenizerWorker
from app.v4.gpu_worker import GPUWorker
from app.v4.writer_worker import WriterWorker
from app.v4.checkpoint import CheckpointManager
from app.v4.shutdown import GracefulShutdown
from app.v4.version import VERSION_NAME

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('batch_processor_v4.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class PipelineV4:
    """
    Оркестратор конвейера v4.
    Запускает все воркеры и управляет их жизненным циклом.
    """
    
    def __init__(self, config_path: str, limit: Optional[int] = None):
        """
        Args:
            config_path: путь к конфигурационному файлу
        """
        self.limit = limit
        self.config_path = config_path
        self.config: Config = load_config(config_path)
        
        # Система остановки
        self.shutdown = GracefulShutdown()
        
        # Менеджер чекпоинтов
        self.checkpoint = CheckpointManager(
            self.config.checkpoint.file,
            save_interval=self.config.checkpoint.save_interval
        )
        self.checkpoint.start_session()  # отмечаем начало сессии
        
        # Очереди
        self.queue1 = Queue(maxsize=self.config.queues.queue1_size)  # сырые документы
        self.queue2 = Queue(maxsize=self.config.queues.queue2_size)  # токенизированные чанки
        self.queue3 = Queue(maxsize=self.config.queues.queue3_size)  # результаты инференса
        
        # Воркеры
        self.reader = None
        self.tokenizers: List[TokenizerWorker] = []
        self.gpu_workers: List[GPUWorker] = []
        self.writer = None
        
        # Статистика
        self.start_time = None
        self.stats = {
            'total_docs': 0,
            'total_chunks': 0,
            'total_entities': 0,
            'entities_by_type': {'LOC': 0, 'PER': 0, 'ORG': 0, 'MISC': 0}
        }
        
        logger.info("=" * 60)
        logger.info("🚀 ИНИЦИАЛИЗАЦИЯ КОНВЕЙЕРА v{VERSION_NAME}")
        logger.info("=" * 60)
        logger.info(f"Конфиг: {config_path}")
        logger.info(f"Лимит документов: {limit if limit else 'без лимита'}")
        logger.info(f"GPU устройств: {len(self.config.gpu_devices)}")
        
        # Детальная информация по каждому GPU
        for i, gpu in enumerate(self.config.gpu_devices):
            logger.info(f"  GPU {i}: device_id={gpu.device_id}, "
                    f"batch_size={gpu.batch_size}, "
                    f"precision={gpu.precision}")
        
        logger.info(f"Токенизаторов: {self.config.tokenizer.num_workers} "
                    f"(на {self.config.tokenizer.num_workers} потоках CPU)")
        logger.info(f"Очереди: q1={self.config.queues.queue1_size}, "
                    f"q2={self.config.queues.queue2_size}, "
                    f"q3={self.config.queues.queue3_size}")
        
        # Добавим информацию о модели
        logger.info(f"Модель: {self.config.model.name}")
        logger.info(f"max_tokens: {self.config.model.max_tokens}, "
                    f"overlap: {self.config.model.overlap_ratio}, "
                    f"min_confidence: {self.config.model.min_confidence}")
        
        # Информация о выходном файле
        logger.info(f"Выходной файл: {self.config.output.path}")
        logger.info(f"Формат: {self.config.output.format}, "
                    f"с confidence: {self.config.output.include_confidence}")
        
        logger.info("=" * 60)
    
    def _create_gpu_config(self, device_config: GPUDeviceConfig):
        """
        Создаёт конфиг для GPUWorker.
        
        Args:
            device_config: конфигурация устройства
            
        Returns:
            object: конфиг для GPUWorker
        """
        class GPUConfig:
            def __init__(self, model, device_id, batch_size, precision):
                self.model = model
                self.device_id = device_id
                self.batch_size = batch_size
                self.precision = precision
        
        return GPUConfig(
            model=self.config.model,
            device_id=device_config.device_id,
            batch_size=device_config.batch_size,
            precision=device_config.precision
        )
    
    def start(self) -> None:
        """Запускает все воркеры."""
        self.start_time = time.time()
        
        # Получаем начальный ID из чекпоинта
        start_id = self.checkpoint.get_start_id()
        processed_before = self.checkpoint.get_processed_count()
        
        logger.info(f"📂 Начало обработки с ID > {start_id} (ранее обработано: {processed_before})")
        
        # Получаем общее количество документов для информации
        temp_reader = ManticoreReader(
            config=self.config.source,
            checkpoint_last_id=0,
            output_queue=Queue(),
            stop_event=self.shutdown.stop_event
        )
        total_docs = temp_reader.get_total_count()
        temp_reader._close()
        logger.info(f"📊 Всего документов в БД: {total_docs}")
        
        # 1. Запускаем Reader
        self.reader = ManticoreReader(
            config=self.config.source,
            checkpoint_last_id=start_id,
            output_queue=self.queue1,
            stop_event=self.shutdown.stop_event,
            name="Reader"
        )
        self.reader.start()
        logger.info("✅ Reader запущен")
        
        # 2. Запускаем Tokenizer-воркеры
        for i in range(self.config.tokenizer.num_workers):
            tokenizer = TokenizerWorker(
                worker_id=i,
                config=self.config.model,
                input_queue=self.queue1,
                output_queue=self.queue2,
                stop_event=self.shutdown.stop_event,
                name=f"Tokenizer-{i}"
            )
            tokenizer.start()
            self.tokenizers.append(tokenizer)
        logger.info(f"✅ {len(self.tokenizers)} Tokenizer-воркеров запущено")
        
        # 3. Запускаем GPU-воркеры (по одному на каждое устройство)
        for i, gpu_config in enumerate(self.config.gpu_devices):
            gpu_worker = GPUWorker(
                device_id=gpu_config.device_id,
                config=self._create_gpu_config(gpu_config),
                input_queue=self.queue2,
                output_queue=self.queue3,
                stop_event=self.shutdown.stop_event,
                name=f"GPU-{gpu_config.device_id}"
            )
            gpu_worker.start()
            self.gpu_workers.append(gpu_worker)
        logger.info(f"✅ {len(self.gpu_workers)} GPU-воркеров запущено")
        
        # 4. Запускаем Writer
        self.writer = WriterWorker(
            config=self.config.output,
            input_queue=self.queue3,
            checkpoint_manager=self.checkpoint,
            stop_event=self.shutdown.stop_event,
            verbose=self.config.logging.verbose,  # NEW
            name="Writer"
        )
        self.writer.start()
        logger.info("✅ Writer запущен")
        
        # Регистрируем функцию остановки
        self.shutdown.register_callback(self._shutdown_callback)
        
        logger.info("🎯 Все воркеры запущены, начинаем обработку...")
        
        # Основной цикл мониторинга
        self._monitor_loop()
    
    def _monitor_loop(self) -> None:
        """Цикл мониторинга и вывода статистики."""
        last_stats_time = time.time()
        
        try:
            while not self.shutdown.is_set():
                time.sleep(5)
                
                # Проверка лимита - берём из writer.stats['completed_docs']
                if self.limit and self.writer:
                    completed = self.writer.stats.get('completed_docs', 0)
                    if completed >= self.limit:
                        logger.info(f"✅ Достигнут лимит в {self.limit} документов (завершено: {completed}), останавливаемся")
                        self.shutdown.stop()
                        break
                
                # Проверяем, живы ли воркеры
                if not self.reader.is_alive():
                    logger.warning("⚠️ Reader остановился, проверяем очередь...")
                    if self.queue1.empty() and self.queue2.empty() and self.queue3.empty():
                        logger.info("✅ Все очереди пусты, завершаем работу")
                        self.shutdown.stop()
                        break
                
                # Выводим статистику каждые 30 секунд
                if time.time() - last_stats_time >= 30:
                    self._print_stats()
                    last_stats_time = time.time()
                    
        except KeyboardInterrupt:
            logger.warning("\n⚠️ Получен Ctrl+C, останавливаем...")
            self.shutdown.stop()
            
    def _print_stats(self) -> None:
        """Выводит текущую статистику с разделением на чанки и документы."""
        elapsed = time.time() - self.start_time
        
        # Собираем статистику от всех воркеров
        reader_stats = self.reader.get_stats() if self.reader else {}
        writer_stats = self.writer.get_stats() if self.writer else {}
        
        # Суммарная статистика по токенизаторам
        tokenizer_docs = sum(w.get_stats()['processed_docs'] for w in self.tokenizers)
        tokenizer_chunks = sum(w.get_stats()['total_chunks'] for w in self.tokenizers)
        tokenizer_tokens = sum(w.get_stats()['total_tokens'] for w in self.tokenizers)
        
        # Суммарная статистика по GPU
        gpu_chunks = sum(w.get_stats()['processed_chunks'] for w in self.gpu_workers)
        gpu_entities = sum(w.get_stats()['entities_found'] for w in self.gpu_workers)
        gpu_docs = len(set().union(*[w.get_stats()['processed_docs'] for w in self.gpu_workers]))
        
        # Размеры очередей
        queue_sizes = {
            'q1': self.queue1.qsize(),
            'q2': self.queue2.qsize(),
            'q3': self.queue3.qsize()
        }
        
        # Расчет скоростей
        chunks_per_sec = gpu_chunks / elapsed if elapsed > 0 else 0
        docs_per_sec = writer_stats.get('completed_docs', 0) / elapsed if elapsed > 0 else 0
        
        logger.info("-" * 60)
        logger.info(f"📊 СТАТИСТИКА (прошло {elapsed:.1f} сек):")
        logger.info(f"  Очереди: q1={queue_sizes['q1']}, q2={queue_sizes['q2']}, q3={queue_sizes['q3']}")
        logger.info(f"  Reader: прочитано {reader_stats.get('read_docs', 0):,} док")
        logger.info(f"  Tokenizer: {tokenizer_docs:,} док -> {tokenizer_chunks:,} чанков ({tokenizer_tokens:,} токенов)")
        logger.info(f"  GPU: {gpu_chunks:,} чанков, {gpu_entities:,} сущностей, уникальных док: {gpu_docs:,}")
        logger.info(f"  Writer: завершено {writer_stats.get('completed_docs', 0):,} док, "
                    f"записано {writer_stats.get('total_entities', 0):,} сущностей")
        logger.info(f"  ⚡ Скорость чанков: {chunks_per_sec:.1f} чанков/сек")
        logger.info(f"  ⚡ Скорость документов: {docs_per_sec:.1f} док/сек")
        logger.info("-" * 60)
    
    def _shutdown_callback(self) -> None:
        """Вызывается при остановке для корректного завершения."""
        logger.info("🛑 Завершение работы воркеров...")
        
        # Даём время на завершение текущих задач
        time.sleep(2)
    
    def wait_for_completion(self) -> None:
        """Ожидает завершения всех воркеров."""
        logger.info("⏳ Ожидание завершения воркеров...")
        
        # Ждём writer (он может дольше всех)
        if self.writer:
            self.writer.join(timeout=30)
        
        # Останавливаем остальных
        if self.reader:
            self.reader.stop()
            self.reader.join(timeout=5)
        
        for tokenizer in self.tokenizers:
            tokenizer.stop()
            tokenizer.join(timeout=5)
        
        for gpu in self.gpu_workers:
            gpu.stop()
            gpu.join(timeout=5)
        
        logger.info("✅ Все воркеры завершены")
    
    def get_final_stats(self) -> Dict[str, Any]:
        """Собирает финальную статистику."""
        elapsed = time.time() - self.start_time
        
        # Собираем статистику от всех воркеров
        writer_stats = self.writer.get_stats() if self.writer else {}
        gpu_stats_list = [w.get_stats() for w in self.gpu_workers]
        
        # Суммарная статистика по GPU
        total_chunks = sum(s.get('processed_chunks', 0) for s in gpu_stats_list)
        total_gpu_docs = len(set().union(*[s.get('processed_docs', set()) for s in gpu_stats_list]))
        
        # Объединяем статистику по сущностям ИЗ GPU (она точнее)
        entities_by_type = {'LOC': 0, 'PER': 0, 'ORG': 0, 'MISC': 0}
        for gpu_stats in gpu_stats_list:
            for etype, count in gpu_stats.get('entities_by_type', {}).items():
                entities_by_type[etype] += count
        
        completed_docs = writer_stats.get('completed_docs', 0)
        
        stats = {
            'elapsed_time': round(elapsed, 2),
            'docs_completed': completed_docs,
            'chunks_processed': total_chunks,
            'unique_docs_in_gpu': total_gpu_docs,
            'total_entities': sum(entities_by_type.values()),  # Берем сумму из GPU
            'entities_by_type': entities_by_type,
            'bytes_written': writer_stats.get('bytes_written', 0),
            'chunks_per_second': round(total_chunks / elapsed if elapsed > 0 else 0, 2),
            'docs_per_second': round(completed_docs / elapsed if elapsed > 0 else 0, 2)
        }
        
        return stats


def main():
    """Точка входа."""
    parser = argparse.ArgumentParser(description='Конвейерная обработка NER v4')
    parser.add_argument('--config', type=str, default='config_v4.yaml',
                       help='Путь к конфигурационному файлу')
    parser.add_argument('--resume', action='store_true',
                       help='Возобновить с последнего чекпоинта')
    parser.add_argument('--limit', type=int,
                       help='Ограничить количество документов (для теста)')
    
    args = parser.parse_args()
    
    # Загружаем конфиг
    if not os.path.exists(args.config):
        logger.error(f"❌ Конфигурационный файл не найден: {args.config}")
        sys.exit(1)
    
    pipeline = PipelineV4(args.config, limit=args.limit)
    
    # Если не resume, сбрасываем чекпоинт
    if not args.resume and os.path.exists(pipeline.config.checkpoint.file):
        logger.warning("⚠️ Режим без возобновления, удаляем существующий чекпоинт")
        os.remove(pipeline.config.checkpoint.file)
        pipeline.checkpoint = CheckpointManager(
            pipeline.config.checkpoint.file,
            save_interval=pipeline.config.checkpoint.save_interval
        )
    
    try:
        # Запускаем конвейер
        pipeline.start()
        
        # Ожидаем завершения
        pipeline.wait_for_completion()
        
        # Финальная статистика
        stats = pipeline.get_final_stats()
        
        logger.info("=" * 60)
        logger.info("🏁 ОБРАБОТКА ЗАВЕРШЕНА")
        logger.info("=" * 60)
        logger.info(f"Обработано документов: {stats['docs_completed']}")
        logger.info(f"Всего сущностей: {stats['total_entities']}")
        logger.info(f"  LOC: {stats['entities_by_type']['LOC']}")
        logger.info(f"  PER: {stats['entities_by_type']['PER']}")
        logger.info(f"  ORG: {stats['entities_by_type']['ORG']}")
        logger.info(f"  MISC: {stats['entities_by_type']['MISC']}")
        logger.info(f"Время: {stats['elapsed_time']} сек")
        logger.info(f"Скорость: {stats['docs_per_second']} док/сек")
        logger.info(f"Файл результатов: {pipeline.config.output.path}")
        logger.info("=" * 60)
        
    except KeyboardInterrupt:
        logger.info("\n👋 Программа остановлена пользователем")
    except Exception as e:
        logger.error(f"❌ Критическая ошибка: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()