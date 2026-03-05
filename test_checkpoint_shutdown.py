#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Тест CheckpointManager и GracefulShutdown
"""

import sys
import time
import logging
import tempfile
import os
from pathlib import Path

# Добавляем путь к проекту
sys.path.insert(0, str(Path(__file__).parent))

from app.v4.checkpoint import CheckpointManager
from app.v4.shutdown import GracefulShutdown, StoppableThread

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_checkpoint():
    """Тестируем CheckpointManager"""
    print("\n" + "=" * 60)
    print("Тест CheckpointManager")
    print("=" * 60)
    
    # Создаём временный файл
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        checkpoint_file = f.name
    
    try:
        # Создаём менеджер
        cp = CheckpointManager(checkpoint_file, save_interval=2)
        
        # Проверяем начальное состояние
        assert cp.last_id == 0
        assert cp.processed == 0
        
        # Сохраняем чекпоинт
        cp.save(last_id=100500, processed=42, force=True)
        
        # Создаём новый менеджер (должен загрузить)
        cp2 = CheckpointManager(checkpoint_file)
        assert cp2.last_id == 100500
        assert cp2.processed == 42
        
        # Обновляем статистику
        cp2.update_stats({
            'tokens': 1500,
            'total': 25,
            'LOC': 10,
            'PER': 10,
            'ORG': 5,
            'MISC': 0
        })
        
        # Сохраняем
        cp2.save(last_id=100501, processed=43)
        
        # Проверяем статистику
        stats = cp2.get_stats()
        assert stats['processed'] == 1  # один документ добавили
        assert stats['total_tokens'] == 1500
        assert stats['entities_found'] == 25
        assert stats['entities_by_type']['LOC'] == 10
        
        # Очищаем
        cp2.clear()
        assert not os.path.exists(checkpoint_file)
        
        print("✅ Все тесты CheckpointManager пройдены")
        
    finally:
        # Удаляем временный файл
        if os.path.exists(checkpoint_file):
            os.unlink(checkpoint_file)


class TestWorker(StoppableThread):
    """Тестовый воркер"""
    
    def __init__(self, name, stop_event):
        super().__init__(name=name, stop_event=stop_event)
        self.counter = 0
    
    def run(self):
        logger.info(f"Рабочий {self.name} запущен")
        while not self.should_stop():
            self.counter += 1
            if not self.safe_sleep(0.1):
                break
        logger.info(f"Рабочий {self.name} завершён, сделано {self.counter} итераций")


def test_shutdown():
    """Тестируем GracefulShutdown и StoppableThread"""
    print("\n" + "=" * 60)
    print("Тест GracefulShutdown и StoppableThread")
    print("=" * 60)
    
    shutdown = GracefulShutdown()
    
    # Создаём несколько тестовых воркеров
    workers = []
    for i in range(3):
        worker = TestWorker(f"Worker-{i}", shutdown.stop_event)
        worker.start()
        workers.append(worker)
    
    # Даём им поработать 1 секунду
    print("Воркеры работают 1 секунду...")
    time.sleep(1)
    
    # Останавливаем
    print("Останавливаем...")
    shutdown.stop()
    
    # Ждём завершения
    for worker in workers:
        worker.join(timeout=2)
        print(f"  {worker.name}: {worker.counter} итераций")
    
    print("✅ Все тесты Shutdown пройдены")


def main():
    """Запуск всех тестов"""
    test_checkpoint()
    test_shutdown()
    print("\n🎉 Все тесты успешно пройдены!")


if __name__ == "__main__":
    main()