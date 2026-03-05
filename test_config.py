#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Тест загрузки конфигурации
"""

import sys
from pathlib import Path

# Добавляем путь к проекту
sys.path.insert(0, str(Path(__file__).parent))

from app.v4.config import load_config, DEFAULT_CONFIG_YAML


def main():
    """Тестируем загрузку конфига"""
    print("=" * 60)
    print("Тест загрузки конфигурации v4")
    print("=" * 60)
    
    # Создаём тестовый конфиг, если его нет
    config_path = Path("config_v4.yaml")
    if not config_path.exists():
        print(f"Создаём тестовый конфиг: {config_path}")
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(DEFAULT_CONFIG_YAML)
    
    # Загружаем конфиг
    print(f"Загрузка конфига из {config_path}...")
    config = load_config(str(config_path))
    
    # Выводим параметры
    print("\n📋 Параметры конфигурации:")
    print(f"  Источник: {config.source.host}:{config.source.port}, таблица: {config.source.table}")
    print(f"  Модель: {config.model.name}")
    print(f"  max_tokens: {config.model.max_tokens}, overlap: {config.model.overlap_ratio}")
    print(f"  Очереди: q1={config.queues.queue1_size}, q2={config.queues.queue2_size}, q3={config.queues.queue3_size}")
    print(f"  Токенизаторов: {config.tokenizer.num_workers}")
    print(f"  GPU устройств: {len(config.gpu_devices)}")
    for gpu in config.gpu_devices:
        print(f"    - GPU {gpu.device_id}: batch_size={gpu.batch_size}, precision={gpu.precision}")
    print(f"  Выходной файл: {config.output.path}")
    print(f"  Чекпоинт: {config.checkpoint.file}")
    
    print("\n✅ Конфигурация успешно загружена!")


if __name__ == "__main__":
    main()