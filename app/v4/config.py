#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Конфигурация для v4 конвейерной обработки
Загрузка параметров из YAML файла
"""

import yaml
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SourceConfig:
    """Конфигурация источника данных (Manticore)"""
    type: str = "manticore"
    host: str = "localhost"
    port: int = 9306
    table: str = "library2026"
    batch_size: int = 1000
    connection_timeout: int = 30
    
    @classmethod
    def from_dict(cls, data: dict):
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class ModelConfig:
    """Конфигурация модели NER"""
    name: str = "Davlan/xlm-roberta-large-ner-hrl"
    max_tokens: int = 512
    overlap_ratio: float = 0.0
    min_confidence: float = 0.5
    include_positions: bool = False  # для отладки, в проде можно отключить
    
    @classmethod
    def from_dict(cls, data: dict):
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class QueueConfig:
    """Размеры очередей-буферов"""
    queue1_size: int = 1000   # сырые документы (после reader)
    queue2_size: int = 20000  # токенизированные чанки (после tokenizer)
    queue3_size: int = 5000   # результаты инференса (после GPU)
    
    @classmethod
    def from_dict(cls, data: dict):
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class TokenizerConfig:
    """Конфигурация токенизаторов (воркеры на CPU)"""
    num_workers: int = 4  # количество параллельных токенизаторов
    prefetch_size: int = 100  # сколько документов заранее готовить
    
    @classmethod
    def from_dict(cls, data: dict):
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class GPUDeviceConfig:
    """Конфигурация конкретного GPU"""
    device_id: int
    batch_size: int
    precision: str = "float16"  # float16 или float32
    
    @classmethod
    def from_dict(cls, data: dict):
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class OutputConfig:
    """Конфигурация выходных данных"""
    format: str = "csv"  # csv или parquet
    path: str = "./results/output.csv"
    delimiter: str = "|"
    include_confidence: bool = True
    include_positions: bool = False
    flush_interval: int = 60  # секунд
    buffer_size: int = 1000  # строк для буферизации перед записью
    
    @classmethod
    def from_dict(cls, data: dict):
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class CheckpointConfig:
    """Конфигурация чекпоинтов"""
    file: str = "./batch_checkpoint_v4.json"
    save_interval: int = 1000  # сохранять каждые N документов
    
    @classmethod
    def from_dict(cls, data: dict):
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class LoggingConfig:
    """Конфигурация логирования"""
    level: str = "INFO"
    file: Optional[str] = "./batch_processor_v4.log"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    @classmethod
    def from_dict(cls, data: dict):
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class Config:
    """Главная конфигурация приложения"""
    source: SourceConfig
    model: ModelConfig
    queues: QueueConfig
    tokenizer: TokenizerConfig
    gpu_devices: List[GPUDeviceConfig]
    output: OutputConfig
    checkpoint: CheckpointConfig
    logging: LoggingConfig
    
    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            source=SourceConfig.from_dict(data.get('source', {})),
            model=ModelConfig.from_dict(data.get('model', {})),
            queues=QueueConfig.from_dict(data.get('queues', {})),
            tokenizer=TokenizerConfig.from_dict(data.get('tokenizer', {})),
            gpu_devices=[GPUDeviceConfig.from_dict(d) for d in data.get('gpu_devices', [])],
            output=OutputConfig.from_dict(data.get('output', {})),
            checkpoint=CheckpointConfig.from_dict(data.get('checkpoint', {})),
            logging=LoggingConfig.from_dict(data.get('logging', {}))
        )


def load_config(path: str) -> Config:
    """
    Загружает конфигурацию из YAML файла
    
    Args:
        path: путь к YAML файлу
        
    Returns:
        Config: объект конфигурации
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Конфигурационный файл не найден: {path}")
    
    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    
    return Config.from_dict(data)


# Пример конфигурации для быстрого старта
DEFAULT_CONFIG_YAML = """
# Конфигурация для v4 конвейерной обработки
source:
  type: manticore
  host: localhost
  port: 9306
  table: library2026
  batch_size: 1000

model:
  name: Davlan/xlm-roberta-large-ner-hrl
  max_tokens: 512
  overlap_ratio: 0.0
  min_confidence: 0.5
  include_positions: false

queues:
  queue1_size: 1000      # сырые документы
  queue2_size: 20000     # токенизированные чанки
  queue3_size: 5000      # результаты инференса

tokenizer:
  num_workers: 4         # количество потоков токенизации
  prefetch_size: 100

gpu_devices:
  - device_id: 0
    batch_size: 32
    precision: float16
  - device_id: 1
    batch_size: 24
    precision: float16

output:
  format: csv
  path: ./results/output.csv
  delimiter: "|"
  include_confidence: true
  include_positions: false
  flush_interval: 60
  buffer_size: 1000

checkpoint:
  file: ./batch_checkpoint_v4.json
  save_interval: 1000

logging:
  level: INFO
  file: ./batch_processor_v4.log
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
"""