#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Константы для v4 конвейерной обработки
"""

from pathlib import Path
from .version import VERSION_FULL as VERSION

# Версия
__version__ = VERSION

# Директории по умолчанию
DEFAULT_RESULTS_DIR = Path("results_v4")
DEFAULT_CHECKPOINT_FILE = "batch_checkpoint_v4.json"
DEFAULT_LOG_FILE = "batch_processor_v4.log"

# Маппинг меток для NER (из конфига модели)
# Будет загружаться динамически, но для справки:
ENTITY_TYPES = ['LOC', 'PER', 'ORG', 'MISC']

# Статусы для очередей (если понадобятся)
STATUS_PENDING = "pending"
STATUS_PROCESSING = "processing"
STATUS_COMPLETED = "completed"
STATUS_ERROR = "error"

# Таймауты (секунды)
QUEUE_GET_TIMEOUT = 1.0
WORKER_JOIN_TIMEOUT = 5.0