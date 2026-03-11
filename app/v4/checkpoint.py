#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Менеджер чекпоинтов для возобновления обработки
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Управление чекпоинтами для возобновления обработки.
    """
    
    def __init__(self, checkpoint_file: str, save_interval: int = 1000):
        self.checkpoint_file = checkpoint_file
        self.save_interval = save_interval
        self.last_id = 0
        self.processed = 0
        self.stats = {
            'total_entities': 0,
            'entities_by_type': {'LOC': 0, 'PER': 0, 'ORG': 0, 'MISC': 0},
            'bytes_written': 0,
            'session_start': None,
            'last_session': None
        }
        self._last_save_counter = 0
        self.load()
    
    def load(self) -> Optional[Dict[str, Any]]:
        """Загружает последний чекпоинт из файла."""
        if not os.path.exists(self.checkpoint_file):
            logger.info("Чекпоинт не найден, начинаем с начала")
            return None
        
        try:
            with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint = json.load(f)
            
            self.last_id = checkpoint.get('last_id', 0)
            self.processed = checkpoint.get('processed', 0)
            
            # Загружаем статистику
            if 'stats' in checkpoint:
                stats = checkpoint['stats']
                self.stats['total_entities'] = stats.get('total_entities', 0)
                self.stats['entities_by_type'] = stats.get('entities_by_type', 
                                                            {'LOC': 0, 'PER': 0, 'ORG': 0, 'MISC': 0})
                self.stats['bytes_written'] = stats.get('bytes_written', 0)
                self.stats['last_session'] = stats.get('session_end', 
                                                       checkpoint.get('timestamp'))
            
            logger.info(f"📂 Загружен чекпоинт: ID > {self.last_id}, "
                       f"обработано {self.processed} документов")
            logger.info(f"   Всего сущностей: {self.stats['total_entities']}")
            
            return checkpoint
            
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки чекпоинта: {e}")
            return None
    
    def save(self, last_id: int, processed: int, stats: Dict, force: bool = False) -> bool:
        """
        Сохраняет чекпоинт.
        
        Args:
            last_id: последний обработанный ID
            processed: общее количество обработанных документов
            stats: статистика обработки за сессию
            force: принудительное сохранение
        """
        self.last_id = max(self.last_id, last_id)
        self.processed = processed
        
        # Обновляем накопленную статистику
        if 'total_entities' in stats:
            self.stats['total_entities'] = stats['total_entities']
        if 'entities_by_type' in stats:
            logger.debug(f"Сохранение entities_by_type: {stats['entities_by_type']}")
            for k, v in stats['entities_by_type'].items():
                self.stats['entities_by_type'][k] = v
        if 'bytes_written' in stats:
            self.stats['bytes_written'] = stats['bytes_written']
        
        # Добавляем время окончания сессии
        self.stats['session_end'] = datetime.now().isoformat()
        
        # Проверяем, нужно ли сохранять
        if not force and self.processed - self._last_save_counter < self.save_interval:
            return False
        
        checkpoint = {
            'last_id': self.last_id,
            'processed': self.processed,
            'stats': self.stats,
            'timestamp': datetime.now().isoformat(),
            'version': '4.1'  # увеличиваем версию
        }
        
        try:
            Path(self.checkpoint_file).parent.mkdir(parents=True, exist_ok=True)
            
            temp_file = f"{self.checkpoint_file}.tmp"
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint, f, ensure_ascii=False, indent=2)
            
            os.replace(temp_file, self.checkpoint_file)
            
            self._last_save_counter = self.processed
            logger.info(f"💾 Чекпоинт сохранён: ID > {self.last_id}, "
                       f"обработано {self.processed}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка сохранения чекпоинта: {e}")
            return False
    
    def start_session(self):
        """Отмечает начало сессии."""
        self.stats['session_start'] = datetime.now().isoformat()
    
    def clear(self) -> None:
        """Удаляет файл чекпоинта после успешного завершения."""
        if os.path.exists(self.checkpoint_file):
            try:
                os.remove(self.checkpoint_file)
                logger.info("🧹 Чекпоинт удалён")
            except Exception as e:
                logger.error(f"Ошибка удаления чекпоинта: {e}")
    
    def get_start_id(self) -> int:
        return self.last_id
    
    def get_processed_count(self) -> int:
        return self.processed