#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Менеджер чекпоинтов для возобновления обработки
Адаптировано из прототипа v3 с улучшениями для v4
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
    Сохраняет последний обработанный ID и статистику.
    """
    
    def __init__(self, checkpoint_file: str, save_interval: int = 1000):
        """
        Args:
            checkpoint_file: путь к файлу чекпоинта
            save_interval: как часто сохранять (в документах)
        """
        self.checkpoint_file = checkpoint_file
        self.save_interval = save_interval
        self.last_id = 0
        self.processed = 0
        self.stats = {
            'processed': 0,
            'total_tokens': 0,
            'entities_found': 0,
            'entities_by_type': {'LOC': 0, 'PER': 0, 'ORG': 0, 'MISC': 0},
            'processing_time': 0,
            'start_time': None
        }
        self._last_save_counter = 0
        
        # Загружаем существующий чекпоинт
        self.load()
    
    def load(self) -> Optional[Dict[str, Any]]:
        """
        Загружает последний чекпоинт из файла.
        
        Returns:
            Dict с данными чекпоинта или None, если файла нет
        """
        if not os.path.exists(self.checkpoint_file):
            logger.info("Чекпоинт не найден, начинаем с начала")
            return None
        
        try:
            with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint = json.load(f)
            
            self.last_id = checkpoint.get('last_id', 0)
            self.processed = checkpoint.get('processed', 0)
            self.stats = checkpoint.get('stats', self.stats)
            
            logger.info(f"📂 Загружен чекпоинт: ID > {self.last_id}, "
                       f"обработано {self.processed} документов")
            logger.info(f"   Статистика: {self.stats['entities_found']} сущностей")
            
            return checkpoint
            
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки чекпоинта: {e}")
            return None
    
    def save(self, last_id: int, processed: Optional[int] = None, 
             stats: Optional[Dict] = None, force: bool = False) -> bool:
        """
        Сохраняет чекпоинт.
        
        Args:
            last_id: последний обработанный ID
            processed: количество обработанных документов
            stats: статистика обработки
            force: принудительное сохранение (игнорирует save_interval)
            
        Returns:
            bool: успешно ли сохранено
        """
        # Обновляем данные
        self.last_id = max(self.last_id, last_id)
        if processed is not None:
            self.processed = processed
        if stats is not None:
            self.stats.update(stats)
        
        # Проверяем, нужно ли сохранять (по интервалу)
        if not force and self.processed - self._last_save_counter < self.save_interval:
            return False
        
        checkpoint = {
            'last_id': self.last_id,
            'processed': self.processed,
            'stats': self.stats,
            'timestamp': datetime.now().isoformat(),
            'version': '4.0'
        }
        
        try:
            # Создаем директорию, если нужно
            Path(self.checkpoint_file).parent.mkdir(parents=True, exist_ok=True)
            
            # Атомарная запись: сначала во временный файл, потом переименовываем
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
    
    def update_stats(self, doc_stats: Dict[str, int]) -> None:
        """
        Обновляет статистику обработки.
        
        Args:
            doc_stats: статистика по документу (токены, сущности по типам)
        """
        self.stats['processed'] += 1
        self.stats['total_tokens'] += doc_stats.get('tokens', 0)
        self.stats['entities_found'] += doc_stats.get('total', 0)
        
        for etype in ['LOC', 'PER', 'ORG', 'MISC']:
            self.stats['entities_by_type'][etype] += doc_stats.get(etype, 0)
    
    def clear(self) -> None:
        """Удаляет файл чекпоинта после успешного завершения."""
        if os.path.exists(self.checkpoint_file):
            try:
                os.remove(self.checkpoint_file)
                logger.info("🧹 Чекпоинт удалён")
            except Exception as e:
                logger.error(f"Ошибка удаления чекпоинта: {e}")
    
    def get_start_id(self) -> int:
        """Возвращает ID, с которого нужно начинать обработку."""
        return self.last_id
    
    def get_processed_count(self) -> int:
        """Возвращает количество обработанных документов."""
        return self.processed
    
    def get_stats(self) -> Dict:
        """Возвращает текущую статистику."""
        return self.stats.copy()