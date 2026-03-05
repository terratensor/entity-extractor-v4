#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Reader для Manticore Search - воркер 1 конвейера
Читает документы пачками, поддерживает чекпоинты и буферизацию
"""

import time
import logging
from queue import Queue, Full
from typing import Optional, Dict, Any, List
from datetime import datetime

import pymysql
from pymysql.cursors import DictCursor

from app.v4.shutdown import StoppableThread

logger = logging.getLogger(__name__)


class ManticoreReader(StoppableThread):
    """
    Читает документы из Manticore и складывает в очередь.
    Поддерживает возобновление с чекпоинта.
    """
    
    def __init__(
        self,
        config,
        checkpoint_last_id: int,
        output_queue: Queue,
        stop_event,
        name: str = "ManticoreReader"
    ):
        """
        Args:
            config: конфигурация источника (SourceConfig)
            checkpoint_last_id: последний обработанный ID (с чего начинать)
            output_queue: очередь для выдачи документов
            stop_event: событие остановки
            name: имя потока
        """
        super().__init__(name=name, stop_event=stop_event)
        
        self.config = config
        self.start_id = checkpoint_last_id
        self.output_queue = output_queue
        
        # Параметры подключения
        self.host = config.host
        self.port = config.port
        self.table = config.table
        self.batch_size = config.batch_size
        self.timeout = getattr(config, 'connection_timeout', 30)
        
        # Статистика
        self.stats = {
            'read_docs': 0,
            'read_batches': 0,
            'last_id': self.start_id,
            'start_time': None,
            'total_time': 0
        }
        
        # Соединение
        self.conn = None
        self._connect()
    
    def _connect(self) -> None:
        """Устанавливает соединение с Manticore."""
        try:
            self.conn = pymysql.connect(
                host=self.host,
                port=self.port,
                user='root',
                charset='utf8mb4',
                cursorclass=DictCursor,
                connect_timeout=self.timeout,
                read_timeout=self.timeout
            )
            logger.info(f"✅ Подключен к Manticore: {self.host}:{self.port}, таблица: {self.table}")
        except Exception as e:
            logger.error(f"❌ Ошибка подключения к Manticore: {e}")
            raise
    
    def _reconnect(self) -> bool:
        """Переподключается при разрыве соединения."""
        try:
            if self.conn:
                self.conn.close()
        except:
            pass
        
        time.sleep(1)  # Пауза перед переподключением
        
        try:
            self._connect()
            return True
        except:
            return False
    
    def get_total_count(self) -> int:
        """
        Возвращает общее количество документов в таблице.
        
        Returns:
            int: количество документов
        """
        try:
            with self.conn.cursor() as cursor:
                cursor.execute(f"SELECT COUNT(*) as cnt FROM {self.table}")
                result = cursor.fetchone()
                return result['cnt'] if result else 0
        except Exception as e:
            logger.error(f"Ошибка получения общего количества: {e}")
            return 0
    
    def get_max_id(self) -> int:
        """
        Возвращает максимальный ID в таблице.
        
        Returns:
            int: максимальный ID
        """
        try:
            with self.conn.cursor() as cursor:
                cursor.execute(f"SELECT MAX(id) as max_id FROM {self.table}")
                result = cursor.fetchone()
                return result['max_id'] if result and result['max_id'] else 0
        except Exception as e:
            logger.error(f"Ошибка получения максимального ID: {e}")
            return 0
    
    def run(self) -> None:
        """
        Основной цикл чтения документов.
        Читает пачками, складывает в output_queue.
        """
        logger.info(f"🚀 Reader запущен, начинаем с ID > {self.start_id}")
        
        last_id = self.start_id
        self.stats['start_time'] = time.time()
        
        # Получаем максимальный ID для прогресса
        max_id = self.get_max_id()
        if max_id:
            logger.info(f"📊 Максимальный ID в таблице: {max_id}")
        
        retry_count = 0
        max_retries = 5
        
        while not self.should_stop():
            try:
                # Читаем пачку документов
                rows = self._fetch_batch(last_id)
                
                if not rows:
                    # Нет больше данных
                    if last_id == self.start_id and retry_count == 0:
                        logger.warning("⚠️  Таблица пуста или нет новых документов")
                    else:
                        logger.info("📭 Все документы прочитаны")
                    break
                
                # Обрабатываем пачку
                for row in rows:
                    # Проверяем остановку
                    if self.should_stop():
                        break
                    
                    # Проверяем, есть ли место в очереди
                    doc = {
                        'id': row['id'],
                        'text': row['content']  # предполагаем, что поле называется content
                    }
                    
                    # Пытаемся положить в очередь с таймаутом
                    try:
                        self.output_queue.put(doc, timeout=1.0)
                    except Full:
                        # Очередь переполнена, ждём и пробуем снова
                        logger.debug(f"Очередь переполнена, ждём... (размер: {self.output_queue.qsize()})")
                        if not self.safe_sleep(0.5):
                            break
                        # Повторяем попытку
                        self.output_queue.put(doc)
                    
                    # Обновляем статистику
                    self.stats['read_docs'] += 1
                    last_id = row['id']
                
                # Обновляем статистику пачки
                self.stats['read_batches'] += 1
                self.stats['last_id'] = last_id
                
                # Логируем прогресс каждые 100 пачек
                if self.stats['read_batches'] % 100 == 0:
                    elapsed = time.time() - self.stats['start_time']
                    speed = self.stats['read_docs'] / elapsed if elapsed > 0 else 0
                    logger.info(f"📈 Прогресс: {self.stats['read_docs']} док, "
                               f"последний ID: {last_id}, "
                               f"скорость: {speed:.1f} док/сек")
                
                # Сброс счётчика повторов при успехе
                retry_count = 0
                
                # Небольшая пауза, чтобы не перегружать Manticore
                if self.stats['read_docs'] % (self.batch_size * 10) == 0:
                    self.safe_sleep(0.1)
                
            except Exception as e:
                logger.error(f"❌ Ошибка чтения из Manticore: {e}", exc_info=True)
                
                retry_count += 1
                if retry_count > max_retries:
                    logger.critical(f"Превышено количество попыток ({max_retries}), останавливаемся")
                    break
                
                # Пытаемся переподключиться
                logger.info(f"Попытка переподключения {retry_count}/{max_retries}...")
                if not self._reconnect():
                    # Не удалось переподключиться, ждём перед следующей попыткой
                    self.safe_sleep(5)
        
        # Завершение работы
        self.stats['total_time'] = time.time() - self.stats['start_time']
        logger.info(f"📊 Reader завершён: прочитано {self.stats['read_docs']} документов, "
                   f"последний ID: {last_id}, время: {self.stats['total_time']:.1f} сек")
        
        # Закрываем соединение
        self._close()
    
    def _fetch_batch(self, last_id: int) -> List[Dict[str, Any]]:
        """
        Получает пачку документов из Manticore.
        
        Args:
            last_id: последний обработанный ID
            
        Returns:
            List[Dict]: список документов с полями id, content
        """
        if not self.conn or not self.conn.open:
            if not self._reconnect():
                return []
        
        try:
            with self.conn.cursor() as cursor:
                query = f"""
                    SELECT id, content 
                    FROM {self.table} 
                    WHERE id > %s 
                    ORDER BY id ASC 
                    LIMIT %s
                    OPTION max_matches=%s
                """
                cursor.execute(query, (last_id, self.batch_size, self.batch_size))
                rows = cursor.fetchall()
                
                # Преобразуем в список словарей с нужными ключами
                result = []
                for row in rows:
                    # Проверяем, что есть оба поля
                    if 'id' in row and 'content' in row:
                        result.append({
                            'id': int(row['id']),
                            'content': str(row['content'])
                        })
                    else:
                        # Возможно, поле называется по-другому
                        # Пробуем найти любой текстовый ключ
                        text_key = None
                        for key in row.keys():
                            if key != 'id' and isinstance(row[key], str):
                                text_key = key
                                break
                        
                        if text_key:
                            result.append({
                                'id': int(row['id']),
                                'content': str(row[text_key])
                            })
                        else:
                            logger.warning(f"Пропускаем запись без текста: {row}")
                
                return result
                
        except pymysql.OperationalError as e:
            # Ошибка соединения
            logger.error(f"Ошибка соединения с Manticore: {e}")
            self.conn = None
            raise
        except Exception as e:
            logger.error(f"Ошибка выполнения запроса: {e}")
            raise
    
    def _close(self) -> None:
        """Закрывает соединение с Manticore."""
        try:
            if self.conn and self.conn.open:
                self.conn.close()
                logger.debug("Соединение с Manticore закрыто")
        except Exception as e:
            logger.error(f"Ошибка при закрытии соединения: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Возвращает статистику работы reader."""
        stats = self.stats.copy()
        if stats['start_time']:
            stats['current_time'] = time.time() - stats['start_time']
        return stats