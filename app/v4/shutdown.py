#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Graceful shutdown для корректного завершения воркеров
Адаптировано из прототипа v3 с улучшениями для v4
"""

import signal
import logging
import threading
import time
from typing import Optional, List, Any

logger = logging.getLogger(__name__)


class GracefulShutdown:
    """
    Обработчик сигналов для корректного завершения.
    Использует threading.Event для оповещения воркеров.
    """
    
    def __init__(self):
        self.stop_event = threading.Event()
        self._shutdown_callbacks = []
        self._registered = False
        
        # Регистрируем обработчики сигналов
        self._register_signal_handlers()
    
    def _register_signal_handlers(self):
        """Регистрирует обработчики SIGINT и SIGTERM."""
        if self._registered:
            return
        
        try:
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
            self._registered = True
            logger.debug("Обработчики сигналов зарегистрированы")
        except ValueError as e:
            # Может быть в потоках, не являющихся главными
            logger.warning(f"Не удалось зарегистрировать обработчики сигналов: {e}")
    
    def _signal_handler(self, signum, frame):
        """
        Обработчик сигналов.
        
        Args:
            signum: номер сигнала
            frame: текущий стек вызовов (не используется)
        """
        signal_name = 'SIGINT' if signum == signal.SIGINT else 'SIGTERM'
        logger.warning(f"\n⚠️  Получен сигнал {signal_name}, инициируем остановку...")
        
        # Устанавливаем событие остановки
        self.stop_event.set()
        
        # Вызываем колбэки
        for callback in self._shutdown_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Ошибка в callback при остановке: {e}")
    
    def register_callback(self, callback):
        """
        Регистрирует функцию, которая будет вызвана при остановке.
        
        Args:
            callback: функция без аргументов
        """
        self._shutdown_callbacks.append(callback)
    
    def is_set(self) -> bool:
        """Проверяет, установлен ли флаг остановки."""
        return self.stop_event.is_set()
    
    def wait_for_stop(self, timeout: Optional[float] = None) -> bool:
        """
        Ожидает сигнала остановки.
        
        Args:
            timeout: таймаут ожидания в секундах
            
        Returns:
            True если сигнал получен, False если таймаут
        """
        return self.stop_event.wait(timeout)
    
    def stop(self):
        """Принудительная установка флага остановки."""
        logger.info("🛑 Принудительная остановка...")
        self.stop_event.set()
    
    def sleep(self, seconds: float):
        """
        Спит с проверкой флага остановки.
        
        Args:
            seconds: сколько секунд спать
            
        Returns:
            True если дождались окончания сна, False если была остановка
        """
        if seconds <= 0:
            return not self.stop_event.is_set()
        
        # Разбиваем сон на маленькие интервалы для проверки флага
        interval = 0.1
        elapsed = 0
        while elapsed < seconds and not self.stop_event.is_set():
            time.sleep(min(interval, seconds - elapsed))
            elapsed += interval
        
        return not self.stop_event.is_set()


class StoppableThread(threading.Thread):
    """
    Поток, который можно остановить через событие.
    Базовый класс для всех воркеров.
    """
    
    def __init__(self, name: Optional[str] = None, 
                 stop_event: Optional[threading.Event] = None):
        """
        Args:
            name: имя потока
            stop_event: событие для сигнала остановки
        """
        super().__init__(name=name)
        self.stop_event = stop_event or threading.Event()
        self.daemon = True  # Потоки-daemon завершатся при выходе из main
    
    def stop(self):
        """Сигнал к остановке."""
        self.stop_event.set()
    
    def should_stop(self) -> bool:
        """Проверяет, нужно ли остановиться."""
        return self.stop_event.is_set()
    
    def safe_sleep(self, seconds: float) -> bool:
        """
        Спит с проверкой флага остановки.
        
        Args:
            seconds: сколько секунд спать
            
        Returns:
            True если доспали до конца, False если прерваны остановкой
        """
        if seconds <= 0:
            return not self.should_stop()
        
        interval = 0.1
        elapsed = 0
        while elapsed < seconds and not self.should_stop():
            time.sleep(min(interval, seconds - elapsed))
            elapsed += interval
        
        return not self.should_stop()