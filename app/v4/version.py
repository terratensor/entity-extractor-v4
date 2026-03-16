#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Версия приложения v4
"""

import subprocess
from pathlib import Path
from datetime import datetime

# =============================================================================
# СТАТИЧЕСКИЕ КОНСТАНТЫ (для совместимости с существующим кодом)
# =============================================================================
# Эти переменные должны оставаться для обратной совместимости
VERSION = "4.1.1"           # полная версия (для checkpoint)
VERSION_NAME = "4.1"        # краткая версия (для main.py логов)
VERSION_FULL = "4.1.1"      # полная версия (альтернативное имя)

# =============================================================================
# ДИНАМИЧЕСКИЕ ПЕРЕМЕННЫЕ (из git)
# =============================================================================
def _get_git_version():
    """Возвращает версию из git tag или dev."""
    try:
        # Пытаемся получить последний тег
        version = subprocess.check_output(
            ['git', 'describe', '--tags', '--abbrev=0'],
            cwd=Path(__file__).parent.parent.parent,
            stderr=subprocess.DEVNULL,
            text=True
        ).strip()
        return version
    except (subprocess.CalledProcessError, FileNotFoundError):
        return VERSION  # fallback к статической версии


def _get_git_commit():
    """Возвращает хеш коммита."""
    try:
        commit = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            cwd=Path(__file__).parent.parent.parent,
            stderr=subprocess.DEVNULL,
            text=True
        ).strip()
        return commit
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "none"


# Динамические переменные (можно использовать, но не обязательно)
GIT_VERSION = _get_git_version()
GIT_COMMIT = _get_git_commit()
BUILD_TIME = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')

# Для обратной совместимости оставляем статические переменные
# но можно обновить их из git, если нужно
if GIT_VERSION != VERSION:
    # Можно залогировать, но не менять статические переменные
    pass

__all__ = ['VERSION', 'VERSION_NAME', 'VERSION_FULL', 
           'GIT_VERSION', 'GIT_COMMIT', 'BUILD_TIME']