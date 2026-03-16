# app/v4/version.py
import subprocess
from pathlib import Path
from datetime import datetime

# Статическая версия (если нужно переопределить)
STATIC_VERSION = None  # например "4.1.1"

def get_version():
    if STATIC_VERSION:
        return STATIC_VERSION
    
    try:
        # Пытаемся получить из git tag
        version = subprocess.check_output(
            ['git', 'describe', '--tags', '--abbrev=0'],
            cwd=Path(__file__).parent.parent.parent,
            stderr=subprocess.DEVNULL,
            text=True
        ).strip()
        return version
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "dev"

__version__ = get_version()
__commit__ = subprocess.getoutput('git rev-parse --short HEAD')[:7]
__build_time__ = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')

__all__ = ['__version__', '__commit__', '__build_time__']