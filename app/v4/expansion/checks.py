#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Методы проверки для расширения слов.
"""

import logging
from typing import Tuple, Optional

from . import constants
from . import utils

logger = logging.getLogger(__name__)


def should_expand(
    text: str,
    start_pos: int,
    end_pos: int,
    original_text: str,
    entity_type: str,
    config: dict,
    verbose: bool = False
) -> Tuple[bool, Optional[str]]:
    """
    Проверяет, нужно ли расширять сущность (в обе стороны).
    
    Returns:
        (bool, str): (нужно ли расширять, причина отказа)
    """
    if verbose:
        logger.warning(f"   _should_expand проверка для '{text}':")
        logger.warning(f"      start_pos={start_pos}, end_pos={end_pos}")
    
    # ----------------------------------------------------------------------
    # ПРОВЕРКА 1: Минимальная длина
    # ----------------------------------------------------------------------
    if len(text) < config['min_token_length']:
        if verbose:
            logger.warning(f"      ❌ rejected_length: длина {len(text)} < {config['min_token_length']}")
        return False, 'length'
    
    # ----------------------------------------------------------------------
    # ПРОВЕРКА 2: Стоп-слова (предлоги, союзы, частицы)
    # ----------------------------------------------------------------------
    if config['enable_stopwords']:
        clean_text = text.lower().strip('▁')
        if clean_text in constants.STOP_WORDS:
            if verbose:
                logger.warning(f"      ❌ rejected_stopword: '{clean_text}' в стоп-словах")
            return False, 'stopword'
    
    # ----------------------------------------------------------------------
    # ПРОВЕРКА 3: Возможность расширения ВЛЕВО
    # ----------------------------------------------------------------------
    can_expand_left = False
    left_reason = None

    if start_pos > 0:
        prev_char = original_text[start_pos - 1]
        if verbose:
            logger.warning(f"      символ слева: '{prev_char}' (код {ord(prev_char)})")
        
        # Условие 3.1: Слева буква или кавычка (не разделитель)
        if prev_char.isalpha() or prev_char in constants.ALL_QUOTES:
            # Условие 3.2: Для букв - всегда разрешаем (часть слова)
            if prev_char.isalpha():
                can_expand_left = True
                if verbose:
                    logger.warning(f"      ✅ можно расширять ВЛЕВО (часть слова)")
            # Для кавычек - только если это начало слова
            elif start_pos - 1 == 0 or original_text[start_pos - 2] in constants.WORD_BREAKS:
                can_expand_left = True
                if verbose:
                    logger.warning(f"      ✅ можно расширять ВЛЕВО (начало слова с кавычкой)")
            else:
                left_reason = "кавычка не в начале слова"
        else:
            left_reason = "не буква и не кавычка"
    else:
        left_reason = "начало текста"
    
    # ----------------------------------------------------------------------
    # ПРОВЕРКА 4: Возможность расширения ВПРАВО
    # ----------------------------------------------------------------------
    can_expand_right = False
    right_reason = None
    
    if end_pos < len(original_text):
        next_char = original_text[end_pos]
        if verbose:
            logger.warning(f"      символ справа: '{next_char}' (код {ord(next_char)})")
        
        # Условие 4.1: Справа буква или кавычка (не разделитель)
        if next_char.isalpha() or next_char in constants.ALL_QUOTES:
            can_expand_right = True
            if verbose:
                logger.warning(f"      ✅ можно расширять ВПРАВО")
        else:
            right_reason = "не буква и не кавычка"
    else:
        right_reason = "конец текста"
    
    # ----------------------------------------------------------------------
    # ПРОВЕРКА 5: Для LOC/PER проверяем заглавные буквы (только для правого расширения)
    # ----------------------------------------------------------------------
    if (config['require_capital'] and entity_type in ['LOC', 'PER'] and 
        can_expand_right and end_pos < len(original_text)):
        next_char = original_text[end_pos]
        if next_char.isalpha() and not next_char.isupper():
            # Если следующая буква строчная - это нормально (продолжение слова)
            # Но если это начало нового слова - должна быть заглавной
            if end_pos > 0 and original_text[end_pos - 1] in constants.WORD_BREAKS:
                if verbose:
                    logger.warning(f"      ❌ rejected_capital: следующая буква '{next_char}' не заглавная")
                return False, 'capital'
    
    # ----------------------------------------------------------------------
    # ИТОГ: расширяем, если есть возможность хотя бы в одну сторону
    # ----------------------------------------------------------------------
    if can_expand_left or can_expand_right:
        if verbose:
            logger.warning(f"      ✅ можно расширять (влево={can_expand_left}, вправо={can_expand_right})")
        return True, None
    else:
        if verbose:
            logger.warning(f"      ❌ нет расширения: влево={left_reason}, вправо={right_reason}")
        return False, None


def check_word_merge(
    original_text: str,
    full_word: str,
    start_pos: int,
    end_pos: int,
    config: dict,
    verbose: bool = False
) -> bool:
    """
    Проверяет, не является ли расширение результатом слияния слов.
    """
    # Если внутри полного слова есть пробелы - это несколько слов
    if ' ' in full_word and '-' not in full_word:
        original_part = original_text[start_pos:end_pos]
        if len(original_part) < len(full_word) * 0.3:
            return True
    
    # Проверка на типичные паттерны слияния
    if end_pos < len(original_text):
        next_char = original_text[end_pos]
        last_char = original_text[end_pos - 1] if end_pos > 0 else ''
        
        # Признаки возможного слияния: согласная + гласная на стыке
        if (last_char.isalpha() and next_char.isalpha() and
            last_char.lower() not in constants.VOWELS and
            next_char.lower() in constants.VOWELS):
            
            # Дополнительные проверки:
            # 1. Проверяем, что после гласной есть буквы (это часть слова, а не окончание)
            if end_pos + 1 < len(original_text):
                next_next = original_text[end_pos + 1]
                if next_next.isalpha():
                    return False
            
            # 2. Проверяем, не является ли это типичным окончанием
            common_endings = {'а', 'я', 'ы', 'и', 'е', 'ё', 'ю', 'й'}
            if next_char.lower() in common_endings:
                if end_pos + 1 >= len(original_text) or original_text[end_pos + 1] in constants.WORD_BREAKS:
                    return False
            
            # 3. Проверяем длину исходного слова
            if len(full_word) < 3:
                if full_word.lower() in constants.STOP_WORDS:
                    return True
            
            return True
    
    return False