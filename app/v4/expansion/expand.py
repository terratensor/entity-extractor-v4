#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Логика расширения слова влево и вправо.
"""

import logging
from typing import Tuple

from . import constants
from . import checks

logger = logging.getLogger(__name__)


def get_word_boundaries(
    start_pos: int,
    end_pos: int,
    original_text: str,
    config: dict,
    verbose: bool = False
) -> Tuple[int, int, bool, bool]:
    """
    Определяет границы слова, расширяясь влево и вправо.
    
    Returns:
        tuple: (word_start, word_end, left_expanded, right_expanded)
    """
    max_left = config['max_search_left']
    max_right = config['max_search_right']
    
    # ----------------------------------------------------------------------
    # РАСШИРЕНИЕ ВЛЕВО
    # ----------------------------------------------------------------------
    word_start = start_pos
    left_expanded = False
    steps_left = 0
    
    if start_pos > 0:
        if verbose:
            logger.warning(f"      поиск влево от {start_pos}:")
        while word_start > 0 and steps_left < max_left:
            prev_char = original_text[word_start - 1]
            if verbose:
                logger.warning(f"        символ {word_start-1}: '{prev_char}'")
            
            if prev_char in constants.WORD_BREAKS and prev_char not in constants.ALL_QUOTES:
                if verbose:
                    logger.warning(f"          стоп - разделитель")
                break
            
            word_start -= 1
            steps_left += 1
            left_expanded = True
            if verbose:
                logger.warning(f"          добавлен влево, теперь начало {word_start}")
    
    # ----------------------------------------------------------------------
    # РАСШИРЕНИЕ ВПРАВО
    # ----------------------------------------------------------------------
    word_end = end_pos
    right_expanded = False
    steps_right = 0
    
    if end_pos < len(original_text):
        if verbose:
            logger.warning(f"      поиск вправо от {end_pos}:")
        while word_end < len(original_text) and steps_right < max_right:
            next_char = original_text[word_end]
            if verbose:
                logger.warning(f"        символ {word_end}: '{next_char}'")
            
            if next_char in constants.WORD_BREAKS and next_char not in constants.ALL_QUOTES:
                if verbose:
                    logger.warning(f"          стоп - разделитель")
                break
            
            word_end += 1
            steps_right += 1
            right_expanded = True
            if verbose:
                logger.warning(f"          добавлен вправо, теперь конец {word_end}")
    
    return word_start, word_end, left_expanded, right_expanded


def check_left_part(
    word_start: int,
    start_pos: int,
    original_text: str,
    verbose: bool = False
) -> Tuple[int, bool]:
    """
    Проверяет левую добавленную часть на разделители.
    
    Returns:
        tuple: (word_start, left_expanded) - обновленные значения
    """
    left_expanded = True
    for pos in range(word_start, start_pos):
        if pos >= len(original_text):
            break
        char = original_text[pos]
        if char in constants.WORD_BREAKS and char not in constants.ALL_QUOTES and char != '-':
            if verbose:
                logger.warning(f"         найден разделитель '{char}' на позиции {pos} в левой части - левое расширение отменяется")
            left_expanded = False
            word_start = start_pos
            break
    return word_start, left_expanded


def check_right_part(
    end_pos: int,
    word_end: int,
    original_text: str,
    verbose: bool = False
) -> Tuple[int, bool]:
    """
    Проверяет правую добавленную часть на разделители.
    
    Returns:
        tuple: (word_end, right_expanded) - обновленные значения
    """
    right_expanded = True
    for pos in range(end_pos, word_end):
        if pos >= len(original_text):
            break
        char = original_text[pos]
        if char in constants.WORD_BREAKS and char not in constants.ALL_QUOTES and char != '-':
            if verbose:
                logger.warning(f"         найден разделитель '{char}' на позиции {pos} в правой части - правое расширение отменяется")
            right_expanded = False
            word_end = end_pos
            break
    return word_end, right_expanded


def get_last_letter_pos(
    word_start: int,
    word_end: int,
    original_text: str
) -> int:
    """
    Находит позицию последней буквы в слове.
    """
    last_letter = word_end
    while last_letter > word_start and not original_text[last_letter - 1].isalpha():
        last_letter -= 1
    return last_letter


def check_inner_range(
    start_pos: int,
    end_pos: int,
    word_start: int,
    word_end: int,
    original_text: str,
    verbose: bool = False
) -> bool:
    """
    Проверяет внутренний диапазон на разделители с учетом реальных границ слова.
    Пробелы внутри слова разрешены (для многословных названий, например "Юрий Милославские").
    
    Returns:
        bool: True если найден разделитель внутри слова
    """
    current_start = word_start
    current_end = word_end
    
    if verbose:
        logger.warning(f"      проверка разделителей внутри диапазона ({start_pos}-{end_pos}) с учетом границ слова ({current_start}-{current_end}):")
    
    # Находим реальный конец слова (последнюю букву)
    last_letter = get_last_letter_pos(word_start, word_end, original_text)
    
    # Проверяем только ту часть, где есть буквы
    check_start = max(start_pos, current_start)
    check_end = min(end_pos, last_letter)
    
    for pos in range(check_start, check_end):
        if pos >= len(original_text):
            break
        char = original_text[pos]
        # Пробелы разрешены внутри слова (для многословных названий)
        # Все остальные разделители (знаки препинания, скобки и т.д.) отменяют расширение
        if char in constants.WORD_BREAKS and char not in constants.ALL_QUOTES and char != '-' and char != ' ':
            if verbose:
                logger.warning(f"         найден разделитель '{char}' на позиции {pos} внутри слова - расширение отменяется")
            return True
    return False

def expand_to_full_word(
    text: str,
    start_pos: int,
    end_pos: int,
    original_text: str,
    entity_type: str,
    config: dict,
    verbose: bool = False
) -> Tuple[str, str, int, int]:
    """
    Расширяет до полного слова в ОБЕ стороны.
    
    Returns:
        tuple: (расширенное слово, тип расширения, новый start, новый end)
    """
    if verbose:
        logger.warning(f"      🔧 _expand_to_full_word для '{text}' ({start_pos}-{end_pos})")
    
    # ----------------------------------------------------------------------
    # ЭТАП 1: ОПРЕДЕЛЕНИЕ ГРАНИЦ СЛОВА
    # ----------------------------------------------------------------------
    word_start, word_end, left_expanded, right_expanded = get_word_boundaries(
        start_pos, end_pos, original_text, config, verbose
    )
    
    # ----------------------------------------------------------------------
    # ЭТАП 2: ПРОВЕРКА ДОБАВЛЕННЫХ ЧАСТЕЙ
    # ----------------------------------------------------------------------
    if left_expanded:
        word_start, left_expanded = check_left_part(
            word_start, start_pos, original_text, verbose
        )
    
    if right_expanded:
        word_end, right_expanded = check_right_part(
            end_pos, word_end, original_text, verbose
        )
    
    # ----------------------------------------------------------------------
    # ЭТАП 3: ПРОВЕРКА ВНУТРЕННЕГО ДИАПАЗОНА
    # ----------------------------------------------------------------------
    full_word = original_text[word_start:word_end]
    
    if (left_expanded or right_expanded) and check_inner_range(
        start_pos, end_pos, word_start, word_end, original_text, verbose
    ):
        # =========================================================================
        # ВАЖНО! Найден разделитель внутри исходного диапазона
        # =========================================================================
        # Правильная стратегия:
        # - Левое расширение сохраняем (буквы слева - часть слова)
        # - Правое расширение отменяем (разделитель справа от start_pos)
        # - Никогда не сбрасываем word_start к start_pos!
        #
        # Пример: 'Алданов… Удивительная' - многоточие внутри слова
        # Левое расширение добавило 'А', его нужно сохранить
        # =========================================================================
        if verbose:
            logger.warning(f"         найден разделитель внутри слова - правое расширение отменяется")
        
        # Отменяем только правое расширение
        right_expanded = False
        # Возвращаем правую границу к исходной
        word_end = end_pos
        # Пересоздаём full_word с сохранённым левым расширением
        full_word = original_text[word_start:word_end]
    
    # ----------------------------------------------------------------------
    # ОПРЕДЕЛЕНИЕ ТИПА РАСШИРЕНИЯ
    # ----------------------------------------------------------------------
    if left_expanded and right_expanded:
        expand_type = 'both'
    elif left_expanded:
        expand_type = 'left'
    elif right_expanded:
        expand_type = 'right'
    else:
        expand_type = 'none'

    if verbose:
        logger.warning(f"      полное слово: '{full_word}'")
        logger.warning(f"      тип расширения: {expand_type}")
        logger.warning(f"      новые позиции: {word_start}-{word_end}")
    
    return full_word, expand_type, word_start, word_end