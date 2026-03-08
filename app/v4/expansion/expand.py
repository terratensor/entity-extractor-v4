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
    
    max_left = config['max_search_left']
    max_right = config['max_search_right']
    
    # ----------------------------------------------------------------------
    # РАСШИРЕНИЕ ВЛЕВО
    # Ищем начало слова, двигаясь влево от start_pos
    # Останавливаемся на разделителях (пробелы, знаки препинания, кроме кавычек)
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
            
            # Останавливаемся на разделителях (но пропускаем кавычки)
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
    # Ищем конец слова, двигаясь вправо от end_pos
    # Останавливаемся на разделителях (пробелы, знаки препинания, кроме кавычек)
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
            
            # Останавливаемся на разделителях (но пропускаем кавычки)
            if next_char in constants.WORD_BREAKS and next_char not in constants.ALL_QUOTES:
                if verbose:
                    logger.warning(f"          стоп - разделитель")
                break
            
            word_end += 1
            steps_right += 1
            right_expanded = True
            if verbose:
                logger.warning(f"          добавлен вправо, теперь конец {word_end}")
    
    # ----------------------------------------------------------------------
    # ФОРМИРОВАНИЕ РЕЗУЛЬТАТА
    # ----------------------------------------------------------------------
    full_word = original_text[word_start:word_end]
    
    # ----------------------------------------------------------------------
    # ПРОВЕРКА НАЛИЧИЯ РАЗДЕЛИТЕЛЕЙ
    # ----------------------------------------------------------------------            
    if left_expanded or right_expanded:
        
        # ----------------------------------------------------------------------
        # ПРОВЕРКА 1: разделители в ЛЕВОЙ добавленной части
        # ----------------------------------------------------------------------
        if left_expanded:
            if verbose:
                logger.warning(f"      проверка левой добавленной части ({word_start}-{start_pos}):")
            
            for pos in range(word_start, start_pos):
                if pos >= len(original_text):
                    break
                char = original_text[pos]
                if char in constants.WORD_BREAKS and char not in constants.ALL_QUOTES and char != '-':
                    if verbose:
                        logger.warning(f"         найден разделитель '{char}' на позиции {pos} в левой части - левое расширение отменяется")
                    left_expanded = False
                    word_start = start_pos
                    full_word = original_text[word_start:word_end]
                    break
        
        # ----------------------------------------------------------------------
        # ПРОВЕРКА 2: разделители в ПРАВОЙ добавленной части
        # ----------------------------------------------------------------------
        if right_expanded:
            if verbose:
                logger.warning(f"      проверка правой добавленной части ({end_pos}-{word_end}):")
            
            for pos in range(end_pos, word_end):
                if pos >= len(original_text):
                    break
                char = original_text[pos]
                if char in constants.WORD_BREAKS and char not in constants.ALL_QUOTES and char != '-':
                    if verbose:
                        logger.warning(f"         найден разделитель '{char}' на позиции {pos} в правой части - правое расширение отменяется")
                    right_expanded = False
                    word_end = end_pos
                    full_word = original_text[word_start:word_end]
                    break
        
        # =========================================================================
        # ПРОВЕРКА 3: разделители ВНУТРИ исходного диапазона
        # =========================================================================
        if left_expanded or right_expanded:
            current_start = word_start
            current_end = word_end
            
            if verbose:
                logger.warning(f"      проверка разделителей внутри диапазона ({start_pos}-{end_pos}) с учетом границ слова ({current_start}-{current_end}):")
            
            check_start = max(start_pos, current_start)
            check_end = min(end_pos, current_end)
            
            for pos in range(check_start, check_end):
                if pos >= len(original_text):
                    break
                char = original_text[pos]
                if char in constants.WORD_BREAKS and char not in constants.ALL_QUOTES and char != '-':
                    if verbose:
                        logger.warning(f"         найден разделитель '{char}' на позиции {pos} внутри слова - расширение отменяется")
                    left_expanded = False
                    right_expanded = False
                    word_start = start_pos
                    word_end = end_pos
                    full_word = original_text[word_start:word_end]
                    break
    
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