#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Финальная очистка расширенных сущностей.
"""

import logging
import re
import unicodedata
from typing import Dict

from . import constants

logger = logging.getLogger(__name__)


def clean_entity(
    text: str, 
    entity_type: str,
    config: dict, 
    verbose: bool = False
) -> str:
    """
    Третий этап: финальная очистка сущности.
    
    Правила:
    1. Сохраняем точки в инициалах и сокращениях (Г.В., г., ул., д.)
    2. Сохраняем дефисы в составных словах (пр-т, Ростов-на-Дону)
    3. Сохраняем парные кавычки, удаляем одиночные и множественные
    4. Удаляем все знаки препинания в начале строки (точки, запятые и т.д.)
    5. Удаляем пробелы в начале после удаления знаков
    6. Нормализуем пробелы
    7. Финальная проверка кавычек в начале/конце
    """
    if not text:
        return text
    
    original = text
    if verbose:
        logger.warning(f"      🧹 финальная очистка: '{text}'")
    
    # ----------------------------------------------------------------------
    # Шаг 0: Нормализация Unicode
    # ----------------------------------------------------------------------
    text = unicodedata.normalize('NFKC', text)
    
    # ----------------------------------------------------------------------
    # Шаг 1: Удаление служебных символов (кроме пробелов)
    # ----------------------------------------------------------------------
    control_cats = {'Cc', 'Cf', 'Cn', 'Co', 'Cs'}
    
    cleaned = []
    removed = []
    for char in text:
        cat = unicodedata.category(char)
        if cat not in control_cats or char in {' ', '\t', '\n', '\r', '\xa0'}:
            cleaned.append(char)
        else:
            removed.append(f"'{char}' (U+{ord(char):04X})")
    
    if removed and verbose:
        logger.warning(f"         удалены служебные символы: {', '.join(removed)}")
    
    text = ''.join(cleaned)
    
    # ----------------------------------------------------------------------
    # Шаг 2: Нормализация пробелов
    # ----------------------------------------------------------------------
    text = re.sub(r'[\s\xa0]+', ' ', text)
    
    # ----------------------------------------------------------------------
    # Шаг 3: Сохраняем точки в инициалах и сокращениях
    # ----------------------------------------------------------------------
    # 3.1 Инициалы (Г.В., А.Т., Т.П.)
    text = re.sub(r'([А-ЯA-Z])\.([А-ЯA-Z])(?=\.|\s|$)', r'\1@DOT@\2', text)
    
    # 3.2 Одиночные инициалы (Г., В., А.)
    text = re.sub(r'([А-ЯA-Z])\.(?=\s|$)', r'\1@DOT@', text)
    
    # 3.3 Сокращения (г., ул., д., пр., т.д., т.п.)
    abbreviations = ['г', 'ул', 'д', 'пр', 'тд', 'тп']
    for abbr in abbreviations:
        pattern = rf'({abbr})\.(?=\s|$)'
        text = re.sub(pattern, r'\1@DOT@', text, flags=re.IGNORECASE)
    
    # 3.4 Числа с точкой (151., 32а.)
    text = re.sub(r'(\d+)\.(?=\s|$)', r'\1@DOT@', text)
    
    if verbose and '@DOT@' in text:
        logger.warning(f"         защищены точки: {text.count('@DOT@')} шт.")
            
    # ----------------------------------------------------------------------
    # Шаг 4: Удаление знаков препинания и скобок с начала и конца
    # ----------------------------------------------------------------------
    REMOVE_START = set(constants.PUNCTUATION) | constants.ALL_BRACKETS

    # Очистка начала
    if text:
        start_removed = 0
        while text and text[0] in REMOVE_START:
            old_text = text
            text = text[1:]
            start_removed += 1
            if verbose and start_removed <= 5:
                logger.warning(f"         удален символ '{old_text[0]}' в начале")
        
        if start_removed > 0 and verbose:
            logger.warning(f"         удалено {start_removed} символов в начале")
        
        while text and text[0] == ' ':
            old_text = text
            text = text[1:]
            if verbose:
                logger.warning(f"         удален пробел в начале: '{old_text}' -> '{text}'")

    # Очистка конца
    if text:
        end_removed = 0
        changed = True
        while changed and text:
            changed = False
            old_text = text
            
            while text and text[-1] == ' ':
                text = text[:-1]
                changed = True
                end_removed += 1
                if verbose and end_removed <= 5:
                    logger.warning(f"         удален пробел в конце")
            
            while text and text[-1] in constants.PUNCTUATION_END:
                removed_char = text[-1]
                text = text[:-1]
                changed = True
                end_removed += 1
                if verbose and end_removed <= 5:
                    logger.warning(f"         удален знак '{removed_char}' в конце")
        
        if end_removed > 0 and verbose:
            logger.warning(f"         удалено {end_removed} символов в конце")
    
    # ----------------------------------------------------------------------
    # Шаг 5: Удаляем лишние дефисы в начале и конце
    # ----------------------------------------------------------------------
    text = text.lstrip('-')
    text = text.rstrip('-')
    
    # ----------------------------------------------------------------------
    # Шаг 6: Восстанавливаем точки
    # ----------------------------------------------------------------------
    text = text.replace('@DOT@', '.')
    
    # ----------------------------------------------------------------------
    # Шаг 7: Работа с парными символами (кавычки и скобки)
    # ----------------------------------------------------------------------
    pairs = [
        ('«', '»'), ('"', '"'), ("'", "'"), ('“', '”'), ('„', '‟'),
        ('(', ')'), ('[', ']'), ('{', '}')
    ]
    
    for open_char, close_char in pairs:
        open_count = text.count(open_char)
        close_count = text.count(close_char)
        
        if verbose and (open_count > 0 or close_count > 0):
            logger.warning(f"         символы {open_char}/{close_char}: {open_count} открыв, {close_count} закрыв")
        
        if open_count != close_count:
            if open_count > 0 and close_count == 0 and open_char == '(':
                if ')' not in text:
                    text = text.replace('(', '')
                    if verbose:
                        logger.warning(f"         удалена открывающая скобка без пары")
            else:
                text = text.replace(open_char, '')
                text = text.replace(close_char, '')
                if verbose and (open_count > 0 or close_count > 0):
                    logger.warning(f"         удалены все символы {open_char}/{close_char} (непарные)")
        else:
            if open_count > 1:
                first_open = text.find(open_char)
                last_close = text.rfind(close_char)
                
                if first_open != -1 and last_close != -1 and first_open < last_close:
                    middle = text[first_open+1:last_close].replace(open_char, '').replace(close_char, '')
                    new_text = text[:first_open] + open_char + middle + close_char + text[last_close+1:]
                    
                    if new_text != text:
                        if verbose:
                            logger.warning(f"         оставлена только одна пара {open_char}{close_char}")
                        text = new_text
    
    # ----------------------------------------------------------------------
    # Шаг 8: Очистка начала от сокращений (только для LOC)
    # ----------------------------------------------------------------------
    if config.get('enable_beginning_cleaning', False):
        # Передаём entity_type в функцию очистки
        text = clean_beginning(
            text, 
            entity_type,
            constants.ABBREVIATIONS, 
            constants.PUNCTUATION, 
            config,
            verbose
        )
    
    # ----------------------------------------------------------------------
    # Шаг 9: Финальный trim
    # ----------------------------------------------------------------------
    text = text.strip()
    text = ' '.join(text.split())
    
    if text != original and verbose:
        logger.warning(f"      🧹 итог очистки: '{original}' -> '{text}'")
    
    return text

def fix_edge_quotes(text: str, verbose: bool = False) -> str:
    """
    Исправляет висящие кавычки в начале и конце строки.
    TODO Не рабоатет так как треубется, отключено
    Правила:
    1. Если в начале нечетное количество кавычек - удаляем одну
    2. Если в конце нечетное количество кавычек - удаляем одну
    3. Четное количество оставляем как есть
    4. Кавычки внутри строки не трогаем вообще
    """
    if not text:
        return text
    
    original = text
    
    # ----------------------------------------------------------------------
    # Проверяем кавычки в начале
    # ----------------------------------------------------------------------
    leading = 0
    for char in text:
        if char in ['"', "'"]:
            leading += 1
        else:
            break
    
    if leading % 2 == 1 and leading > 0:
        text = text[1:]
        if verbose:
            logger.warning(f"         удалена одна кавычка в начале")
    
    # ----------------------------------------------------------------------
    # Проверяем кавычки в конце
    # ----------------------------------------------------------------------
    trailing = 0
    for char in reversed(text):
        if char in ['"', "'"]:
            trailing += 1
        else:
            break
    
    if trailing % 2 == 1 and trailing > 0:
        text = text[:-1]
        if verbose:
            logger.warning(f"         удалена одна кавычка в конце")
    
    return text

def strip_abbreviations(text: str, abbreviations: list, verbose: bool = False) -> str:
    """
    Удаляет сокращения из начала строки (только если они в начале и с учетом регистра).
    
    Args:
        text: исходная строка
        abbreviations: список сокращений для удаления
        verbose: флаг отладки
    
    Returns:
        str: строка без сокращения в начале
    """
    if not text:
        return text
    
    original = text
    text_lower = text.lower()
    
    for abbr in abbreviations:
        if text_lower.startswith(abbr):
            # Проверяем, что это действительно слово (после сокращения пробел или конец)
            remainder = text[len(abbr):]
            if not remainder or remainder[0] in [' ', '\t', '\n', '\r']:
                text = remainder.lstrip()
                if verbose:
                    logger.warning(f"         удалено сокращение '{abbr}' из начала")
                break
    
    return text


def strip_punctuation(text: str, punctuation: str, verbose: bool = False) -> str:
    """
    Рекурсивно удаляет знаки пунктуации из начала и конца строки.
    
    Args:
        text: исходная строка
        punctuation: строка со знаками пунктуации для удаления
        verbose: флаг отладки
    
    Returns:
        str: строка без пунктуации в начале и конце
    """
    if not text:
        return text
    
    original = text
    changed = True
    
    while changed and text:
        changed = False
        old_text = text
        
        # Удаляем с начала
        while text and text[0] in punctuation:
            text = text[1:]
            changed = True
        
        # Удаляем с конца
        while text and text[-1] in punctuation:
            text = text[:-1]
            changed = True
        
        if changed and verbose:
            logger.warning(f"         удалены знаки: '{old_text}' -> '{text}'")
    
    return text


def clean_beginning(
    text: str, 
    entity_type: str,
    abbreviations: list, 
    punctuation: str, 
    config: dict,
    verbose: bool = False
) -> str:
    """
    Комплексная очистка начала строки (только для LOC).
    
    Args:
        text: исходная строка
        entity_type: тип сущности (LOC, PER, ORG)
        abbreviations: список сокращений для удаления
        punctuation: строка со знаками пунктуации
        config: конфиг с параметрами
        verbose: флаг отладки
    
    Returns:
        str: очищенная строка
    """
    if not text:
        return text
    
    # Очищаем только для LOC
    if entity_type != 'LOC':
        return text
    
    original = text
    changed = True
    iteration = 0
    
    while changed:
        changed = False
        iteration += 1
        old_text = text
        
        # Шаг 1: удаляем сокращения
        new_text = strip_abbreviations(text, abbreviations, verbose)
        if new_text != text:
            text = new_text
            changed = True
            if verbose:
                logger.warning(f"         итерация {iteration}: после удаления сокращений")
        
        # Шаг 2: trim пробелов
        new_text = text.strip()
        if new_text != text:
            text = new_text
            changed = True
            if verbose:
                logger.warning(f"         итерация {iteration}: после trim")
        
        # Шаг 3: рекурсивно удаляем пунктуацию
        new_text = strip_punctuation(text, punctuation, verbose)
        if new_text != text:
            text = new_text
            changed = True
            if verbose:
                logger.warning(f"         итерация {iteration}: после удаления пунктуации")
    
    return text