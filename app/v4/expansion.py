#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Модуль для расширения сущностей до полных слов.
Отдельный этап пост-обработки в WriterWorker.
"""

import logging
from typing import List, Dict, Optional, Set

logger = logging.getLogger(__name__)


class WordExpander:
    """
    Расширяет найденные сущности до полных слов, используя оригинальный текст.
    Работает только forward expansion (добавляет окончания).
    """
    
    # [ЭКСПЕРИМЕНТАЛЬНЫЙ ПАРАМЕТР] Минимальная длина токена для расширения
    MIN_TOKEN_LENGTH = 2
    
    # [ЭКСПЕРИМЕНТАЛЬНЫЙ ПАРАМЕТР] Стоп-слова (предлоги, союзы, частицы)
    STOP_WORDS = {
        'в', 'на', 'с', 'у', 'к', 'о', 'об', 'от', 'до', 'по', 'за', 'над',
        'под', 'без', 'для', 'ради', 'из', 'через', 'и', 'а', 'но', 'же',
        'бы', 'ли', 'вот', 'это', 'что', 'как', 'так', 'все', 'еще', 'уже'
    }
    
    # [ЭКСПЕРИМЕНТАЛЬНЫЙ ПАРАМЕТР] Гласные для проверки слияния
    VOWELS = set('аеёиоуыэюя')
    
    # [ЭКСПЕРИМЕНТАЛЬНЫЙ ПАРАМЕТР] Разделители слов
    WORD_BREAKS = set(' .,!?;:()[]{}«»""\'\n\r\t')
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Args:
            config: словарь с параметрами расширения
        """
        self.config = config or {}
        self.stats = {
            'attempts': 0,
            'expanded': 0,
            'rejected': 0
        }
    
    def expand_entity(self, entity: Dict, original_text: str) -> Dict:
        """
        Расширяет одну сущность, если это необходимо.
        
        Args:
            entity: словарь сущности с полями text, type, confidence, (опционально positions)
            original_text: полный текст документа
            
        Returns:
            Dict: сущность с возможным расширенным текстом
        """
        # Расширяем только LOC и PER
        if entity['type'] not in ['LOC', 'PER']:
            return entity
        
        # Нужны позиции для расширения
        if 'positions' not in entity or not entity['positions']:
            return entity
        
        self.stats['attempts'] += 1
        
        # Берем последнюю позицию
        last_pos = entity['positions'][-1]
        start_pos = entity['positions'][0]['start']
        end_pos = last_pos['end']
        
        # Проверяем, нужно ли расширять
        if not self._should_expand(entity['text'], end_pos, original_text):
            return entity
        
        # Расширяем
        expanded = self._expand_to_full_word(
            entity['text'], start_pos, end_pos, original_text, entity['type']
        )
        
        if expanded != entity['text']:
            self.stats['expanded'] += 1
            # Создаем копию с расширенным текстом
            result = entity.copy()
            result['text'] = expanded
            result['expanded'] = True
            result['original_text'] = entity['text']
            return result
        
        self.stats['rejected'] += 1
        return entity
    
    def _should_expand(self, text: str, end_pos: int, original_text: str) -> bool:
        """
        Проверяет, нужно ли расширять сущность.
        
        [ЭКСПЕРИМЕНТАЛЬНЫЕ ПРОВЕРКИ]
        """
        # Проверка 1: текст не должен быть слишком коротким
        if len(text) < self.MIN_TOKEN_LENGTH:
            return False
        
        # Проверка 2: не расширяем стоп-слова
        if text.lower().strip('▁') in self.STOP_WORDS:
            return False
        
        # Проверка 3: если после end_pos сразу пробел - слово полное
        if end_pos >= len(original_text):
            return False
        
        next_char = original_text[end_pos]
        if next_char in self.WORD_BREAKS:
            return False
        
        # Проверка 4: следующий символ должен быть буквой
        if not next_char.isalpha():
            return False
        
        # Проверка 5: следующая буква должна быть строчной
        # (новые имена обычно с заглавной)
        if not next_char.islower():
            return False
        
        return True
    
    def _expand_to_full_word(self, text: str, start_pos: int, end_pos: int,
                            original_text: str, entity_type: str) -> str:
        """
        Расширяет до полного слова.
        
        [ЭКСПЕРИМЕНТАЛЬНЫЙ АЛГОРИТМ]
        """
        # Ищем конец слова (до пробела или разделителя)
        word_end = end_pos
        while word_end < len(original_text):
            char = original_text[word_end]
            if char in self.WORD_BREAKS:
                break
            word_end += 1
        
        # Расширяем текст
        expanded = original_text[start_pos:word_end]
        
        # Проверка на слияние (последний символ исходного - согласная,
        # первый добавленный - гласная)
        if len(text) > 0 and word_end > end_pos:
            last_orig = text[-1]
            first_added = original_text[end_pos]
            
            if (last_orig.isalpha() and first_added.isalpha() and
                last_orig.lower() not in self.VOWELS and
                first_added.lower() in self.VOWELS):
                # Потенциальное слияние, возвращаем исходное
                return text
        
        return expanded
    
    def get_stats(self) -> Dict:
        """Возвращает статистику расширений."""
        return self.stats.copy()