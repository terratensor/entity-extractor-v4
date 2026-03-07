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
    Работает в обе стороны (влево и вправо) с экспериментальными параметрами.
    """
    
    # [ЭКСПЕРИМЕНТАЛЬНЫЕ ПАРАМЕТРЫ ЗНАЧЕНИЯ ПО УМОЛЧАНИЮ]
    DEFAULTS = {
        # Минимальная длина токена для расширения
        'min_token_length': 2,
        
        # Максимальное расстояние поиска влево/вправо (символов)
        'max_search_left': 30,
        'max_search_right': 30,
        
        # Минимальная доля исходного текста в расширенном (0.3 = 30%)
        'min_coverage': 0.3,
        
        # Требовать заглавную букву для LOC/PER
        'require_capital': True,
        
        # Включить проверку на стоп-слова
        'enable_stopwords': True,
        
        # Включить проверку на слияние
        'enable_merge_check': True,
        
        # Максимальное соотношение длин (расширенное / исходное)
        'max_length_ratio': 3.0
    }
    
    # Стоп-слова (предлоги, союзы, частицы)
    STOP_WORDS = {
        'в', 'на', 'с', 'у', 'к', 'о', 'об', 'от', 'до', 'по', 'за', 'над',
        'под', 'без', 'для', 'ради', 'из', 'через', 'и', 'а', 'но', 'же',
        'бы', 'ли', 'вот', 'это', 'что', 'как', 'так', 'все', 'еще', 'уже',
        'даже', 'только', 'чтобы', 'если', 'потому', 'поэтому'
    }
    
    # Гласные для проверки слияния
    VOWELS = set('аеёиоуыэюя')
    
    # Разделители слов
    WORD_BREAKS = set(' .,!?;:()[]{}«»""\'\n\r\t')
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Args:
            config: словарь с параметрами расширения (переопределяют DEFAULTS)
        """
        self.config = self.DEFAULTS.copy()
        if config:
            self.config.update(config)
        
        self.stats = {
            'attempts': 0,
            'expanded_left': 0,
            'expanded_right': 0,
            'expanded_both': 0,
            'rejected': 0,
            'rejected_stopword': 0,
            'rejected_capital': 0,
            'rejected_merge': 0,
            'rejected_coverage': 0,
            'rejected_length': 0
        }
        
        logger.info(f"🤖 WordExpander инициализирован с параметрами:")
        for key, value in self.config.items():
            logger.info(f"   {key}: {value}")
    
    def expand_entity(self, entity: Dict, original_text: str) -> Dict:
        """
        Расширяет одну сущность, если это необходимо.
        
        Args:
            entity: словарь сущности с полями text, type, confidence, positions
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
        
        # Берем первую и последнюю позицию
        first_pos = entity['positions'][0]
        last_pos = entity['positions'][-1]
        start_pos = first_pos['start']
        end_pos = last_pos['end']
        
        # [ОТЛАДКА] Печатаем информацию о сущности
        logger.warning(f"🔍 РАСШИРЕНИЕ: '{entity['text']}' ({entity['type']}) "
                    f"позиции: {start_pos}-{end_pos}")
        logger.warning(f"   Текст вокруг: '{original_text[max(0, start_pos-20):min(len(original_text), end_pos+20)]}'")

        # Проверяем, нужно ли расширять
        should, reason = self._should_expand(
            entity['text'], start_pos, end_pos, original_text, entity['type']
        )

        logger.warning(f"   _should_expand: {should}, причина: {reason}")
        
        if not should:
            self.stats['rejected'] += 1
            if reason:
                self.stats[f'rejected_{reason}'] += 1
            return entity
        
        # Расширяем
        expanded, expand_type = self._expand_to_full_word(
            entity['text'], start_pos, end_pos, original_text, entity['type']
        )
        
        if expanded != entity['text']:
            # Обновляем статистику по типу расширения
            if expand_type == 'left':
                self.stats['expanded_left'] += 1
            elif expand_type == 'right':
                self.stats['expanded_right'] += 1
            elif expand_type == 'both':
                self.stats['expanded_both'] += 1
            
            # Создаем копию с расширенным текстом
            result = entity.copy()
            result['text'] = expanded
            result['expanded'] = True
            result['expansion_type'] = expand_type
            result['original_text'] = entity['text']
            return result
        
        return entity
    
    def _should_expand(self, text: str, start_pos: int, end_pos: int,
                      original_text: str, entity_type: str) -> tuple:
        """
        Проверяет, нужно ли расширять сущность.
        
        Returns:
            (bool, str): (нужно ли расширять, причина отказа)
        """
        logger.warning(f"   _should_expand проверка для '{text}':")

        # [ПАРАМЕТР] Проверка 1: минимальная длина
        if len(text) < self.config['min_token_length']:
            logger.warning(f"      ❌ rejected_length: длина {len(text)} < {self.config['min_token_length']}")
            return False, 'length'
        
        # [ПАРАМЕТР] Проверка 2: стоп-слова
        if self.config['enable_stopwords']:
            clean_text = text.lower().strip('▁')
            if clean_text in self.STOP_WORDS:
                logger.warning(f"      ❌ rejected_stopword: '{clean_text}' в стоп-словах")
                return False, 'stopword'
        
        # Проверка 3: если после end_pos сразу пробел - слово полное
        if end_pos >= len(original_text):
            logger.warning(f"      ❌ конец текста")
            return False, None
        
        next_char = original_text[end_pos]
        logger.warning(f"      следующий символ: '{next_char}' (код {ord(next_char)})")

        if next_char in self.WORD_BREAKS:
            logger.warning(f"      ❌ следующий символ - разделитель")
            return False, None
        
        # Проверка 4: следующий символ должен быть буквой
        if not next_char.isalpha():
            logger.warning(f"      ❌ следующий символ не буква")
            return False, None
        
        # [ПАРАМЕТР] Проверка 5: следующая буква должна быть строчной
        # (новые имена обычно с заглавной)
        if self.config['require_capital'] and entity_type in ['LOC', 'PER']:
            if not next_char.islower():
                logger.warning(f"      ❌ rejected_capital: следующая буква '{next_char}' не строчная")
                return False, 'capital'
        
        logger.warning(f"      ✅ можно расширять")
        return True, None
    
    def _expand_to_full_word(self, text: str, start_pos: int, end_pos: int,
                            original_text: str, entity_type: str) -> tuple:
        """
        Расширяет до полного слова в ОБЕ стороны.
        """
        logger.warning(f"      🔧 _expand_to_full_word для '{text}' ({start_pos}-{end_pos})")
        
        # [ПАРАМЕТР] Максимальное расстояние поиска
        max_left = self.config['max_search_left']
        max_right = self.config['max_search_right']
        
        # 1. Ищем начало слова (влево)
        word_start = start_pos
        left_expanded = False
        steps_left = 0
        
        logger.warning(f"      поиск влево от {start_pos}:")
        while word_start > 0 and steps_left < max_left:
            prev_char = original_text[word_start - 1]
            logger.warning(f"        символ {word_start-1}: '{prev_char}'")
            if prev_char in self.WORD_BREAKS:
                logger.warning(f"          стоп - разделитель")
                break
            if not prev_char.isalpha():
                logger.warning(f"          стоп - не буква")
                break
            word_start -= 1
            steps_left += 1
            left_expanded = True
            logger.warning(f"          добавлен влево, теперь начало {word_start}")
        
        # 2. Ищем конец слова (вправо)
        word_end = end_pos
        right_expanded = False
        steps_right = 0
        
        logger.warning(f"      поиск вправо от {end_pos}:")
        while word_end < len(original_text) and steps_right < max_right:
            next_char = original_text[word_end]
            logger.warning(f"        символ {word_end}: '{next_char}'")
            if next_char in self.WORD_BREAKS:
                logger.warning(f"          стоп - разделитель")
                break
            if not next_char.isalpha():
                logger.warning(f"          стоп - не буква")
                break
            word_end += 1
            steps_right += 1
            right_expanded = True
            logger.warning(f"          добавлен вправо, теперь конец {word_end}")
        
        # Полное слово из оригинала
        full_word = original_text[word_start:word_end]
        
        # Определяем тип расширения
        if left_expanded and right_expanded:
            expand_type = 'both'
        elif left_expanded:
            expand_type = 'left'
        elif right_expanded:
            expand_type = 'right'
        else:
            expand_type = 'none'
        
        logger.warning(f"      полное слово: '{full_word}'")
        logger.warning(f"      тип расширения: {expand_type}")
        
        # [ПАРАМЕТР] Проверка на слишком длинное расширение
        if len(full_word) > len(text) * self.config['max_length_ratio']:
            logger.warning(f"      ❌ слишком длинное: {len(full_word)} > {len(text)} * {self.config['max_length_ratio']}")
            return text, 'none'
        
        # Проверка: исходный текст должен быть подстрокой полного слова
        if text not in full_word:
            logger.warning(f"      ❌ исходный текст '{text}' не подстрока '{full_word}'")
            return text, 'none'
        
        # [ПАРАМЕТР] Проверка на минимальное покрытие
        coverage = len(text) / len(full_word) if len(full_word) > 0 else 0
        if coverage < self.config['min_coverage']:
            logger.warning(f"      ❌ покрытие {coverage:.2f} < {self.config['min_coverage']}")
            return text, 'none'
        
        # [ПАРАМЕТР] Проверка на заглавные для LOC/PER
        if self.config['require_capital'] and entity_type in ['LOC', 'PER']:
            # Первая буква полного слова должна быть заглавной
            words = full_word.split()
            for w in words:
                if w and w[0].isalpha() and not w[0].isupper():
                    logger.warning(f"      ❌ слово '{w}' начинается с маленькой буквы")
                    return text, 'none'
        
        # [ПАРАМЕТР] Проверка на слияние
        if self.config['enable_merge_check']:
            if self._check_word_merge(original_text, full_word, start_pos, end_pos):
                logger.warning(f"      ❌ обнаружено слияние слов")
                return text, 'none'
        
        logger.warning(f"      ✅ расширение: '{text}' -> '{full_word}'")
        return full_word, expand_type
    
    def _check_word_merge(self, original_text: str, full_word: str,
                        start_pos: int, end_pos: int) -> bool:
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
                last_char.lower() not in self.VOWELS and
                next_char.lower() in self.VOWELS):
                
                # ДОПОЛНИТЕЛЬНЫЕ ПРОВЕРКИ:
                
                # 1. Проверяем, что после гласной есть буквы (это часть слова, а не окончание)
                if end_pos + 1 < len(original_text):
                    next_next = original_text[end_pos + 1]
                    if next_next.isalpha():
                        # Если после гласной еще буквы - это часть слова, а не окончание
                        # Значит, это НЕ слияние
                        return False
                
                # 2. Проверяем, не является ли это типичным окончанием
                # Типичные окончания в русском языке: а, я, ы, и, е, ё, ю, й
                common_endings = {'а', 'я', 'ы', 'и', 'е', 'ё', 'ю', 'й'}
                if next_char.lower() in common_endings:
                    # Проверяем, что после окончания идет пробел
                    if end_pos + 1 >= len(original_text) or original_text[end_pos + 1] in self.WORD_BREAKS:
                        # Это нормальное окончание, а не слияние
                        return False
                
                # 3. Проверяем длину исходного слова
                # Если исходное слово очень короткое (1-2 буквы), это может быть предлог
                if len(full_word) < 3:
                    # Проверяем по стоп-словам
                    if full_word.lower() in self.STOP_WORDS:
                        return True  # это предлог - возможно слияние
                
                # Если все проверки пройдены - это подозрительно
                return True
        
        return False
    
    def get_stats(self) -> Dict:
        """Возвращает статистику расширений."""
        stats = self.stats.copy()
        
        # Добавляем проценты для удобства
        if stats['attempts'] > 0:
            stats['expanded_total'] = (stats['expanded_left'] + 
                                       stats['expanded_right'] + 
                                       stats['expanded_both'])
            stats['expand_percent'] = round(
                stats['expanded_total'] / stats['attempts'] * 100, 1
            )
        
        return stats