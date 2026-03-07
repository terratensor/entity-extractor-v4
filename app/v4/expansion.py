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
        'max_length_ratio': 5.0,
        
        # Типы сущностей для расширения
        'expand_entity_types': ['LOC', 'PER', 'ORG'],
        
        # Включить финальную очистку
        'enable_final_cleaning': True
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
    
    # Разделители слов (пробелы и знаки препинания, НО не кавычки!)
    WORD_BREAKS = set(' .,!?;:()[]{}""\'\n\r\t')
    
    # Кавычки (разные типографские варианты)
    OPEN_QUOTES = {'«', '“', '„', '"', "'"}
    CLOSE_QUOTES = {'»', '”', '‟', '"', "'"}
    ALL_QUOTES = OPEN_QUOTES | CLOSE_QUOTES
    
    # Пунктуация для удаления в начале/конце (кроме кавычек)
    PUNCTUATION_START = '.,!?;:'
    PUNCTUATION_END = '.,!?;:'
    
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
            'rejected_length': 0,
            'cleaned': 0
        }
        
        logger.info(f"🤖 WordExpander инициализирован с параметрами:")
        for key, value in self.config.items():
            logger.info(f"   {key}: {value}")
    
    def expand_entity(self, entity: Dict, original_text: str) -> Dict:
        """
        Расширяет одну сущность, если это необходимо, затем очищает.
        
        Args:
            entity: словарь сущности с полями text, type, confidence, positions
            original_text: полный текст документа
            
        Returns:
            Dict: сущность с возможным расширенным текстом
        """
        # Проверяем, нужно ли расширять этот тип сущности
        expand_types = self.config.get('expand_entity_types', ['LOC', 'PER'])
        if entity['type'] not in expand_types:
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
        
        # Отладка
        logger.warning(f"🔍 РАСШИРЕНИЕ: '{entity['text']}' ({entity['type']}) "
                      f"позиции: {start_pos}-{end_pos}")
        logger.warning(f"   Текст вокруг: '{original_text[max(0, start_pos-20):min(len(original_text), end_pos+20)]}'")

        # Проверяем, нужно ли расширять
        should, reason = self._should_expand(
            entity['text'], start_pos, end_pos, original_text, entity['type']
        )

        logger.warning(f"   _should_expand: {should}, причина: {reason}")
        
        result_entity = entity.copy()
        
        if should:
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
                
                result_entity['text'] = expanded
                result_entity['expanded'] = True
                result_entity['expansion_type'] = expand_type
                result_entity['original_text'] = entity['text']
        
        # ----------------------------------------------------------------------
        # ТРЕТИЙ ЭТАП: финальная очистка результата
        # ----------------------------------------------------------------------
        if self.config.get('enable_final_cleaning', True):
            cleaned_text = self._clean_entity(result_entity['text'])
            if cleaned_text != result_entity['text']:
                logger.warning(f"   🧹 финальная очистка: '{result_entity['text']}' -> '{cleaned_text}'")
                result_entity['text'] = cleaned_text
                result_entity['cleaned'] = True
                self.stats['cleaned'] += 1
        
        return result_entity
    
    def _clean_entity(self, text: str) -> str:
        """
        Третий этап: финальная очистка сущности от лишних знаков препинания.
        Вызывается после всех расширений.
        
        Правила:
        1. Удалить знаки препинания в начале и конце (.,!?;:)
        2. Для кавычек: если только открывающая или только закрывающая - удалить
        3. Если есть и открывающая и закрывающая - оставить обе
        """
        if not text:
            return text
        
        original = text
        logger.warning(f"      🧹 финальная очистка: '{text}'")
        
        # ----------------------------------------------------------------------
        # Правило 1: Удаляем знаки препинания в начале
        # ----------------------------------------------------------------------
        text = text.lstrip(self.PUNCTUATION_START)
        if text != original:
            logger.warning(f"         удалены знаки в начале: '{original}' -> '{text}'")
            original = text
        
        # ----------------------------------------------------------------------
        # Правило 2: Удаляем знаки препинания в конце
        # ----------------------------------------------------------------------
        text = text.rstrip(self.PUNCTUATION_END)
        if text != original:
            logger.warning(f"         удалены знаки в конце: '{original}' -> '{text}'")
            original = text
        
        # ----------------------------------------------------------------------
        # Правило 3: Проверяем парность кавычек
        # ----------------------------------------------------------------------
        # Проверяем первый символ
        has_open = False
        has_close = False
        open_char = None
        close_char = None
        
        if text and text[0] in self.OPEN_QUOTES:
            has_open = True
            open_char = text[0]
            logger.warning(f"         найдена открывающая кавычка в начале: '{open_char}'")
        
        if text and text[-1] in self.CLOSE_QUOTES:
            has_close = True
            close_char = text[-1]
            logger.warning(f"         найдена закрывающая кавычка в конце: '{close_char}'")
        
        # Если есть открывающая, но нет закрывающей - удаляем открывающую
        if has_open and not has_close:
            text = text[1:]
            logger.warning(f"         удалена открывающая кавычка без пары: '{original}' -> '{text}'")
            original = text
        
        # Если есть закрывающая, но нет открывающей - удаляем закрывающую
        if has_close and not has_open:
            text = text[:-1]
            logger.warning(f"         удалена закрывающая кавычка без пары: '{original}' -> '{text}'")
            original = text
        
        # ----------------------------------------------------------------------
        # Правило 4: Дополнительная проверка - нет ли точки в начале после всех чисток
        # ----------------------------------------------------------------------
        if text and text[0] in self.PUNCTUATION_START:
            text = text.lstrip(self.PUNCTUATION_START)
            logger.warning(f"         повторная очистка начала: '{original}' -> '{text}'")
        
        return text
    
    def _should_expand(self, text: str, start_pos: int, end_pos: int,
                      original_text: str, entity_type: str) -> tuple:
        """
        Проверяет, нужно ли расширять сущность (в обе стороны).
        Каждая проверка отдельно и четко прокомментирована.
        
        Returns:
            (bool, str): (нужно ли расширять, причина отказа)
        """
        logger.warning(f"   _should_expand проверка для '{text}':")
        logger.warning(f"      start_pos={start_pos}, end_pos={end_pos}")
        
        # ----------------------------------------------------------------------
        # ПРОВЕРКА 1: Минимальная длина
        # ----------------------------------------------------------------------
        if len(text) < self.config['min_token_length']:
            logger.warning(f"      ❌ rejected_length: длина {len(text)} < {self.config['min_token_length']}")
            return False, 'length'
        
        # ----------------------------------------------------------------------
        # ПРОВЕРКА 2: Стоп-слова (предлоги, союзы, частицы)
        # ----------------------------------------------------------------------
        if self.config['enable_stopwords']:
            clean_text = text.lower().strip('▁')
            if clean_text in self.STOP_WORDS:
                logger.warning(f"      ❌ rejected_stopword: '{clean_text}' в стоп-словах")
                return False, 'stopword'
        
        # ----------------------------------------------------------------------
        # ПРОВЕРКА 3: Возможность расширения ВЛЕВО
        # ----------------------------------------------------------------------
        can_expand_left = False
        left_reason = None
        
        if start_pos > 0:
            prev_char = original_text[start_pos - 1]
            logger.warning(f"      символ слева: '{prev_char}' (код {ord(prev_char)})")
            
            # Условие 3.1: Слева буква или кавычка (не разделитель)
            if prev_char.isalpha() or prev_char in self.ALL_QUOTES:
                # Условие 3.2: Это начало слова (перед ним разделитель или начало текста)
                if start_pos - 1 == 0 or original_text[start_pos - 2] in self.WORD_BREAKS:
                    can_expand_left = True
                    logger.warning(f"      ✅ можно расширять ВЛЕВО")
                else:
                    left_reason = "не начало слова"
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
            logger.warning(f"      символ справа: '{next_char}' (код {ord(next_char)})")
            
            # Условие 4.1: Справа буква или кавычка (не разделитель)
            if next_char.isalpha() or next_char in self.ALL_QUOTES:
                can_expand_right = True
                logger.warning(f"      ✅ можно расширять ВПРАВО")
            else:
                right_reason = "не буква и не кавычка"
        else:
            right_reason = "конец текста"
        
        # ----------------------------------------------------------------------
        # ПРОВЕРКА 5: Для LOC/PER проверяем заглавные буквы (только для правого расширения)
        # ----------------------------------------------------------------------
        if (self.config['require_capital'] and entity_type in ['LOC', 'PER'] and 
            can_expand_right and end_pos < len(original_text)):
            next_char = original_text[end_pos]
            if next_char.isalpha() and not next_char.isupper():
                # Если следующая буква строчная - это нормально (продолжение слова)
                # Но если это начало нового слова - должна быть заглавной
                if end_pos > 0 and original_text[end_pos - 1] in self.WORD_BREAKS:
                    logger.warning(f"      ❌ rejected_capital: следующая буква '{next_char}' не заглавная")
                    return False, 'capital'
        
        # ----------------------------------------------------------------------
        # ИТОГ: расширяем, если есть возможность хотя бы в одну сторону
        # ----------------------------------------------------------------------
        if can_expand_left or can_expand_right:
            logger.warning(f"      ✅ можно расширять (влево={can_expand_left}, вправо={can_expand_right})")
            return True, None
        else:
            logger.warning(f"      ❌ нет расширения: влево={left_reason}, вправо={right_reason}")
            return False, None
    
    def _expand_to_full_word(self, text: str, start_pos: int, end_pos: int,
                            original_text: str, entity_type: str) -> tuple:
        """
        Расширяет до полного слова в ОБЕ стороны.
        """
        logger.warning(f"      🔧 _expand_to_full_word для '{text}' ({start_pos}-{end_pos})")
        
        max_left = self.config['max_search_left']
        max_right = self.config['max_search_right']
        
        # ----------------------------------------------------------------------
        # РАСШИРЕНИЕ ВЛЕВО
        # ----------------------------------------------------------------------
        word_start = start_pos
        left_expanded = False
        steps_left = 0
        
        if start_pos > 0:
            logger.warning(f"      поиск влево от {start_pos}:")
            while word_start > 0 and steps_left < max_left:
                prev_char = original_text[word_start - 1]
                logger.warning(f"        символ {word_start-1}: '{prev_char}'")
                
                if prev_char in self.WORD_BREAKS and prev_char not in self.ALL_QUOTES:
                    logger.warning(f"          стоп - разделитель")
                    break
                
                word_start -= 1
                steps_left += 1
                left_expanded = True
                logger.warning(f"          добавлен влево, теперь начало {word_start}")
        
        # ----------------------------------------------------------------------
        # РАСШИРЕНИЕ ВПРАВО
        # ----------------------------------------------------------------------
        word_end = end_pos
        right_expanded = False
        steps_right = 0
        
        if end_pos < len(original_text):
            logger.warning(f"      поиск вправо от {end_pos}:")
            while word_end < len(original_text) and steps_right < max_right:
                next_char = original_text[word_end]
                logger.warning(f"        символ {word_end}: '{next_char}'")
                
                if next_char in self.WORD_BREAKS and next_char not in self.ALL_QUOTES:
                    logger.warning(f"          стоп - разделитель")
                    break
                
                word_end += 1
                steps_right += 1
                right_expanded = True
                logger.warning(f"          добавлен вправо, теперь конец {word_end}")
        
        # ----------------------------------------------------------------------
        # ФОРМИРОВАНИЕ РЕЗУЛЬТАТА
        # ----------------------------------------------------------------------
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
        
        # ----------------------------------------------------------------------
        # ПРОВЕРКИ РЕЗУЛЬТАТА
        # ----------------------------------------------------------------------
        
        # Проверка 1: слишком длинное расширение
        if len(full_word) > len(text) * self.config['max_length_ratio']:
            logger.warning(f"      ❌ слишком длинное: {len(full_word)} > {len(text)} * {self.config['max_length_ratio']}")
            return text, 'none'
        
        # ----------------------------------------------------------------------
        # ПРОВЕРКА 2: для расширенных слов доверяем оригинальному тексту
        # ----------------------------------------------------------------------
        if left_expanded or right_expanded:
            # Мы расширили слово, используя оригинальный текст
            # Доверяем оригиналу, даже если оно не содержит исходный текст модели
            logger.warning(f"      ✅ слово расширено, доверяем оригиналу")
        else:
            # Слово не расширялось, проверяем подстроку как обычно
            if text not in full_word:
                logger.warning(f"      ❌ исходный текст не подстрока (без расширения)")
                return text, 'none'
        
        # ----------------------------------------------------------------------
        # ПРОВЕРКА 3: минимальное покрытие
        # ----------------------------------------------------------------------
        coverage = len(text) / len(full_word) if len(full_word) > 0 else 0
        if coverage < self.config['min_coverage']:
            logger.warning(f"      ❌ покрытие {coverage:.2f} < {self.config['min_coverage']}")
            return text, 'none'
        
        # ----------------------------------------------------------------------
        # ПРОВЕРКА 4: заглавные буквы для LOC/PER (пропуская кавычки)
        # ----------------------------------------------------------------------
        if (self.config['require_capital'] and entity_type in ['LOC', 'PER'] and left_expanded):
            # Ищем первую букву (пропуская кавычки)
            first_letter = None
            for char in full_word:
                if char.isalpha():
                    first_letter = char
                    break
            
            if first_letter is None:
                logger.warning(f"      ❌ в слове нет букв")
                return text, 'none'
            
            if not first_letter.isupper():
                logger.warning(f"      ❌ первая буква '{first_letter}' не заглавная")
                return text, 'none'
            else:
                logger.warning(f"      ✅ первая буква '{first_letter}' заглавная")
        
        # ----------------------------------------------------------------------
        # ПРОВЕРКА 5: проверка на слияние
        # ----------------------------------------------------------------------
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
                
                # Дополнительные проверки:
                # 1. Проверяем, что после гласной есть буквы (это часть слова, а не окончание)
                if end_pos + 1 < len(original_text):
                    next_next = original_text[end_pos + 1]
                    if next_next.isalpha():
                        return False
                
                # 2. Проверяем, не является ли это типичным окончанием
                common_endings = {'а', 'я', 'ы', 'и', 'е', 'ё', 'ю', 'й'}
                if next_char.lower() in common_endings:
                    if end_pos + 1 >= len(original_text) or original_text[end_pos + 1] in self.WORD_BREAKS:
                        return False
                
                # 3. Проверяем длину исходного слова
                if len(full_word) < 3:
                    if full_word.lower() in self.STOP_WORDS:
                        return True
                
                return True
        
        return False
    
    def _get_entity_type(self, label: str) -> str:
        """Извлекает тип сущности из BIO-тега."""
        if label.startswith(('B-', 'I-')):
            return label[2:]
        return label
    
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