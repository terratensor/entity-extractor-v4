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
    WORD_BREAKS = set(' .,!?;:…\n\r\t') # Убраны все виды скобок ()[]{}<> | Убраны кавычки ""\'
    
    # Кавычки (разные типографские варианты)
    OPEN_QUOTES = {'«', '“', '„', '"', "'"}
    CLOSE_QUOTES = {'»', '”', '‟', '"', "'"}
    ALL_QUOTES = OPEN_QUOTES | CLOSE_QUOTES

    # Открывающие скобки
    OPEN_BRACKETS = {'(', '[', '{', '<'}
    # Закрывающие скобки
    CLOSE_BRACKETS = {')', ']', '}', '>'}
    # Все скобки для проверки
    ALL_BRACKETS = OPEN_BRACKETS | CLOSE_BRACKETS
    
    # Пунктуация для удаления в начале/конце (включая все виды тире)
    PUNCTUATION = '.,!?;:…—–-•=|\\/*$%^&@~<>' # TODO проверить и добавить дефис -
    PUNCTUATION_START = PUNCTUATION  # для lstrip
    PUNCTUATION_END = PUNCTUATION + '№'   # для rstrip
    
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
        # ----------------------------------------------------------------------
        # ПРОВЕРКА ТИПА СУЩНОСТИ
        # Расширяем только указанные в конфиге типы (LOC, PER, ORG)
        # ----------------------------------------------------------------------
        expand_types = self.config.get('expand_entity_types', ['LOC', 'PER'])
        if entity['type'] not in expand_types:
            return entity
        
        # ----------------------------------------------------------------------
        # ПРОВЕРКА НАЛИЧИЯ ПОЗИЦИЙ
        # Без позиций невозможно определить, где в тексте находится сущность
        # ----------------------------------------------------------------------
        if 'positions' not in entity or not entity['positions']:
            return entity
        
        self.stats['attempts'] += 1
        
        # ----------------------------------------------------------------------
        # ОПРЕДЕЛЕНИЕ ГРАНИЦ СУЩНОСТИ
        # Берем первую и последнюю позицию из списка
        # ----------------------------------------------------------------------
        first_pos = entity['positions'][0]
        last_pos = entity['positions'][-1]
        start_pos = first_pos['start']
        end_pos = last_pos['end']
        
        # ----------------------------------------------------------------------
        # ОТЛАДКА: вывод информации о сущности и контекста
        # ----------------------------------------------------------------------
        if self.config.get('verbose', False):
            logger.warning(f"🔍 РАСШИРЕНИЕ: '{entity['text']}' ({entity['type']}) "
                        f"позиции: {start_pos}-{end_pos}")
            logger.warning(f"   Текст вокруг: '{original_text[max(0, start_pos-20):min(len(original_text), end_pos+20)]}'")

        # ----------------------------------------------------------------------
        # ПРОВЕРКА НЕОБХОДИМОСТИ РАСШИРЕНИЯ
        # Анализируем символы слева и справа от сущности
        # ----------------------------------------------------------------------
        should, reason = self._should_expand(
            entity['text'], start_pos, end_pos, original_text, entity['type']
        )

        if self.config.get('verbose', False):
            logger.warning(f"   _should_expand: {should}, причина: {reason}")
        
        result_entity = entity.copy()
        
        if should:
            # ----------------------------------------------------------------------
            # ЭТАП 1: РАСШИРЕНИЕ
            # Пытаемся расширить сущность влево и/или вправо
            # ----------------------------------------------------------------------
            expanded, expand_type, new_start, new_end = self._expand_to_full_word(
                entity['text'], start_pos, end_pos, original_text, entity['type']
            )
            
            if expanded != entity['text']:
                # ----------------------------------------------------------------------
                # ОБНОВЛЕНИЕ СТАТИСТИКИ
                # ----------------------------------------------------------------------
                if expand_type == 'left':
                    self.stats['expanded_left'] += 1
                elif expand_type == 'right':
                    self.stats['expanded_right'] += 1
                elif expand_type == 'both':
                    self.stats['expanded_both'] += 1
                
                # ----------------------------------------------------------------------
                # СОХРАНЕНИЕ РЕЗУЛЬТАТА РАСШИРЕНИЯ
                # ----------------------------------------------------------------------
                result_entity['text'] = expanded
                result_entity['expanded'] = True
                result_entity['expansion_type'] = expand_type
                result_entity['original_text'] = entity['text']
                
                # ----------------------------------------------------------------------
                # [НОВОЕ] ДОБАВЛЕНИЕ НОВЫХ ПОЗИЦИЙ
                # Сохраняем новые границы сущности после расширения
                # ----------------------------------------------------------------------
                if new_start is not None and new_end is not None:
                    result_entity['positions'] = [{'start': new_start, 'end': new_end}]
                    result_entity['original_positions'] = entity['positions']
                    
                    if self.config.get('verbose', False):
                        logger.warning(f"      новые позиции: {new_start}-{new_end}")
        
        # ----------------------------------------------------------------------
        # ЭТАП 2: ФИНАЛЬНАЯ ОЧИСТКА РЕЗУЛЬТАТА
        # Удаляем лишние знаки препинания, проверяем парность кавычек
        # ----------------------------------------------------------------------
        if self.config.get('enable_final_cleaning', True):
            cleaned_text = self._clean_entity(result_entity['text'])
            if cleaned_text != result_entity['text']:
                if self.config.get('verbose', False):
                    logger.warning(f"   🧹 финальная очистка: '{result_entity['text']}' -> '{cleaned_text}'")
                result_entity['text'] = cleaned_text
                result_entity['cleaned'] = True
                self.stats['cleaned'] += 1
        
        # ----------------------------------------------------------------------
        # ИТОГОВЫЙ ВЫВОД
        # ----------------------------------------------------------------------
        if self.config.get('verbose', False):
            logger.warning(f"   📝 ИТОГ: '{entity['text']}' -> '{result_entity['text']}'")
            if result_entity.get('expanded'):
                logger.warning(f"      новые позиции: {result_entity['positions'][0]['start']}-{result_entity['positions'][0]['end']}")
            if result_entity.get('cleaned'):
                logger.warning(f"      очищено: да")

        return result_entity

    def _clean_entity(self, text: str) -> str:
        """
        Третий этап: финальная очистка сущности.
        
        Правила:
        1. Сохраняем точки в инициалах и сокращениях (Г.В., г., ул., д.)
        2. Сохраняем дефисы в составных словах (пр-т, Ростов-на-Дону)
        3. Сохраняем парные кавычки, удаляем одиночные и множественные
        4. Удаляем все знаки препинания в начале строки (точки, запятые и т.д.)
        5. Удаляем пробелы в начале после удаления знаков
        6. Нормализуем пробелы
        """
        if not text:
            return text
        
        original = text
        if self.config.get('verbose', False):
            logger.warning(f"      🧹 финальная очистка: '{text}'")
        
        import unicodedata
        import re
        
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
        
        if removed and self.config.get('verbose', False):
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
        
        if self.config.get('verbose', False) and '@DOT@' in text:
            logger.warning(f"         защищены точки: {text.count('@DOT@')} шт.")
                
        # ----------------------------------------------------------------------
        # Шаг 4: Удаление знаков препинания и скобок с начала и конца
        # ----------------------------------------------------------------------
        # REMOVE_START = set(self.PUNCTUATION) | self.ALL_BRACKETS | {'"', "'", '«', '»', '“', '”', '„', '‟'}
        REMOVE_START = set(self.PUNCTUATION) | self.ALL_BRACKETS

        # Очистка начала
        if text:
            # Циклически удаляем все нежелательные символы в начале
            start_removed = 0
            while text and text[0] in REMOVE_START:
                old_text = text
                text = text[1:]
                start_removed += 1
                if self.config.get('verbose', False) and start_removed <= 5:
                    logger.warning(f"         удален символ '{old_text[0]}' в начале")
            
            if start_removed > 0 and self.config.get('verbose', False):
                logger.warning(f"         удалено {start_removed} символов в начале")
            
            # Удаляем пробелы после удаления символов
            while text and text[0] == ' ':
                old_text = text
                text = text[1:]
                if self.config.get('verbose', False):
                    logger.warning(f"         удален пробел в начале: '{old_text}' -> '{text}'")

        # Очистка конца
        if text:
            end_removed = 0
            changed = True
            while changed and text:
                changed = False
                old_text = text
                
                # Сначала пробелы
                while text and text[-1] == ' ':
                    text = text[:-1]
                    changed = True
                    end_removed += 1
                    if self.config.get('verbose', False) and end_removed <= 5:
                        logger.warning(f"         удален пробел в конце")
                
                # Потом знаки препинания
                while text and text[-1] in self.PUNCTUATION_END:
                    removed_char = text[-1]
                    text = text[:-1]
                    changed = True
                    end_removed += 1
                    if self.config.get('verbose', False) and end_removed <= 5:
                        logger.warning(f"         удален знак '{removed_char}' в конце")
            
            if end_removed > 0 and self.config.get('verbose', False):
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
            # Кавычки
            ('«', '»'),
            ('"', '"'),
            ("'", "'"),
            ('“', '”'),
            ('„', '‟'),
            # Скобки
            ('(', ')'),
            ('[', ']'),
            ('{', '}')
        ]
        
        for open_char, close_char in pairs:
            open_count = text.count(open_char)
            close_count = text.count(close_char)
            
            if self.config.get('verbose', False) and (open_count > 0 or close_count > 0):
                logger.warning(f"         символы {open_char}/{close_char}: {open_count} открыв, {close_count} закрыв")
            
            # Если количество не совпадает - удаляем все вхождения этого типа
            if open_count != close_count:
                # Но сохраняем, если это часть слова (например, Gauss в скобках)
                # Проверяем, не является ли это частью контента
                if open_count > 0 and close_count == 0 and open_char == '(':
                    # Проверяем, есть ли закрывающая скобка дальше в тексте
                    if ')' not in text:
                        text = text.replace('(', '')
                        if self.config.get('verbose', False):
                            logger.warning(f"         удалена открывающая скобка без пары")
                else:
                    text = text.replace(open_char, '')
                    text = text.replace(close_char, '')
                    if self.config.get('verbose', False) and (open_count > 0 or close_count > 0):
                        logger.warning(f"         удалены все символы {open_char}/{close_char} (непарные)")
            else:
                # Если парные, но есть множественные - оставляем только одну пару
                if open_count > 1:
                    # Находим первую открывающую и последнюю закрывающую
                    first_open = text.find(open_char)
                    last_close = text.rfind(close_char)
                    
                    if first_open != -1 and last_close != -1 and first_open < last_close:
                        # Оставляем только эти символы
                        middle = text[first_open+1:last_close].replace(open_char, '').replace(close_char, '')
                        new_text = text[:first_open] + open_char + middle + close_char + text[last_close+1:]
                        
                        if new_text != text:
                            if self.config.get('verbose', False):
                                logger.warning(f"         оставлена только одна пара {open_char}{close_char}")
                            text = new_text
        
        # ----------------------------------------------------------------------
        # Шаг 8: Финальная очистка
        # ----------------------------------------------------------------------
        text = text.strip()
        text = ' '.join(text.split())
        
        if text != original and self.config.get('verbose', False):
            logger.warning(f"      🧹 итог очистки: '{original}' -> '{text}'")
        
        return text
        
    def _should_expand(self, text: str, start_pos: int, end_pos: int,
                      original_text: str, entity_type: str) -> tuple:
        """
        Проверяет, нужно ли расширять сущность (в обе стороны).
        Каждая проверка отдельно и четко прокомментирована.
        
        Returns:
            (bool, str): (нужно ли расширять, причина отказа)
        """
        # Нормализуем текст для проверки
        import unicodedata
        text_norm = unicodedata.normalize('NFKC', text)
        original_text_norm = unicodedata.normalize('NFKC', original_text)

        if self.config.get('verbose', False):
            logger.warning(f"   _should_expand проверка для '{text}':")
            logger.warning(f"      start_pos={start_pos}, end_pos={end_pos}")
        
        # ----------------------------------------------------------------------
        # ПРОВЕРКА 1: Минимальная длина
        # ----------------------------------------------------------------------
        if len(text) < self.config['min_token_length']:
            if self.config.get('verbose', False):
                logger.warning(f"      ❌ rejected_length: длина {len(text)} < {self.config['min_token_length']}")
            return False, 'length'
        
        # ----------------------------------------------------------------------
        # ПРОВЕРКА 2: Стоп-слова (предлоги, союзы, частицы)
        # ----------------------------------------------------------------------
        if self.config['enable_stopwords']:
            clean_text = text.lower().strip('▁')
            if clean_text in self.STOP_WORDS:
                if self.config.get('verbose', False):
                    logger.warning(f"      ❌ rejected_stopword: '{clean_text}' в стоп-словах")
                return False, 'stopword'
        
        # ----------------------------------------------------------------------
        # ПРОВЕРКА 3: Возможность расширения ВЛЕВО
        # ----------------------------------------------------------------------
        can_expand_left = False
        left_reason = None

        if start_pos > 0:
            prev_char = original_text[start_pos - 1]
            if self.config.get('verbose', False):
                logger.warning(f"      символ слева: '{prev_char}' (код {ord(prev_char)})")
            
            # Условие 3.1: Слева буква или кавычка (не разделитель)
            if prev_char.isalpha() or prev_char in self.ALL_QUOTES:
                # Условие 3.2: Для букв - всегда разрешаем (часть слова)
                if prev_char.isalpha():
                    can_expand_left = True
                    if self.config.get('verbose', False):
                        logger.warning(f"      ✅ можно расширять ВЛЕВО (часть слова)")
                # Для кавычек - только если это начало слова
                elif start_pos - 1 == 0 or original_text[start_pos - 2] in self.WORD_BREAKS:
                    can_expand_left = True
                    if self.config.get('verbose', False):
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
            if self.config.get('verbose', False):
                logger.warning(f"      символ справа: '{next_char}' (код {ord(next_char)})")
            
            # Условие 4.1: Справа буква или кавычка (не разделитель)
            if next_char.isalpha() or next_char in self.ALL_QUOTES:
                can_expand_right = True
                if self.config.get('verbose', False):
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
                    if self.config.get('verbose', False):
                        logger.warning(f"      ❌ rejected_capital: следующая буква '{next_char}' не заглавная")
                    return False, 'capital'
        
        # ----------------------------------------------------------------------
        # ИТОГ: расширяем, если есть возможность хотя бы в одну сторону
        # ----------------------------------------------------------------------
        if can_expand_left or can_expand_right:
            if self.config.get('verbose', False):
                logger.warning(f"      ✅ можно расширять (влево={can_expand_left}, вправо={can_expand_right})")
            return True, None
        else:
            if self.config.get('verbose', False):
                logger.warning(f"      ❌ нет расширения: влево={left_reason}, вправо={right_reason}")
            return False, None
    
    def _expand_to_full_word(self, text: str, start_pos: int, end_pos: int,
                            original_text: str, entity_type: str) -> tuple:
        """
        Расширяет до полного слова в ОБЕ стороны.
        
        Returns:
            tuple: (расширенное слово, тип расширения, новый start, новый end)
        """
        if self.config.get('verbose', False):
            logger.warning(f"      🔧 _expand_to_full_word для '{text}' ({start_pos}-{end_pos})")
        
        max_left = self.config['max_search_left']
        max_right = self.config['max_search_right']
        
        # ----------------------------------------------------------------------
        # РАСШИРЕНИЕ ВЛЕВО
        # Ищем начало слова, двигаясь влево от start_pos
        # Останавливаемся на разделителях (пробелы, знаки препинания, кроме кавычек)
        # ----------------------------------------------------------------------
        word_start = start_pos
        left_expanded = False
        steps_left = 0
        
        if start_pos > 0:
            if self.config.get('verbose', False):
                logger.warning(f"      поиск влево от {start_pos}:")
            while word_start > 0 and steps_left < max_left:
                prev_char = original_text[word_start - 1]
                if self.config.get('verbose', False):
                    logger.warning(f"        символ {word_start-1}: '{prev_char}'")
                
                # Останавливаемся на разделителях (но пропускаем кавычки)
                if prev_char in self.WORD_BREAKS and prev_char not in self.ALL_QUOTES:
                    if self.config.get('verbose', False):
                        logger.warning(f"          стоп - разделитель")
                    break
                
                word_start -= 1
                steps_left += 1
                left_expanded = True
                if self.config.get('verbose', False):
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
            if self.config.get('verbose', False):
                logger.warning(f"      поиск вправо от {end_pos}:")
            while word_end < len(original_text) and steps_right < max_right:
                next_char = original_text[word_end]
                if self.config.get('verbose', False):
                    logger.warning(f"        символ {word_end}: '{next_char}'")
                
                # Останавливаемся на разделителях (но пропускаем кавычки)
                if next_char in self.WORD_BREAKS and next_char not in self.ALL_QUOTES:
                    if self.config.get('verbose', False):
                        logger.warning(f"          стоп - разделитель")
                    break
                
                word_end += 1
                steps_right += 1
                right_expanded = True
                if self.config.get('verbose', False):
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
            # ПРОВЕРКА 1: разделители ВНУТРИ исходного диапазона
            # Проверяем символы между start_pos и end_pos.
            # Если там есть разделители (например, многоточие), значит исходная
            # сущность уже содержит разрыв и расширение отменяется.
            # ----------------------------------------------------------------------
            if self.config.get('verbose', False):
                logger.warning(f"      проверка разделителей внутри исходного диапазона ({start_pos}-{end_pos}):")
            
            for pos in range(start_pos, end_pos):
                if pos >= len(original_text):
                    break
                char = original_text[pos]
                if char in self.WORD_BREAKS and char not in self.ALL_QUOTES and char != '-':
                    if self.config.get('verbose', False):
                        logger.warning(f"         найден разделитель '{char}' на позиции {pos} - отмена расширения")
                    return text, 'none', start_pos, end_pos
            
            # ----------------------------------------------------------------------
            # ПРОВЕРКА 2: разделители в ЛЕВОЙ добавленной части
            # Проверяем символы от нового начала (word_start) до исходного (start_pos).
            # Если там есть разделители, значит слева другое слово.
            # ----------------------------------------------------------------------
            if left_expanded:
                if self.config.get('verbose', False):
                    logger.warning(f"      проверка левой добавленной части ({word_start}-{start_pos}):")
                
                for pos in range(word_start, start_pos):
                    if pos >= len(original_text):
                        break
                    char = original_text[pos]
                    if char in self.WORD_BREAKS and char not in self.ALL_QUOTES and char != '-':
                        if self.config.get('verbose', False):
                            logger.warning(f"         найден разделитель '{char}' на позиции {pos} в левой части - отмена расширения")
                        return text, 'none', start_pos, end_pos
            
            # ----------------------------------------------------------------------
            # ПРОВЕРКА 3: разделители в ПРАВОЙ добавленной части
            # Проверяем символы от исходного конца (end_pos) до нового (word_end).
            # Если там есть разделители, значит справа другое слово.
            # ----------------------------------------------------------------------
            if right_expanded:
                if self.config.get('verbose', False):
                    logger.warning(f"      проверка правой добавленной части ({end_pos}-{word_end}):")
                
                for pos in range(end_pos, word_end):
                    if pos >= len(original_text):
                        break
                    char = original_text[pos]
                    if char in self.WORD_BREAKS and char not in self.ALL_QUOTES and char != '-':
                        if self.config.get('verbose', False):
                            logger.warning(f"         найден разделитель '{char}' на позиции {pos} в правой части - отмена расширения")
                        return text, 'none', start_pos, end_pos
        
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

        if self.config.get('verbose', False):
            logger.warning(f"      полное слово: '{full_word}'")
            logger.warning(f"      тип расширения: {expand_type}")
            logger.warning(f"      новые позиции: {word_start}-{word_end}")
        
        # ----------------------------------------------------------------------
        # ПРОВЕРКИ РЕЗУЛЬТАТА
        # ----------------------------------------------------------------------
        
        # ----------------------------------------------------------------------
        # ПРОВЕРКА 1: слишком длинное расширение
        # Если расширенное слово превышает исходное более чем в max_length_ratio раз,
        # это похоже на ошибку (например, захват целого предложения)
        # ----------------------------------------------------------------------
        if len(full_word) > len(text) * self.config['max_length_ratio']:
            if self.config.get('verbose', False):
                logger.warning(f"      ❌ слишком длинное: {len(full_word)} > {len(text)} * {self.config['max_length_ratio']}")
            return text, 'none', start_pos, end_pos
        
        # ----------------------------------------------------------------------
        # ПРОВЕРКА 2: многоуровневая проверка подстроки
        # Если мы НЕ расширяли слово, проверяем, что исходный текст является подстрокой
        # с учетом нормализации пробелов, кавычек и последовательности слов
        # ----------------------------------------------------------------------
        if left_expanded or right_expanded:
            # Мы расширили слово, используя оригинальный текст
            # доверяем оригиналу, даже если оно не содержит исходный текст модели
            if self.config.get('verbose', False):
                logger.warning(f"      ✅ слово расширено, доверяем оригиналу")
        else:
            # Слово не расширялось - применяем многоуровневую проверку
            # Уровень 1: нормализация пробелов
            text_norm = ' '.join(text.split())
            full_norm = ' '.join(full_word.split())
            
            if self.config.get('verbose', False):
                logger.warning(f"      уровень 1 (норм. пробелов): '{text_norm}' vs '{full_norm}'")
            
            is_substring = False
            level = 0
            
            if text_norm in full_norm:
                is_substring = True
                level = 1
                if self.config.get('verbose', False):
                    logger.warning(f"      ✅ подстрока (уровень 1)")
            else:
                # Уровень 2: удаляем кавычки с обеих сторон
                text_unquoted = text_norm.strip('«»""')
                full_unquoted = full_norm.strip('«»""')
                
                if self.config.get('verbose', False):
                    logger.warning(f"      уровень 2 (без кавычек): '{text_unquoted}' vs '{full_unquoted}'")
                
                if text_unquoted in full_unquoted:
                    is_substring = True
                    level = 2
                    if self.config.get('verbose', False):
                        logger.warning(f"      ✅ подстрока (уровень 2)")
                else:
                    # Уровень 3: удаляем ВСЕ пробелы в начале и конце
                    text_clean = text_unquoted.strip()
                    full_clean = full_unquoted.strip()
                    
                    if self.config.get('verbose', False):
                        logger.warning(f"      уровень 3 (strip): '{text_clean}' vs '{full_clean}'")
                    
                    if text_clean in full_clean:
                        is_substring = True
                        level = 3
                        if self.config.get('verbose', False):
                            logger.warning(f"      ✅ подстрока (уровень 3)")
                    else:
                        # Уровень 4: разбиваем на слова и проверяем последовательность
                        text_words = text_clean.split()
                        full_words = full_clean.split()
                        
                        if self.config.get('verbose', False):
                            logger.warning(f"      уровень 4 (слова): {text_words} vs {full_words}")
                        
                        if len(text_words) > 0:
                            for i in range(len(full_words) - len(text_words) + 1):
                                if full_words[i:i+len(text_words)] == text_words:
                                    is_substring = True
                                    level = 4
                                    if self.config.get('verbose', False):
                                        logger.warning(f"      ✅ последовательность слов найдена на позиции {i}")
                                    break
            
            if not is_substring:
                if self.config.get('verbose', False):
                    logger.warning(f"      ❌ исходный текст не подстрока после всех уровней")
                return text, 'none', start_pos, end_pos
        
        # ----------------------------------------------------------------------
        # ПРОВЕРКА 3: минимальное покрытие
        # Исходный текст должен составлять не менее min_coverage от расширенного
        # Это защищает от случаев, когда мы добавляем слишком много лишнего
        # ----------------------------------------------------------------------
        coverage = len(text) / len(full_word) if len(full_word) > 0 else 0
        if coverage < self.config['min_coverage']:
            if self.config.get('verbose', False):
                logger.warning(f"      ❌ покрытие {coverage:.2f} < {self.config['min_coverage']}")
            return text, 'none', start_pos, end_pos
        
        # ----------------------------------------------------------------------
        # ПРОВЕРКА 4: заглавные буквы для LOC/PER (пропуская кавычки)
        # Для имен собственных первая буква должна быть заглавной
        # Пропускаем кавычки в начале при поиске первой буквы
        # ----------------------------------------------------------------------
        if (self.config['require_capital'] and entity_type in ['LOC', 'PER'] and left_expanded):
            # Ищем первую букву (пропуская кавычки)
            first_letter = None
            for char in full_word:
                if char.isalpha():
                    first_letter = char
                    break
            
            if first_letter is None:
                if self.config.get('verbose', False):
                    logger.warning(f"      ❌ в слове нет букв")
                return text, 'none', start_pos, end_pos
            
            if not first_letter.isupper():
                if self.config.get('verbose', False):
                    logger.warning(f"      ❌ первая буква '{first_letter}' не заглавная")
                return text, 'none', start_pos, end_pos
        
        # ----------------------------------------------------------------------
        # ПРОВЕРКА 5: проверка на слияние
        # Проверяем, не является ли расширение результатом слияния слов
        # (например, "вМоскве" должно остаться "вМоскве", а не расширяться)
        # ----------------------------------------------------------------------
        if self.config['enable_merge_check']:
            if self._check_word_merge(original_text, full_word, start_pos, end_pos):
                if self.config.get('verbose', False):
                    logger.warning(f"      ❌ обнаружено слияние слов")
                return text, 'none', start_pos, end_pos
        
        if self.config.get('verbose', False):
            logger.warning(f"      ✅ расширение: '{text}' -> '{full_word}'")
        return full_word, expand_type, word_start, word_end  

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