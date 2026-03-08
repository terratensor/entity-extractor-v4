#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Основной класс WordExpander для расширения сущностей.
"""

import logging
from typing import Dict, Optional, List, Any

from . import constants
from . import checks
from . import expand
from . import clean
from . import utils

logger = logging.getLogger(__name__)


class WordExpander:
    """
    Расширяет найденные сущности до полных слов, используя оригинальный текст.
    Работает в обе стороны (влево и вправо) с экспериментальными параметрами.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Args:
            config: словарь с параметрами расширения (переопределяют DEFAULTS)
        """
        self.config = constants.DEFAULTS.copy()
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
        verbose = self.config.get('verbose', False)
        
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
        if verbose:
            logger.warning(f"🔍 РАСШИРЕНИЕ: '{entity['text']}' ({entity['type']}) "
                        f"позиции: {start_pos}-{end_pos}")
            logger.warning(f"   Текст вокруг: '{original_text[max(0, start_pos-20):min(len(original_text), end_pos+20)]}'")

        # ----------------------------------------------------------------------
        # ПРОВЕРКА НЕОБХОДИМОСТИ РАСШИРЕНИЯ
        # Анализируем символы слева и справа от сущности
        # ----------------------------------------------------------------------
        should, reason = checks.should_expand(
            entity['text'], start_pos, end_pos, original_text, 
            entity['type'], self.config, verbose
        )

        if verbose:
            logger.warning(f"   _should_expand: {should}, причина: {reason}")
        
        result_entity = entity.copy()
        
        if should:
            # ----------------------------------------------------------------------
            # ЭТАП 1: РАСШИРЕНИЕ
            # Пытаемся расширить сущность влево и/или вправо
            # ----------------------------------------------------------------------
            expanded, expand_type, new_start, new_end = expand.expand_to_full_word(
                entity['text'], start_pos, end_pos, original_text, 
                entity['type'], self.config, verbose
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
                    
                    if verbose:
                        logger.warning(f"      новые позиции: {new_start}-{new_end}")
        
        # ----------------------------------------------------------------------
        # ЭТАП 2: ФИНАЛЬНАЯ ОЧИСТКА РЕЗУЛЬТАТА
        # Удаляем лишние знаки препинания, проверяем парность кавычек
        # ----------------------------------------------------------------------
        if self.config.get('enable_final_cleaning', True):
            cleaned_text = clean.clean_entity(result_entity['text'], self.config, verbose)
            if cleaned_text != result_entity['text']:
                if verbose:
                    logger.warning(f"   🧹 финальная очистка: '{result_entity['text']}' -> '{cleaned_text}'")
                result_entity['text'] = cleaned_text
                result_entity['cleaned'] = True
                self.stats['cleaned'] += 1
        
        # ----------------------------------------------------------------------
        # ИТОГОВЫЙ ВЫВОД
        # ----------------------------------------------------------------------
        if verbose:
            logger.warning(f"   📝 ИТОГ: '{entity['text']}' -> '{result_entity['text']}'")
            if result_entity.get('expanded'):
                logger.warning(f"      новые позиции: {result_entity['positions'][0]['start']}-{result_entity['positions'][0]['end']}")
            if result_entity.get('cleaned'):
                logger.warning(f"      очищено: да")

        return result_entity
    
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