# 🏗️ Entity Extractor V4 — Технический мануал

## 📋 Содержание
1. [Архитектура системы](#архитектура-системы)
2. [Конвейер обработки](#конвейер-обработки)
3. [Детальное описание компонентов](#детальное-описание-компонентов)
4. [Модель NER](#модель-ner)
5. [Логика расширения слов](#логика-расширения-слов)
6. [Очистка сущностей](#очистка-сущностей)
7. [Конфигурация](#конфигурация)
8. [Форматы данных](#форматы-данных)
9. [Производительность](#производительность)
10. [Диагностика и отладка](#диагностика-и-отладка)

---

## 1. Архитектура системы

### 1.1 Общая схема
```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Reader    │────▶│  Tokenizer   │────▶│     GPU      │────▶│    Writer    │
│  (воркер 1) │     │  (воркер 2)  │     │  (воркер 3)  │     │  (воркер 4)  │
└─────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
       │                    │                    │                    │
       ▼                    ▼                    ▼                    ▼
   Очередь 1            Очередь 2            Очередь 3           Файл CSV
   (Queue)              (Queue)              (Queue)
```

### 1.2 Ключевые принципы
- **Конвейерная обработка**: каждый компонент работает независимо
- **Буферизация**: очереди сглаживают неравномерность поступления данных
- **Graceful shutdown**: корректное завершение при SIGINT/SIGTERM
- **Чекпоинты**: возможность возобновления с последнего обработанного ID

### 1.3 Технологический стек
- Python 3.13+
- PyTorch 2.0+
- Transformers 4.30+
- CUDA 11.8+ (для GPU)
- Manticore Search (источник данных)
- CSV/Parquet (выходной формат)

---

## 2. Конвейер обработки

### 2.1 Поток данных
```
Manticore ──┐
            ▼
    ┌───────────────┐
    │    Reader     │───┐
    │   batch=1000  │   │
    └───────────────┘   │
                        ▼
                 ┌──────────────┐
                 │   Queue 1    │
                 │  maxsize=N   │
                 └──────────────┘
                        │
                        ▼
    ┌─────────────────────────────────┐
    │       Tokenizer Workers         │
    │  ┌─────┐ ┌─────┐ ┌─────┐        │
    │  │ T1  │ │ T2  │ │ T3  │  ...   │
    │  └─────┘ └─────┘ └─────┘        │
    └─────────────────────────────────┘
                        │
                        ▼
                 ┌──────────────┐
                 │   Queue 2    │
                 │  maxsize=N   │
                 └──────────────┘
                        │
                        ▼
    ┌─────────────────────────────────┐
    │         GPU Workers             │
    │  ┌─────┐ ┌─────┐ ┌─────┐        │
    │  │GPU0 │ │GPU1 │ │GPU2 │  ...   │
    │  └─────┘ └─────┘ └─────┘        │
    └─────────────────────────────────┘
                        │
                        ▼
                 ┌──────────────┐
                 │   Queue 3    │
                 │  maxsize=N   │
                 └──────────────┘
                        │
                        ▼
    ┌───────────────┐
    │    Writer     │───┐
    └───────────────┘   │
                        ▼
                   CSV файл
```

### 2.2 Размеры очередей (эмпирические)
- **Queue 1** (сырые документы): 1000-5000
- **Queue 2** (токенизированные чанки): 20000-100000
- **Queue 3** (результаты): 5000-10000

### 2.3 Чекпоинты
```json
{
  "last_id": 6056452479959178072,
  "processed": 7000,
  "stats": {
    "processed": 7000,
    "total_tokens": 4356279,
    "entities_found": 99900,
    "entities_by_type": {
      "LOC": 42716,
      "PER": 53576,
      "ORG": 3608,
      "MISC": 0
    }
  },
  "timestamp": "2026-03-05T14:52:29.550034"
}
```

---

## 3. Детальное описание компонентов

### 3.1 Reader (воркер 1) — `reader.py`

**Назначение**: Чтение документов из Manticore Search

**Параметры**:
- `batch_size`: 1000 (настраивается)
- `connection_timeout`: 30 сек

**Логика работы**:
```python
while not stop_event.is_set():
    rows = fetch_batch(last_id)  # SELECT id, content FROM table WHERE id > %s ORDER BY id LIMIT %s
    for row in rows:
        while queue1.full():
            time.sleep(0.1)
        queue1.put({'id': row['id'], 'text': row['content']})
    last_id = rows[-1]['id']
```

**Особенности**:
- Автоматическое переподключение при ошибках
- Поддержка разных имен полей в таблице
- Сбор статистики (read_docs, read_batches)

### 3.2 Tokenizer (воркер 2) — `tokenizer_worker.py`

**Назначение**: Токенизация и разбивка на чанки

**Параметры**:
- `num_workers`: количество потоков (рекомендуется 4-24)
- `max_tokens`: 512
- `overlap_ratio`: 0.0

**Логика работы**:
```python
inputs = tokenizer(
    text,
    return_overflowing_tokens=True,
    return_offsets_mapping=True,
    max_length=max_tokens,
    stride=int(max_tokens * overlap_ratio)
)

for i in range(len(inputs['input_ids'])):
    # Определяем глобальные позиции чанка
    valid_starts = [o[0] for o in offsets if o and o[1] > 0]
    valid_ends = [o[1] for o in offsets if o and o[1] > 0]
    global_start = min(valid_starts)
    global_end = max(valid_ends)
    
    chunks.append({
        'id': doc_id,
        'chunk_id': i,
        'total_chunks': total,
        'input_ids': input_ids,
        'attention_mask': mask,
        'text': text[global_start:global_end],
        'global_start': global_start,
        'global_end': global_end
    })
```

**Критически важный момент**: Используется `min(valid_starts)` и `max(valid_ends)` для захвата всего диапазона, включая многоточия и спецсимволы.

### 3.3 GPU Worker (воркер 3) — `gpu_worker.py`

**Назначение**: Инференс модели на GPU

**Параметры**:
- `batch_size`: зависит от GPU (32-96 для RTX 5060 Ti)
- `precision`: float16/float32
- `min_confidence`: 0.5

**Логика работы**:
```python
# Используем пайплайн с aggregation_strategy=None (как в v1)
if not hasattr(self, 'ner_pipeline'):
    self.ner_pipeline = pipeline(
        "ner",
        model=self.model,
        tokenizer=self.tokenizer,
        device=self.device,
        aggregation_strategy=None
    )

# Получаем сырые предсказания
batch_results = self.ner_pipeline(texts, batch_size=len(texts))

# Корректируем позиции с учетом глобального смещения
for t in token_entities:
    t['start'] += global_start
    t['end'] += global_start

# Извлекаем сущности
entities = self._extract_entities_v1(token_entities, chunk_text)
```

### 3.4 Writer (воркер 4) — `writer_worker.py`

**Назначение**: Сборка чанков и запись в CSV

**Логика работы**:
```python
# Сохраняем текст каждого чанка
self.doc_text_parts[doc_id][chunk_id] = text

# Когда все чанки получены - собираем полный текст
if len(self.doc_text_parts[doc_id]) == total_chunks:
    full_text = ''.join(self.doc_text_parts[doc_id][i] for i in range(total_chunks))

# Собираем сущности из чанков
all_entities = []
for i in range(total_chunks):
    all_entities.extend(self.pending_docs[doc_id][i])

# Записываем с использованием csv.writer (правильное экранирование)
writer.writerow([doc_id, type, text, confidence, start, end])
```

---

## 4. Модель NER

### 4.1 Базовая модель
```
Davlan/xlm-roberta-large-ner-hrl
```
- **Архитектура**: XLM-RoBERTa-large
- **Параметры**: 560M
- **Языки**: мультиязычная (поддержка русского)
- **Метки**: O, B-DATE, I-DATE, B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC

### 4.2 Метки (id2label)
```json
{
  "0": "O",
  "1": "B-DATE",
  "2": "I-DATE",
  "3": "B-PER",
  "4": "I-PER",
  "5": "B-ORG",
  "6": "I-ORG",
  "7": "B-LOC",
  "8": "I-LOC"
}
```

### 4.3 Особенности модели
- **DATE** распознается, но в выходных данных фильтруется
- **MISC** отсутствует в конфиге, но оставлен для совместимости
- **B-/I-** префиксы важны для правильной сборки

---

## 5. Логика расширения слов — `expansion.py`

### 5.1 Общая структура модуля
```
expansion/
├── __init__.py      # экспортирует WordExpander
├── constants.py     # все константы
├── core.py          # основной класс
├── checks.py        # проверки (_should_expand, _check_word_merge)
├── expand.py        # логика расширения
├── clean.py         # финальная очистка
└── utils.py         # вспомогательные функции
```

### 5.2 Алгоритм расширения — `expand_to_full_word`

#### Этап 1: Определение границ
```python
# Расширение влево
while word_start > 0:
    prev_char = original_text[word_start - 1]
    if prev_char in WORD_BREAKS and prev_char not in ALL_QUOTES:
        break
    word_start -= 1

# Расширение вправо
while word_end < len(original_text):
    next_char = original_text[word_end]
    if next_char in WORD_BREAKS and next_char not in ALL_QUOTES:
        break
    word_end += 1
```

#### Этап 2: Проверка добавленных частей
```python
# Левая часть
for pos in range(word_start, start_pos):
    char = original_text[pos]
    if char in WORD_BREAKS and char not in ALL_QUOTES and char != '-':
        left_expanded = False
        word_start = start_pos

# Правая часть
for pos in range(end_pos, word_end):
    char = original_text[pos]
    if char in WORD_BREAKS and char not in ALL_QUOTES and char != '-':
        right_expanded = False
        word_end = end_pos
```

#### Этап 3: Проверка внутреннего диапазона
```python
# Находим последнюю букву
last_letter = word_end
while last_letter > word_start and not original_text[last_letter-1].isalpha():
    last_letter -= 1

# Проверяем только до последней буквы
check_end = min(end_pos, last_letter)
for pos in range(start_pos, check_end):
    char = original_text[pos]
    # Пробелы внутри разрешены (для многословных названий)
    if char in WORD_BREAKS and char not in ALL_QUOTES and char != '-' and char != ' ':
        return text, 'none', start_pos, end_pos
```

#### Этап 4: Защита от "срыва" в предложения
```python
def is_likely_sentence(text, original_length):
    words = text.split()
    if len(words) > 8: return True
    if len(text) / original_length > 5.0: return True
    if any(c in text for c in '.!?'): return True
    return False
```

### 5.3 Ключевые параметры расширения
```python
DEFAULTS = {
    'min_token_length': 2,
    'max_search_left': 30,
    'max_search_right': 30,
    'min_coverage': 0.3,
    'require_capital': True,
    'enable_stopwords': True,
    'enable_merge_check': True,
    'max_length_ratio': 5.0,
    'expand_entity_types': ['LOC', 'PER', 'ORG'],
    'enable_final_cleaning': True,
    'max_sentence_ratio': 5.0
}
```

### 5.4 Разделители и специальные символы
```python
WORD_BREAKS = set(' .,!?;:…\n\r\t')
OPEN_QUOTES = {'«', '“', '„', '"', "'"}
CLOSE_QUOTES = {'»', '”', '‟', '"', "'"}
ALL_QUOTES = OPEN_QUOTES | CLOSE_QUOTES
PUNCTUATION = '.,!?;:…—–-•=|\\/*$%^&@~<>'
```

---

## 6. Очистка сущностей — `clean.py`

### 6.1 Этапы очистки (в порядке выполнения)

#### Шаг 0: Нормализация Unicode
```python
text = unicodedata.normalize('NFKC', text)
```

#### Шаг 1: Удаление служебных символов
```python
control_cats = {'Cc', 'Cf', 'Cn', 'Co', 'Cs'}
```

#### Шаг 2: Нормализация пробелов
```python
text = re.sub(r'[\s\xa0]+', ' ', text)
```

#### Шаг 3: Сохранение точек в инициалах
```python
# Г.В. → Г@DOT@В
text = re.sub(r'([А-ЯA-Z])\.([А-ЯA-Z])(?=\.|\s|$)', r'\1@DOT@\2', text)
# г. → г@DOT@
text = re.sub(r'([А-ЯA-Z])\.(?=\s|$)', r'\1@DOT@', text)
```

#### Шаг 4: Удаление пунктуации с концов
```python
REMOVE_START = set(PUNCTUATION) | ALL_BRACKETS
while text and text[0] in REMOVE_START:
    text = text[1:]
```

#### Шаг 5: Удаление дефисов
```python
text = text.lstrip('-').rstrip('-')
```

#### Шаг 6: Восстановление точек
```python
text = text.replace('@DOT@', '.')
```

#### Шаг 7: Работа с парными символами
```python
pairs = [('«', '»'), ('"', '"'), ("'", "'"), ('(', ')')]
for open_char, close_char in pairs:
    open_count = text.count(open_char)
    close_count = text.count(close_char)
    if open_count != close_count:
        text = text.replace(open_char, '').replace(close_char, '')
```

#### Шаг 8: Очистка начала от сокращений (только LOC)
```python
ABBREVIATIONS = ['г.', 'пос.', 'дер.', 'п.', 'с.', 'д.']
for abbr in abbreviations:
    if text.lower().startswith(abbr):
        text = text[len(abbr):].lstrip()
```

#### Шаг 9: Финальный trim
```python
text = text.strip()
text = ' '.join(text.split())
```

---

## 7. Конфигурация

### 7.1 Полный пример `config.yaml`
```yaml
source:
  type: manticore
  host: localhost
  port: 9306
  table: library2026
  batch_size: 1000
  connection_timeout: 30

model:
  name: Davlan/xlm-roberta-large-ner-hrl
  max_tokens: 512
  overlap_ratio: 0.0
  min_confidence: 0.5
  include_positions: true

queues:
  queue1_size: 1000
  queue2_size: 20000
  queue3_size: 5000

tokenizer:
  num_workers: 8
  prefetch_size: 100

gpu_devices:
  - device_id: 0
    batch_size: 64
    precision: float16
  - device_id: 1
    batch_size: 32
    precision: float16

output:
  format: csv
  path: ./results/output.csv
  delimiter: "|"
  include_confidence: true
  include_positions: true
  flush_interval: 30
  buffer_size: 5000
  enable_expansion: true
  expansion_params:
    min_token_length: 2
    max_search_left: 30
    max_search_right: 30
    min_coverage: 0.3
    require_capital: true
    enable_stopwords: true
    enable_merge_check: true
    max_length_ratio: 5.0
    max_sentence_ratio: 5.0
    expand_entity_types: ['LOC', 'PER', 'ORG']
    enable_final_cleaning: true
    enable_beginning_cleaning: true

checkpoint:
  file: ./checkpoint.json
  save_interval: 1000

logging:
  level: INFO
  file: ./batch.log
  verbose: false
```

### 7.2 Параметры расшифровка

#### source
- `batch_size`: сколько документов читать за раз из Manticore
- `connection_timeout`: таймаут соединения

#### model
- `max_tokens`: максимальная длина последовательности для модели
- `overlap_ratio`: перекрытие окон (0.0 = без перекрытия)
- `min_confidence`: минимальная уверенность для сохранения сущности
- `include_positions`: сохранять ли позиции в оригинале

#### queues
- Размеры очередей-буферов между воркерами

#### tokenizer
- `num_workers`: количество параллельных токенизаторов
- `prefetch_size`: сколько документов заранее готовить

#### gpu_devices
- `device_id`: ID GPU
- `batch_size`: размер батча для этого GPU
- `precision`: точность вычислений (float16 для скорости)

#### output.expansion_params
- `min_token_length`: минимальная длина для расширения
- `max_search_left/right`: максимальное расстояние поиска
- `min_coverage`: минимальная доля исходного текста в расширенном
- `require_capital`: требовать заглавную букву для LOC/PER
- `max_length_ratio`: максимальное соотношение длин
- `max_sentence_ratio`: защита от срыва в предложения
- `expand_entity_types`: типы сущностей для расширения

---

## 8. Форматы данных

### 8.1 Входные данные (из Manticore)
```sql
SELECT id, content FROM table WHERE id > %s ORDER BY id LIMIT %s
```

### 8.2 Промежуточные форматы

**Queue 1** (Reader → Tokenizer):
```python
{'id': 12345, 'text': 'Текст документа'}
```

**Queue 2** (Tokenizer → GPU):
```python
{
    'id': 12345,
    'chunk_id': 0,
    'total_chunks': 3,
    'input_ids': [0, 343, 567, 2],
    'attention_mask': [1, 1, 1, 0],
    'text': 'Текст чанка',
    'global_start': 0,
    'global_end': 512,
    'token_count': 4
}
```

**Queue 3** (GPU → Writer):
```python
{
    'id': 12345,
    'chunk_id': 0,
    'total_chunks': 3,
    'entities': [
        {'text': 'Москва', 'type': 'LOC', 'confidence': 0.99},
        {'text': 'Россия', 'type': 'LOC', 'confidence': 0.98}
    ],
    'stats': {'tokens': 4, 'entities_count': 2}
}
```

### 8.3 Выходной CSV
```csv
doc_id|entity_type|entity_text|confidence|start_pos|end_pos
6056452479959171073|PER|Анджей Калиш|0.9178|0|12
6056452479959171073|PER|Ба Цзыc Наидой Шариповой|0.9512|102|126
6056452479959171073|PER|Мантэка Чиа|0.9988|143|154
6056452479959171073|LOC|Таогардена|0.9686|158|168
```

**Правила экранирования** (через `csv.writer`):
- Разделитель `|` внутри поля экранируется кавычками
- Кавычки `"` удваиваются (`"` → `""`)
- Поля с разделителями или кавычками заключаются в кавычки

---

## 9. Производительность

### 9.1 Ожидаемые показатели

| Компонент | Скорость | Узкое место |
|-----------|----------|-------------|
| Reader | 2000+ док/сек | Сеть, Manticore |
| Tokenizer | 500+ док/сек | CPU |
| GPU (RTX 5060 Ti) | 160+ чанков/сек | GPU compute |
| GPU (RTX 3060) | 90+ чанков/сек | GPU compute |
| Writer | 2000+ записей/сек | Диск I/O |
| **Общая** | **50-70 док/сек** | **GPU** |

### 9.2 Формулы расчета
```
документов/сек = чанков/сек / (среднее число чанков на документ)
среднее число чанков = длина_текста_в_символах / 4 / 512

Пример: 225 чанков/сек, средний документ 2000 символов → 2000/4/512 ≈ 1 чанк
→ 225 док/сек (но это идеальный случай)

Реально: 225 чанков/сек, средний документ 2200 символов → 2200/4/512 ≈ 4.3 чанка
→ 225/4.3 ≈ 52 док/сек
```

### 9.3 Профилирование памяти

**GPU Memory** (float16):
- Модель: ~3.1 GB
- Активации для batch=32: ~800 MB
- Итого: ~4-5 GB

**CPU Memory**:
- Токенизаторы: ~500 MB каждый
- Очереди: зависит от размера
- Рекомендуется: 32+ GB RAM

---

## 10. Диагностика и отладка

### 10.1 Уровни логирования

**production** (`verbose: false`):
- Только ключевые события
- Статистика каждые 100 документов
- Ошибки и предупреждения

**debug** (`verbose: true`):
- Детальная трассировка расширения
- Показывает каждый шаг проверок
- Позиции и контекст

### 10.2 Ключевые метрики в логах

```
📊 СТАТИСТИКА (прошло 300.1 сек):
  Очереди: q1=0, q2=0, q3=0
  Reader: прочитано 10,000 док
  Tokenizer: 10,000 док -> 35,000 чанков (2,560,445 токенов)
  GPU: 35,000 чанков, 45,000 сущностей
  Writer: завершено 10,000 док, записано 45,000 сущностей
  ⚡ Скорость чанков: 116.6 чанков/сек
  ⚡ Скорость документов: 33.3 док/сек
```

### 10.3 Статистика расширений
```
📊 Статистика расширений: попыток=833,356, расширено=108,089 (13.0%)
```

### 10.4 Типичные проблемы и их решение

#### "Алданов… Удивительная" → "Алданов"
- **Причина**: многоточие внутри исходного диапазона
- **Решение**: проверка внутреннего диапазона (шаг 3 в expand)

#### "та" → "а" (потеря буквы)
- **Причина**: сброс word_start при отмене
- **Решение**: сохранять левое расширение, отменять только правое

#### "« мирной литературы" → "мирной литературы"
- **Причина**: пробел внутри слова
- **Решение**: разрешить пробелы в check_inner_range

#### Геленджика би → целое предложение
- **Причина**: срыв расширения
- **Решение**: защита is_likely_sentence

### 10.5 Команды для мониторинга

```bash
# Следим за прогрессом
tail -f batch.log | grep "СТАТИСТИКА\|Скорость\|расширено"

# Следим за GPU
watch -n 2 nvidia-smi

# Следим за размером выходного файла
watch -n 10 "ls -lh results/output.csv && wc -l results/output.csv"

# Поиск конкретного документа в логах
grep -B 10 -A 10 "6056452479959171073" batch.log

# Статистика расширений
grep "Статистика расширений" batch.log | tail
```

---

## 📚 Заключение

Система представляет собой высокопроизводительный конвейер для извлечения именованных сущностей из больших текстовых коллекций. Ключевые особенности:

1. **Модульность** — каждый компонент выполняет одну задачу
2. **Масштабируемость** — легко добавить GPU или токенизаторы
3. **Надёжность** — чекпоинты и graceful shutdown
4. **Качество** — многоуровневая проверка и очистка
5. **Производительность** — конвейерная обработка без простоев

**Важные файлы для понимания:**
- `app/v4/main.py` — оркестратор
- `app/v4/expansion/expand.py` — сердце расширения
- `app/v4/expansion/clean.py` — финальная очистка
- `app/v4/writer_worker.py` — запись результатов

**Критически важные моменты:**
- Всегда проверяйте позиции при разбивке на чанки
- Не сбрасывайте левое расширение при внутренних разделителях
- Разрешайте пробелы внутри многословных названий
- Защищайтесь от срыва в предложения