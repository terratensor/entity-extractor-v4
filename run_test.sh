#!/bin/bash

# Тестовый запуск конвейера v4 на малом объёме

echo "========================================="
echo "🚀 Запуск теста конвейера v4"
echo "========================================="

# Активируем виртуальное окружение
source venv/bin/activate

# Создаём тестовый конфиг если нужно
if [ ! -f "config_test.yaml" ]; then
    cat > config_test.yaml << 'EOF'
source:
  host: localhost
  port: 9306
  table: library2026
  batch_size: 100

model:
  name: Davlan/xlm-roberta-large-ner-hrl
  max_tokens: 512
  overlap_ratio: 0.0
  min_confidence: 0.5
  include_positions: false

queues:
  queue1_size: 100
  queue2_size: 1000
  queue3_size: 500

tokenizer:
  num_workers: 2

gpu_devices:
  - device_id: 0
    batch_size: 8
    precision: float16
  - device_id: 1
    batch_size: 6
    precision: float16

output:
  format: csv
  path: ./results_test/output.csv
  delimiter: "|"
  include_confidence: true
  include_positions: false
  flush_interval: 10
  buffer_size: 100

checkpoint:
  file: ./checkpoint_test.json
  save_interval: 100

logging:
  level: INFO
  file: ./batch_test.log
EOF
    echo "✅ Создан тестовый конфиг"
fi

# Создаём директорию для результатов
mkdir -p results_test

# Запускаем с ограничением в 10000 документов
python -m app.v4.main --config config_test.yaml --limit 10

echo "========================================="
echo "✅ Тест завершён"
echo "========================================="