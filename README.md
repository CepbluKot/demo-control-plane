# Demo Control Plane

Демо-control-plane для обнаружения аномалий метрик, суммаризации логов и генерации уведомления для SRE.

## Что делает пайплайн

1. Берет фактические метрики (`actual`) за lookback-окно.
2. Берет прогноз (`predicted`) для этой же метрики.
3. Находит аномалии на фактических данных.
4. Для Top-N аномалий делает map-reduce суммаризацию логов.
5. Генерирует текст уведомления.
6. Показывает результат в Streamlit UI (в том числе live-этапы обработки).

Дополнительно есть отдельная UI-страница `Logs Summarizer` для ручного запуска map-reduce по логам (без аномалий main-pipeline).

## Источники данных

### 1) Фактические метрики (`actual`)

Переключается через `CONTROL_PLANE_METRICS_SOURCE`:

- `prometheus` — читает из Prometheus.
- `clickhouse` — читает через `clickhouse_connect.get_client(...).query_df(...)`.

Контракт для `clickhouse`-запроса:

- должен вернуть колонки `timestamp` и `value`;
- в `CONTROL_PLANE_CLICKHOUSE_METRICS_QUERY` задается полный SQL для исходных метрик
  (с `WHERE/ORDER BY/LIMIT` как нужно в вашем контуре).
- поддерживаются плейсхолдеры: `{start}`, `{end}`, `{start_ts}`, `{end_ts}`.

### 2) Прогнозы (`predictions`)

Читаются из таблицы `metrics_forecast` через `control_plane/predictions_db.py`.
Используются поля таблицы:

- `timestamp`
- `service`
- `metric_name`
- `generated_at`
- `value` (переименовывается в `predicted`)

Опциональные фильтры применяются только если колонки реально есть:

- `prediction_kind`
- `forecast_type`

Это совместимо с таблицей, где есть базовые поля без этих двух колонок.

Дополнительно можно задать полный SQL для чтения прогнозов:

- `CONTROL_PLANE_CLICKHOUSE_PREDICTIONS_QUERY`
- запрос должен вернуть `timestamp` и `predicted` (или `value`, тогда переименуется автоматически)
- поддерживаются плейсхолдеры: `{service}`, `{metric_name}`, `{forecast_type}`, `{prediction_kind}`, `{start}`, `{end}`, `{start_ts}`, `{end_ts}`, `{table}`

### 3) Доступ к ClickHouse (важно)

- Все runtime-запросы к ClickHouse идут через `clickhouse_connect.get_client(...).query_df(...)`.
- `control_plane/predictions_db.py` читает `metrics_forecast` через `query_df`.
- `my_summarizer.py` читает логи батчами через `query_df` по `CONTROL_PLANE_CLICKHOUSE_LOGS_QUERY`.
- `control_plane/actuals.py` читает исходные метрики (`timestamp`, `value`) через `query_df`.

## Расширяемые интерфейсы

### Детектор аномалий

- Интерфейс: `AnomalyDetector` в `control_plane/anomaly_detectors.py`.
- Подключение: `CONTROL_PLANE_ANOMALY_DETECTOR`.
- По умолчанию: `rolling_iqr`.
- Доступные варианты: `rolling_iqr`, `residual_zscore`, `pyod_ecod`, `pyod_iforest`, `ruptures_pelt`.

### Суммаризатор логов

- Базовый адаптер: `control_plane/summarizer.py` (`do_summary`).
- Кастомный путь: `CONTROL_PLANE_SUMMARIZER_CALLABLE`.
- Готовый шаблон для старта: `my_summarizer.py::summarize_logs`.
- В `my_summarizer.py` реализован map-reduce summarizer:
  - batched-fetch логов из ClickHouse;
  - MAP: LLM-анализ каждого чанка (хронологический порядок, секции: TIMELINE / FIRST_SYMPTOM / CAUSAL_HINTS / EVIDENCE / OPEN_QUESTIONS);
  - REDUCE: объединение partial summary в post-mortem отчёт (INCIDENT_TIMELINE / FIRST_SIGNAL / ROOT_CAUSE_HYPOTHESES / CAUSAL_CHAIN / SUPPORTING_EVIDENCE / PRIORITY_ACTIONS);
  - финальный freeform-нарратив (3-5 абзацев связным текстом для SRE-команды, без секций);
  - если LLM недоступна, включается эвристический fallback (pipeline не падает);
  - в `.env` задается один полный SQL: `CONTROL_PLANE_CLICKHOUSE_LOGS_QUERY`;
  - SQL должен возвращать логовые поля (минимум `timestamp`, остальное опционально);
  - `anomaly['service']` обязателен;
  - поддерживаются плейсхолдеры: `{period_start}`, `{period_end}`, `{limit}`, `{offset}`, `{service}`;
  - **keyset-пейджинг** (рекомендуется): используй `{last_ts}` вместо `{offset}` — ClickHouse идёт по индексу, не скануирует с начала, нет MEMORY_LIMIT_EXCEEDED на больших объёмах;
  - если нет ни `{offset}`, ни `{last_ts}` — single-shot (одна страница);
  - промежуточные батчи не хранятся в RAM (выгружаются через progress callback в JSONL).

### Ручная страница `Logs Summarizer` (важно)

- Поддерживает один или несколько SQL-шаблонов.
- Для нескольких запросов используй разделитель строкой:
  - `-- QUERY --`
- Запросы исполняются параллельно, результаты объединяются и сортируются по `timestamp` (хронологически).
- При нескольких источниках каждая строка получает тег `_source` (query_1, query_2...) — LLM видит откуда запись.
- Перед запуском выполняется `SELECT count()` по каждому шаблону — прогресс-бар и ETA работают с первого батча.
- Если отдельный ClickHouse-запрос падает, он пропускается, ошибка показывается в UI, общий процесс не прерывается.
- Метрики-контекст ограничен 50 000 строк (защита от OOM).
- Во время выполнения параметры формы блокируются и разблокируются после завершения.
- В UI есть live-прогресс, ETA (экстраполяция по фактической скорости обработки), итоговый красивый отчет и файлы-артефакты.
- Поддерживаемые плейсхолдеры: `{period_start}`, `{period_end}`, `{limit}`, `{offset}`;
  - **keyset-пейджинг**: `{last_ts}` вместо `{offset}` — рекомендуется для больших таблиц.

Ожидаемый callable (один из поддерживаемых вариантов сигнатуры):

- `fn(period_start: str, period_end: str, anomaly: dict)`
- `fn(start_dt: datetime, end_dt: datetime, anomaly: dict)`
- `fn(start_dt: datetime, end_dt: datetime)`

Возвращаемое значение:

- строка summary, или
- объект/словарь с полем `summary`.

Для live-map в UI желательно вернуть также один из списков:

- `map_summaries`
- `batch_summaries`
- `chunk_summaries`
- `partial_summaries`
- `map_results`
- `chunks`

### Отправка уведомления

- Базовый интерфейс-заглушка: `control_plane/alerts.py` (`make_alert`).
- Кастомный путь: `CONTROL_PLANE_ALERT_CALLABLE`.
- Готовый шаблон для старта: `my_alert.py::send_sre_alert`.

Поддерживаемая сигнатура:

- `fn(text: str)`

## Запуск

## 1) Установка

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 2) Настройка

```bash
cp .env.example .env
# отредактировать .env
```

## 3) Streamlit UI

```bash
streamlit run streamlit_app.py
```

## 4) CLI loop (бесконечный цикл)

```bash
python control-plane.py
```

## Режимы

### Тестовый режим

```env
CONTROL_PLANE_TEST_MODE=true
CONTROL_PLANE_PROCESS_ALERTS=true
```

В этом режиме:

- `actual` и `predictions` генерируются синтетически;
- map/reduce и alert проходят через mock-сценарий;
- удобно для smoke-тестов UI.

### Боевой режим

```env
CONTROL_PLANE_TEST_MODE=false
CONTROL_PLANE_METRICS_SOURCE=clickhouse  # или prometheus
CONTROL_PLANE_PROCESS_ALERTS=true
```

И дополнительно настраиваются реальные креды/запросы.

## UI: что отображается

На странице `Control Plane` строго в таком порядке:

1. График прошлое + будущее с отмеченными аномалиями.
2. Табличка с найденными аномалиями.
3. Пошаговый чат-анализ Top-N аномалий (live map/reduce и уведомление).

Дополнительно:

- карточки аномалий в пункте 3 появляются последовательно, по факту обработки;
- предикт на графиках показывается только в будущем (после последней actual-точки).

На странице `Logs Summarizer`:

1. Пошаговый интерактивный чат map/reduce.
2. Итоговый отчет (структурированный summary + свободный summary + метрики/артефакты).

## Полезные переменные .env

Минимум для твоего кейса (ClickHouse actual + таблица forecast):

```env
CONTROL_PLANE_TEST_MODE=false
CONTROL_PLANE_METRICS_SOURCE=clickhouse

CONTROL_PLANE_CLICKHOUSE_METRICS_HOST=...
CONTROL_PLANE_CLICKHOUSE_METRICS_PORT=8123
CONTROL_PLANE_CLICKHOUSE_METRICS_USERNAME=...
CONTROL_PLANE_CLICKHOUSE_METRICS_PASSWORD=...
CONTROL_PLANE_CLICKHOUSE_METRICS_QUERY=SELECT timestamp, value FROM your_actual_table ORDER BY timestamp
CONTROL_PLANE_CLICKHOUSE_PREDICTIONS_HOST=...
CONTROL_PLANE_CLICKHOUSE_PREDICTIONS_PORT=8123
CONTROL_PLANE_CLICKHOUSE_PREDICTIONS_USERNAME=...
CONTROL_PLANE_CLICKHOUSE_PREDICTIONS_PASSWORD=...
CONTROL_PLANE_CLICKHOUSE_PREDICTIONS_SECURE=false
CONTROL_PLANE_CLICKHOUSE_PREDICTIONS_QUERY=SELECT timestamp, value AS predicted FROM metrics_forecast WHERE service='{service}' AND metric_name='{metric_name}' AND timestamp > now() ORDER BY timestamp
CONTROL_PLANE_FORECAST_SERVICE=airflow-test-v1
CONTROL_PLANE_FORECAST_METRIC_NAME=memory
CONTROL_PLANE_FORECAST_TYPE=short
CONTROL_PLANE_PREDICTION_KIND=forecast

# Опционально: свои адаптеры
CONTROL_PLANE_SUMMARIZER_CALLABLE=my_summarizer.summarize_logs
CONTROL_PLANE_ALERT_CALLABLE=my_alert.send_sre_alert

# Логи для batched-fetch суммаризатора
CONTROL_PLANE_LOGS_CLICKHOUSE_HOST=...
CONTROL_PLANE_LOGS_CLICKHOUSE_PORT=8123
CONTROL_PLANE_LOGS_CLICKHOUSE_USERNAME=...
CONTROL_PLANE_LOGS_CLICKHOUSE_PASSWORD=...
# LIMIT/OFFSET (классика, но на больших объёмах может упасть с MEMORY_LIMIT_EXCEEDED):
CONTROL_PLANE_CLICKHOUSE_LOGS_QUERY=SELECT timestamp, level, message FROM logs_airflow_test_v1 WHERE timestamp >= parseDateTimeBestEffort('{period_start}') AND timestamp < parseDateTimeBestEffort('{period_end}') ORDER BY timestamp LIMIT {limit} OFFSET {offset}

# Keyset-пейджинг (рекомендуется — ClickHouse использует индекс, нет OOM):
# CONTROL_PLANE_CLICKHOUSE_LOGS_QUERY=SELECT timestamp, level, message FROM logs_airflow_test_v1 WHERE timestamp > parseDateTimeBestEffort('{last_ts}') AND timestamp < parseDateTimeBestEffort('{period_end}') ORDER BY timestamp LIMIT {limit}

CONTROL_PLANE_LOGS_PAGE_LIMIT=1000
CONTROL_PLANE_UI_LOGS_SUMMARY_DEFAULT_SQL=SELECT timestamp, level, message FROM logs_airflow_test_v1 WHERE timestamp > parseDateTimeBestEffort('{last_ts}') AND timestamp < parseDateTimeBestEffort('{period_end}') ORDER BY timestamp LIMIT {limit}
CONTROL_PLANE_UI_LOGS_SUMMARY_DB_BATCH_SIZE=1000
CONTROL_PLANE_UI_LOGS_SUMMARY_LLM_BATCH_SIZE=200
```

## Диагностика

### Быстрая проверка синтаксиса

```bash
python -m py_compile control-plane.py streamlit_app.py control_plane/*.py
```

### Логи в файлы

В файл пишутся только уровни `INFO/WARNING/ERROR`:

- `artifacts/logs/control-plane.log`

### Если Matplotlib ругается на кэш-директорию

Задай writable-директорию:

```bash
export MPLCONFIGDIR=/tmp/matplotlib
```

## Структура проекта

- `control_plane/pipeline.py` — CLI loop.
- `streamlit_app.py` — UI + одноразовый прогон пайплайна.
- `control_plane/actuals.py` — источник actual-метрик (prometheus/clickhouse).
- `control_plane/predictions_db.py` — чтение forecast из БД.
- `control_plane/anomaly_detectors.py` — интерфейс и реализации детекторов.
- `control_plane/processing.py` — orchestration map/reduce + alert.
- `control_plane/summarizer.py` — адаптер внешнего суммаризатора.
- `control_plane/alerts.py` — интерфейс/заглушка отправки алерта.
- `control_plane/visualization.py` — построение графиков.
- `control_plane/config.py` — конфиг из `.env`.
