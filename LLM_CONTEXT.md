# LLM Context: Demo Control Plane

Этот файл предназначен для LLM/агентов, которые подключаются к проекту и должны быстро понять архитектуру, контракты и ограничения.

## 1) Цель проекта

Pipeline для SRE-кейса:

1. Получить `actual` метрику за окно времени.
2. Получить `predictions` (forecast).
3. Найти аномалии на фактических данных.
4. По Top-N аномалиям сделать map-reduce суммаризацию логов.
5. Сгенерировать уведомление.
6. Показать live-прогресс и результаты в Streamlit UI.

## 2) Точки входа

- CLI loop: `control-plane.py` -> `control_plane.pipeline.main()`.
- Streamlit: `streamlit_app.py`.
- Streamlit содержит 2 страницы:
  - `Control Plane` (основной end-to-end pipeline по аномалиям).
  - `Logs Summarizer` (ручной map-reduce анализ логов).

## 3) Основные модули

- `control_plane/config.py`
  - читает runtime-конфиг через `settings.py` (Pydantic Settings)
  - хранит runtime-конфиг
- `control_plane/actuals.py`
  - единый интерфейс получения `actual`
  - источники: `prometheus` и `clickhouse`
- `control_plane/predictions_db.py`
  - чтение forecast из БД (`metrics_forecast`)
- `control_plane/anomaly_detectors.py`
  - интерфейс детектора + реестр
  - built-ins: `rolling_iqr`, `residual_zscore`, `pyod_ecod`, `pyod_iforest`, `ruptures_pelt`
- `control_plane/processing.py`
  - orchestration обработки аномалий
  - события map/reduce/alert
- `ui/pages/logs_summary_page.py`
  - ручной интерактивный map-reduce UI
  - поддержка нескольких SQL-шаблонов, объединение результатов
  - live-прогресс, ETA, итоговый отчет и сохранение артефактов
- `control_plane/summarizer.py`
  - дефолтный адаптер к `my_summarizer.summarize_logs`
- `control_plane/alerts.py`
  - интерфейс отправки алерта (сейчас stub)
- `control_plane/visualization.py`
  - генерация графиков
- `control_plane/predictions_db.py`, `control_plane/actuals.py`, `my_summarizer.py`
  - ClickHouse-запросы выполняются через `clickhouse_connect.get_client(...).query_df(...)`

## 4) Ключевые DataFrame-контракты

### 4.1 `actual_df`

Обязательные колонки:

- `timestamp` (datetime-like, UTC)
- `value` (numeric)

### 4.2 `predictions_df`

Обязательные колонки:

- `timestamp` (datetime-like, UTC)
- `predicted` (numeric)

### 4.3 `merged_df`

После merge содержит минимум:

- `timestamp`
- `value`
- `predicted`
- `residual` (после детекции)
- `is_anomaly` (после детекции)

### 4.4 `anomalies_df`

Подмножество `merged_df` плюс:

- `source` (для actual-аномалий обычно `actual`)

## 5) Контракты расширения

## 5.1 Аномали-детектор

Протокол: `AnomalyDetector` (`control_plane/anomaly_detectors.py`)

```python
def detect(actual_df: pd.DataFrame, predictions_df: pd.DataFrame, step: str) -> DetectionResult
```

Возвращает `DetectionResult(merged_df, anomalies_df)`.

## 5.2 Суммаризатор

Используется один из вариантов:

1. `CONTROL_PLANE_SUMMARIZER_CALLABLE` (приоритет)
2. дефолт `control_plane/summarizer.py::do_summary`
3. шаблон для быстрой интеграции: `my_summarizer.py::summarize_logs`

`my_summarizer.py::summarize_logs` делает map-reduce суммаризацию логов:

- берёт service из `anomaly['service']` (обязательное поле);
- использует один полный SQL-шаблон из `CONTROL_PLANE_CLICKHOUSE_LOGS_QUERY`;
- SQL возвращает логовые поля (минимум `timestamp`);
- поддерживает плейсхолдеры: `{period_start}`, `{period_end}`, `{limit}`, `{offset}`, `{service}`, `{last_ts}`;
- **keyset-пейджинг** (`{last_ts}`): вместо LIMIT/OFFSET курсором служит max(timestamp) предыдущей страницы — ClickHouse использует индекс, нет MEMORY_LIMIT_EXCEEDED; рекомендуется для больших таблиц;
- если нет ни `{offset}`, ни `{last_ts}` — single-shot (одна страница);
- ошибки ClickHouse перехватываются: страница пропускается, `on_error` callback, результат содержит `fetch_errors`;
- `keep_map_batches_in_memory=False` по умолчанию — сырые строки не хранятся в RAM;
- промпты ориентированы на incident investigation:
  - MAP: TIMELINE / FIRST_SYMPTOM / CAUSAL_HINTS / EVIDENCE / OPEN_QUESTIONS (хронологический порядок);
  - REDUCE: post-mortem (INCIDENT_TIMELINE / FIRST_SIGNAL / ROOT_CAUSE_HYPOTHESES / CAUSAL_CHAIN / SUPPORTING_EVIDENCE / PRIORITY_ACTIONS);
  - freeform-нарратив: 3-5 абзацев связным текстом для SRE-команды (отдельный LLM-вызов после reduce);
- возвращает словарь с `summary`, `freeform_summary`, `chunk_summaries`, `fetch_errors`, `stats`.

Поддерживаемые сигнатуры callable:

- `fn(period_start: str, period_end: str, anomaly: dict)`
- `fn(start_dt: datetime, end_dt: datetime, anomaly: dict)`
- `fn(start_dt: datetime, end_dt: datetime)`

Ожидаемый результат:

- строка, или
- объект/dict с `summary`

Для красивого live map UI желательно возвращать один из list-полей:

- `map_summaries`
- `batch_summaries`
- `chunk_summaries`
- `partial_summaries`
- `map_results`
- `chunks`

## 5.3 Alert sender

Используется один из вариантов:

1. `CONTROL_PLANE_ALERT_CALLABLE` (приоритет)
2. `control_plane/alerts.py::make_alert` (stub)
3. шаблон для быстрой интеграции: `my_alert.py::send_sre_alert`

Поддерживаемые сигнатуры callable:

- `fn(text: str)`

## 6) Runtime-события обработки аномалий

Источник: `process_anomalies(..., on_event=...)` в `control_plane/processing.py`.

Основные события:

- `process_start`
- `anomaly_start`
- `summary_start`
- `map_start`
- `map_batch_start`
- `map_batch`
- `map_done`
- `reduce_start`
- `reduce_group_start`
- `reduce_group_done`
- `summary_done`
- `reduce_done`
- `notification_ready`
- `alert_start`
- `alert_done`
- `anomaly_done`
- `anomaly_error`
- `process_done`

`streamlit_app.py` ожидает эти события и рендерит live-чат по ним.

Для `Logs Summarizer`-страницы (ручной режим) дополнительно:
- `page_fetched` — страница загружена из ClickHouse
- `map_batch_start` / `map_batch` — начало/конец LLM-анализа чанка
- `map_start` / `map_done`
- `reduce_start` / `reduce_group_start` / `reduce_group_done` / `reduce_done`
- `freeform_start` / `freeform_done` — генерация свободного нарратива
- `fetch_error` — ошибка ClickHouse на отдельной странице (не ломает пайплайн)

## 7) Ручная страница Logs Summarizer: поведение

- В SQL-поле можно указывать несколько запросов; разделитель между ними:
  - `-- QUERY --`
- Запросы выполняются параллельно и объединяются в общий поток через min-heap по `timestamp`.
- При нескольких источниках каждая строка получает тег `_source` (query_1, query_2...) — LLM учитывает источник.
- После объединения строки сортируются по `timestamp` в хронологическом порядке.
- Перед запуском выполняется `SELECT count()` по каждому шаблону — прогресс-бар и ETA точны с первого батча.
- Ошибка отдельного ClickHouse-запроса не ломает весь процесс:
  - запрос пропускается;
  - ошибка логируется и показывается в UI;
  - в итоговый отчёт попадает список `fetch_errors`.
- Метрики-контекст ограничен 50 000 строк суммарно (защита от OOM).
- На время выполнения параметры формы блокируются.
- После завершения:
  - появляется структурированный итоговый summary (post-mortem формат);
  - дополнительно генерируется freeform-нарратив (3-5 абзацев для SRE-команды);
  - результаты сохраняются в `json/md` + `jsonl` live-артефакты.
- Поддерживаемые плейсхолдеры в SQL: `{period_start}`, `{period_end}`, `{limit}`, `{offset}`, `{last_ts}`.
  - `{last_ts}` — keyset-пейджинг: рекомендуется для больших таблиц во избежание MEMORY_LIMIT_EXCEEDED.

## 8) UI-инварианты (важно не ломать)

На странице в main-area должен быть порядок:

1. График прошлое + будущее + аномалии.
2. Таблица аномалий.
3. Чат-процесс обработки Top-N аномалий.

Дополнительные требования:

- карточки аномалий появляются последовательно, по мере обработки;
- никаких лишних блоков в main-area;
- предикт на графиках должен быть только в будущем (после последней actual-точки).

## 9) Конфиг, который критичен для интеграции

- `CONTROL_PLANE_METRICS_SOURCE=prometheus|clickhouse`
- `CONTROL_PLANE_CLICKHOUSE_METRICS_HOST|PORT|USERNAME|PASSWORD|SECURE|QUERY` (если source=clickhouse)
- `CONTROL_PLANE_CLICKHOUSE_PREDICTIONS_HOST|PORT|USERNAME|PASSWORD|SECURE`
- `CONTROL_PLANE_CLICKHOUSE_PREDICTIONS_QUERY` (опциональный полный SQL для forecast)
- `CONTROL_PLANE_FORECAST_*` для таблицы прогнозов
- `CONTROL_PLANE_SUMMARIZER_CALLABLE` (опционально)
- `CONTROL_PLANE_ALERT_CALLABLE` (опционально)
- `CONTROL_PLANE_LOGS_CLICKHOUSE_HOST|PORT|USERNAME|PASSWORD`
- `CONTROL_PLANE_CLICKHOUSE_LOGS_QUERY`
- `CONTROL_PLANE_LOGS_PAGE_LIMIT`
- `CONTROL_PLANE_UI_LOGS_SUMMARY_DEFAULT_SQL`
- `CONTROL_PLANE_UI_LOGS_SUMMARY_DB_BATCH_SIZE`
- `CONTROL_PLANE_UI_LOGS_SUMMARY_LLM_BATCH_SIZE`

## 10) Ограничения/риски

- В окружениях без writable `~/.config/matplotlib` возможен warning; используйте `MPLCONFIGDIR`.
- Реальные интеграции суммаризатора/алертов не тестируются в `TEST_MODE`.
- Для `clickhouse`-source query обязан вернуть корректные `timestamp/value`.
- Запросы с `LIMIT N OFFSET M` на больших таблицах ClickHouse могут вызывать MEMORY_LIMIT_EXCEEDED (code 241): ClickHouse читает все строки до OFFSET в RAM. Решение — keyset-пейджинг через `{last_ts}`.
- При keyset-пейджинге дубли строк с одинаковым timestamp на границе страниц теоретически возможны — используй миллисекундную точность в `timestamp`.
- Метрики-контекст в `Logs Summarizer` ограничен 50 000 строк (константа `MAX_METRICS_ROWS_TOTAL`).

## 11) Рекомендуемый smoke-check перед merge

```bash
python -m py_compile control-plane.py streamlit_app.py control_plane/*.py
```

И базовый прогон синтетики:

```bash
CONTROL_PLANE_TEST_MODE=true CONTROL_PLANE_PROCESS_ALERTS=false python -c "from streamlit_app import run_single_iteration; from control_plane.config import *; r=run_single_iteration(test_mode=True, query=PROM_QUERY, detector_name=ANOMALY_DETECTOR, data_lookback_minutes=DATA_LOOKBACK_MINUTES, prediction_lookahead_minutes=PREDICTION_LOOKAHEAD_MINUTES, analyze_top_n=ANALYZE_TOP_N_ANOMALIES, process_lookback_minutes=LOOPBACK_MINUTES, process_alerts=False); print(len(r['actual_df']), len(r['predictions_df']), len(r['anomalies']))"
```
