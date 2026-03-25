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

## 3) Основные модули

- `control_plane/config.py`
  - загружает `.env`
  - хранит runtime-конфиг
- `control_plane/actuals.py`
  - единый интерфейс получения `actual`
  - источники: `prometheus` и `clickhouse`
- `control_plane/predictions_db.py`
  - чтение forecast из БД (`metrics_forecast`)
- `control_plane/anomaly_detectors.py`
  - интерфейс детектора + реестр
  - built-ins: `rolling_iqr`, `residual_zscore`
- `control_plane/processing.py`
  - orchestration обработки аномалий
  - события map/reduce/alert
- `control_plane/summarizer.py`
  - дефолтный адаптер к `llm_log_summarizer`
- `control_plane/alerts.py`
  - интерфейс отправки алерта (сейчас stub)
- `control_plane/visualization.py`
  - генерация графиков

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

`my_summarizer.py::summarize_logs` делает batched-fetch логов из ClickHouse:

- берёт service из `anomaly['service']` или `CONTROL_PLANE_LOGS_DEFAULT_SERVICE`;
- определяет таблицу через `CONTROL_PLANE_LOGS_TABLE_BY_SERVICE_JSON`;
- строит запрос из `CONTROL_PLANE_LOGS_BASE_SELECT` + авто `WHERE period` + `ORDER BY` + `LIMIT/OFFSET`;
- возвращает `chunk_summaries` для live UI.

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

- `fn(summary_text: str, summary: str, anomaly: dict)`
- `fn(summary_text: str)`
- `fn(message: str)`

## 6) Runtime-события обработки аномалий

Источник: `process_anomalies(..., on_event=...)` в `control_plane/processing.py`.

Основные события:

- `process_start`
- `anomaly_start`
- `summary_start`
- `map_start`
- `map_batch`
- `map_done`
- `reduce_start`
- `summary_done`
- `reduce_done`
- `notification_ready`
- `alert_start`
- `alert_done`
- `anomaly_done`
- `anomaly_error`
- `process_done`

`streamlit_app.py` ожидает эти события и рендерит live-чат по ним.

## 7) UI-инварианты (важно не ломать)

На странице в main-area должен быть порядок:

1. График прошлое + будущее + аномалии.
2. Таблица аномалий.
3. Чат-процесс обработки Top-N аномалий.

Дополнительные требования:

- карточки аномалий появляются последовательно, по мере обработки;
- никаких лишних блоков в main-area;
- предикт на графиках должен быть только в будущем (после последней actual-точки).

## 8) Конфиг, который критичен для интеграции

- `CONTROL_PLANE_METRICS_SOURCE=prometheus|clickhouse`
- `CONTROL_PLANE_CLICKHOUSE_*` (если source=clickhouse)
- `CONTROL_PLANE_FORECAST_*` для таблицы прогнозов
- `CONTROL_PLANE_SUMMARIZER_CALLABLE` (опционально)
- `CONTROL_PLANE_ALERT_CALLABLE` (опционально)
- `CONTROL_PLANE_LOGS_*` (если используется `my_summarizer.py`)

## 9) Ограничения/риски

- В окружениях без writable `~/.config/matplotlib` возможен warning; используйте `MPLCONFIGDIR`.
- Реальные интеграции суммаризатора/алертов не тестируются в `TEST_MODE`.
- Для `clickhouse`-source query обязан вернуть корректные `timestamp/value`.

## 10) Рекомендуемый smoke-check перед merge

```bash
python -m py_compile control-plane.py streamlit_app.py control_plane/*.py
```

И базовый прогон синтетики:

```bash
CONTROL_PLANE_TEST_MODE=true CONTROL_PLANE_PROCESS_ALERTS=false python -c "from streamlit_app import run_single_iteration; from control_plane.config import *; r=run_single_iteration(test_mode=True, query=PROM_QUERY, detector_name=ANOMALY_DETECTOR, data_lookback_minutes=DATA_LOOKBACK_MINUTES, prediction_lookahead_minutes=PREDICTION_LOOKAHEAD_MINUTES, analyze_top_n=ANALYZE_TOP_N_ANOMALIES, process_lookback_minutes=LOOPBACK_MINUTES, process_alerts=False); print(len(r['actual_df']), len(r['predictions_df']), len(r['anomalies']))"
```
