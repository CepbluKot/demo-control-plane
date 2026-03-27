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
  - поддержка нескольких SQL-шаблонов с двухуровневой суммаризацией (see §7)
  - live-прогресс, ETA по timestamp, итоговый отчет и сохранение артефактов
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

## 5.2 Суммаризатор (`my_summarizer.py`)

Используется один из вариантов:

1. `CONTROL_PLANE_SUMMARIZER_CALLABLE` (приоритет)
2. дефолт `control_plane/summarizer.py::do_summary`
3. шаблон для быстрой интеграции: `my_summarizer.py::summarize_logs`

### Алгоритм `PeriodLogSummarizer`

MAP-REDUCE пайплайн внутри `my_summarizer.py`:

- **MAP**: постраничная загрузка из ClickHouse; каждая страница делится на LLM-чанки (`llm_chunk_rows`); LLM анализирует каждый чанк независимо
- Параллельный MAP: `map_workers > 1` → `ThreadPoolExecutor`; внутри каждого источника батчи отправляются параллельно; порядок результатов сохраняется
- **REDUCE**: адаптивное объединение MAP-результатов (single-pass → adaptive group-merge при переполнении контекста)
- **Freeform**: финальный нарратив (3-5 абзацев) после REDUCE

### LLM-промпты (инцидент-ориентированные)

**System prompt**: senior SRE-инженер, специализирующийся на расследовании инцидентов.
- Факты из логов → `[ФАКТ]`, гипотезы → `[ГИПОТЕЗА]`
- Если данных нет → честно пишет "данных недостаточно"

**MAP-промпт** (`_build_chunk_prompt`): секции:
- `СОБЫТИЯ` — конкретные события с timestamp и цитатами из логов
- `ПРИЗНАКИ ИНЦИДЕНТА` — привязка к алертам из контекста (формат: "Алерт X ← строка N: цитата")
- `АНОМАЛИИ` — что выглядит ненормально
- `ВОПРОСЫ` — что непонятно, что искать дальше
- Если в данных есть `_source`: статистика по источникам + инструкция искать межисточниковые связи

**REDUCE-промпт** (`_build_reduce_prompt`): секции:
- `ХРОНОЛОГИЯ` — события с реальными timestamp'ами из логов (не сочинёнными)
- `ПЕРВОПРИЧИНА` — одно утверждение + `[ФАКТ]`/`[ГИПОТЕЗА]` + доказательство
- `ОБЪЯСНЕНИЕ АЛЕРТОВ` — для каждого алерта из контекста: лог-объяснение + цитата
- `ЦЕПОЧКА СОБЫТИЙ` — конкретная цепочка A(ts) → B(ts) → алерт C (только если подтверждена)
- `ПРОБЕЛЫ` — необъяснённое, что проверить дополнительно

**Freeform-промпт** (внутренний, `_build_freeform_prompt`): черновой нарратив из REDUCE-результата.

**Финальный freeform** (UI-уровень, `_build_freeform_summary_prompt`): структурированный отчёт для SRE:
- Что произошло / Хронология / Объяснение алертов / Первопричина / Что делать прямо сейчас
- Все утверждения привязаны к конкретным алертам из пользовательского контекста

**Контекст-инцидента** (`context_prefix`): prepend к каждому LLM-запросу в директивном формате:
```
ИНЦИДЕНТ ДЛЯ РАССЛЕДОВАНИЯ:
{user_goal}

ТВОЯ ЗАДАЧА: найти в логах конкретные события, которые объясняют перечисленные алерты...
```

### LLM retry-логика

`_make_llm_call(max_retries, retry_base_delay, on_retry)`:
- При ошибке LLM делает retry с экспоненциальной задержкой: `2^attempt × retry_base_delay` секунд
- `on_retry(attempt, total, exc)` callback → событие в UI-чате
- После исчерпания попыток → `_heuristic_llm_call(prompt, error=str(exc))` (показывает реальный текст ошибки)
- Из воркер-потоков (параллельный MAP) Streamlit-рендер НЕ вызывается (защита от thread-safety)

### Пагинация и keyset

Плейсхолдеры SQL: `{period_start}`, `{period_end}`, `{limit}`, `{offset}`, `{last_ts}`, `{service}`

- `{last_ts}` → keyset-пейджинг: cursor = max(timestamp) предыдущей страницы, первая страница = period_start
- Если нет ни `{offset}`, ни `{last_ts}` → single-shot (одна страница)
- `uses_paging_template=True` если `{last_ts}` ИЛИ оба `{limit}+{offset}` → шаблон управляет пейджингом сам (без внешней обёртки)

### API `summarize_logs`

```python
summarize_logs(period_start, period_end, anomaly, on_progress) -> dict
```

Возвращает: `summary`, `freeform_summary`, `chunk_summaries`, `fetch_errors`, `stats`.

### `build_cross_source_reduce_prompt`

Standalone-функция для финального cross-source REDUCE (два-уровневая суммаризация):

```python
build_cross_source_reduce_prompt(
    summaries_by_source: Dict[str, str],
    period_start: str,
    period_end: str,
) -> str
```

Секции: ХРОНОЛОГИЯ / ПЕРВОПРИЧИНА / ОБЪЯСНЕНИЕ АЛЕРТОВ / ЦЕПОЧКА СОБЫТИЙ / ПРОБЕЛЫ — с явными ссылками на источник каждого события.

Поддерживаемые сигнатуры callable суммаризатора:

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
- `freeform_start` / `freeform_done` — генерация внутреннего нарратива (per-source)
- `fetch_error` — ошибка ClickHouse на отдельной странице (не ломает пайплайн)

## 7) Ручная страница Logs Summarizer: поведение

### 7.1 Общее

- SQL-поле принимает несколько запросов; разделитель: `-- QUERY --`
- Одиночный запрос → single-query mode; два и более → multi-query mode
- Перед запуском: `SELECT count()` по каждому шаблону → точный прогресс-бар с первого батча
- Прогресс: **timestamp-based** (по `last_batch_ts` vs период), без подсчёта батчей заранее
- Все timestamps в UI отображаются в **МСК (UTC+3)**; дефолты полей ввода тоже в МСК
- Метрики-контекст ограничен 50 000 строк суммарно (константа `MAX_METRICS_ROWS_TOTAL`)
- На время выполнения параметры формы блокируются

### 7.2 Single-query mode

1. `_db_fetch_page` → `PeriodLogSummarizer.summarize_period`
2. Все строки из одного источника
3. MAP → REDUCE → финальный freeform (UI-уровень)

### 7.3 Multi-query mode (двухуровневая суммаризация)

**Алгоритм**:

1. **Per-source MAP→REDUCE** (последовательно по источникам, параллельно внутри каждого):
   - Для каждого SQL-шаблона создаётся независимый `_db_fetch_page` с собственным keyset-курсором
   - Создаётся отдельный `PeriodLogSummarizer` → `summarize_period(total_rows_estimate=None)`
   - MAP-воркеры (`map_workers`) параллельны внутри одного источника; источники обрабатываются последовательно (нет конкурентных ThreadPoolExecutor)
   - В UI-чате появляется разделитель `--- Источник: query_N (N/M) ---`; `last_batch_ts` сбрасывается перед каждым источником

2. **Cross-source REDUCE** (один LLM-вызов):
   - После обработки всех источников вызывается `build_cross_source_reduce_prompt(summaries_by_source, period_start, period_end)`
   - Промпт явно требует найти межисточниковые причинно-следственные связи
   - Контекст инцидента (alert context) prepend'ится к cross-source промпту аналогично MAP/REDUCE

3. **Финальный freeform** (UI-уровень, общий для обоих режимов):
   - `_build_freeform_summary_prompt(final_summary, user_goal, period, stats, metrics_context)`
   - Структурированный шаблон отчёта SRE

**Важно**: `_StreamingLogsMerger` (`_db_fetch_page` multi-query ветка) больше НЕ используется в основном пайплайне. Он остался в коде для возможного переиспользования, но `streaming_logs_merger = None` всегда.

### 7.4 Ошибки и неполная суммаризация

- Ошибка ClickHouse на отдельной странице: `_register_query_error` → событие в чате + `state["query_errors"]`
- В UI во время выполнения: `st.warning("Ошибок ClickHouse: N")` если `query_errors` непустой
- После MAP: если `query_errors` непустой → в events добавляется `⚠️ ВНИМАНИЕ: часть данных не получена из БД — суммаризация неполная`
- Финальный отчёт: раскрывающийся expander "Ошибки ClickHouse" с текстами всех ошибок

### 7.5 После завершения

- Структурированный итоговый summary (post-mortem формат из REDUCE)
- Дополнительный freeform-нарратив (секции: Что произошло / Хронология / Объяснение алертов / Первопричина / Что делать)
- Результаты сохраняются в `json/md` + `jsonl` live-артефакты

### 7.6 Поддерживаемые плейсхолдеры в SQL

| Плейсхолдер | Назначение |
|---|---|
| `{period_start}` | Начало временного окна (ISO) |
| `{period_end}` | Конец временного окна (ISO) |
| `{limit}` | Размер страницы |
| `{offset}` | Смещение (LIMIT/OFFSET пагинация) |
| `{last_ts}` | Keyset-пагинация: max timestamp предыдущей страницы |

**Рекомендация**: использовать `{last_ts}` вместо `{offset}` на больших таблицах — ClickHouse использует индекс, нет MEMORY_LIMIT_EXCEEDED.

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

### Базовый

- `CONTROL_PLANE_METRICS_SOURCE=prometheus|clickhouse`
- `CONTROL_PLANE_CLICKHOUSE_METRICS_HOST|PORT|USERNAME|PASSWORD|SECURE|QUERY` (если source=clickhouse)
- `CONTROL_PLANE_CLICKHOUSE_PREDICTIONS_HOST|PORT|USERNAME|PASSWORD|SECURE`
- `CONTROL_PLANE_CLICKHOUSE_PREDICTIONS_QUERY` (опциональный полный SQL для forecast)
- `CONTROL_PLANE_FORECAST_*` для таблицы прогнозов
- `CONTROL_PLANE_SUMMARIZER_CALLABLE` (опционально)
- `CONTROL_PLANE_ALERT_CALLABLE` (опционально)

### Logs Summarizer (my_summarizer pipeline)

- `CONTROL_PLANE_LOGS_CLICKHOUSE_HOST|PORT|USERNAME|PASSWORD`
- `CONTROL_PLANE_CLICKHOUSE_LOGS_QUERY` — полный SQL с плейсхолдерами
- `CONTROL_PLANE_LOGS_PAGE_LIMIT` — размер страницы (default: 1000)
- `CONTROL_PLANE_LOGS_FETCH_MODE=time_window|tail_n_logs`
- `CONTROL_PLANE_LOGS_TAIL_LIMIT` — tail-limit для tail_n_logs режима (default: 1000)

### Logs Summarizer UI

- `CONTROL_PLANE_UI_LOGS_SUMMARY_DEFAULT_SQL` — дефолтный SQL в UI
- `CONTROL_PLANE_UI_LOGS_SUMMARY_DEFAULT_METRICS_SQL` — дефолтный SQL метрик в UI
- `CONTROL_PLANE_UI_LOGS_SUMMARY_DB_BATCH_SIZE` — страница из ClickHouse (default: 1000)
- `CONTROL_PLANE_UI_LOGS_SUMMARY_LLM_BATCH_SIZE` — строк на один LLM-запрос (default: 200)
- `CONTROL_PLANE_UI_LOGS_SUMMARY_MAP_WORKERS` — параллельных LLM-воркеров в MAP (default: 1)
- `CONTROL_PLANE_UI_LOGS_SUMMARY_MAX_RETRIES` — ретраев LLM при ошибке (default: 3)

### LLM

- `OPENAI_API_BASE_DB` — base URL LLM API (OpenAI-совместимый)
- `OPENAI_API_KEY_DB` — API-ключ
- `LLM_MODEL_ID` — идентификатор модели

## 10) Ограничения/риски

- В окружениях без writable `~/.config/matplotlib` возможен warning; используйте `MPLCONFIGDIR`.
- Реальные интеграции суммаризатора/алертов не тестируются в `TEST_MODE`.
- Для `clickhouse`-source query обязан вернуть корректные `timestamp/value`.
- **MEMORY_LIMIT_EXCEEDED (code 241)**: запросы с `LIMIT N OFFSET M` на больших таблицах ClickHouse читают все строки до OFFSET в RAM. Симптом: суммаризация внезапно останавливается после первой страницы. Решение — keyset-пейджинг через `{last_ts}`.
- При keyset-пейджинге дубли строк с одинаковым timestamp на границе страниц теоретически возможны — используй миллисекундную точность в `timestamp`.
- Метрики-контекст в `Logs Summarizer` ограничен 50 000 строк (константа `MAX_METRICS_ROWS_TOTAL`).
- **Demo mode и размер выборки**: в demo-режиме генерируется `max(LOGS_TAIL_LIMIT, DB_BATCH_SIZE×4, 4000)` строк; значения меньше `DB_BATCH_SIZE` не дадут пагинации (одна страница → стоп).
- **Параллельный MAP и Streamlit**: `on_retry` callback из воркер-потоков не вызывает `_render_logs_summary_chat` — рендер только из main thread. GIL защищает `state["events"].append()`.
- В multi-query mode источники обрабатываются **последовательно** (не параллельно между собой). Добавление межисточниковой параллельности потребует thread-safe аккумуляции `state` и единого executor.

## 11) Рекомендуемый smoke-check перед merge

```bash
python -m py_compile control-plane.py streamlit_app.py control_plane/*.py my_summarizer.py ui/pages/logs_summary_page.py
```

И базовый прогон синтетики:

```bash
CONTROL_PLANE_TEST_MODE=true CONTROL_PLANE_PROCESS_ALERTS=false python -c "from streamlit_app import run_single_iteration; from control_plane.config import *; r=run_single_iteration(test_mode=True, query=PROM_QUERY, detector_name=ANOMALY_DETECTOR, data_lookback_minutes=DATA_LOOKBACK_MINUTES, prediction_lookahead_minutes=PREDICTION_LOOKAHEAD_MINUTES, analyze_top_n=ANALYZE_TOP_N_ANOMALIES, process_lookback_minutes=LOOPBACK_MINUTES, process_alerts=False); print(len(r['actual_df']), len(r['predictions_df']), len(r['anomalies']))"
```
