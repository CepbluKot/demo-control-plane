# LLM Context: Log Summarizer Pipeline

Этот файл — полная документация для LLM/агентов. Прочитай его перед любым изменением кода.

---

## 1. Продукт: что это и зачем

**Log Summarizer** — пайплайн для SRE, который автоматически расследует инциденты по логам.

**Проблема, которую он решает:** Во время инцидента у дежурного инженера есть сотни тысяч строк логов за нужный период, несколько сработавших алертов и вопрос: «что именно сломалось и почему?» Листать логи вручную — медленно. LLM за один вызов видит только часть данных — слишком много строк не помещаются в контекст.

**Решение:** MAP-REDUCE пайплайн. Логи нарезаются на чанки, каждый анализируется LLM параллельно (MAP), затем результаты итеративно сворачиваются в единый анализ (REDUCE), из которого генерируется структурированный отчёт в 14 секциях.

**Результат:** Markdown-отчёт с хронологией событий, причинно-следственными цепочками, объяснением каждого алерта, гипотезами первопричин и конкретными рекомендациями для SRE.

**Точка входа:** `run_pipeline.py` — задай режим `MODE` и запусти `python run_pipeline.py`.

---

## 2. Архитектура: полный поток данных

```
ClickHouse
    ↓  (DataLoader.iter_log_pages — keyset pagination по группам)
list[LogRow]   ← каждая строка с zone-меткой (context_before / incident / context_after)
    ↓  (Chunker.chunk — нарезка по токеновому бюджету)
list[Chunk]   ← батчи строк, помещающихся в MAP-промпт
    ↓  (MapProcessor.process_all — параллельно, asyncio semaphore)
list[BatchAnalysis]   ← события, гипотезы, доказательства за каждый батч
    ↓  (TreeReducer.reduce — итеративный LLM merge)
MergedAnalysis   ← единый дедуплицированный анализ всего периода
    ↓
    ├── MultipassReportGenerator.generate()  →  report_multipass.md  (14 секций, ~13 LLM-вызовов)
    ├── ReportGenerator.generate()           →  report.md            (монолитный нарратив, 1 вызов)
    └── MarkdownRenderer.render()            →  report_data.md       (детерминированный, без LLM)
```

При нескольких инцидентах дополнительно:
```
list[(name, MergedAnalysis, PipelineConfig)]
    ↓  (CrossIncidentAnalyzer.generate_combined_report)
combined_report.md   ← 14 секций по объединённым данным всех инцидентов
```

---

## 3. Файловая структура проекта

```
run_pipeline.py                    ← ТОЧКА ВХОДА: настройки, список инцидентов, запуск
log_summarizer/
  config.py                        ← PipelineConfig — единственное место для всех настроек
  models.py                        ← Pydantic-модели: LogRow, Chunk, BatchAnalysis, MergedAnalysis, ...
  orchestrator.py                  ← PipelineOrchestrator — управляет 6 стадиями пайплайна
  data_loader.py                   ← DataLoader — выгрузка логов и метрик из ClickHouse
  chunker.py                       ← Chunker — нарезка LogRow на Chunk по токеновому бюджету
  map_processor.py                 ← MapProcessor — параллельная MAP-фаза
  tree_reducer.py                  ← TreeReducer — итеративная REDUCE-фаза
  llm_client.py                    ← LLMClient — HTTP к LLM, retry, JSON/TOOLS mode
  multipass_report_generator.py    ← MultipassReportGenerator — 14-секционный отчёт
  report_generator.py              ← ReportGenerator — монолитный нарратив (1 LLM-вызов)
  markdown_renderer.py             ← MarkdownRenderer — программный отчёт без LLM
  cross_incident_analyzer.py       ← CrossIncidentAnalyzer — объединённый анализ инцидентов
  prompts/
    map_system.py                  ← MAP system prompt (роль SRE-аналитика, алерты, зоны)
    map_user.py                    ← MAP user prompt (форматирование строк лога)
    reduce_merge.py                ← REDUCE merge system prompt
    reduce_compress.py             ← REDUCE compress system prompt
  utils/
    logging.py                     ← setup_pipeline_logging(), get_logger()
    progress.py                    ← ProgressTracker, TimeProgress, fmt_dur(), bar()
    tokens.py                      ← estimate_tokens(), tokens_to_chars()
```

**Артефакты каждого запуска:**
```
runs/
  {run_timestamp}/               ← одна папка на весь вызов python run_pipeline.py (МСК)
    pipeline.log                 ← лог всего запуска
    combined_report.md           ← (только при ≥2 инцидентах) объединённый отчёт
    _combined/                   ← артефакты combined_report (промпты, промежуточные данные)
    {incident_name}/
      {artifact_timestamp}/      ← одна папка на один прогон оркестратора (МСК)
        report_multipass.md      ← ГЛАВНЫЙ отчёт (14 секций)
        report.md                ← монолитный нарратив
        report_data.md           ← программный детерминированный отчёт
        chunks_meta.json         ← метаданные чанков (без сырых строк)
        raw/                     ← страницы сырых данных из ClickHouse
          page_001.txt           ← строки первой страницы в читаемом виде
          page_002.txt           ← ...
        llm/                     ← промпты и ответы LLM
          call_NNNN_{type}_{system/user/response}.txt
          call_NNNN_{type}_error.txt  ← при ошибке
        map/                     ← BatchAnalysis по каждому чанку (chunk_NNN.json)
        reduce/                  ← MergedAnalysis после каждого merge-шага
```

Каждый аудит-файл LLM начинается с заголовка:
```
# tokens: system=2,341  user=18,432  response=1,205  total=21,978
```

Временны́е метки папок используют МСК (UTC+3).

---

## 4. Ключевые модели данных (models.py)

### 4.1 Входные данные

**`LogRow`** — одна строка лога из ClickHouse:
- `timestamp: datetime`
- `level: Optional[str]` — error/warning/info/debug
- `source: Optional[str]` — `namespace/pod_name` (если pod_name доступен) или container_name
- `message: str`
- `raw_line: str` — оригинальная строка целиком (передаётся в LLM); включает `namespace/pod_name` и время
- `zone: str` — `"context_before"` | `"incident"` | `"context_after"` (проставляется DataLoader)

**`Chunk`** — нарезанный батч строк (входная единица MAP):
- `rows: list[LogRow]`
- `time_range: tuple[datetime, datetime]`
- `token_estimate: int`
- `batch_zone: str` — зона батча в целом (`"incident"` | `"context_before"` | `"context_after"` | `"mixed"`)

**`Alert`** — алерт, сработавший во время инцидента:
- `id: str` — `"alert-001"`, `"alert-002"`, ... (присваивается через `make_alerts()`)
- `name: str` — имя алерта, e.g. `"AirflowKubernetesExecutorFailed"`
- `fired_at: Optional[datetime]`
- `severity: Severity`
- `description: Optional[str]`

**`MetricRow`** — одна точка метрики:
- `timestamp: datetime`, `service: str`, `metric_name: str`, `value: float`

### 4.2 MAP-фаза: BatchAnalysis

**`BatchAnalysis`** — результат LLM-анализа одного чанка логов:
- `time_range: tuple[datetime, datetime]`
- `narrative: str` — связный текст (3-5 предложений) что происходило
- `events: list[Event]` — ключевые события, выделенные LLM
- `evidence: list[Evidence]` — дословные цитаты из логов (не сжимаются через LLM)
- `hypotheses: list[Hypothesis]` — гипотезы о причинах
- `anomalies: list[Anomaly]` — что выглядит ненормально
- `alert_refs: list[AlertRef]` — статус объяснённости каждого алерта
- `preliminary_recommendations: list[str]`
- `metrics_context: Optional[str]` — что показывают метрики в этом батче (заполняет LLM)
- `data_quality: Optional[str]` — `"processing_error"` если произошла ошибка
- `batch_zone: str` — проставляется MapProcessor программно, не LLM

**`Event`** — ключевое событие:
- `id: str` — e.g. `"evt-007-001"`
- `timestamp: datetime`
- `source: str` — `namespace/pod_name` (из raw_line; LLM использует это для группировки по podам)
- `description: str` (English) / `description_ru: Optional[str]` (русский, заполняется REDUCE)
- `severity: Severity` — critical/high/medium/low/info
- `importance: float` — 0.0–1.0, релевантность для данного расследования
- `tags: list[str]` — oom/connection/timeout/...

**`Evidence`** — дословная цитата:
- `id: str`, `timestamp: datetime`, `source: str`
- `raw_line: str` — точная строка из лога; **никогда не проходит через LLM-сжатие**
- `severity: Severity`, `linked_event_id: Optional[str]`

**`Hypothesis`** — гипотеза о причине инцидента. Три смысловых поля:
- `id: str`, `title: str` (≤60 символов), `title_ru: Optional[str]`
- `description: str` — **что произошло** по версии этой гипотезы (нарратив сценария сбоя)
- `description_ru: Optional[str]`
- `reasoning: str` — **почему считаем это верным** (конкретные доказательства, строки логов, тайминги)
- `reasoning_ru: Optional[str]`
- `confidence: str` — `"low"` | `"medium"` | `"high"`
- `supporting_event_ids: list[str]`, `contradicting_event_ids: list[str]`
- `related_alert_ids: list[str]`

**Разделение `description` / `reasoning` важно:** по `title` часто непонятно о чём гипотеза; `description` даёт нарратив, `reasoning` — обоснование. Это позволяет SRE быстро читать гипотезы в отчёте.

### 4.3 REDUCE-фаза: MergedAnalysis

**`MergedAnalysis`** — результат REDUCE, объединённый анализ:
- `time_range: tuple[datetime, datetime]`
- `narrative: str` / `narrative_ru: Optional[str]`
- `events: list[Event]` — дедуплицированный timeline
- `causal_chains: list[CausalLink]` — причинно-следственные связи между событиями
- `hypotheses: list[Hypothesis]`
- `anomalies: list[Anomaly]`
- `gaps: list[TimeGap]` — разрывы в данных
- `impact_summary: str` / `impact_summary_ru: Optional[str]`
- `preliminary_recommendations: list[str]` / `preliminary_recommendations_ru: list[str]`
- `zones_covered: list[str]` — проставляется программно, не LLM
- `evidence_bank: list[Evidence]` — НЕ проходит через LLM-merge; concat + dedup
- `alert_refs: list[AlertRef]` — лучший статус по каждому алерту; программный merge

**`CausalLink`** — причинно-следственная связь:
- `from_event_id: str`, `to_event_id: str`
- `description: str` / `description_ru: Optional[str]`
- `mechanism: Optional[str]` — КАК именно одно событие привело к другому
- `confidence: str` — low/medium/high

**`TimeGap`** — разрыв в данных:
- `start: datetime`, `end: datetime`
- `description: str` / `description_ru: Optional[str]`

**`AlertRef`** — статус объяснённости алерта:
- `alert_id: str`, `status: AlertStatus`, `comment: Optional[str]`
- `AlertStatus`: `EXPLAINED` > `PARTIAL` > `NOT_EXPLAINED` > `NOT_SEEN` (priority для merge)

### 4.4 Сериализация: что и когда исключается

| Метод | Исключает | Причина |
|---|---|---|
| `BatchAnalysis.to_json_str()` | `evidence`, `alert_refs` | Evidence → evidence_bank, alert_refs → программный merge |
| `MergedAnalysis.to_json_str()` | `evidence_bank`, `alert_refs` | Не гонять через LLM |
| `MergedAnalysis.to_json_str_with_evidence()` | ничего | Полная сериализация |

---

## 5. PipelineConfig — конфигурация

Создаётся один раз в `run_pipeline.py` и передаётся во все модули.

### Обязательные поля

| Поле | Тип | Описание |
|---|---|---|
| `logs_sql_template` | `str` | SQL с плейсхолдерами `{last_ts}`, `{period_end}`, `{limit}`, `{raw_limit}` |
| `incident_context` | `str` | Свободный текст: что случилось, какие алерты |
| `model` | `str` | Название модели |
| `api_base` | `str` | URL LLM API (например `http://localhost:8000`) |
| `api_key` | `str` | API-ключ |

### Временные окна

Пайплайн работает с двумя окнами:

**Узкое (incident window)** — когда наблюдался инцидент и сработали алерты:
- `incident_start: Optional[datetime]`
- `incident_end: Optional[datetime]`
- Алерты должны попасть внутрь этого окна

**Широкое (context window)** — откуда грузить логи (предыстория + постсоставная):
- `context_start: Optional[datetime]` и `context_end: Optional[datetime]` — явное задание
- Или автоматически: `incident ± context_auto_expand_hours` (по умолчанию ±1 час)
- `context_auto_expand_hours: float = 1.0`

Пример: инцидент 14:15–14:35 → логи грузятся 13:15–15:35.

Методы: `context_start_actual()`, `context_end_actual()`, `has_context_window()`, `validate_windows()`.

### MAP-параметры

| Поле | Дефолт | Описание |
|---|---|---|
| `batch_size` | 200 | Кол-во **групп** (агрегированных строк) из ClickHouse за один запрос |
| `batch_raw_multiplier` | 50 | Множитель для лимита сырых строк: `raw_limit = batch_size × multiplier`. Гарантирует достаточно сырых строк для формирования `batch_size` групп даже при высокой повторяемости. |
| `map_concurrency` | 5 | Параллельных LLM-вызовов в MAP |
| `max_batch_tokens` | None | Токенов на MAP-батч (None → 55% от max_context_tokens) |
| `max_split_depth` | 6 | Максимум рекурсивных делений чанка при overflow |
| `min_batch_lines` | 20 | Минимум строк — меньше этого не делить |

### REDUCE-параметры

| Поле | Дефолт | Описание |
|---|---|---|
| `max_group_size` | 4 | Максимум элементов в одной REDUCE-группе |
| `max_item_chars` | 40 000 | Лимит символов на item перед merge |
| `compression_target_pct` | 50 | % сжатия при overflow |
| `max_reduce_rounds` | 15 | Максимум раундов дерева |
| `max_events_per_merge` | 30 | Максимум событий после каждого merge-шага |
| `pre_compress_threshold` | 50 000 | Символов — порог превентивной компрессии |

### Отчёт

| Поле | Дефолт | Описание |
|---|---|---|
| `report_budget_analysis_pct` | 0.50 | Доля контекста под MergedAnalysis |
| `report_budget_evidence_pct` | 0.30 | Доля контекста под evidence_bank |
| `report_budget_early_pct` | 0.20 | Доля контекста под early_summaries (ранние MAP-саммари) |
| `report_response_reserve_tokens` | 30 000 | Резерв токенов на ответ LLM |
| `report_system_prompt_tokens` | 3 000 | Оценка размера system prompt (вычитается из бюджета) |
| `early_summaries_budget_chars` | 40 000 | Лимит символов на early_summaries в запросах к финальному отчёту |

### Прочие

| Поле | Дефолт | Описание |
|---|---|---|
| `max_context_tokens` | 150 000 | Размер контекстного окна модели |
| `use_instructor` | True | Использовать instructor для JSON-вывода |
| `model_supports_tool_calling` | False | False → JSON mode, True → TOOLS mode |
| `temperature_map` | 0.2 | Температура MAP |
| `temperature_reduce` | 0.2 | Температура REDUCE |
| `temperature_report` | 0.3 | Температура финального отчёта |
| `max_retries` | 3 | Попыток при 500/timeout (exponential backoff) |
| `retry_backoff_base` | 2.0 | База экспоненциального backoff (2, 4, 8 секунд) |
| `runs_dir` | `"runs"` | Корень для артефактов (пустая строка → не сохранять) |
| `total_log_rows` | 0 | Заполняется оркестратором после загрузки; выводится в отчёте |
| `alerts` | `[]` | Список Alert (создавать через `make_alerts()`) |
| `query_time_slice_hours` | 0.0 | Разбивка периода на слайсы (0 = без разбивки) |

---

## 6. Пайплайн: 6 стадий (PipelineOrchestrator)

`orchestrator.run()` — async, выполняет 6 стадий последовательно.

### Стадия 1: Загрузка данных

`DataLoader.iter_log_pages()` постранично выгружает логи из ClickHouse за **context-окно** (не incident). Все страницы собираются в `all_rows`, затем Chunker нарезает их все сразу по токенам — это гарантирует полный токеновый бюджет в каждом чанке.

#### Keyset-пагинация

Пагинация идёт по **группам** (агрегированным строкам после GROUP BY), а не по сырым строкам.

**Формула:** внутренний подзапрос берёт до `{raw_limit}` сырых строк → оконные функции → GROUP BY → внешний `LIMIT {limit}` групп.

Watermark `last_ts` обновляется после каждой страницы по двум правилам:
- **Последняя страница** (`len(rows) < batch_size`): `last_ts = max(end_time)` по всем строкам страницы. Предотвращает «утечку» хвостов длинных групп на следующую страницу.
- **Промежуточная страница** (`len(rows) == batch_size`): `last_ts = end_time` последней строки (с максимальным `start_time`). Предотвращает гэп, который возникает если ранняя группа имеет большой `end_time` (повторяющееся сообщение на протяжении многих часов).

**Почему так:** группы PARTITION BY container могут перекрываться по времени — группа из container A с `start_time=01:30, end_time=19:22` и группа из container B с `start_time=15:01, end_time=15:01` оба могут быть на одной странице. `max(end_time)` при промежуточных страницах создаст гэп (следующая страница начнётся с 19:22, пропустив всё от 15:01 до 19:22). `rows[-1]["end_time"]` в этом случае корректен.

Каждой строке проставляется `zone`:
- `"context_before"` — до `incident_start`
- `"incident"` — в окне инцидента
- `"context_after"` — после `incident_end`

Параллельно загружаются метрики (`DataLoader.fetch_metrics()`), если задан `metrics_sql_template`.

#### SQL-плейсхолдеры

| Плейсхолдер | Что подставляется |
|---|---|
| `{last_ts}` | Keyset: `end_time` конца предыдущей страницы |
| `{period_end}` / `{end_time}` / `{start_time}` / `{period_start}` | Границы context-окна |
| `{limit}` | Кол-во групп на страницу (= `batch_size`) |
| `{raw_limit}` | Лимит сырых строк во внутреннем подзапросе (= `batch_size × batch_raw_multiplier`) |
| `{offset}` | OFFSET-пагинация (не рекомендуется — использовать `{last_ts}`) |

#### Обязательные колонки в SELECT

- `timestamp` — DataLoader ищет это имя для zone-разметки
- `end_time` — **обязательно при keyset**: используется для вычисления watermark `last_ts`. Должен содержать максимальный сырой timestamp группы.
- `raw_line` — текст лога, передаётся в LLM

#### Рекомендуемые колонки

- `namespace` — Kubernetes-неймспейс
- `container_name` — имя контейнера
- `pod_name` — имя pod-инстанса (приоритет над container_name в поле `source`)
- `image_tag` — короткое имя образа с версией (e.g. `cert-manager-controller:v1.12.3`)

#### Прогресс загрузки

`TimeProgress` — трекер по временной оси. Прогресс = `(last_ts − start) / (end − start)`. Логирует после каждой страницы:
```
LOAD  стр.3  ████████░░░░░░  57%  7,641 гр.  ~42,300 tok  запрос 37.5s  elapsed 7m 23s  ETA ~5m 10s
...
LOAD  100%  ✓  15 стр.  22,690 гр.  18m 43s
```

`~42,300 tok` — оценка токенов для строк этой страницы (используется для мониторинга плотности данных).

### Стадия 2: MAP-фаза (MapProcessor)

`MapProcessor.process_all()` обрабатывает чанки **параллельно** через `asyncio.Semaphore(map_concurrency)`.

Для каждого чанка:
1. Строит system prompt (MAP_SYSTEM_TEMPLATE с контекстом инцидента, алертами, зонами)
2. Строит user prompt (отформатированные строки лога + метрики)
3. Вызывает `LLMClient.call_json()` → `BatchAnalysis`
4. Проставляет `batch_zone` программно (не доверяет LLM)
5. При `ContextOverflowError` — делит чанк пополам рекурсивно (до `max_split_depth`)

После split оба результата идут в REDUCE как **отдельные элементы** — LLM свяжет их через causal_chains.

**Что LLM извлекает из каждого батча:**
- `events` — ключевые события с timestamp, severity, importance (0–1)
- `evidence` — дословные цитаты строк лога
- `hypotheses` — гипотезы о причинах (каждая содержит `description` + `reasoning`)
- `anomalies` — аномалии
- `alert_refs` — статус каждого алерта (EXPLAINED / PARTIAL / NOT_EXPLAINED / NOT_SEEN)
- `preliminary_recommendations` — ранние рекомендации

**Прогресс:** `ProgressTracker(total=len(chunks))` логирует ASCII-бар + ETA + токены чанка после каждого:
```
MAP  3/12  ████████░░░░░░  25%  elapsed 45s  ETA ~2m 15s  chunk-003  14:15→14:20  200 строк  ~18,400 tok  7 событий  [15s]
```

### Стадия 3: REDUCE-фаза (TreeReducer)

Итеративно сворачивает `list[BatchAnalysis]` → один `MergedAnalysis`.

**Алгоритм одного раунда:**
1. `_adaptive_group_size()` — вычисляет сколько items влезает в 55% контекста (по среднему токенов первых 5 элементов)
2. `_make_groups()` — нарезает список на группы
3. Для каждой группы из >1 элемента → `_merge_group()` → один `MergedAnalysis`
4. `_trim_events()` — обрезает события до `max_events_per_merge` по severity (без LLM)
5. `_maybe_compress()` — сжимает через LLM если результат > `compression_target_pct` контекста
6. Повторяем до одного элемента или `max_reduce_rounds`

**Работа с evidence и alert_refs:**
- `evidence` из всех батчей собирается в `evidence_bank` **до** reduce-цикла и **никогда** не проходит через LLM
- `alert_refs` объединяются программно: для каждого `alert_id` берётся лучший статус (EXPLAINED > PARTIAL > NOT_EXPLAINED > NOT_SEEN)

**При ContextOverflowError в merge:**
- Группа >2 элементов → рекурсивный split на две половины
- Пара (2 элемента) → `_compress_and_merge()`: сжимаем по одному и пробуем merge
- Если всё равно overflow → `_programmatic_merge()` без LLM (программная конкатенация); счётчик `programmatic_merge_count`

**Превентивная компрессия:** если суммарный размер payload группы > `pre_compress_threshold` символов — сжимаем все входы ДО отправки в LLM.

**Кросс-зональные связи:** при merge группы, покрывающей зоны `context_before` + `incident`, в user-промпт добавляется явная инструкция искать causal_chains через границу зон.

**Финальный результат** хранится в `orchestrator.last_merged`.

### Стадия 4: Программный отчёт (MarkdownRenderer)

Детерминированный, без LLM. Сохраняется как `report_data.md`. Полезен для отладки.

### Стадия 5: Многопроходный LLM-отчёт (MultipassReportGenerator)

14 секций, каждая — отдельный LLM-вызов. Строго последовательно.

### Стадия 6: Монолитный LLM-отчёт (ReportGenerator)

Один LLM-вызов → нарративный текст. Сохраняется как `report.md`.

---

## 7. Многопроходный отчёт: 14 секций (MultipassReportGenerator)

Главный выход пайплайна. ~13 LLM-вызовов (секция 7 — программная заглушка).

### Порядок генерации

```
sec1  ← программно (контекст инцидента, алерты, строки логов)
sec3  ← data_coverage (покрытие данных)
sec4  ← chronology (хронология событий)
sec5  ← causal_chains (технические цепочки с ID/timestamp/механизмами)
sec5a ← root_cause_explanation (объяснение первопричины для человека)
sec6  ← alert_links (sec5a как контекст)
sec7  ← metrics (программная заглушка)
sec8  ← hypotheses (sec5a как контекст)
sec9  ← conflicts (sec5a как контекст)
sec10 ← gaps (sec5a как контекст)
sec11 ← impact (sec5a как контекст)
sec12 ← recommendations (sec5a как контекст)
sec13 ← limitations
sec14 ← coverage_recommendations
sec2  ← summary (последним; получает sec13 + sec5a)
events_ref ← программный справочник событий (таблица ID → детали)
```

### Секции отчёта

| № | Название | LLM | Ключевой контент |
|---|---|---|---|
| 1 | Контекст инцидента | ❌ | Incident context, время, алерты, строк логов |
| 2 | Резюме инцидента | ✅ | TL;DR для SRE (генерируется последним) |
| 3 | Покрытие данных | ✅ | Период, зоны, объём, SQL |
| 4 | Хронология событий | ✅ | Все события хронологически, цитаты [ФАКТ] |
| 5 | Причинно-следственные цепочки | ✅ | Технические цепочки с event ID + механизм |
| 5а | Объяснение первопричины | ✅ | Человеческий язык: симптомы → деградация → корень |
| 6 | Связь с алертами | ✅ | Статус каждого алерта + где в цепочке сработал |
| 7 | Аномалии метрик | ❌ | Заглушка (метрики не передаются в MergedAnalysis) |
| 8 | Гипотезы первопричин | ✅ | Ранжированные гипотезы + сверка с цепочкой |
| 9 | Конфликтующие версии | ✅ | Аномалии и альтернативные версии событий |
| 10 | Разрывы в цепочках | ✅ | Пробелы в данных + где разрыв в цепочке |
| 11 | Масштаб и влияние | ✅ | Затронутые сервисы, длительность, масштаб |
| 12 | Рекомендации для SRE | ✅ | P0/P1/P2, каждая рвёт звено цепочки |
| 13 | Уровень уверенности | ✅ | Честная оценка ограничений и пробелов |
| 14 | Рекомендации по анализу | ✅ | Расширить окно? Добавить сервисы? |
| — | Справочник событий | ❌ | Таблица ID → время/сервис/описание |

### Роль секции 5а (root_cause_explanation)

Связующий элемент между техническим анализом (sec5) и человеческим пониманием. Структура:
1. **Симптомы** — что наблюдал дежурный (алерты, ошибки)
2. **Цепочка деградации** — как одно событие привело к другому шаг за шагом
3. **Корень проблемы** — где и почему дал сбой
4. **Почему именно сейчас** — что изменилось или достигло предела

Текст 5а передаётся как `root_cause_text` в sec6, sec8, sec9, sec10, sec11, sec12, sec2.

### Fallback при ContextOverflowError

- `sec4` (хронология): обрезка до importance > 0.7, затем только top-20, затем `_call_or_stub()`
- `sec5` (цепочки): обрезка до confidence > 0.5
- `sec8` (гипотезы): обрезка до medium/high confidence
- Остальные секции: `_call_or_stub()` — при любой ошибке возвращает `_PLACEHOLDER`

---

## 8. LLMClient — HTTP к LLM

Единственное место, которое знает про HTTP, retry и JSON-парсинг.

### URL и /v1

`api_base` передаётся без `/v1` (например `http://localhost:8000`). LLMClient автоматически добавляет `/v1` через `_build_openai_base_url()`: если URL не заканчивается на `/v1` — дописывает. Клиент OpenAI SDK создаётся с этим нормализованным URL.

### Два метода

**`call_json(system, user, response_model, temperature)`** → Pydantic-объект

Режимы:
- `use_instructor=True` + `model_supports_tool_calling=False` → **JSON mode** через instructor
- `use_instructor=True` + `model_supports_tool_calling=True` → **TOOLS mode** через instructor
- Автоматический fallback TOOLS→JSON при grammar ошибках vLLM (`invalid grammar` / `tool_call_parser`)
- `use_instructor=False` → прямой вызов с `response_format={"type": "json_object"}` + ручной parse

**`call_text(system, user, temperature)`** → `str` — plain text для финального отчёта.

Таймаут HTTP-запросов: `timeout=600.0` секунд (передаётся в OpenAI SDK).

### Retry-логика

- Retry на: 500/502/503/504/timeout/connection error
- Exponential backoff: `retry_backoff_base ** attempt` секунд (2, 4, 8 по умолчанию)
- **Не retry** на: 400 (→ `ContextOverflowError`)
- JSON parse error → одна попытка с temperature=0.0
- После исчерпания попыток → `LLMUnavailableError`

### ContextOverflowError

Бросается при HTTP 400 с маркерами: `context_length_exceeded`, `context length`, `maximum context`, `prompt is too long`, `input is too long`, `invalid grammar`.

Это сигнал для MapProcessor (split чанка) и TreeReducer (split группы или компрессия). **Не retry.**

### Аудит промптов

Каждый вызов сохраняет файлы в `{run_dir}/llm/`:
- `call_NNNN_{kind}_system.txt` — system prompt
- `call_NNNN_{kind}_user.txt` — user prompt
- `call_NNNN_{kind}_response.txt` — ответ (или `_error.txt` при ошибке)

Каждый файл начинается с заголовка подсчёта токенов:
```
# tokens: system=2,341  user=18,432  response=1,205  total=21,978
```

Это позволяет немедленно увидеть насколько близко к лимиту контекста каждый вызов.

---

## 9. CrossIncidentAnalyzer — объединённый анализ

При ≥2 инцидентах в INCIDENTS запускается автоматически.

### generate() — вспомогательный кросс-анализ

1. **Шаг 1/2:** LLM-карточка для каждого инцидента (название, период, первопричина, топ-рекомендации)
2. **Шаг 2/2:** LLM ищет мета-цепочку между инцидентами

### generate_combined_report() — объединённый 14-секционный отчёт

1. Вызывает `generate()` для кросс-инцидентного текста
2. `TreeReducer._programmatic_merge(all_merged)` → объединённый `MergedAnalysis`
3. `_make_combined_config()` → `PipelineConfig` охватывающий все инциденты
4. Запускает `MultipassReportGenerator` → та же 14-секционная структура

---

## 10. Настройка в run_pipeline.py

### Подключение к LLM

```python
API_BASE = "http://localhost:8000"   # vLLM / OpenAI-совместимый сервер (без /v1)
API_KEY  = "sk-placeholder"          # для vLLM — любая строка
MODEL    = "Qwen3-235B"              # точное имя из /v1/models
MAX_CONTEXT_TOKENS = 100_000         # размер контекстного окна модели
MODEL_SUPPORTS_TOOL_CALLING = False  # False = JSON mode (безопаснее с vLLM)
MAX_EVENTS_PER_MERGE = 30            # макс событий после каждого REDUCE-шага
MAX_BATCH_TOKENS = None              # None → 55% от MAX_CONTEXT_TOKENS
```

### Подключение к ClickHouse

```python
CH_HOST     = "localhost"
CH_PORT     = 8123
CH_USER     = "default"
CH_PASSWORD = ""
CH_DATABASE = "default"
```

### Параметры загрузки из ClickHouse

```python
MAP_CONCURRENCY      = 5      # параллельных LLM-вызовов в MAP
BATCH_SIZE           = 1000   # кол-во групп на страницу (LIMIT на outer GROUP BY)
BATCH_RAW_MULTIPLIER = 50     # raw_limit = BATCH_SIZE × BATCH_RAW_MULTIPLIER сырых строк
                               # Обеспечивает достаточно сырых строк для BATCH_SIZE групп
                               # при высокой повторяемости (компрессия до 50:1)
QUERY_TIME_SLICE_HOURS = 0    # 0 = без разбивки на слайсы
```

### SQL-шаблон (LOGS_SQL)

#### Формат строки, который видит LLM

Каждая агрегированная запись в `raw_line` выглядит так:
```
[2026-03-18 14:48:07.694 → 2026-03-18 15:28:21.324] ×51  k-ndp-p01-runsur-airflow/airflow-scheduler-6d8f9b-xk2pq  [2026-03-18T14:48:07.693+0300] WARNING - Retrying...
```

Структура: `[start → end] ×N  namespace/pod_name  <оригинальный текст лога>`

- `[start → end]` — временной диапазон группы (повторения одного сообщения)
- `×N` — сколько раз подряд встретилось это сообщение
- `namespace/pod_name` — откуда (позволяет отличить реплики одного деплоя)
- оригинальный текст — как есть из таблицы

#### Агрегация строк (GROUP BY)

SQL группирует подряд идущие одинаковые строки (по последним 10 символам) в пределах каждого контейнера (PARTITION BY namespace, container_name). Это сжимает повторяющиеся ошибки типа "×847 Error connecting to DB" в одну запись.

**Почему PARTITION BY важен:** без него сравнение идёт между строками разных контейнеров — группировка становится семантически неверной.

#### Текущий LOGS_SQL (упрощённая схема)

```sql
SELECT
    start_time AS timestamp,
    end_time,
    namespace,
    container_name,
    pod_name,
    image_tag,    -- 'cert-manager-controller:v1.12.3'
    concat('[', toString(start_time), ' → ', toString(end_time), ']',
           ' ×', toString(cnt),
           '  ', namespace, '/', pod_name,
           '  ', log_text) AS raw_line
FROM (
    SELECT
        min(timestamp) AS start_time, max(timestamp) AS end_time,
        min(log) AS log_text, count() AS cnt,
        any(kubernetes_namespace_name) AS namespace,
        any(kubernetes_container_name) AS container_name,
        any(kubernetes_pod_name) AS pod_name,
        arrayElement(splitByChar('/', any(kubernetes_container_image)), -1) AS image_tag
    FROM (
        SELECT *, sum(is_new_group) OVER (
            PARTITION BY kubernetes_namespace_name, kubernetes_container_name
            ORDER BY timestamp ASC ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS group_id
        FROM (
            SELECT *, if(right(log, 10) != ifNull(lagInFrame(right(log, 10)) OVER (
                PARTITION BY kubernetes_namespace_name, kubernetes_container_name
                ORDER BY timestamp ASC
            ), ''), 1, 0) AS is_new_group
            FROM raw_lm.log_k8s_containers_MT
            WHERE timestamp > parseDateTime64BestEffort('{last_ts}')
              AND timestamp <= parseDateTime64BestEffort('{period_end}')
              AND ... -- фильтры по кластеру, контейнерам, ключевым словам
            ORDER BY timestamp ASC
            LIMIT {raw_limit}    -- сырые строки: batch_size × batch_raw_multiplier
        )
    )
    GROUP BY group_id, kubernetes_namespace_name, kubernetes_container_name
)
ORDER BY start_time ASC
LIMIT {limit}    -- групп на страницу: batch_size
```

#### Ключевые плейсхолдеры

| Плейсхолдер | Где | Значение |
|---|---|---|
| `{last_ts}` | WHERE внутреннего запроса | Watermark предыдущей страницы |
| `{period_end}` | WHERE внутреннего запроса | Конец context-окна (верхняя граница) |
| `{raw_limit}` | LIMIT внутреннего запроса | `batch_size × batch_raw_multiplier` |
| `{limit}` | LIMIT внешнего запроса | `batch_size` (кол-во групп на страницу) |

**Важно:** `{raw_limit}` на внутренний запрос предотвращает OOM при оконных функциях на больших периодах. `{limit}` на внешний GROUP BY гарантирует стабильное кол-во групп на страницу.

### Режим запуска (MODE)

```python
MODE = "incidents"   # "incidents" | "freeform"
```

**`"incidents"`** — обрабатывает список `INCIDENTS`. Поддерживает `--only name` и `--list`.

**`"freeform"`** — один инцидент с явным периодом `FREEFORM_START..FREEFORM_END`. Удобно для разбора произвольного временного окна без добавления в `INCIDENTS`.

### Обрезка периода по последнему алерту (AUTO_TRIM_AFTER_LAST_ALERT)

```python
AUTO_TRIM_AFTER_LAST_ALERT = False
TRIM_BUFFER_MINUTES        = 15
```

При `True` — `context_end` обрезается до `max(alert.fired_at) + TRIM_BUFFER_MINUTES`. Работает в обоих режимах. Реализовано в `_apply_trim()`.

### Список инцидентов (INCIDENTS) — режим "incidents"

```python
INCIDENTS = [
    {
        "name": "my-incident-2026-04-17",
        "context": "Описание: что случилось, какие сервисы затронуты.",
        "alerts": [
            {
                "name": "AlertName",
                "fired_at": datetime(2026, 4, 17, 14, 15, 0, tzinfo=MSK),
                "severity": "critical",
                "description": "Текст алерта",
            },
        ],
        "incident_start": datetime(2026, 4, 17, 14, 10, 0, tzinfo=MSK),
        "incident_end":   datetime(2026, 4, 17, 14, 50, 0, tzinfo=MSK),
        "context_start":  datetime(2026, 4, 17, 13, 0, 0, tzinfo=MSK),   # опционально
        "context_end":    datetime(2026, 4, 17, 16, 0, 0, tzinfo=MSK),   # опционально
    },
]
```

`fired_at` алертов должны попасть внутрь `incident_start...incident_end`. Время в МСК: `MSK = timezone(timedelta(hours=3))`.

### Freeform-инцидент — режим "freeform"

```python
FREEFORM_START   = datetime(2026, 4, 17, 1, 30, 0, tzinfo=MSK)
FREEFORM_END     = datetime(2026, 4, 17, 19, 30, 0, tzinfo=MSK)
FREEFORM_CONTEXT = "Разобраться что происходило с кластером в этот период"
FREEFORM_ALERTS  = [
    {
        "name": "AlertName",
        "fired_at": datetime(2026, 4, 17, 5, 8, 0, tzinfo=MSK),
        "severity": "critical",
        "description": "Описание алерта",
    },
]
```

Валидация при старте: `FREEFORM_START`, `FREEFORM_END` и хотя бы один алерт обязательны. Все `fired_at` должны попасть в период.

### Запуск

```bash
python run_pipeline.py                         # все инциденты из INCIDENTS (или freeform)
python run_pipeline.py --only my-incident-name # один инцидент (режим incidents)
python run_pipeline.py --list                  # список инцидентов
```

---

## 11. Логирование и прогресс

Все модули используют `get_logger(name)` — стандартный Python logging с префиксом `log_summarizer.{name}`.

Настройка: `setup_pipeline_logging(level, log_file)` — пишет на stderr и в файл одновременно.

**Что логируется на уровне INFO:**

- **LOAD:** `TimeProgress` — после каждой страницы: `стр.N  ████░░  57%  7,641 гр.  ~42k tok  запрос 37.5s  elapsed Xm  ETA ~Ym` и итоговый `LOAD  100%  ✓  15 стр.  22,690 гр.  18m 43s`
- **Оркестратор:** шапка с параметрами, старт/конец каждой стадии с elapsed
- **MAP:** `ProgressTracker` — после каждого чанка: `MAP  3/12  ████░░  25%  elapsed 45s  ETA ~2m`
- **REDUCE:** каждый раунд, каждый merge (период, события до → после), финал
- **REPORT:** `ProgressTracker` — после каждой секции

**Что НЕ логируется (пишется в файлы):**
- Промпты (→ `llm/call_NNNN_*_system.txt` с заголовком токенов)
- Ответы LLM (→ `llm/call_NNNN_*_response.txt` с заголовком токенов)
- Сырые страницы из ClickHouse (→ `raw/page_NNN.txt`)
- BatchAnalysis по чанкам (→ `map/chunk_NNN.json`)
- MergedAnalysis после каждого merge (→ `reduce/round_NN_group_NN.json`)

---

## 12. Зоны логов (zone system)

| Зона | Когда | Назначение |
|---|---|---|
| `context_before` | До `incident_start` | Предыстория: деплои, изменения |
| `incident` | В окне инцидента | Основные события |
| `context_after` | После `incident_end` | Восстановление, последствия |

**Применение:**
- `LogRow.zone` — проставляется DataLoader
- `Chunk.batch_zone` — зона батча; `"mixed"` если батч пересекает границу
- В mixed-батче MAP-промпт добавляет префиксы `[CB]` / `[INC]` / `[CA]` к каждой строке
- При REDUCE-merge с `context_before` + `incident` — LLM получает инструкцию строить кросс-зональные causal_chains

---

## 13. Ограничения и риски

### OOM в ClickHouse при оконных функциях
Оконные функции (`lagInFrame`, `sum OVER`) держат в памяти все строки периода. Решение: `LIMIT {raw_limit}` на внутренний подзапрос — ClickHouse обрабатывает не более `raw_limit` сырых строк через оконные функции за один запрос.

### Гэпы в данных при keyset-пагинации с GROUP BY
Группы PARTITION BY container перекрываются по времени. Ранняя группа с большим `end_time` (повторяющееся сообщение весь день) при `max(end_time)` как watermark создала бы гэп в несколько часов. Решение: `rows[-1]["end_time"]` (end_time последней группы по start_time) для промежуточных страниц; `max(end_time)` только для последней страницы.

### Дубли на границах страниц
Группа с `start_time=T1, end_time=T2` где `T2 > rows[-1]["end_time"]` может иметь «хвост» сырых строк в следующей странице. Они формируют дополнительную ×M группу — незначительный дубль. Не влияет на анализ.

### Параллельный MAP и GIL
asyncio — однопоточный. LLM-вызовы в `run_in_executor()` — I/O-bound, GIL не мешает.

### Программный merge (fallback)
Если merge не помещается в контекст — `_programmatic_merge()` без LLM. Счётчик: `orchestrator._tree_reducer.programmatic_merge_count`. Отражается в секции 13 отчёта.

### Висячие ссылки в MergedAnalysis
После `_trim_events()` ID в `causal_chains` и `hypotheses` могут ссылаться на удалённые события. `_check_referential_integrity()` логирует WARNING — следующий REDUCE-раунд исправит.

### vLLM context overflow
vLLM возвращает HTTP 400 при превышении контекста → `ContextOverflowError` → split чанка. Если vLLM падает (5xx) — retry. `MAX_CONTEXT_TOKENS` должен соответствовать реальному контексту модели: при заниженном значении чанки маленькие (лишние round-trips), при завышенном — чанки слишком большие и LLM возвращает 400 (автоматически обрабатывается).

---

## 14. Расширение: добавление нового SQL-шаблона или источника

1. В `LOGS_SQL` обязательны колонки: `timestamp`, `end_time` (для keyset), `raw_line`
2. Рекомендуемые: `namespace`, `container_name`, `pod_name` (DataLoader строит из них `source`)
3. Плейсхолдеры: `{last_ts}`, `{period_end}`, `{limit}`, `{raw_limit}`
4. DataLoader автоматически парсит zone и source из стандартных имён колонок
5. `pod_name` приоритетен над `container_name` для поля `source` в событиях

---

## 15. Smoke-check перед изменениями

```bash
python -m py_compile run_pipeline.py log_summarizer/*.py log_summarizer/utils/*.py log_summarizer/prompts/*.py
```

Проверка импортов:
```bash
python -c "from log_summarizer.orchestrator import PipelineOrchestrator; print('ok')"
python -c "from log_summarizer.cross_incident_analyzer import CrossIncidentAnalyzer; print('ok')"
python -c "from log_summarizer.multipass_report_generator import MultipassReportGenerator; print('ok')"
```
