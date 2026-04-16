# LLM Context: Log Summarizer Pipeline

Этот файл — полная документация для LLM/агентов. Прочитай его перед любым изменением кода.

---

## 1. Продукт: что это и зачем

**Log Summarizer** — пайплайн для SRE, который автоматически расследует инциденты по логам.

**Проблема, которую он решает:** Во время инцидента у дежурного инженера есть сотни тысяч строк логов за нужный период, несколько сработавших алертов и вопрос: «что именно сломалось и почему?» Листать логи вручную — медленно. LLM за один вызов видит только часть данных — слишком много строк не помещаются в контекст.

**Решение:** MAP-REDUCE пайплайн. Логи нарезаются на чанки, каждый анализируется LLM параллельно (MAP), затем результаты итеративно сворачиваются в единый анализ (REDUCE), из которого генерируется структурированный отчёт в 14 секциях.

**Результат:** Markdown-отчёт с хронологией событий, причинно-следственными цепочками, объяснением каждого алерта, гипотезами первопричин и конкретными рекомендациями для SRE.

**Точка входа:** `run_pipeline.py` — заполни `INCIDENTS` и запусти `python run_pipeline.py`.

---

## 2. Архитектура: полный поток данных

```
ClickHouse
    ↓  (DataLoader.iter_log_pages — keyset pagination)
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
    progress.py                    ← ProgressTracker, fmt_dur(), bar()
    tokens.py                      ← estimate_tokens(), tokens_to_chars()
```

**Артефакты каждого запуска:**
```
runs/
  {run_timestamp}/               ← одна папка на весь вызов python run_pipeline.py
    pipeline.log                 ← лог всего запуска
    combined_report.md           ← (только при ≥2 инцидентах) объединённый отчёт
    _combined/                   ← артефакты combined_report (промпты, промежуточные данные)
    {incident_name}/
      {artifact_timestamp}/      ← одна папка на один прогон оркестратора
        report_multipass.md      ← ГЛАВНЫЙ отчёт (14 секций)
        report.md                ← монолитный нарратив
        report_data.md           ← программный детерминированный отчёт
        chunks_meta.json         ← метаданные чанков (без сырых строк)
        llm/                     ← промпты и ответы LLM (call_NNNN_{type}_{system/user/response}.txt)
        map/                     ← BatchAnalysis по каждому чанку (chunk_NNN.json)
        reduce/                  ← MergedAnalysis после каждого merge-шага
```

---

## 4. Ключевые модели данных (models.py)

### 4.1 Входные данные

**`LogRow`** — одна строка лога из ClickHouse:
- `timestamp: datetime`
- `level: Optional[str]` — error/warning/info/debug
- `source: Optional[str]` — сервис / pod / контейнер
- `message: str`
- `raw_line: str` — оригинальная строка целиком (передаётся в LLM)
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
- `data_quality: Optional[str]` — `"processing_error"` если произошла ошибка
- `batch_zone: str` — проставляется MapProcessor программно, не LLM

**`Event`** — ключевое событие:
- `id: str` — e.g. `"evt-007-001"`
- `timestamp: datetime`
- `source: str` — сервис/pod
- `description: str` (English) / `description_ru: Optional[str]` (русский, заполняется REDUCE)
- `severity: Severity` — critical/high/medium/low/info
- `importance: float` — 0.0–1.0, релевантность для данного расследования
- `tags: list[str]` — oom/connection/timeout/...

**`Evidence`** — дословная цитата:
- `id: str`, `timestamp: datetime`, `source: str`
- `raw_line: str` — точная строка из лога; **никогда не проходит через LLM-сжатие**
- `severity: Severity`, `linked_event_id: Optional[str]`

**`Hypothesis`** — гипотеза:
- `id: str`, `title: str`, `title_ru: Optional[str]`
- `description: str`, `description_ru: Optional[str]`
- `confidence: str` — `"low"` | `"medium"` | `"high"`
- `supporting_event_ids: list[str]`, `contradicting_event_ids: list[str]`
- `related_alert_ids: list[str]`

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
| `logs_sql_template` | `str` | SQL с плейсхолдерами `{start_time}`, `{end_time}`, `{last_ts}`, `{limit}` |
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
| `batch_size` | 200 | Строк из ClickHouse за один запрос |
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
| `report_response_reserve_tokens` | 30 000 | Резерв токенов на ответ LLM |

### Прочие

| Поле | Дефолт | Описание |
|---|---|---|
| `max_context_tokens` | 150 000 | Размер контекстного окна модели |
| `use_instructor` | True | Использовать instructor для JSON-вывода |
| `model_supports_tool_calling` | False | False → JSON mode, True → TOOLS mode |
| `temperature_map` | 0.2 | Температура MAP |
| `temperature_reduce` | 0.2 | Температура REDUCE |
| `temperature_report` | 0.3 | Температура финального отчёта |
| `runs_dir` | `"runs"` | Корень для артефактов (пустая строка → не сохранять) |
| `total_log_rows` | 0 | Заполняется оркестратором после загрузки; выводится в отчёте |
| `alerts` | `[]` | Список Alert (создавать через `make_alerts()`) |

---

## 6. Пайплайн: 6 стадий (PipelineOrchestrator)

`orchestrator.run()` — async, выполняет 6 стадий последовательно.

### Стадия 1: Загрузка данных

`DataLoader.iter_log_pages()` постранично выгружает логи из ClickHouse за **context-окно** (не incident).

- Поддерживает **keyset-пагинацию** через `{last_ts}` — каждая следующая страница начинается с `max(timestamp)` предыдущей. Это критично для больших таблиц: `OFFSET` вызывает `MEMORY_LIMIT_EXCEEDED`.
- Каждой строке проставляется `zone`:
  - `"context_before"` — до `incident_start`
  - `"incident"` — в окне инцидента
  - `"context_after"` — после `incident_end`
- Параллельно загружаются метрики (`DataLoader.fetch_metrics()`), если задан `metrics_sql_template`.
- `config.total_log_rows` заполняется суммарным числом строк.

**SQL-плейсхолдеры:**

| Плейсхолдер | Что подставляется |
|---|---|
| `{start_time}` / `{period_start}` | Начало context-окна |
| `{end_time}` / `{period_end}` | Конец context-окна |
| `{limit}` | Размер страницы |
| `{offset}` | Смещение (LIMIT/OFFSET пагинация) |
| `{last_ts}` | Keyset: max timestamp предыдущей страницы |

**Рекомендация:** всегда использовать `{last_ts}` вместо `{offset}`.

Обязательные колонки в SELECT:
- `timestamp` — DataLoader ищет это имя для keyset и zone-разметки
- `raw_line` — текст лога, передаётся в LLM; fallback: `message` / `msg` / `log` / `value`

Опциональные для `source`: `source` / `service` / `kubernetes_container_name`; или `namespace` + `container_name`.

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
- `hypotheses` — гипотезы о причинах
- `anomalies` — аномалии
- `alert_refs` — статус каждого алерта (EXPLAINED / PARTIAL / NOT_EXPLAINED / NOT_SEEN)
- `preliminary_recommendations` — ранние рекомендации

**Прогресс:** `ProgressTracker(total=len(chunks))` логирует ASCII-бар + ETA после каждого чанка.

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

**Кросс-зональные связи:** при merge группы, покрывающей зоны `context_before` + `incident`, в user-промпт добавляется явная инструкция искать causal_chains через границу зон — это самые ценные связи: изменение/деплой до инцидента, которое его породило.

**Zones_covered** проставляется программно как union зон входных items.

**Финальный результат** хранится в `orchestrator.last_merged`.

### Стадия 4: Программный отчёт (MarkdownRenderer)

Детерминированный, без LLM. Сохраняется как `report_data.md`. Полезен для отладки — показывает точно то, что извлечено из MergedAnalysis.

### Стадия 5: Многопроходный LLM-отчёт (MultipassReportGenerator)

14 секций, каждая — отдельный LLM-вызов. Строго последовательно.

### Стадия 6: Монолитный LLM-отчёт (ReportGenerator)

Один LLM-вызов → нарративный текст. Сохраняется как `report.md`.

---

## 7. Многопроходный отчёт: 14 секций (MultipassReportGenerator)

Главный выход пайплайна. ~13 LLM-вызовов (секция 7 — программная заглушка).

### Порядок генерации

Сначала генерируются секции 5 и 5а — они служат контекстом для всех последующих.

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

Это связующий элемент между техническим анализом (sec5) и человеческим пониманием. Структура:
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

### Прогресс

`ProgressTracker(total=13, label="REPORT")` — логирует бар + ETA после каждой секции.

---

## 8. LLMClient — HTTP к LLM

Единственное место, которое знает про HTTP, retry и JSON-парсинг.

### Два метода

**`call_json(system, user, response_model, temperature)`** → Pydantic-объект

Режимы (определяются при инициализации):
- `use_instructor=True` + `model_supports_tool_calling=False` → **JSON mode** через instructor
- `use_instructor=True` + `model_supports_tool_calling=True` → **TOOLS mode** через instructor
- Автоматический fallback TOOLS→JSON при grammar ошибках vLLM (`invalid grammar` / `tool_call_parser`)
- `use_instructor=False` → прямой вызов с `response_format={"type": "json_object"}` + ручной parse

**`call_text(system, user, temperature)`** → `str`

Plain text — используется для финального отчёта.

### Retry-логика

- Retry на: 500/502/503/504/timeout/connection error
- Exponential backoff: `retry_backoff_base ** attempt` секунд (2, 4, 8 по умолчанию)
- **Не retry** на: 400 (→ `ContextOverflowError`)
- JSON parse error → одна попытка с temperature=0.0
- После исчерпания попыток → `LLMUnavailableError`

### ContextOverflowError

Бросается при HTTP 400 с маркерами: `context_length_exceeded`, `context length`, `maximum context`, `prompt is too long`, `input is too long`, `invalid grammar` (vLLM grammar overflow).

Это сигнал для MapProcessor (split чанка) и TreeReducer (split группы или компрессия).

### Аудит промптов

Если `audit_dir` задан, каждый вызов сохраняет:
- `call_NNNN_json_system.txt` — system prompt
- `call_NNNN_json_user.txt` — user prompt
- `call_NNNN_json_response.txt` — ответ LLM (или `_error.txt` при ошибке)

Аудит-папка: `{run_dir}/llm/` (передаётся оркестратором).

---

## 9. CrossIncidentAnalyzer — объединённый анализ

При ≥2 инцидентах в INCIDENTS запускается автоматически.

### generate() — вспомогательный кросс-анализ

Входные данные: `list[(name, MergedAnalysis)]`.

1. **Шаг 1/2:** LLM-карточка для каждого инцидента (название, период, первопричина, ключевой механизм, топ-рекомендации)
2. **Шаг 2/2:** LLM ищет мета-цепочку — реальные причинно-следственные связи между инцидентами. Критерии: один инцидент создал условия для следующего / общая первопричина в разных формах / побочный эффект восстановления. Если связей нет — явно пишет «Связей не обнаружено».

Результат: `cross_incident_report.md` в папке запуска.

### generate_combined_report() — объединённый 14-секционный отчёт

Входные данные: `list[(name, MergedAnalysis, PipelineConfig)]`.

1. Вызывает `generate()` для кросс-инцидентного текста (используется как контекст)
2. `TreeReducer._programmatic_merge(all_merged)` → объединённый `MergedAnalysis`
3. `_make_combined_config()` → `PipelineConfig` охватывающий все инциденты:
   - `incident_start = min(all starts)`, `incident_end = max(all ends)`
   - `context_start = min(all context_starts)`, `context_end = max(all context_ends)`
   - `context_auto_expand_hours = 0` (окна заданы явно)
   - `alerts` — union всех алертов без дублей по id
   - `total_log_rows = sum(all total_log_rows)`
   - `incident_context` — сводный текст всех инцидентов + первые 1500 символов кросс-анализа
4. Запускает `MultipassReportGenerator` → та же 14-секционная структура

Результат: `combined_report.md` в папке запуска.

---

## 10. Настройка в run_pipeline.py

### Подключение к LLM

```python
API_BASE = "http://localhost:8000"   # vLLM / OpenAI-совместимый сервер (без /v1)
API_KEY  = "sk-placeholder"          # для vLLM — любая строка
MODEL    = "Qwen3-235B"              # точное имя из /v1/models
MAX_CONTEXT_TOKENS = 100_000         # размер контекстного окна модели
MODEL_SUPPORTS_TOOL_CALLING = False  # False = JSON mode (безопаснее с vLLM)
```

### Подключение к ClickHouse

```python
CH_HOST     = "localhost"
CH_PORT     = 8123          # HTTP-порт
CH_USER     = "default"
CH_PASSWORD = ""
CH_DATABASE = "default"
```

### SQL-шаблон (LOGS_SQL)

Обязательные колонки в SELECT: `timestamp`, `raw_line`.

Рекомендуемый паттерн с keyset-пагинацией:
```sql
SELECT
    timestamp,
    source,
    raw_line
FROM your_logs_table
WHERE timestamp BETWEEN '{start_time}' AND '{end_time}'
  AND timestamp > '{last_ts}'   -- keyset pagination
  -- ... фильтры по сервисам/ошибкам
ORDER BY timestamp ASC
LIMIT {limit}
```

**Не используй `{offset}`** — на больших таблицах ClickHouse читает все строки до смещения в RAM → `MEMORY_LIMIT_EXCEEDED`.

### Список инцидентов (INCIDENTS)

```python
INCIDENTS = [
    {
        "name": "my-incident-2026-04-16",    # slug: только латиница, цифры, дефисы
        "context": """
            Описание: что случилось, какие сервисы затронуты.
            Чем конкретнее — тем точнее анализ.
        """,
        "alerts": [
            {
                "name": "AlertName",
                "fired_at": datetime(2026, 4, 16, 14, 15, 0, tzinfo=MSK),
                "severity": "critical",              # critical/high/medium/low/info
                "description": "Текст алерта",       # опционально
            },
        ],
        "incident_start": datetime(2026, 4, 16, 14, 10, 0, tzinfo=MSK),
        "incident_end":   datetime(2026, 4, 16, 14, 50, 0, tzinfo=MSK),
        # Опциональное широкое окно (по умолчанию ±1 час от incident):
        "context_start":  datetime(2026, 4, 16, 13, 0, 0, tzinfo=MSK),
        "context_end":    datetime(2026, 4, 16, 16, 0, 0, tzinfo=MSK),
    },
]
```

**Важно:** `fired_at` алертов должны попасть внутрь `incident_start...incident_end`. Время задавать в МСК (UTC+3): `MSK = timezone(timedelta(hours=3))`.

### Запуск

```bash
python run_pipeline.py                         # все инциденты из INCIDENTS
python run_pipeline.py --only my-incident-name # один инцидент
python run_pipeline.py --list                  # список инцидентов
```

---

## 11. Логирование и прогресс

Все модули используют `get_logger(name)` — стандартный Python logging с префиксом `log_summarizer.{name}`.

Настройка: `setup_pipeline_logging(level, log_file)` — пишет на stderr и в файл одновременно.

**Что логируется на уровне INFO:**

- **Оркестратор:** шапка с параметрами, старт/конец каждой из 6 стадий с elapsed, итоговый summary
- **MAP:** `ProgressTracker` — после каждого чанка: N/M ████ % elapsed ETA | chunk-NNN HH:MM→HH:MM строк событий
- **REDUCE:** каждый раунд (items → groups), каждый merge (период, события, гипотезы до → после), финал
- **REPORT:** `ProgressTracker` — после каждой секции: N/M ████ % elapsed ETA → путь к файлу
- **Cross-incident:** шаги 1/2 и 2/2 с elapsed

**Что НЕ логируется (пишется в файлы):**
- Промпты (→ `llm/call_NNNN_*_system.txt`)
- Ответы LLM (→ `llm/call_NNNN_*_response.txt`)
- BatchAnalysis по чанкам (→ `map/chunk_NNN.json`)
- MergedAnalysis после каждого merge (→ `reduce/round_NN_group_NN.json`)

**`ProgressTracker`:**
```
MAP  3/12  ████████░░░░░░  25%  elapsed 45s  ETA ~2m 15s  | chunk-003  14:15:00→14:20:00  200 строк  7 событий  [15s]
```

**`fmt_dur(seconds)`:** `"45s"` | `"3m 12s"` | `"1h 04m"` | `"<1s"`

---

## 12. Зоны логов (zone system)

Пайплайн разделяет логи на три зоны относительно incident-окна:

| Зона | Когда | Назначение |
|---|---|---|
| `context_before` | До `incident_start` | Предыстория: деплои, изменения, накопленная нагрузка |
| `incident` | В окне инцидента | Основные события инцидента |
| `context_after` | После `incident_end` | Восстановление, последствия |

**Применение:**
- `LogRow.zone` — проставляется DataLoader по каждой строке
- `Chunk.batch_zone` — зона батча; `"mixed"` если батч пересекает границу
- `MergedAnalysis.zones_covered` — union зон; проставляется TreeReducer программно
- В REDUCE-промпте: при merge группы с `context_before` + `incident` LLM получает явную инструкцию строить кросс-зональные causal_chains — это самые ценные связи

---

## 13. Ограничения и риски

### MEMORY_LIMIT_EXCEEDED в ClickHouse
`LIMIT N OFFSET M` читает все строки до M в RAM. Симптом: суммаризация останавливается после первой страницы. **Решение: всегда использовать `{last_ts}`** keyset-пагинацию.

### Дубли строк при keyset-пагинации
При одинаковом timestamp у нескольких строк на границе страниц возможны дубли. Используй миллисекундную точность в `timestamp`.

### Параллельный MAP и GIL
asyncio — однопоточный, поэтому `ProgressTracker.tick()` из параллельных чанков безопасен (нет гонок). LLM-вызовы выполняются в `run_in_executor()`, но GIL не мешает — вызовы I/O-bound.

### Программный merge (fallback)
Если после нескольких раундов компрессии merge всё равно не помещается в контекст — включается `_programmatic_merge()` без LLM: простая конкатенация без семантического объединения. Счётчик: `orchestrator._tree_reducer.programmatic_merge_count`. Факт программного merge логируется и отражается в секции 13 отчёта.

### Висячие ссылки в MergedAnalysis
После `_trim_events()` ID в `causal_chains` и `hypotheses` могут ссылаться на удалённые события. `MergedAnalysis._check_referential_integrity()` логирует WARNING, но не бросает исключение — REDUCE-фаза на следующем раунде исправляет.

### Температуры
- MAP и REDUCE: `0.2` — детерминированность важна для точности извлечения событий
- Отчёт: `0.3` — небольшая вариативность для читаемости

---

## 14. Расширение: добавление нового SQL-шаблона или источника

1. В `LOGS_SQL` (или в `incident["context"]`) добавить нужный фильтр
2. Обязательные колонки: `timestamp` и `raw_line` (или `message`/`msg`/`log`)
3. DataLoader автоматически парсит zone и source из стандартных имён колонок
4. Для нескольких источников: создай несколько записей в `INCIDENTS` или запусти несколько раз

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
