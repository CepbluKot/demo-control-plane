CREATE DATABASE IF NOT EXISTS summary_test;

CREATE TABLE IF NOT EXISTS summary_test.summary_job_events
(
    event_id UUID DEFAULT generateUUIDv4(),
    job_id String,
    event_time DateTime64(3, 'UTC') DEFAULT now64(3),
    event_type LowCardinality(String),
    job_status LowCardinality(String),
    actor LowCardinality(String) DEFAULT '',
    message String DEFAULT '',
    payload String DEFAULT '{}'
)
ENGINE = MergeTree
ORDER BY (job_id, event_time, event_id);

CREATE TABLE IF NOT EXISTS summary_test.summary_node_events
(
    event_id UUID DEFAULT generateUUIDv4(),
    job_id String,
    node_id String,
    event_time DateTime64(3, 'UTC') DEFAULT now64(3),
    event_type LowCardinality(String),
    node_status LowCardinality(String),
    node_type LowCardinality(String),
    level UInt32 DEFAULT 0,
    node_index UInt32 DEFAULT 0,
    attempt UInt32 DEFAULT 0,
    actor LowCardinality(String) DEFAULT '',
    message String DEFAULT '',
    payload String DEFAULT '{}'
)
ENGINE = MergeTree
ORDER BY (job_id, node_type, level, node_index, node_id, event_time, event_id);

CREATE TABLE IF NOT EXISTS summary_test.summary_artifacts
(
    artifact_id UUID DEFAULT generateUUIDv4(),
    job_id String,
    node_id String DEFAULT '',
    artifact_type LowCardinality(String),
    stage LowCardinality(String),
    level UInt32 DEFAULT 0,
    content_hash String,
    content String,
    metadata String DEFAULT '{}',
    created_at DateTime64(3, 'UTC') DEFAULT now64(3)
)
ENGINE = MergeTree
ORDER BY (job_id, stage, level, node_id, artifact_type, created_at, artifact_id);

CREATE TABLE IF NOT EXISTS summary_test.summary_input_segments
(
    job_id String,
    segment_index UInt64,
    source_type LowCardinality(String),
    source_format LowCardinality(String),
    content_hash String,
    content String,
    rows_count UInt64,
    chars UInt64,
    metadata String DEFAULT '{}',
    created_at DateTime64(3, 'UTC') DEFAULT now64(3)
)
ENGINE = MergeTree
ORDER BY (job_id, segment_index);

CREATE TABLE IF NOT EXISTS summary_test.summary_llm_calls
(
    call_id UUID DEFAULT generateUUIDv4(),
    job_id String,
    node_id String DEFAULT '',
    created_at DateTime64(3, 'UTC') DEFAULT now64(3),
    provider LowCardinality(String) DEFAULT '',
    model String DEFAULT '',
    status LowCardinality(String),
    error_class LowCardinality(String) DEFAULT '',
    http_status UInt16 DEFAULT 0,
    latency_ms UInt32 DEFAULT 0,
    pool_wait_ms UInt32 DEFAULT 0,
    provider_latency_ms UInt32 DEFAULT 0,
    prompt_tokens UInt32 DEFAULT 0,
    completion_tokens UInt32 DEFAULT 0,
    total_tokens UInt32 DEFAULT 0,
    request_hash String DEFAULT '',
    response_hash String DEFAULT '',
    request_json String DEFAULT '{}',
    response_json String DEFAULT '{}',
    error_message String DEFAULT ''
)
ENGINE = MergeTree
ORDER BY (job_id, node_id, created_at, call_id);

CREATE VIEW IF NOT EXISTS summary_test.summary_job_current_v AS
SELECT
    job_id,
    argMax(job_status, (event_time, event_id)) AS job_status,
    argMax(event_type, (event_time, event_id)) AS last_event_type,
    max(event_time) AS updated_at,
    count() AS events_count
FROM summary_test.summary_job_events
GROUP BY job_id;

CREATE VIEW IF NOT EXISTS summary_test.summary_node_current_v AS
SELECT
    job_id,
    node_id,
    argMax(node_type, (event_time, event_id)) AS node_type,
    argMax(level, (event_time, event_id)) AS level,
    argMax(node_index, (event_time, event_id)) AS node_index,
    argMax(node_status, (event_time, event_id)) AS node_status,
    argMax(event_type, (event_time, event_id)) AS last_event_type,
    max(event_time) AS updated_at,
    count() AS events_count
FROM summary_test.summary_node_events
GROUP BY job_id, node_id;

CREATE TABLE IF NOT EXISTS summary_test.monitoring_profiles
(
    snapshot_id UUID DEFAULT generateUUIDv4(),
    profile_id String,
    name String,
    service String,
    description String DEFAULT '',
    workflow_id String,
    workflow_inputs String DEFAULT '{}',
    metadata String DEFAULT '{}',
    is_archived UInt8 DEFAULT 0,
    created_at DateTime64(3, 'UTC') DEFAULT now64(3),
    updated_at DateTime64(3, 'UTC') DEFAULT now64(3)
)
ENGINE = MergeTree
ORDER BY (profile_id, updated_at, snapshot_id);

CREATE TABLE IF NOT EXISTS summary_test.monitoring_schedules
(
    snapshot_id UUID DEFAULT generateUUIDv4(),
    profile_id String,
    cron String DEFAULT '',
    timezone String DEFAULT 'UTC',
    is_enabled UInt8 DEFAULT 0,
    max_active_runs UInt32 DEFAULT 1,
    next_run_at Nullable(DateTime64(3, 'UTC')),
    last_run_at Nullable(DateTime64(3, 'UTC')),
    created_at DateTime64(3, 'UTC') DEFAULT now64(3),
    updated_at DateTime64(3, 'UTC') DEFAULT now64(3)
)
ENGINE = MergeTree
ORDER BY (profile_id, updated_at, snapshot_id);

CREATE TABLE IF NOT EXISTS summary_test.monitoring_runs
(
    snapshot_id UUID DEFAULT generateUUIDv4(),
    run_id String,
    profile_id String,
    status LowCardinality(String),
    trigger_type LowCardinality(String),
    workflow_id String,
    workflow_run_id String DEFAULT '',
    task_id String DEFAULT '',
    requested_at DateTime64(3, 'UTC') DEFAULT now64(3),
    started_at Nullable(DateTime64(3, 'UTC')),
    finished_at Nullable(DateTime64(3, 'UTC')),
    scheduled_for Nullable(DateTime64(3, 'UTC')),
    inputs_json String DEFAULT '{}',
    output_json String DEFAULT '{}',
    error_message String DEFAULT '',
    metadata String DEFAULT '{}',
    updated_at DateTime64(3, 'UTC') DEFAULT now64(3)
)
ENGINE = MergeTree
ORDER BY (profile_id, run_id, updated_at, snapshot_id);

INSERT INTO summary_test.summary_job_events
    (job_id, event_type, job_status, actor, message, payload)
VALUES
    (
        'smoke-job',
        'JOB_CREATED',
        'CREATED',
        'init',
        'Seed row for test-infra smoke checks',
        '{"source":"001_summary_pipeline.sql"}'
    );
