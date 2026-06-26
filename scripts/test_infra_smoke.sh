#!/usr/bin/env bash
set -euo pipefail

COMPOSE_FILE="${COMPOSE_FILE:-docker-compose.test-infra.yml}"

docker compose -f "$COMPOSE_FILE" ps

docker compose -f "$COMPOSE_FILE" exec -T clickhouse \
  clickhouse-client \
  --user "${SUMMARY_TEST_CLICKHOUSE_USER:-default}" \
  --password "${SUMMARY_TEST_CLICKHOUSE_PASSWORD:-}" \
  --database summary_test \
  --query "
    SELECT
      'clickhouse_ok' AS check_name,
      count() AS smoke_jobs
    FROM summary_job_events
    WHERE job_id = 'smoke-job'
    FORMAT PrettyCompact
  "

docker compose -f "$COMPOSE_FILE" exec -T clickhouse \
  clickhouse-client \
  --user "${SUMMARY_TEST_CLICKHOUSE_USER:-default}" \
  --password "${SUMMARY_TEST_CLICKHOUSE_PASSWORD:-}" \
  --database summary_test \
  --query "
    SELECT
      'input_segments_table_ok' AS check_name,
      count() AS rows_count
    FROM summary_input_segments
    FORMAT PrettyCompact
  "

docker compose -f "$COMPOSE_FILE" exec -T redis redis-cli ping
