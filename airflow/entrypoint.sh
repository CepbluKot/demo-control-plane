#!/bin/sh
set -e

if [ -n "$LOGS_SQL_B64" ]; then
    export LOGS_SQL=$(echo "$LOGS_SQL_B64" | base64 -d)
fi

if [ -n "$METRICS_SQL_B64" ]; then
    export METRICS_SQL=$(echo "$METRICS_SQL_B64" | base64 -d)
fi

exec python -m log_summarizer.main "$@"
