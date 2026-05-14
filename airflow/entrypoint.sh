#!/bin/sh
set -e

python -m log_summarizer.main "$@"

# Пишем отчёт в XCom чтобы Airflow мог вернуть его через API
OUTPUT_ARG=""
for arg in "$@"; do
    if [ -n "$NEXT_IS_OUTPUT" ]; then
        OUTPUT_ARG="$arg"
        break
    fi
    [ "$arg" = "--output" ] && NEXT_IS_OUTPUT=1
done

if [ -n "$OUTPUT_ARG" ] && [ -f "$OUTPUT_ARG" ]; then
    python3 -c "
import json, sys
text = open('$OUTPUT_ARG').read()
json.dump({'report': text}, open('/airflow/xcom/return.json', 'w'))
"
fi
