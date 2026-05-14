#!/bin/sh
set -e
exec python -m log_summarizer.main "$@"
