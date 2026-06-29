# PA workspace context

This repository is part of the PA workspace at `/root/MIS_VIEWS/pa`.

For the full workspace map, see the root-local file:

```text
/root/MIS_VIEWS/pa/PA_WORKSPACE_OVERVIEW_2026-06-29.md
```

Repository role:

- FastAPI + Dramatiq backend for Summary jobs.
- Executes MAP/REDUCE/FINAL LLM pipeline over large input, staged uploads, and ClickHouse query ingestion.
- Persists jobs, events, nodes, artifacts, and input segments in ClickHouse.
- Uses Redis broker and a shared LLM concurrency pool.

Important local commands:

```bash
python -m unittest tests.test_summary_backend
python -m compileall -q summary_backend summary_frontend tests scripts
bash scripts/run_summary_stack.sh
bash scripts/stop_summary_stack.sh
```

Important files:

- `summary_backend/api.py`
- `summary_backend/pipeline.py`
- `summary_backend/tasks.py`
- `summary_backend/store.py`
- `summary_backend/llm_client.py`
- `summary_backend/llm_pool.py`
- `summary_backend/config.py`

Operational note:

- After each future change request, update relevant docs, run reasonable checks, commit, and push.
- Subagent delegation is disabled while subagent startup repeatedly fails with authentication or refresh-token errors. Continue locally until subagent authentication is explicitly fixed.
