"""Локальный тест map_split.main без Dify."""

import sys
import json
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parents[1]))

from nodes.map_split import main

ROWS = [
    {"timestamp": "2025-04-01 09:00:01.000", "end_time": "2025-04-01 09:00:01.000", "cnt": "1",
     "namespace": "kubesphere-logging-system", "pod_name": "fluent-bit-rql5p",
     "container_name": "fluent-bit",
     "log_text": "[error] [output:http:http.3] host:7196, HTTP status=429"},
    {"timestamp": "2025-04-01 09:00:02.000", "end_time": "2025-04-01 09:00:03.000", "cnt": "3",
     "namespace": "airflow", "pod_name": "airflow-worker-7f9d4",
     "container_name": "worker",
     "log_text": "Error: ImagePullBackOff for image registry.internal/airflow:2.8.1"},
    {"timestamp": "2025-04-01 09:00:05.000", "end_time": "2025-04-01 09:00:05.000", "cnt": "1",
     "namespace": "airflow", "pod_name": "airflow-scheduler-6b8c2",
     "container_name": "scheduler",
     "log_text": "Task instance PID 4821 heartbeat timed out"},
    {"timestamp": "2025-04-01 09:00:07.000", "end_time": "2025-04-01 09:00:07.000", "cnt": "1",
     "namespace": "airflow", "pod_name": "airflow-worker-3a1f7",
     "container_name": "worker",
     "log_text": "OOMKilled: container exceeded memory limit 2Gi"},
    {"timestamp": "2025-04-01 09:00:10.000", "end_time": "2025-04-01 09:00:10.000", "cnt": "1",
     "namespace": "kube-system", "pod_name": "coredns-5d78c9869d-xk2pq",
     "container_name": "coredns",
     "log_text": "SERVFAIL reply for registry.internal.: read udp timeout"},
]


def run(label, **kwargs):
    result = main(**kwargs)
    print(f"\n=== {label} ===")
    for key, val in result.items():
        if key.startswith("batch_") and isinstance(val, list):
            print(f"  {key}: {len(val)} rows")
            for line in val:
                obj = json.loads(line)
                print(f"    {obj.get('timestamp','')}  {obj.get('pod_name','')}  {obj.get('log_text','')[:60]}")
        else:
            print(f"  {key}: {val!r}")


# ── тест 1: первая итерация, 2 батча по 2 строки ─────────────────────────────
run("iter 1 (offset=0, n=2, budget=200)",
    rows=ROWS, offset=0, token_budget="200", max_batch="2", n_parallel="2")

# ── тест 2: продолжение (offset=4, последняя строка) ─────────────────────────
run("iter 2 (offset=4, n=2)",
    rows=ROWS, offset=4, token_budget="200", max_batch="2", n_parallel="2")

# ── тест 3: rows как JSON-строка (как Dify иногда передаёт) ──────────────────
run("rows as JSON string",
    rows=json.dumps(ROWS), offset=0, token_budget="9999", max_batch="29", n_parallel="3")

# ── тест 4: offset за пределами — has_more должен быть 0 ─────────────────────
run("offset past end",
    rows=ROWS, offset=99, token_budget="6000", max_batch="29", n_parallel="3")
