#!/usr/bin/env python3
"""Кодирует .sql файл в base64 для вставки в Airflow DAG param."""
import base64
import sys

path = sys.argv[1] if len(sys.argv) > 1 else input("Путь к .sql файлу: ")
print(base64.b64encode(open(path, "rb").read()).decode())
