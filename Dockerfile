FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY summary_backend ./summary_backend
COPY summary_frontend ./summary_frontend

RUN mkdir -p /app/artifacts/summary_backend/logs \
    /app/artifacts/summary_backend/audit \
    /app/artifacts/summary_backend/uploads

EXPOSE 8088

CMD ["python", "-m", "summary_backend"]
