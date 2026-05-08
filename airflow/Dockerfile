FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY log_summarizer/ ./log_summarizer/

ENTRYPOINT ["python", "-m", "log_summarizer.main"]
