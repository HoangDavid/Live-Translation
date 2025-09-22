
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /Live-Translation

COPY requirements.bot.txt .

RUN pip install --no-cache-dir -r requirements.bot_orchestrator.txt

COPY . .

ENTRYPOINT ["python" ,"transport.py"]

