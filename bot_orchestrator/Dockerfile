
# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /bot_orchestrator

COPY requirements.bot_orchestrator.txt .

# install the required dependencies
RUN pip install --no-cache-dir -r requirements.bot_orchestrator.txt

COPY . .

EXPOSE 7001

CMD ["python", "-m", "bot_orchestrator.main"]



