
FROM python:3.12-slim

# Install ffmpeg (required for pydub to convert Telegram voice messages)
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the correct python file
COPY TelegramAlyaBotOpenAI.py .

CMD ["sh", "-c", "uvicorn TelegramAlyaBotOpenAI:app --host 0.0.0.0 --port ${PORT}"]
