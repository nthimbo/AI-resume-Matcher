# Streamlit + SentenceTransformers app
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \    PYTHONUNBUFFERED=1 \    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y \    build-essential poppler-utils libglib2.0-0 libsm6 libxext6 libxrender-dev \    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . /app

EXPOSE 8501
CMD ["streamlit", "run", "app_mvp.py", "--server.port=8501", "--server.address=0.0.0.0"]
