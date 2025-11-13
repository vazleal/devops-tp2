FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Instala dependências
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/train.py ./train.py

ENV DATASET_DIR=/datasets \
    OUTPUT_PATH=/mnt/model/model.pkl

CMD ["python", "train.py", "--output", "/mnt/model/model.pkl"]