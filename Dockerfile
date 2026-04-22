FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    make \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m pip install --upgrade pip && \
    pip install -r requirements.txt

COPY . .

RUN mkdir -p /app/output /app/logs

CMD ["python", "-m", "src.main", "--input", "input/synthetic_data.json", "--outdir", "output", "--diagram-mode", "route"]