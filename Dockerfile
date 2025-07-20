
FROM python:3.11-slim as builder

WORKDIR /app


RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*


COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


RUN python -m spacy download en_core_web_sm

FROM python:3.11-slim

WORKDIR /app


RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*


COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages


COPY . .


RUN mkdir -p uploads


ENV PYTHONPATH=/usr/local/lib/python3.11/site-packages


EXPOSE 8000


CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
