FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    apt-get update && apt-get install -y jq

COPY app/ ./app/

WORKDIR /app/app

RUN chmod +x start.sh

EXPOSE 8002 

CMD ["./start.sh"]
