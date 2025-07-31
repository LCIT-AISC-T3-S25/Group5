FROM python:3.11-slim

WORKDIR /app

COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt && \
    apt-get update && apt-get install -y jq && \
    chmod +x /app/start.sh

EXPOSE 9000

CMD ["./start.sh"]
