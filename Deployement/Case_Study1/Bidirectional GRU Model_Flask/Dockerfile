FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN apt-get update && apt-get install -y gcc libglib2.0-0 libsm6 libxrender1 libxext6 \
 && pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy entire project including SavedModel folder
COPY . .

EXPOSE 5000

CMD ["python", "app.py"]
