FROM python:3.10-slim

WORKDIR /app

# Copy requirements if you have one, else create it
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

EXPOSE 5002

CMD ["python", "app/index.py"]