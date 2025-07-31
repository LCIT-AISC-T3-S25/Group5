# Base Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy config.json explicitly into /app/app/
COPY app/config.json ./app/

# Copy entire app folder
COPY app/ ./app/

# Set working directory to app/
WORKDIR /app/app

# Expose Flask port (read from config.json — 8002 in your case)
EXPOSE 8002

# Run Flask app
CMD ["python", "main.py"]
