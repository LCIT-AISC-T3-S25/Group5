# Use Python 3.10 slim image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY main.py .
COPY index.faiss .
COPY id_to_passage.pkl .
COPY embedder_model/ ./embedder_model/

# Expose port
EXPOSE 8000

# Set environment variable for GROQ API KEY
ENV GROQ_API_KEY=gsk_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
