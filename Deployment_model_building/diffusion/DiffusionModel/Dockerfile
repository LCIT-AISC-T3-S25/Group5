# Use official slim Python base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    jq \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


# Copy dependency file and install Python packages
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose the port specified in config.json
EXPOSE 8003

# Run app using dynamic host and port from config.json
CMD ["sh", "-c", "HOST=$(jq -r .host config.json) && PORT=$(jq -r .port config.json) && echo Running on $HOST:$PORT && python app.py"]
