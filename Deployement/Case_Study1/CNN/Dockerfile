# Use official Python image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy all project files into container
COPY . .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port the app runs on
EXPOSE 8000

# Command to run the FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
