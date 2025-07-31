#!/bin/bash

# RAG API Deployment Script

echo "🚀 Building and deploying RAG Streaming API..."

# Build the Docker image
echo "📦 Building Docker image..."
docker build -t rag-streaming-api .

# Stop and remove existing container if running
echo "🛑 Stopping existing container..."
docker stop rag-api-container 2>/dev/null || true
docker rm rag-api-container 2>/dev/null || true

# Run the new container
echo "🔄 Starting new container..."
docker run -d \
  --name rag-api-container \
  -p 8000:8000 \
  -e GROQ_API_KEY=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 \
  rag-streaming-api

echo "✅ Deployment complete!"
echo "🌐 API is available at: http://localhost:8000"
echo "📚 API documentation: http://localhost:8000/docs"
echo "🔍 Check container logs: docker logs rag-api-container"
