#!/bin/bash

source .env

echo "Building Docker image..."
docker build -t langgraph-exp .

echo "Starting container in background..."
docker run -d \
  --name langgraph-container \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  -v $(pwd):/app \
  --entrypoint tail \
  langgraph-exp -f /dev/null

echo "Container is ready. Use ./start.sh <path/to/file.py> to run scripts."
