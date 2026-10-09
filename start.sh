#!/bin/bash

if [ -z "$1" ]; then
  echo "Usage: ./start.sh <path/to/file.py>"
  echo "Example: ./start.sh stage1_fundamentals/01_why_langgraph.py"
  exit 1
fi

source .env

echo "Running $1..."
docker exec langgraph-container python $1
