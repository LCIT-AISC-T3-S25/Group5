#!/bin/bash

HOST=$(jq -r '.host' config.json)
PORT=$(jq -r '.port' config.json)

echo "Launching API on $HOST:$PORT ..."
exec uvicorn main:app --host "$HOST" --port "$PORT"
