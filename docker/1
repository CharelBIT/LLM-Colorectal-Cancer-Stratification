#!/bin/bash

/bin/ollama serve &
pid=$!

sleep 5

echo "> Download deepseek model..."
# ollama pull deepseek-r1:7b
ollama pull deepseek-r1:32b

wait $pid
