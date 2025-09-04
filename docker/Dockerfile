FROM ollama/ollama:latest

RUN apt-get update && apt-get install -y bash

ENV OLLAMA_KEEP_ALIVE=24h
ENV OLLAMA_HOST=0.0.0.0

EXPOSE 11434
