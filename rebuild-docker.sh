#!/bin/bash

# Pre-create the directory to avoid docker daemon creating it as root
mkdir -p ./instances/briefings/.nanobot

# Clean up legacy raw docker container if it exists
docker rm -f nanobot 2>/dev/null || true

docker compose down
docker compose build
docker compose up -d
