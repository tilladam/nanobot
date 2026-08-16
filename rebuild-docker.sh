#!/bin/bash

docker rm -f nanobot
docker build -t nanobot .
docker run -d \
  --name nanobot \
  --restart always \
  -v ~/.nanobot:/home/nanobot/.nanobot \
  -v /mnt/HC_Volume_104301936/obsidian:/mnt/obsidian \
  nanobot gateway
