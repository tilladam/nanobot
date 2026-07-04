#!/bin/bash

docker rm -f nanobot
docker build -t nanobot .
docker run -d \
  --name nanobot \
  --restart always \
  -v ~/.nanobot:/home/nanobot/.nanobot \
  nanobot gateway
