#!/bin/bash
python3 ./inference/main.py
python3 -m http.server 3000 &