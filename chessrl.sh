#!/bin/bash
# Chess RL UCI Engine launcher for Linux/Mac
# Use this in chess GUIs: Add the full path to this script as the engine

cd "$(dirname "$0")"
python3 uci.py --model checkpoints/model_final --simulations 400 "$@"
