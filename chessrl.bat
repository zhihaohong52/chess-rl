@echo off
REM Chess RL UCI Engine launcher for Windows
REM Use this in Arena: Add the full path to this .bat file as the engine

cd /d "%~dp0"
python uci.py --model checkpoints/model_final --simulations 400 %*
