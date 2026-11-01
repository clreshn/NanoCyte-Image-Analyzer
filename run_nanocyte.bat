@echo off
cd /d "%~dp0"
python analyzer_gui_v3.py
if errorlevel 1 (
    echo.
    echo The analyzer encountered an error.
    pause
)