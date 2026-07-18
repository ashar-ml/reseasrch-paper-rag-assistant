@echo off
title Research Paper RAG Assistant Launcher
echo ===================================================
echo   Starting Research Paper RAG Assistant...
echo ===================================================
echo.

:: Start FastAPI Backend
echo [1/2] Starting FastAPI Backend on Port 8005...
start "RAG Backend Server" cmd /k "cd /d "%~dp0backend" && venv\Scripts\python.exe -m app.main"

:: Small delay to let the backend start up
timeout /t 3 /nobreak >nul

:: Start Streamlit Frontend
echo [2/2] Starting Streamlit Frontend...
start "RAG Streamlit Frontend" cmd /k "cd /d "%~dp0frontend" && venv\Scripts\streamlit.exe run app.py"

echo.
echo ===================================================
echo   Project successfully launched in separate windows!
echo   You can close this launcher window now.
echo ===================================================
timeout /t 5
