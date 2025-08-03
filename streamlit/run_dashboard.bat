@echo off
echo Starting PyPSA HRES Model Dashboard...
echo.

REM Try to find and activate virtual environment
if exist "..\venv\Scripts\activate.bat" (
    echo Activating virtual environment from ..\venv\
    call "..\venv\Scripts\activate.bat"
) else if exist "..\..\venv\Scripts\activate.bat" (
    echo Activating virtual environment from ..\..\venv\
    call "..\..\venv\Scripts\activate.bat"
) else if exist "..\..\..\venv\Scripts\activate.bat" (
    echo Activating virtual environment from ..\..\..\venv\
    call "..\..\..\venv\Scripts\activate.bat"
) else if exist "venv\Scripts\activate.bat" (
    echo Activating virtual environment from venv\
    call "venv\Scripts\activate.bat"
) else (
    echo Warning: Virtual environment not found.
    echo Please make sure you have created a virtual environment or modify this script.
    echo Continuing with system Python...
)

echo.
echo Make sure you have installed all requirements:
echo pip install -r ../requirements.txt
echo.
echo The dashboard will open in your default web browser.
echo Press Ctrl+C to stop the server.
echo.
cd /d "%~dp0"
python main.py
pause
