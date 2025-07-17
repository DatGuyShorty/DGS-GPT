@echo off
setlocal enabledelayedexpansion

REM DGS-GPT Setup Script for Windows
REM This script sets up the development environment for DGS-GPT

echo 🚀 Setting up DGS-GPT development environment...

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Error: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

echo 🐍 Using Python:
python --version

REM Check Python version (minimum 3.8)
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
for /f "tokens=1,2 delims=." %%a in ("%PYTHON_VERSION%") do (
    set MAJOR=%%a
    set MINOR=%%b
)

if %MAJOR% lss 3 (
    echo ❌ Error: Python %PYTHON_VERSION% detected, but Python 3.8+ is required
    pause
    exit /b 1
)
if %MAJOR% equ 3 if %MINOR% lss 8 (
    echo ❌ Error: Python %PYTHON_VERSION% detected, but Python 3.8+ is required
    pause
    exit /b 1
)

echo ✅ Python version check passed

REM Create virtual environment
echo 📦 Creating virtual environment...
if exist ".venv" (
    echo ⚠️  Virtual environment already exists. Removing old environment...
    rmdir /s /q .venv
)

python -m venv .venv

REM Activate virtual environment
echo 🔌 Activating virtual environment...
call .venv\Scripts\activate.bat

REM Verify activation
if "%VIRTUAL_ENV%"=="" (
    echo ❌ Failed to activate virtual environment
    pause
    exit /b 1
) else (
    echo ✅ Virtual environment activated: %VIRTUAL_ENV%
)

REM Upgrade pip to latest version
echo ⬆️  Upgrading pip...
python -m pip install --upgrade pip

REM Install PyTorch first (it's the largest dependency)
echo 🔥 Installing PyTorch...
pip install "torch>=2.0.0"

REM Install remaining dependencies
echo 📚 Installing dependencies from requirements.txt...
if not exist "requirements.txt" (
    echo ❌ Error: requirements.txt not found
    pause
    exit /b 1
)

pip install -r requirements.txt

REM Check if vocab.txt exists
echo 📄 Checking for dataset files...
if not exist "vocab.txt" (
    echo ⚠️  vocab.txt not found. You may need to prepare your dataset.
    echo    Run: python dataset.py ^(to download and prepare a sample dataset^)
    echo    Or place your own text data in vocab.txt
)

REM Run a quick import test
echo 🧪 Testing imports...
python -c "import torch; import optuna; import matplotlib; from gui import GPT_GUI; print('✅ All core modules import successfully')" 2>nul
if %errorlevel% neq 0 (
    echo ❌ Import test failed. There may be missing dependencies.
    pause
    exit /b 1
) else (
    echo ✅ Import test passed
)

echo.
echo 🎉 Setup completed successfully!
echo.
echo 📋 Next steps:
echo    1. Activate the environment: .venv\Scripts\activate.bat
echo    2. Prepare dataset ^(if needed^): python dataset.py
echo    3. Run the GUI: python gui.py
echo    4. Or run command line: python ShitGPT.py
echo.
echo 💡 Tips:
echo    - The GUI provides the easiest way to train and generate text
echo    - Use hyperparameter optimization to find the best settings
echo    - Check README.md for detailed usage instructions
echo.

pause
