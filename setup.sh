#!/bin/bash

# DGS-GPT Setup Script
# This script sets up the development environment for DGS-GPT

set -e  # Exit on any error

echo "🚀 Setting up DGS-GPT development environment..."

# Check if Python is installed
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    echo "❌ Error: Python is not installed or not in PATH"
    echo "Please install Python 3.8+ from https://python.org"
    exit 1
fi

# Use python3 if available, otherwise python
PYTHON_CMD="python3"
if ! command -v python3 &> /dev/null; then
    PYTHON_CMD="python"
fi

echo "🐍 Using Python: $($PYTHON_CMD --version)"

# Check Python version (minimum 3.8)
PYTHON_VERSION=$($PYTHON_CMD -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
REQUIRED_VERSION="3.8"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo "❌ Error: Python $PYTHON_VERSION detected, but Python $REQUIRED_VERSION+ is required"
    exit 1
fi

echo "✅ Python version check passed"

# Create virtual environment
echo "📦 Creating virtual environment..."
if [ -d ".venv" ]; then
    echo "⚠️  Virtual environment already exists. Removing old environment..."
    rm -rf .venv
fi

$PYTHON_CMD -m venv .venv

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source .venv/bin/activate

# Verify activation
if [ "$VIRTUAL_ENV" != "" ]; then
    echo "✅ Virtual environment activated: $VIRTUAL_ENV"
else
    echo "❌ Failed to activate virtual environment"
    exit 1
fi

# Upgrade pip to latest version
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install PyTorch first (it's the largest dependency)
echo "🔥 Installing PyTorch..."
pip install torch>=2.0.0

# Install remaining dependencies
echo "📚 Installing dependencies from requirements.txt..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "❌ Error: requirements.txt not found"
    exit 1
fi

# Check if vocab.txt exists, if not, suggest running dataset preparation
echo "📄 Checking for dataset files..."
if [ ! -f "vocab.txt" ]; then
    echo "⚠️  vocab.txt not found. You may need to prepare your dataset."
    echo "   Run: python dataset.py (to download and prepare a sample dataset)"
    echo "   Or place your own text data in vocab.txt"
fi

# Run a quick import test
echo "🧪 Testing imports..."
if python -c "
import torch
import optuna
import matplotlib
from gui import GPT_GUI
print('✅ All core modules import successfully')
" 2>/dev/null; then
    echo "✅ Import test passed"
else
    echo "❌ Import test failed. There may be missing dependencies."
    exit 1
fi

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "📋 Next steps:"
echo "   1. Activate the environment: source .venv/bin/activate"
echo "   2. Prepare dataset (if needed): python dataset.py"
echo "   3. Run the GUI: python gui.py"
echo "   4. Or run command line: python ShitGPT.py"
echo ""
echo "💡 Tips:"
echo "   - The GUI provides the easiest way to train and generate text"
echo "   - Use hyperparameter optimization to find the best settings"
echo "   - Check README.md for detailed usage instructions"
echo ""


