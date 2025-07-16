#!/bin/bash

# Create virtual environment
python -m venv .venv

# Activate it
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies from requirements.txt (if it exists)
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
fi

echo "✅ Virtual environment created and activated."


pip install -r requirements.txt

echo "✅ Dependencies installed."


