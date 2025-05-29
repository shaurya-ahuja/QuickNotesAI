#!/bin/bash

# Get the directory where this script is located
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

# Check for virtual environment in common locations
if [ -d ".venv" ]; then
    VENV_PATH=".venv"
elif [ -d "venv" ]; then
    VENV_PATH="venv"
fi

if [ ! -z "$VENV_PATH" ]; then
    echo "✅ Found virtual environment: $VENV_PATH"
    source "$VENV_PATH/bin/activate"
    
    # Check if streamlit is installed in venv
    if ! python -c "import streamlit" 2>/dev/null; then
        echo "⚠️  Streamlit not found in virtual environment. Installing..."
        pip install streamlit
    fi
    
    echo "🚀 Starting QuickNotes-AI with $(python --version)..."
    python -m streamlit run app.py
else
    echo "⚠️  No virtual environment found (.venv or venv)."
    echo "   Using system Python: $(python3 --version)"
    echo "   If this fails, please create a venv: python3 -m venv .venv"
    
    python3 -m streamlit run app.py
fi
