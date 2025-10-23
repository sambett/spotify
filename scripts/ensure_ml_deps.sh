#!/bin/bash
# Ensure ML dependencies are installed before running Python scripts

if ! python3 -c "import numpy" 2>/dev/null; then
    echo "📦 Installing ML dependencies (first-time setup)..."
    pip3 install --no-cache-dir -q numpy==1.24.4 scikit-learn==1.3.2 pandas==2.0.3 matplotlib==3.7.5 seaborn==0.13.0
    echo "✅ ML dependencies installed"
fi

# Run the python script passed as argument
exec python3 "$@"
