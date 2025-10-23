#!/bin/bash
# Entrypoint script for Spotify Analytics Pipeline

set -e

echo "================================================"
echo "🎵 Spotify Analytics Pipeline - Starting..."
echo "================================================"
echo ""

# Print environment info
echo "📦 Environment:"
echo "  Python: $(python --version)"
echo "  PySpark: $(python -c 'import pyspark; print(pyspark.__version__)' 2>/dev/null || echo 'Not available')"
echo "  Delta Lake: $(python -c 'import delta; print(delta.__version__)' 2>/dev/null || echo 'Not available')"
echo ""

# Check for required environment variables
echo "🔍 Checking configuration..."
if [ -z "$CLIENT_ID" ]; then
    echo "❌ ERROR: CLIENT_ID environment variable is required"
    exit 1
fi

if [ -z "$CLIENT_SECRET" ]; then
    echo "❌ ERROR: CLIENT_SECRET environment variable is required"
    exit 1
fi

echo "✅ Configuration looks good"
echo ""

# Create directories if they don't exist
mkdir -p /app/data/bronze
mkdir -p /app/data/kaggle
mkdir -p /app/data/logs
mkdir -p /app/data/.tokens

# Check if Kaggle dataset exists
if [ -f "/app/data/kaggle/dataset.csv" ]; then
    echo "✅ Kaggle dataset found"
    ROWS=$(wc -l < /app/data/kaggle/dataset.csv)
    echo "   Lines in dataset: $ROWS"
else
    echo "⚠️  Kaggle dataset not found at /app/data/kaggle/dataset.csv"
    echo "   The pipeline will still run but won't load Kaggle data"
fi
echo ""

# Check if tokens exist
if [ -f "/app/data/.spotify_tokens.json" ]; then
    echo "✅ Spotify tokens found (will be validated)"
else
    echo "⚠️  No Spotify tokens found"
    echo "   If running interactively, authentication will be required"
fi
echo ""

echo "================================================"
echo "🚀 Starting application..."
echo "================================================"
echo ""

# Execute the command passed to docker run
exec "$@"
