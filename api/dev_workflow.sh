#!/bin/bash
echo "🚀 Deploying Gender Prediction API..."

# Verifica directory
if [ ! -f "modal_deployment.py" ]; then
    echo "❌ Error: Run this script from the api/ directory"
    exit 1
fi

# Verifica Modal
if ! command -v modal &> /dev/null; then
    echo "❌ Modal CLI not found. Install with: pip install modal"
    exit 1
fi

# Verifica config
if [ ! -f "config.py" ]; then
    echo "❌ config.py not found. Copy from config.py.template and update paths"
    exit 1
fi

echo "📦 Deploying to Modal..."
modal deploy modal_deployment.py

echo ""
echo "✅ Deployment complete!"
echo "📊 Monitor: https://modal.com/apps"
echo "📝 Logs: modal logs gender-prediction-v3"
