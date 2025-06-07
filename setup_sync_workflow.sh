#!/bin/bash
# setup_sync_workflow.sh - Setup automatico per sincronizzazione

echo "🔧 Setting up Gender Prediction Sync Workflow"
echo "=============================================="

# Verifica di essere nella directory corretta
if [ ! -d "src/gender_predict" ] || [ ! -f "scripts/final_predictor.py" ]; then
    echo "❌ Error: Run this script from the gender-predict/ root directory"
    echo "   Required: src/gender_predict/ and scripts/final_predictor.py"
    exit 1
fi

echo "📁 Creating API directory structure..."

# Crea directory api se non esiste
mkdir -p api

# Sposta file esistenti se sono nella root
if [ -f "modal_deployment.py" ]; then
    echo "📦 Moving modal_deployment.py to api/"
    mv modal_deployment.py api/
fi

if [ -f "index.html" ]; then
    echo "🌐 Moving index.html to api/web_interface.html"
    mv index.html api/web_interface.html
fi

echo "📝 Creating configuration files..."

# Crea config.py.template (per GitHub)
cat > api/config.py.template << 'EOF'
"""
Configuration template for API deployment.
Copy this to config.py and update with your actual paths.
"""

import os
from pathlib import Path

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent
SRC_PATH = PROJECT_ROOT / "src"
MODELS_PATH = PROJECT_ROOT / "models"

# Model configuration - UPDATE THESE PATHS
MODEL_CONFIG = {
    'model_path': '/app/models/best_v3_model/YOUR_EXPERIMENT_ID/models/model.pth',
    'preprocessor_path': '/app/models/best_v3_model/YOUR_EXPERIMENT_ID/preprocessor.pkl',
    'optimal_threshold': 0.480,
    'unicode_preprocessing': True,
    'expected_performance': {
        'f1_score': 0.8976,
        'accuracy': 0.9207,
        'bias_ratio': 0.9999,
        'bias_deviation': 0.01
    }
}

# Modal configuration
MODAL_CONFIG = {
    'app_name': 'gender-prediction-v3',
    'gpu_type': 'T4',
    'container_idle_timeout': 300,
    'concurrency_limit': 10
}

# Environment variables (optional)
API_KEY = os.getenv('GENDER_API_KEY')
ALLOWED_ORIGINS = os.getenv('ALLOWED_ORIGINS', '*').split(',')
EOF

# Cerca il path del modello automaticamente
echo "🔍 Detecting model paths..."
MODEL_DIR=""
if [ -d "models/best_v3_model" ]; then
    MODEL_DIR=$(find models/best_v3_model -name "*.pth" -type f | head -1)
    if [ -n "$MODEL_DIR" ]; then
        MODEL_BASE=$(dirname "$MODEL_DIR")
        PREPROCESSOR_PATH="${MODEL_BASE%/models}/preprocessor.pkl"
        echo "✅ Found model: $MODEL_DIR"
        echo "✅ Expected preprocessor: $PREPROCESSOR_PATH"
        
        # Crea config.py con path reali
        cat > api/config.py << EOF
"""
Configuration for API deployment.
KEEP THIS FILE PRIVATE - contains actual paths and settings.
"""

import os
from pathlib import Path

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent
SRC_PATH = PROJECT_ROOT / "src"
MODELS_PATH = PROJECT_ROOT / "models"

# Model configuration - AUTO-DETECTED PATHS
MODEL_CONFIG = {
    'model_path': '/app/${MODEL_DIR}',
    'preprocessor_path': '/app/${PREPROCESSOR_PATH}',
    'optimal_threshold': 0.480,
    'unicode_preprocessing': True,
    'expected_performance': {
        'f1_score': 0.8976,
        'accuracy': 0.9207,
        'bias_ratio': 0.9999,
        'bias_deviation': 0.01
    }
}

# Modal configuration
MODAL_CONFIG = {
    'app_name': 'gender-prediction-v3',
    'gpu_type': 'T4',
    'container_idle_timeout': 300,
    'concurrency_limit': 10
}

# Environment variables (optional)
API_KEY = os.getenv('GENDER_API_KEY')
ALLOWED_ORIGINS = os.getenv('ALLOWED_ORIGINS', '*').split(',')
EOF
        echo "✅ Created api/config.py with auto-detected paths"
    else
        echo "⚠️  Model files not found in models/best_v3_model/"
        echo "   You'll need to update paths manually in api/config.py"
        cp api/config.py.template api/config.py
    fi
else
    echo "⚠️  models/best_v3_model directory not found"
    echo "   You'll need to update paths manually in api/config.py"
    cp api/config.py.template api/config.py
fi

# Crea deploy script
cat > api/deploy.sh << 'EOF'
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
EOF

chmod +x api/deploy.sh

# Aggiorna/crea .gitignore
echo "📝 Updating .gitignore..."

if [ -f ".gitignore" ]; then
    # Aggiungi config.py al gitignore esistente se non c'è già
    if ! grep -q "api/config.py" .gitignore; then
        echo "" >> .gitignore
        echo "# API Configuration (keep private)" >> .gitignore
        echo "api/config.py" >> .gitignore
    fi
else
    # Crea .gitignore nuovo
    cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
.env
.venv
env/
venv/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo

# Jupyter
.ipynb_checkpoints

# Experiments
experiments/*/logs/
experiments/*/checkpoints/
*.log

# OS
.DS_Store
Thumbs.db

# API Configuration (keep private)
api/config.py

# Secrets
*.key
*.pem
*.env.local
*.env.production
EOF
fi

# Crea README per API
cat > api/README.md << 'EOF'
# Gender Prediction API Deployment

This directory contains the deployment configuration for the Gender Prediction V3 model with **automatic synchronization**.

## 🔄 Auto-Sync Architecture

The deployment automatically imports your latest code:
- `modal_deployment.py` imports directly from `../scripts/final_predictor.py`
- Any change to your local code is reflected in deployment
- Zero manual synchronization required

## ⚙️ Configuration

1. **Copy template**: `cp config.py.template config.py`
2. **Update paths**: Edit `config.py` with your model paths
3. **Keep private**: `config.py` is gitignored for privacy

## 🚀 Quick Deploy

```bash
cd api/
./deploy.sh
```

## 🧪 Development Workflow

```bash
# 1. Make changes to ../scripts/final_predictor.py
# 2. Test locally
python ../scripts/final_predictor.py --single_name "Mario Rossi"

# 3. Deploy with changes automatically synced
./deploy.sh

# 4. Test deployed API
modal run modal_deployment.py::test_prediction
```

## 📁 Files

- `modal_deployment.py`: Main deployment (auto-syncs from ../scripts/)
- `config.py`: Private configuration (gitignored)
- `config.py.template`: Template for new setups
- `deploy.sh`: Deployment script
- `web_interface.html`: Research interface

## 🔍 Monitoring

- Dashboard: https://modal.com/apps
- Logs: `modal logs gender-prediction-v3`
- Stats: `modal stats gender-prediction-v3`
EOF

echo ""
echo "✅ Setup complete!"
echo ""
echo "📁 New structure:"
echo "   gender-predict/"
echo "   ├── api/                    # 🚀 Deployment (config.py private)"
echo "   │   ├── modal_deployment.py # ← Auto-syncs from ../scripts/"
echo "   │   ├── config.py          # ← Private (gitignored)"
echo "   │   ├── config.py.template # ← Public template"
echo "   │   ├── deploy.sh          # ← Deployment script"
echo "   │   └── README.md          # ← API documentation"
echo "   ├── scripts/               # 📦 Your code (public)"
echo "   │   └── final_predictor.py # ← Source of truth"
echo "   └── ..."
echo ""
echo "🔧 Next steps:"
echo "   1. Verify paths in api/config.py"
echo "   2. Install Modal: pip install modal"
echo "   3. Setup Modal: modal token new"
echo "   4. Deploy: cd api && ./deploy.sh"
echo ""

# Verifica finale
echo "🔍 Verification:"
if [ -f "api/config.py" ]; then
    echo "   ✅ api/config.py created"
else
    echo "   ❌ api/config.py missing"
fi

if [ -f "scripts/final_predictor.py" ]; then
    echo "   ✅ scripts/final_predictor.py found"
else
    echo "   ❌ scripts/final_predictor.py missing"
fi

if grep -q "api/config.py" .gitignore 2>/dev/null; then
    echo "   ✅ api/config.py added to .gitignore"
else
    echo "   ❌ .gitignore not configured"
fi

echo ""
echo "🎯 Ready for auto-sync deployment!"
