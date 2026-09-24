#!/bin/bash

# Script di deployment per Gender Prediction API
# Esegui dalla directory api/: ./deploy.sh

echo "🚀 Gender Prediction API - Deployment"
echo "===================================="
echo ""

# Colori
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Verifica di essere nella directory api/
if [ ! -f "modal_deployment.py" ] || [ ! -f "config.py" ]; then
    echo -e "${RED}❌ Errore: esegui questo script dalla directory api/${NC}"
    echo "   Directory corrente: $(pwd)"
    exit 1
fi

echo "📍 Esecuzione dalla directory: $(pwd)"
echo ""

# 1. Verifica file necessari
echo "1️⃣ Verifica file necessari..."
echo ""

FILES=(
    "../models/best_v3_model/preprocessor.pkl"
    "../models/best_v3_model/feature_extractor.pkl"
    "../models/best_v3_model/models/model.pth"
    "../scripts/final_predictor.py"
    "./config.py"
    "./modal_deployment.py"
)

all_ok=true
for file in "${FILES[@]}"; do
    if [ -f "$file" ]; then
        size=$(du -h "$file" 2>/dev/null | cut -f1)
        echo -e "${GREEN}✅${NC} $file ($size)"
    else
        echo -e "${RED}❌${NC} $file MANCANTE!"
        all_ok=false
    fi
done

if [ "$all_ok" = false ]; then
    echo -e "\n${RED}Errore: alcuni file necessari mancano!${NC}"
    exit 1
fi

# 2. Test file montati
echo ""
echo "2️⃣ Test dei file montati su Modal..."
echo ""
modal run modal_deployment.py::test_model_files

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Test dei file fallito!${NC}"
    exit 1
fi

# 3. Deploy
echo ""
echo "3️⃣ Deploy dell'applicazione..."
echo ""
modal deploy modal_deployment.py

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Deploy fallito!${NC}"
    exit 1
fi

# 4. Test funzionalità
echo ""
echo "4️⃣ Test delle funzionalità API..."
echo ""
modal run modal_deployment.py::test_academic_features

# 5. Ottieni URL e test endpoint
echo ""
echo "5️⃣ Verifica deployment..."
echo ""

APP_URL=$(modal app list | grep "gender-prediction-academic" | grep -oE 'https://[^ ]+' | head -1)

if [ -z "$APP_URL" ]; then
    echo -e "${YELLOW}⚠️  Non riesco a trovare l'URL automaticamente.${NC}"
    echo "   Esegui: modal app list"
else
    echo -e "${GREEN}✅ App deployata su:${NC} $APP_URL"
    echo ""
    
    # Test health endpoint
    echo "📊 Test Health Check:"
    HEALTH_RESPONSE=$(curl -s "$APP_URL/health")
    if [ $? -eq 0 ]; then
        echo "$HEALTH_RESPONSE" | python3 -m json.tool 2>/dev/null | head -20
        echo ""
        
        # Test prediction
        echo "📊 Test Prediction:"
        curl -s -X POST "$APP_URL/predict" \
            -H "Content-Type: application/json" \
            -d '{"names": "Mario Rossi", "return_metadata": true}' | python3 -m json.tool 2>/dev/null | head -30
    else
        echo -e "${RED}❌ Impossibile connettersi all'API${NC}"
    fi
fi

echo ""
echo -e "${GREEN}✅ Deployment completato!${NC}"
echo ""
echo "📌 Comandi utili:"
echo "   • Logs: modal app logs gender-prediction-academic"
echo "   • Stop: modal app stop gender-prediction-academic"
echo "   • Docs: $APP_URL/docs"
echo "   • Verifica modello: python verify_model_version.py"
echo ""