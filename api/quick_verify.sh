#!/bin/bash

# Verifica rapida del modello deployato usando curl
# Esegui dalla directory api/: ./quick_verify.sh

echo "🔍 Verifica Rapida Modello Deployato"
echo "===================================="
echo ""

# Colori
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

# Trova URL dell'app
APP_URL=$(modal app list | grep "gender-prediction-academic" | grep -oE 'https://[^ ]+' | head -1)

if [ -z "$APP_URL" ]; then
    echo -e "${RED}❌ API non trovata! Esegui prima: make deploy${NC}"
    exit 1
fi

echo "📍 API URL: $APP_URL"
echo ""

# Test nomi
echo "🧪 Test predizioni..."
echo ""

# Singola predizione
echo "Test 1: Mario Rossi"
curl -s -X POST "$APP_URL/predict" \
    -H "Content-Type: application/json" \
    -d '{"names": "Mario Rossi"}' | python3 -c "
import sys, json
data = json.load(sys.stdin)
pred = data['predictions'][0]
print(f\"  Genere: {pred['predicted_gender']}\")
print(f\"  Probabilità F: {pred['probability_female']:.3f}\")
print(f\"  Confidence: {pred['confidence']:.3f}\")
"

echo ""

# Batch test
echo "Test 2: Batch internazionale"
curl -s -X POST "$APP_URL/predict" \
    -H "Content-Type: application/json" \
    -d '{
        "names": [
            "Anna Bianchi",
            "John Smith",
            "María López",
            "François Dupont",
            "أحمد"
        ]
    }' | python3 -c "
import sys, json
data = json.load(sys.stdin)
print('  Nome               Genere    Prob F    Confidence')
print('  ' + '-'*50)
for pred in data['predictions']:
    name = pred['name'][:18].ljust(18)
    gender = pred['predicted_gender'].ljust(8)
    prob_f = f\"{pred['probability_female']:.3f}\".ljust(8)
    conf = f\"{pred['confidence']:.3f}\"
    print(f'  {name} {gender} {prob_f} {conf}')
"

echo ""

# Metadata check
echo "📊 Verifica metadata modello..."
curl -s "$APP_URL/stats" | python3 -c "
import sys, json
try:
    stats = json.load(sys.stdin)
    model_info = stats.get('model_info', {})
    print(f\"  Model Path: {model_info.get('model_path', 'N/A')}\")
    print(f\"  Threshold: {model_info.get('threshold', 'N/A')}\")
    print(f\"  Expected Accuracy: {model_info.get('expected_accuracy', 'N/A')}\")
except:
    print('  ⚠️  Non riesco a leggere i metadata')
"

echo ""
echo -e "${GREEN}✅ Verifica completata!${NC}"
echo ""
echo "Per un test più approfondito con confronto locale:"
echo "  1. Installa requests: pip install requests"
echo "  2. Esegui: python verify_model_version.py"
echo ""