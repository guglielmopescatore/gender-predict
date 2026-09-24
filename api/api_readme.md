# Gender Prediction API - Deployment Guide

Questa directory contiene i file per il deployment dell'API su Modal.

## 📋 Prerequisiti

1. **Modal CLI installato e configurato**:
   ```bash
   pip install modal
   modal token new
   ```

2. **File necessari** (verificati automaticamente):
   - `../models/best_v3_model/preprocessor.pkl`
   - `../models/best_v3_model/feature_extractor.pkl`
   - `../models/best_v3_model/models/model.pth`
   - `../scripts/final_predictor.py`

3. **Opzionale - per testing completo**:
   ```bash
   pip install requests  # Per verify_model_version.py
   # O installa tutte le dipendenze:
   pip install -r requirements.txt
   ```

## 🚀 Quick Start

### Opzione 1: Usando Make (Consigliato)
```bash
cd api/
make all        # Deploy completo con test
make verify     # Verifica che il modello sia corretto
```

### Opzione 2: Usando lo script bash
```bash
cd api/
chmod +x deploy.sh
./deploy.sh
```

### Opzione 3: Comandi manuali
```bash
cd api/
modal deploy modal_deployment.py
modal run modal_deployment.py::test_academic_features
python verify_model_version.py
```

## 📦 File nella directory

- `modal_deployment.py` - Codice principale del deployment
- `config.py` - Configurazione del modello e dell'API
- `deploy.sh` - Script di deployment automatico
- `verify_model_version.py` - Verifica che il modello deployato sia corretto
- `Makefile` - Comandi semplificati

## 🔧 Comandi utili

```bash
# Status e logs
make status     # Vedi stato dell'app
make logs       # Vedi logs in tempo reale

# Gestione deployment
make stop       # Ferma l'app
make clean      # Pulisci tutto e ferma l'app

# Test e verifica
make test       # Esegui test
make verify     # Verifica versione modello (richiede requests)
make quick-verify  # Verifica rapida con curl (senza requests)
```

## 🧪 Test dell'API

Dopo il deployment, testa l'API:

```bash
# Singola predizione
curl -X POST "https://YOUR-APP-URL/predict" \
  -H "Content-Type: application/json" \
  -d '{"names": "Mario Rossi", "return_metadata": true}'

# Batch prediction
curl -X POST "https://YOUR-APP-URL/predict" \
  -H "Content-Type: application/json" \
  -d '{"names": ["Anna Rossi", "Marco Verdi", "Sara Neri"]}'
```

## 📊 Verifica del deployment

Il comando `make verify` o `python verify_model_version.py` confronta le predizioni del modello locale con quelle dell'API per assicurarsi che sia stato deployato il modello corretto.

## ⚠️ Troubleshooting

1. **"34000 files uploaded"**: Se vedi questo messaggio, il file `modal_deployment.py` sta montando troppi file. Verifica che stia usando `.add_local_file()` per file singoli invece di `.add_local_dir("..")`.

2. **"Model not found"**: Verifica che i path in `config.py` corrispondano alla struttura montata:
   ```python
   'model_path': '/app/models/best_v3_model/models/model.pth'
   ```

3. **Import errors**: Assicurati che il PYTHONPATH sia configurato correttamente in `modal_deployment.py`.

4. **"No module named 'requests'"**: Per il comando `make verify`, installa requests:
   ```bash
   pip install requests
   # O usa la verifica rapida che non richiede requests:
   make quick-verify
   ```

## 📝 Note

- L'API usa rate limiting per uso accademico (1000 req/ora)
- Supporta batch fino a 500 nomi per richiesta
- Include metadata sul modello se richiesto con `return_metadata=true`

## 🔗 Links

- API Docs: `https://YOUR-APP-URL/docs`
- Health Check: `https://YOUR-APP-URL/health`
- Stats: `https://YOUR-APP-URL/stats`
