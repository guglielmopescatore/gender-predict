#!/bin/bash
# Script ottimizzato per RTX 4090 - Target >92.5% accuracy
# Con opzione per analisi degli errori

# Parsing degli argomenti
ENABLE_ERROR_ANALYSIS=false
CUSTOM_NOTE=""
SHOW_HELP=false

# Funzione di aiuto
show_help() {
    echo "🚀 Enhanced Training Script per RTX 4090"
    echo ""
    echo "Utilizzo:"
    echo "  $0 [opzioni]"
    echo ""
    echo "Opzioni:"
    echo "  --error-analysis     Abilita analisi dettagliata degli errori"
    echo "  --note TESTO         Aggiunge una nota personalizzata all'esperimento"
    echo "  --help               Mostra questo messaggio di aiuto"
    echo ""
    echo "Esempi:"
    echo "  $0                                    # Training standard"
    echo "  $0 --error-analysis                  # Training con analisi errori"
    echo "  $0 --error-analysis --note \"test_v2\"  # Training con nota personalizzata"
    echo ""
}

# Parsing argomenti
while [[ $# -gt 0 ]]; do
    case $1 in
        --error-analysis)
            ENABLE_ERROR_ANALYSIS=true
            shift
            ;;
        --note)
            CUSTOM_NOTE="$2"
            shift 2
            ;;
        --help|-h)
            SHOW_HELP=true
            shift
            ;;
        *)
            echo "⚠️  Opzione sconosciuta: $1"
            echo "Usa --help per vedere le opzioni disponibili"
            exit 1
            ;;
    esac
done

# Mostra aiuto se richiesto
if [ "$SHOW_HELP" = true ]; then
    show_help
    exit 0
fi

echo "🚀 Launching ENHANCED training - Target >92.5%"
echo "💪 Sfruttando al massimo RTX 4090"

# Mostra configurazione
if [ "$ENABLE_ERROR_ANALYSIS" = true ]; then
    echo "🔍 Analisi degli errori: ABILITATA"
else
    echo "🔍 Analisi degli errori: DISABILITATA"
fi

if [ -n "$CUSTOM_NOTE" ]; then
    echo "📝 Nota personalizzata: $CUSTOM_NOTE"
fi

echo ""

# Verifica che siamo nella directory corretta
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

# Verifica che esista il file di training
if [ ! -f "train_improved_model_v2.py" ]; then
    echo "❌ Errore: train_improved_model_v2.py non trovato nella directory corrente"
    echo "   Directory attuale: $(pwd)"
    exit 1
fi

# Verifica che esista il dataset
DATASET_PATH="../training_dataset_final_nodiac.csv"
if [ ! -f "$DATASET_PATH" ]; then
    echo "❌ Errore: Dataset non trovato: $DATASET_PATH"
    echo "   Verifica che il file esista nella directory parent"
    exit 1
fi

# Verifica che esista error_analysis_tool.py se richiesto
if [ "$ENABLE_ERROR_ANALYSIS" = true ]; then
    if [ ! -f "../error_analysis_tool.py" ]; then
        echo "❌ Errore: error_analysis_tool.py non trovato nella directory parent"
        echo "   Necessario per l'analisi degli errori"
        exit 1
    fi
fi

# Costruisci la nota finale
FINAL_NOTE="enhanced_training"
if [ -n "$CUSTOM_NOTE" ]; then
    FINAL_NOTE="${FINAL_NOTE}_${CUSTOM_NOTE}"
fi
if [ "$ENABLE_ERROR_ANALYSIS" = true ]; then
    FINAL_NOTE="${FINAL_NOTE}_with_errors"
fi

# Mostra avvio
echo "🎯 Avvio training con configurazione:"
echo "   Dataset: $DATASET_PATH"
echo "   Nota: $FINAL_NOTE"
echo "   Error Analysis: $ENABLE_ERROR_ANALYSIS"
echo ""

# Array con i parametri base
TRAINING_ARGS=(
    --round 3
    --data_file "$DATASET_PATH"
    --save_dir ..
    --epochs 50
    --early_stop 5
    --seed 43
    --note "$FINAL_NOTE"
    
    # Architettura ottimizzata
    --embedding_dim 64
    --hidden_size 256
    --n_layers 3
    --num_heads 8
    --dropout 0.35
    --dual_input
    
    # Training ottimizzato per RTX 4090
    --batch_size 1024
    --lr 5e-4
    --min_lr 1e-7
    --warmup_epochs 5
    --gradient_clip 1.0
    --freeze_epochs 5
    
    # Data augmentation
    --augment_prob 0.25
    
    # Loss function ottimizzata per dataset sbilanciato
    --loss focal
    --focal_gamma 2.0
    --focal_alpha 0.49
    --label_smooth 0.0
)

# Aggiungi flag per error analysis se abilitato
if [ "$ENABLE_ERROR_ANALYSIS" = true ]; then
    TRAINING_ARGS+=(--enable_error_analysis)
fi

# Controlla se GPU è disponibile
if command -v nvidia-smi &> /dev/null; then
    echo "📊 GPU Status:"
    nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits | head -1
    echo ""
else
    echo "⚠️  nvidia-smi non disponibile - verifica installazione CUDA"
    echo ""
fi

# Timestamp di inizio
START_TIME=$(date)
echo "⏰ Inizio training: $START_TIME"
echo ""

# Esegui il training
echo "🚀 Esecuzione comando:"
echo "python train_improved_model_v2.py ${TRAINING_ARGS[*]}"
echo ""

python train_improved_model_v2.py "${TRAINING_ARGS[@]}"

# Risultato del training
TRAINING_EXIT_CODE=$?
END_TIME=$(date)

echo ""
echo "⏰ Fine training: $END_TIME"
echo ""

if [ $TRAINING_EXIT_CODE -eq 0 ]; then
    echo "✅ Training completato con successo!"
    
    # Suggerimenti post-training
    echo ""
    echo "🎯 Prossimi passi suggeriti:"
    echo "   1. Controlla i risultati:"
    echo "      python experiment_tools.py list --detail"
    echo ""
    echo "   2. Confronta con esperimenti precedenti:"
    echo "      python experiment_tools.py compare --metric test_f1"
    echo ""
    echo "   3. Analizza il bias:"
    echo "      python experiment_tools.py bias"
    echo ""
    
    if [ "$ENABLE_ERROR_ANALYSIS" = true ]; then
        echo "   4. Esamina i pattern di errore:"
        echo "      python experiment_tools.py errors"
        echo ""
        echo "   5. Trova l'ultimo esperimento e guarda gli errori:"
        echo "      ls -la ../experiments/ | tail -5"
        echo "      # Poi apri il file error_analysis.csv nell'esperimento"
        echo ""
    fi
    
    echo "   Per risultati dettagliati, apri il report HTML generato!"
    
else
    echo "❌ Training fallito con codice di uscita: $TRAINING_EXIT_CODE"
    echo ""
    echo "🔍 Suggerimenti per il debug:"
    echo "   1. Controlla i log sopra per errori specifici"
    echo "   2. Verifica la disponibilità della GPU: nvidia-smi"
    echo "   3. Controlla lo spazio su disco: df -h"
    echo "   4. Verifica la memoria RAM: free -h"
    echo ""
    exit $TRAINING_EXIT_CODE
fi

echo ""
echo "🎯 Con questi parametri potenziati dovresti raggiungere:"
echo "   - Accuracy: 92.5-93%"
echo "   - F1: 90-91%"
echo "   - Training time: ~90-120 min"
if [ "$ENABLE_ERROR_ANALYSIS" = true ]; then
    echo "   - Analisi errori dettagliata disponibile nei log"
fi
echo ""
