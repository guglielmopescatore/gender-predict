#!/bin/bash
# Script ottimizzato per RTX 4090 - Target >92.5% accuracy

echo "🚀 Launching ENHANCED training - Target >92.5%"
echo "💪 Sfruttando al massimo RTX 4090"
echo ""

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

# Versione 1: Più capacità del modello
python train_improved_model_v2.py \
    --round 3 \
    --data_file ../training_dataset_final_nodiac.csv \
    --save_dir .. \
    --epochs 50 \
    --early_stop 5 \
    --seed 43 \
    \
    --embedding_dim 64 \
    --hidden_size 256 \
    --n_layers 3 \
    --num_heads 8 \
    --dropout 0.35 \
    --dual_input \
    \
    --batch_size 1024 \
    --lr 5e-4 \
    --min_lr 1e-7 \
    --warmup_epochs 5 \
    --gradient_clip 1.0 \
    --freeze_epochs 5 \
    \
    --augment_prob 0.25 \
    \
    --loss focal \
    --focal_gamma 2.0 \
    --focal_alpha 0.49 \
    --label_smooth 0.0

echo ""
echo "🎯 Con questi parametri potenziati dovresti raggiungere:"
echo "   - Accuracy: 92.5-93%"
echo "   - F1: 90-91%"
echo "   - Training time: ~90-120 min"
