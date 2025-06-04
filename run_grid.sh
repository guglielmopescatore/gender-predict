#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# run_grid.sh – Batch launcher for overnight experiments
# -----------------------------------------------------------------------------
# Usage (inside the virtual‑env, from the project root):
#   chmod +x run_grid.sh
#   nohup ./run_grid.sh > grid.log 2>&1 &
#   disown                                # optional – detach from shell
# -----------------------------------------------------------------------------
# If you need to activate the env from inside the script, uncomment the line
# below and adjust the path:
# source "$(dirname "$0")/../env/bin/activate"
# -----------------------------------------------------------------------------

# Location of the training CSV
DATA="data/raw/training_dataset_final_nodiac.csv"

# Arguments shared by *all* experiments
BASE_ARGS=(
  --round 3
  --data_file "$DATA"
  --seed 43
  --epochs 50
  --early_stop 5
  --embedding_dim 64
  --hidden_size 256
  --n_layers 3
  --num_heads 8
  --batch_size 1024
  --lr 0.0005
  --min_lr 1e-07
  --warmup_epochs 5
  --gradient_clip 1.0
  --dropout 0.35
  --dual_input
  --freeze_epochs 5
  --augment_prob 0.25
  --enable_error_analysis
  --find_lr --lr_finder_iters 200
)

# -----------------------------------------------------------------------------
# Helper to launch a single experiment.
# Arguments:
#   $1 : experiment tag (for console echo only)
#   $@ : remaining CLI flags *after* the tag
# -----------------------------------------------------------------------------
run () {
  local tag=$1; shift                       # tag is just for the echo below
  echo -e "\n>>> avvio esperimento ${tag}\n"

  # Pipe "n" to automatically skip the LR‑Finder prompt
  yes n | python scripts/train_model.py "${BASE_ARGS[@]}" "$@"
}

# -----------------------------------------------------------------------------
# Grid of experiments (no quotes around multi‑word flag sequences!)
# -----------------------------------------------------------------------------
run B1 --note exp_B1_tta5 --loss bce --use_tta --tta_n_aug 5

run B2 --note exp_B2_tta5_do0.2 --loss bce --dropout 0.20 \
       --use_tta --tta_n_aug 5

run M0 --note exp_M0_mix0.1 --loss bce \
       --use_mixup --mixup_alpha 0.1

run M1 --note exp_M1_mix0.1_tta5 --loss bce \
       --use_mixup --mixup_alpha 0.1 \
       --use_tta --tta_n_aug 5

run M2 --note exp_M2_mix0.2_tta5 --loss bce \
       --use_mixup --mixup_alpha 0.2 \
       --use_tta --tta_n_aug 5

run F0 --note exp_F0_focal1 --loss focal --alpha 0.5 --gamma 1

run F1 --note exp_F1_focal1_tta5 --loss focal --alpha 0.5 --gamma 1 \
       --use_tta --tta_n_aug 5

run FM1 --note exp_FM1_mix0.1_focal1_tta5 --loss focal --alpha 0.5 --gamma 1 \
        --use_mixup --mixup_alpha 0.1 \
        --use_tta --tta_n_aug 5

run FM2 --note exp_FM2_mix0.1_focal1_do0.2_tta5 --loss focal --alpha 0.5 --gamma 1 \
        --dropout 0.20 \
        --use_mixup --mixup_alpha 0.1 \
        --use_tta --tta_n_aug 5

run H1 --note exp_H1_ttaSmart --loss bce \
       --use_tta --tta_strategy smart --tta_min_aug 3 --tta_max_aug 10 --tta_std 0.15

run H2 --note exp_H2_mix0.1_ttaSmart --loss bce \
       --use_mixup --mixup_alpha 0.1 \
       --use_tta --tta_strategy smart --tta_min_aug 3 --tta_max_aug 10 --tta_std 0.15

