# Production model

Weights and preprocessor loaded by `scripts/final_predictor.py`.

- `model.pth` — V3 model, experiment `20250603_192912_r3_bce_h256_l3_dual_frz5` (BCE loss, hidden 256, 3 layers, dual input). Test F1 ≈ 0.90.
- `preprocessor.pkl` — name preprocessor fitted on the training data.
- `parameters.json` — training configuration of the experiment.

Runs on CPU; no GPU required.
