# Production model

Weights and preprocessor loaded by `scripts/final_predictor.py` and mounted by `api/modal_deployment.py`.

- `model.pth` — V3 architecture (BCE loss, hidden 256, 3 layers, dual input), variant **V4-R1**:
  trained with advanced name preprocessing (diacritic normalisation, hyphen and surname-prefix handling).
  Selected on 2025-06-19 over the B0 baseline on a 40k comparison set; decision threshold 0.52
  (accuracy ≈ 0.92, F1 ≈ 0.90, gender bias ratio ≈ 1.00). Metrics in `logs/`.
- `preprocessor.pkl`, `feature_extractor.pkl` — fitted name preprocessor and feature extractor.
- `parameters.json` — training configuration.

Non-Latin names are transliterated before prediction by `scripts/transliteration_wrapper.py`
(see `scripts/enhanced_predictor.py`). Runs on CPU; no GPU required (~5 ms per name).
