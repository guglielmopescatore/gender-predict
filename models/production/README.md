# Production model — gender-predict 4.0 (S-D1)

- `model.pth` — single-string model (V3S): one character-level BiLSTM with multi-head attention on the whole
  normalised name, plus suffix and phonetic features; hidden size 256, 3 layers, 8 attention heads, 4.6 M weights.
  Trained for 20 epochs on 6.97 M names (IMDb, Wikidata, registers of elected officials and candidates of Italy,
  France and Brazil, Olympic athletes), with soft targets from three teacher models (distillation). Seed 43.
- `preprocessor.pkl`, `feature_extractor.pkl` — fitted preprocessor (name normalisation 2, single string of 48
  characters) and feature extractor.
- `first_names.json.gz` — first names seen at least 10 times in the training file, with the number of persons and of
  women: used for the reliability level of each prediction.
- `config.json` — files, decision threshold (0.46), reference metrics. `parameters.json` — training configuration.
- `logs/` — metrics on the held-out split of the training file.

Selected on 18 November 2026. The previous production model (V4-R1, June 2025) is in `models/v4r1/`.
Runs on CPU; no GPU required.
