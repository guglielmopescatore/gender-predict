# Gender Prediction from Names

A PyTorch model that predicts the gender (W/M) associated with a full name, trained on IMDb name data. The repository ships the trained production model, a Python API, a command-line tool, and the full training and evaluation framework used to build it.

The model runs on CPU (about 5 ms per name); no GPU is required.

## Installation

```bash
git clone https://github.com/guglielmopescatore/gender-predict.git
cd gender-predict
pip install -e .
```

Python ≥ 3.8. The trained model (`models/production/`, ~36 MB) is included in the repository, so inference works right after cloning.

## Quick start

### Command line

```bash
# one or more names
gender-predict "Maria Rossi" "Анна Каренина" "张伟"

# a CSV file: adds predicted_gender, probability_female, confidence
gender-predict --input names.csv --output results.csv --name-column primaryName
```

`python scripts/final_predictor.py ...` works the same way without installing. Options: `--threshold`, `--no-transliterate`, `--model-dir`, `--device`, `-q`.

### Python

```python
from gender_predict import GenderPredictor

predictor = GenderPredictor()          # loads models/production
predictor.predict("Maria Rossi")
# {'name': 'Maria Rossi', 'predicted_gender': 'W', 'probability_female': 0.986,
#  'confidence': 0.986, 'threshold_used': 0.52, 'transliterated_name': 'maria rossi',
#  'detected_script': 'LAT', 'was_transliterated': False}

predictor.predict_many(["张伟", "Иван Петров"])
df = predictor.predict_dataframe(df, name_column="primaryName")
```

`probability_female` is the model output; `predicted_gender` is `W` when it is at or above the decision threshold (0.52), `M` otherwise. `confidence` is the probability of the predicted class.

## Production model

`models/production/config.json` is the single source of truth for file names, threshold and reference metrics.

| | |
|---|---|
| Architecture | V3: character BiLSTM with multi-head attention, dual input (first name / surname), suffix and phonetic features |
| Weights | experiment `r3_bce_h256_l3_dual_frz5`, variant **V4-R1** (advanced preprocessing: diacritics, hyphens, surname prefixes), selected 2025-06-19 |
| Threshold | 0.52, chosen on a 40k comparison set for the best accuracy/F1 with gender error rates as equal as possible |
| Test set | accuracy 0.925, F1 0.903 (class W), bias ratio 1.006 |
| Comparison set (40k) | accuracy 0.922, F1 0.899, bias ratio 0.999 |

Inference pipeline: (1) non-Latin scripts (Cyrillic, Chinese, Japanese, Korean) are transliterated to a romanised form, and all names are lower-cased and stripped of diacritics and apostrophes; (2) robust cleaning (encoding fixes, control characters, Unicode normalisation) and the preprocessor fitted at training time; (3) model forward pass; (4) decision at the threshold. Arabic script is currently not transliterated and Japanese kanji are read as Chinese, so predictions for those scripts are unreliable.

Regression tests (`pytest tests/`) check that the pipeline reproduces the reference probabilities stored in `tests/regression/baseline_v4r1.json`.

## Training and evaluation

### Training a Model

```bash
# Basic model (Round 0)
python scripts/train_model.py --round 0 --data_file data/training.csv

# Enhanced model with focal loss (Round 1)
python scripts/train_model.py --round 1 --data_file data/training.csv \
    --loss focal --balanced_sampler

# Enhanced architecture (Round 2)  
python scripts/train_model.py --round 2 --data_file data/training.csv \
    --n_layers 2 --hidden_size 80

# Advanced V3 model (Round 3)
python scripts/train_model.py --round 3 --data_file data/training.csv \
    --embedding_dim 64 --hidden_size 256 --n_layers 3

# For complete parameter list
python scripts/train_model.py --help
```

### Evaluating a Model

```bash
python scripts/evaluate_model.py \
    --model experiments/[experiment_id]/models/model.pth \
    --preprocessor experiments/[experiment_id]/preprocessor.pkl \
    --test_data data/test.csv
```

### Comparing Experiments

```bash
# Compare F1 scores across experiments
python scripts/experiment_tools.py compare --metric test_f1

# Analyze bias metrics
python scripts/experiment_tools.py bias

# Generate full report
python scripts/experiment_tools.py report
```

## Model Architectures

### Round 0: Base Model
- BiLSTM with attention
- Character-level embeddings
- Simple architecture for baseline

### Round 1: Enhanced Training  
- Focal loss for imbalanced data
- Label smoothing
- Balanced batch sampling
- Early stopping with min_delta

### Round 2: Enhanced Architecture
- Multi-layer BiLSTM
- Improved attention mechanisms
- Larger capacity models
- Layer normalization

### Round 3: Advanced V3 Model
- Multi-head attention (4-8 heads)
- Feature engineering (suffixes, phonetics)
- Data augmentation with mixup
- Cosine annealing scheduler with warmup
- Advanced preprocessing
- Embedding layer freezing
- Test Time Augmentation support

## Training Features

### Finding Optimal Learning Rate
Automatically find the best learning rate before training:
```bash
python scripts/train_model.py --data_file data.csv --find_lr --lr_finder_iters 200
```

### Mixup Data Augmentation
Improve generalization by training on interpolated samples:
```bash
python scripts/train_model.py --data_file data.csv --use_mixup --mixup_alpha 0.2
```

### Embedding Layer Freezing
Stabilize early training by freezing embeddings:
```bash
python scripts/train_model.py --data_file data.csv --freeze_epochs 5
```

### Advanced Loss Functions
Configure focal loss for imbalanced datasets:
```bash
python scripts/train_model.py --data_file data.csv \
    --loss focal --alpha 0.492 --gamma 2.0
```

### Error Analysis
Generate detailed error analysis reports:
```bash
python scripts/train_model.py --data_file data.csv --enable_error_analysis
```

This generates:
- `error_analysis.csv` - All prediction errors
- `error_analysis_results.json` - Statistical analysis
- `error_summary.json` - Key insights
- `error_analysis.png` - Visualizations

## Test Time Augmentation (TTA)

### Standard TTA
Fixed number of augmentations for all samples:
```bash
python scripts/train_model.py --data_file data.csv --use_tta --tta_n_aug 5
```

### Smart TTA  
Adaptive augmentation based on prediction uncertainty:
```bash
python scripts/train_model.py --data_file data.csv \
    --use_tta --tta_strategy smart \
    --tta_min_aug 3 --tta_max_aug 10 --tta_std 0.15
```

## Experiment Tracking

The package includes comprehensive experiment tracking:

- **Automatic ID generation** based on parameters
- **Full configuration logging** 
- **Training history tracking with plots**
- **Bias analysis and fairness metrics**
- **Model checkpointing with best model selection**
- **HTML reports with visualizations**
- **Experiment comparison tools**

## Data Format

Expected CSV format:
```csv
primaryName,gender
John Smith,M
Jane Doe,W
Marco Rossi,M
Giulia Bianchi,W
```

## Repository structure

```
gender-predict/
├── models/production/          # trained model, preprocessor, config.json, metrics
├── src/gender_predict/         # the package
│   ├── inference/              # GenderPredictor, CLI, transliteration, robust preprocessing
│   ├── data/                   # preprocessing, datasets, augmentation, feature extraction
│   ├── models/                 # architectures (base, enhanced, V3)
│   ├── training/               # losses, samplers, schedulers
│   ├── evaluation/             # evaluator, error analysis, TTA, post-processing
│   └── experiments/            # experiment manager and comparison
├── scripts/                    # train_model.py, evaluate_model.py, experiment_tools.py, final_predictor.py
├── tools/                      # threshold and bias analysis utilities
├── examples/                   # sample-data preparation
├── tests/                      # regression tests and baseline probabilities
├── data/                       # datasets (only small samples are tracked)
└── experiments/                # training outputs (not tracked)
```

## Utility Tools

### Analysis tools (in `tools/`)

```bash
# Compute F1-optimal thresholds for experiments
python tools/calc_thresholds.py --exp_dir ./experiments --n_last 12

# Summarize experiment results
python tools/summarize_grid_results.py --exp_dir ./experiments --n_last 12 \
    --out_csv grid_metrics.csv --out_md grid_metrics.md

# Run validation inference for an experiment
python tools/infer_validation.py --exp_dir ./experiments/[experiment_id]

# Batch evaluation of multiple experiments
bash tools/batch_evaluate.sh
```

### Data Preparation (in `examples/`)

```bash
# Create sample datasets for testing
python examples/create_sample_data.py

# Prepare data for training
python examples/prepare_data.py --input raw_data.csv --output processed_data.csv
```

## Advanced Usage

### Python API

```python
from gender_predict import create_model, NamePreprocessor, ModelEvaluator
from gender_predict.evaluation.tta import TestTimeAugmentation
from gender_predict.data import NameAugmenter, NameFeatureExtractor

# Create and train a model
preprocessor = NamePreprocessor()
model = create_model('v3', vocab_size=preprocessor.vocab_size, ...)

# Evaluate with TTA
augmenter = NameAugmenter(augment_prob=0.2)
feature_extractor = NameFeatureExtractor()
tta_evaluator = TestTimeAugmentation(
    model, preprocessor, augmenter, device='cuda', 
    feature_extractor=feature_extractor
)
prob, confidence = tta_evaluator.predict_single("John Smith", n_aug=5)
```

### Custom Training Loop

```python
from gender_predict.training import FocalLossImproved, CosineAnnealingWarmupScheduler
from gender_predict.experiments import ExperimentManager
from gender_predict.utils.lr_finder import find_optimal_lr

# Find optimal learning rate
optimal_lr = find_optimal_lr(model, train_loader, criterion, device)

# Setup custom training
criterion = FocalLossImproved(alpha=0.7, gamma=2.0)
scheduler = CosineAnnealingWarmupScheduler(
    optimizer, warmup_epochs=3, max_epochs=30
)
experiment = ExperimentManager(args)

# Your training loop here...
```

## Complete Parameter Reference

For a complete list of all available parameters and their descriptions:

```bash
python scripts/train_model.py --help
```

Key parameter categories:
- **Model Architecture**: `--embedding_dim`, `--hidden_size`, `--n_layers`, `--num_heads`
- **Training**: `--epochs`, `--batch_size`, `--lr`, `--early_stop`
- **Loss Functions**: `--loss`, `--alpha`, `--gamma`, `--pos_weight`
- **Data Augmentation**: `--augment_prob`, `--use_mixup`, `--mixup_alpha`
- **Optimization**: `--freeze_epochs`, `--gradient_clip`, `--warmup_epochs`
- **Evaluation**: `--enable_error_analysis`, `--use_tta`, `--tta_strategy`
- **Hardware**: `--num_workers`, `--pin_memory`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

GNU General Public License v3.0 - see LICENSE file for details.

## Citation

If you use this package in your research, please cite:

```bibtex
@software{gender_predict,
  title={Gender Prediction from Names},
  author={Guglielmo Pescatore},
  year={2025},
  url={https://github.com/guglielmopescatore/gender-predict}
}
```

## Ethical Considerations

This tool is intended for research purposes. Users should be aware of:
- **Bias**: Models may reflect biases present in training data
- **Cultural Sensitivity**: Name-gender associations vary across cultures
- **Privacy**: Consider privacy implications when processing personal data
- **Fairness**: Regular bias evaluation and mitigation strategies are recommended

For academic use, please ensure compliance with your institution's ethics guidelines.
