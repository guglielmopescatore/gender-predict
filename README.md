# Gender Prediction Package

A comprehensive deep learning framework for gender prediction from names using PyTorch.

## 🚀 Features

- **Multiple Model Architectures**: From basic BiLSTM to advanced multi-head attention models
- **Advanced Training Techniques**: 
  - Focal loss with label smoothing
  - Mixup data augmentation
  - Balanced sampling
  - Cosine annealing with warmup
  - Gradient clipping & early stopping
  - Embedding layer freezing
- **Automatic Hyperparameter Optimization**: Learning rate finder
- **Test Time Augmentation (TTA)**: Smart adaptive augmentation for improved accuracy
- **Comprehensive Evaluation**: 
  - Bias analysis & fairness metrics
  - Detailed error analysis with visualizations
  - Confusion matrices
- **Experiment Management**: Full experiment tracking, comparison and reporting
- **Data Processing**: Advanced name preprocessing with diacritic handling and augmentation

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/guglielmopescatore/gender-predict.git
cd gender-predict

# Install in development mode
pip install -e .

# Or install from PyPI (if published)
pip install gender-predict
```

## 🎯 Quick Start

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

# Advanced V3 model (Round 3) - see Training Features section for all options
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

## 🏗️ Package Structure

```
src/gender_predict/
├── models/              # Model architectures
│   ├── base.py         # GenderPredictor, GenderPredictorEnhanced
│   ├── v3.py           # GenderPredictorV3 with advanced features
│   └── layers.py       # Attention layers
├── data/               # Data handling
│   ├── preprocessing.py       # Name preprocessing
│   ├── datasets.py           # PyTorch datasets  
│   ├── augmentation.py       # Data augmentation
│   └── feature_extraction.py # Feature engineering
├── training/           # Training utilities
│   ├── losses.py       # Loss functions (Focal, Label Smoothing)
│   ├── samplers.py     # Balanced batch sampling
│   └── schedulers.py   # Learning rate scheduling & mixup
├── evaluation/         # Evaluation and analysis
│   ├── evaluator.py         # Unified evaluation
│   ├── postprocess.py       # Post-processing, calibration
│   ├── error_analysis.py    # Detailed error analysis
│   └── tta.py              # Test Time Augmentation
├── experiments/        # Experiment management
│   ├── manager.py      # Experiment tracking
│   └── comparison.py   # Experiment comparison
└── utils/              # General utilities
    ├── common.py       # Common utilities
    ├── lr_finder.py    # Learning rate finder
    └── data_cleaning.py # Data cleaning tools
```

## 🧠 Model Architectures

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

## 📊 Training Features

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

### DataLoader Optimization
Optimize data loading for your hardware:
```bash
# For GPU training on Linux
python scripts/train_model.py --data_file data.csv --num_workers 8 --pin_memory

# For Windows (avoid multiprocessing issues)
python scripts/train_model.py --data_file data.csv --num_workers 0
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

## 🎯 Test Time Augmentation (TTA)

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

## 📊 Experiment Tracking

The package includes comprehensive experiment tracking:

- **Automatic ID generation** based on parameters
- **Full configuration logging** 
- **Training history tracking with plots**
- **Bias analysis and fairness metrics**
- **Model checkpointing with best model selection**
- **HTML reports with visualizations**
- **Experiment comparison tools**

## 🎯 Performance

The models achieve:
- **Accuracy**: 92-94% on gender prediction
- **F1 Score**: 90-92% 
- **With TTA**: Additional +0.3-0.8% accuracy improvement
- **Bias Balance**: Configurable fairness constraints
- **Speed**: Fast inference with GPU acceleration

## 📝 Data Format

Expected CSV format:
```csv
primaryName,gender
John Smith,M
Jane Doe,W
Marco Rossi,M
Giulia Bianchi,W
```

## 🔧 Advanced Usage

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

## 📋 Complete Parameter Reference

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

## 🛠️ Utility Scripts

A set of ready-to-use Python scripts for batch analysis and post-processing of experiments.  
All are found in the `scripts/` directory.

| Script                      | Purpose                                                                                          |
|-----------------------------|--------------------------------------------------------------------------------------------------|
| **infer_validation.py**     | Runs validation inference for an existing experiment, reproducing the original split and outputting per-sample probabilities and labels. |
| **calc_thresholds.py**      | Computes the F1-optimal threshold for the last N experiments, writing the result to `val_threshold.json` in each experiment.             |
| **summarize_grid_results.py** | Summarizes the latest N experiments into a single CSV and Markdown table for easy comparison (F1, accuracy, threshold, etc.).            |

### Example usage

```bash
# Run validation inference for an experiment
python scripts/infer_validation.py --exp_dir ./experiments/20250604_192834_r3_bce_h256_l3_dual_frz5

# Find F1-optimal threshold for the latest 12 experiments
python scripts/calc_thresholds.py --exp_dir ./experiments --n_last 12

# Summarize latest 12 experiments as CSV and Markdown
python scripts/summarize_grid_results.py --exp_dir ./experiments --n_last 12 \
    --out_csv grid_metrics.csv --out_md grid_metrics.md

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.

## 📚 Citation

If you use this package in your research, please cite:

```bibtex
@software{gender_predict,
  title={Gender Prediction Package},
  author={Guglielmo Pescatore},
  year={2024},
  url={https://github.com/guglielmopescatore/gender-predict}
}
```
