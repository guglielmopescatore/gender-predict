# Gender Prediction from Names

A deep learning framework for gender prediction from names using PyTorch. This package implements multiple model architectures with advanced training techniques and comprehensive evaluation tools.

## Features

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
- **Production API**: Ready-to-deploy REST API with auto-sync capabilities

## Installation

```bash
# Clone the repository
git clone https://github.com/guglielmopescatore/gender-predict.git
cd gender-predict

# Install in development mode
pip install -e .

# Or install from PyPI (when published)
pip install gender-predict
```

## Quick Start

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

### Production Inference

```bash
# Single name prediction
python scripts/final_predictor.py --single_name "Mario Rossi"

# Batch prediction
python scripts/final_predictor.py --input data.csv --output results.csv
```

### API Deployment

The package includes a production-ready REST API with auto-sync capabilities:

#### Quick Setup
```bash
cd api/
cp config.py.template config.py
# Edit config.py with your model paths
./dev_workflow.sh
```

#### Auto-Sync Architecture
- **Zero duplication**: Direct import from `scripts/final_predictor.py`
- **Automatic sync**: Changes to local code reflected in production API
- **Single source**: One file to maintain for prediction logic

#### Development Workflow
```bash
# 1. Make changes to scripts/final_predictor.py
# 2. Test locally
python scripts/final_predictor.py --single_name "Mario Rossi"

# 3. Deploy with automatic synchronization
cd api/ && ./dev_workflow.sh

# 4. Test deployed API
modal run modal_deployment.py::test_prediction
```

#### Configuration Files
- `modal_deployment.py`: Main deployment with auto-sync
- `config.py`: Private configuration (gitignored)
- `config.py.template`: Setup template
- `web_interface.html`: Web interface for testing

#### Monitoring
- **Dashboard**: https://modal.com/apps
- **Logs**: `modal logs gender-prediction-v3`
- **Health**: `/health` endpoint for status checks

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

## Repository Structure

```
gender-predict/
├── api/                        # Production API deployment
│   ├── modal_deployment.py    # Modal deployment configuration
│   ├── dev_workflow.sh        # Deployment automation
│   ├── web_interface.html     # Web interface
│   └── config.py.template     # Configuration template
├── scripts/                    # Core training and evaluation scripts
│   ├── train_model.py         # Main training script
│   ├── evaluate_model.py      # Model evaluation
│   ├── final_predictor.py     # Production inference
│   └── experiment_tools.py    # Experiment management
├── tools/                      # Utility scripts for analysis
│   ├── calc_thresholds.py     # Threshold optimization
│   ├── summarize_grid_results.py # Experiment summarization
│   ├── infer_validation.py    # Validation inference
│   └── batch_evaluate.sh      # Batch evaluation
├── examples/                   # Example scripts and data preparation
│   ├── create_sample_data.py  # Generate sample datasets
│   └── prepare_data.py        # Data preparation utilities
├── src/gender_predict/         # Core package modules
│   ├── models/                # Model architectures
│   ├── data/                  # Data handling and preprocessing
│   ├── training/              # Training utilities and loss functions
│   ├── evaluation/            # Evaluation and analysis tools
│   ├── experiments/           # Experiment management
│   └── utils/                 # General utilities
├── experiments/               # Trained models and results
├── models/                    # Production-ready models
│   └── best_v3_model/        # Current best performing model
├── data/                      # Datasets and preprocessing
│   ├── raw/                  # Raw data and samples
│   ├── processed/            # Processed datasets
│   └── external/             # External data sources
└── tests/                     # Unit tests
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

## Performance

Based on our evaluation datasets, the models achieve:
- **Accuracy**: 90-94% on gender prediction tasks
- **F1 Score**: 88-92% depending on dataset and model configuration
- **Bias Metrics**: Configurable fairness constraints with bias analysis
- **Inference Speed**: Optimized for both CPU and GPU deployment

Note: Performance may vary significantly depending on dataset characteristics, name origins, and linguistic diversity.

## Data Format

Expected CSV format:
```csv
primaryName,gender
John Smith,M
Jane Doe,W
Marco Rossi,M
Giulia Bianchi,W
```

## Utility Tools

### Analysis Tools (in `tools/`)

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
