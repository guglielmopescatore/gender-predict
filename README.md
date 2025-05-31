# Gender Prediction Package

A comprehensive deep learning framework for gender prediction from names using PyTorch.

## 🚀 Features

- **Multiple Model Architectures**: From basic BiLSTM to advanced multi-head attention models
- **Advanced Training Techniques**: Focal loss, label smoothing, balanced sampling, cosine annealing
- **Comprehensive Evaluation**: Bias analysis, error analysis, fairness metrics
- **Experiment Management**: Full experiment tracking and comparison
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
    --loss focal --alpha 0.7 --gamma 2.0 --balanced_sampler

# Enhanced architecture (Round 2)  
python scripts/train_model.py --round 2 --data_file data/training.csv \
    --n_layers 2 --hidden_size 80 --dual_input

# Advanced V3 model (Round 3)
python scripts/train_model.py --round 3 --data_file data/training.csv \
    --advanced_preprocessing --augment_prob 0.15 --num_heads 4
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
# Compare accuracy across experiments
python scripts/experiment_tools.py compare --metric test_accuracy

# Analyze bias metrics
python scripts/experiment_tools.py bias

# Generate full report
python scripts/experiment_tools.py report
```

## 🏗️ Package Structure

```
src/gender_predict/
├── models/           # Model architectures
│   ├── base.py      # GenderPredictor, GenderPredictorEnhanced
│   ├── v3.py        # GenderPredictorV3 with advanced features
│   └── layers.py    # Attention layers
├── data/            # Data handling
│   ├── preprocessing.py    # Name preprocessing
│   ├── datasets.py        # PyTorch datasets
│   ├── augmentation.py    # Data augmentation
│   └── feature_extraction.py  # Feature engineering
├── training/        # Training utilities
│   ├── losses.py    # Loss functions (Focal, Label Smoothing)
│   ├── samplers.py  # Balanced batch sampling
│   └── schedulers.py # Learning rate scheduling
├── evaluation/      # Evaluation and analysis
│   ├── evaluator.py      # Unified evaluation
│   ├── postprocess.py    # Post-processing, calibration
│   └── error_analysis.py # Error analysis tools
├── experiments/     # Experiment management
│   ├── manager.py   # Experiment tracking
│   └── comparison.py # Experiment comparison
└── utils/           # General utilities
    ├── common.py    # Common utilities
    └── data_cleaning.py  # Data cleaning tools
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
- Early stopping

### Round 2: Enhanced Architecture
- Multi-layer BiLSTM
- Improved attention mechanisms
- Larger capacity models
- Layer normalization

### Round 3: Advanced V3 Model
- Multi-head attention (4+ heads)
- Feature engineering (suffixes, phonetics)
- Data augmentation
- Cosine annealing scheduler
- Advanced preprocessing

## 📊 Experiment Tracking

The package includes comprehensive experiment tracking:

- **Automatic ID generation** based on parameters
- **Full configuration logging** 
- **Training history tracking**
- **Bias analysis and fairness metrics**
- **HTML reports with visualizations**
- **Model comparison tools**

## 🎯 Performance

The models achieve:
- **Accuracy**: 92-94% on gender prediction
- **F1 Score**: 90-92% 
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

## 🔧 Advanced Usage

### Python API

```python
from gender_predict import create_model, NamePreprocessor, ModelEvaluator

# Create and train a model
preprocessor = NamePreprocessor()
model = create_model('v3', vocab_size=preprocessor.vocab_size, ...)

# Evaluate model
evaluator = ModelEvaluator(model, preprocessor)
results = evaluator.evaluate_dataset(test_dataset)
```

### Custom Training Loop

```python
from gender_predict.training import FocalLossImproved, CosineAnnealingWarmupScheduler
from gender_predict.experiments import ExperimentManager

# Setup custom training
criterion = FocalLossImproved(alpha=0.7, gamma=2.0)
scheduler = CosineAnnealingWarmupScheduler(optimizer, warmup_epochs=3, max_epochs=30)
experiment = ExperimentManager(args)

# Your training loop here...
```

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
