#!/usr/bin/env python3
"""
Crea script unificati per il nuovo package.
"""

import os

def create_file_with_content(path: str, content: str):
    """Crea un file con il contenuto specificato."""
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"✅ Created: {path}")

def create_train_script():
    """Crea script di training unificato."""
    
    content = '''#!/usr/bin/env python3
"""
Unified training script for gender prediction models.

This script provides a unified interface for training all model types:
- Round 0: Basic models
- Round 1: Enhanced training with focal loss, balanced sampling
- Round 2: Enhanced architectures  
- Round 3: Advanced V3 models with feature engineering
"""

import argparse
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from gender_predict.models import create_model, MODEL_REGISTRY
from gender_predict.data import (
    NamePreprocessor, ImprovedNamePreprocessor,
    NameGenderDataset, ImprovedNameGenderDataset,
    NameFeatureExtractor, NameAugmenter
)
from gender_predict.training import (
    FocalLoss, FocalLossImproved, LabelSmoothing,
    BalancedBatchSampler, CosineAnnealingWarmupScheduler
)
from gender_predict.utils import EarlyStopping
from gender_predict.experiments import ExperimentManager

def setup_model_and_data(args):
    """Setup model and data based on round/configuration."""
    
    # Load data
    df = pd.read_csv(args.data_file)
    print(f"Loaded {len(df)} records")
    
    # Split data
    train_val_df, test_df = train_test_split(
        df, test_size=0.1, random_state=args.seed, stratify=df['gender']
    )
    train_df, val_df = train_test_split(
        train_val_df, test_size=0.1, random_state=args.seed, stratify=train_val_df['gender']
    )
    
    # Setup based on round
    if args.round >= 3:
        # V3 model with advanced features
        preprocessor = ImprovedNamePreprocessor() if args.advanced_preprocessing else NamePreprocessor()
        feature_extractor = NameFeatureExtractor()
        augmenter = NameAugmenter(args.augment_prob) if args.augment_prob > 0 else None
        
        train_dataset = ImprovedNameGenderDataset(
            train_df, preprocessor, feature_extractor, 
            mode='train', augmenter=augmenter, augment_prob=args.augment_prob
        )
        val_dataset = ImprovedNameGenderDataset(
            val_df, preprocessor, feature_extractor, mode='val'
        )
        test_dataset = ImprovedNameGenderDataset(
            test_df, preprocessor, feature_extractor, mode='test'
        )
        
        # V3 model
        model = create_model('v3',
            vocab_size=preprocessor.vocab_size,
            suffix_vocab_size=len(feature_extractor.suffix_to_idx),
            embedding_dim=args.embedding_dim,
            hidden_size=args.hidden_size,
            n_layers=args.n_layers,
            dropout_rate=args.dropout,
            num_attention_heads=args.num_heads
        )
        
    else:
        # Standard preprocessing and models
        preprocessor = NamePreprocessor()
        
        train_dataset = NameGenderDataset(train_df, preprocessor, mode='train')
        val_dataset = NameGenderDataset(val_df, preprocessor, mode='val')
        test_dataset = NameGenderDataset(test_df, preprocessor, mode='test')
        
        if args.round >= 2:
            # Enhanced model
            model = create_model('enhanced',
                vocab_size=preprocessor.vocab_size,
                embedding_dim=args.embedding_dim,
                hidden_size=args.hidden_size,
                n_layers=args.n_layers,
                dual_input=args.dual_input
            )
        else:
            # Base model
            model = create_model('base',
                vocab_size=preprocessor.vocab_size,
                embedding_dim=args.embedding_dim,
                hidden_size=args.hidden_size
            )
    
    return model, preprocessor, train_dataset, val_dataset, test_dataset

def setup_training_components(args, train_dataset):
    """Setup loss function, optimizer, scheduler, etc."""
    
    # Loss function
    if args.loss == 'focal':
        if args.round >= 3:
            criterion = FocalLossImproved(
                alpha=args.alpha, gamma=args.gamma, auto_weight=args.auto_weight
            )
        else:
            criterion = FocalLoss(alpha=args.alpha, gamma=args.gamma)
            
        if args.label_smooth > 0:
            criterion = LabelSmoothing(criterion, epsilon=args.label_smooth)
    else:
        pos_weight = torch.tensor(args.pos_weight) if args.pos_weight != 1.0 else None
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # DataLoader
    if args.balanced_sampler and args.round >= 1:
        sampler = BalancedBatchSampler(train_dataset, args.batch_size)
        train_loader = DataLoader(train_dataset, batch_sampler=sampler, num_workers=4)
    else:
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
        )
    
    return criterion, train_loader

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, 
                num_epochs, device, experiment, early_stopping=None):
    """Unified training loop."""
    
    model.to(device)
    best_val_f1 = 0
    history = {'train_loss': [], 'val_loss': [], 'val_f1': []}
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            # Handle different model input requirements
            if len(batch) > 3:  # V3 model
                outputs = model(
                    batch['first_name'].to(device),
                    batch['last_name'].to(device), 
                    batch['first_suffix'].to(device),
                    batch['last_suffix'].to(device),
                    batch['phonetic_features'].to(device)
                )
            else:  # Standard models
                outputs = model(
                    batch['first_name'].to(device),
                    batch['last_name'].to(device)
                )
            
            loss = criterion(outputs, batch['gender'].to(device))
            loss.backward()
            
            if hasattr(args, 'gradient_clip') and args.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            
            optimizer.step()
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_preds, val_targets = [], []
        
        with torch.no_grad():
            for batch in val_loader:
                if len(batch) > 3:  # V3 model
                    outputs = model(
                        batch['first_name'].to(device),
                        batch['last_name'].to(device),
                        batch['first_suffix'].to(device), 
                        batch['last_suffix'].to(device),
                        batch['phonetic_features'].to(device)
                    )
                else:  # Standard models
                    outputs = model(
                        batch['first_name'].to(device),
                        batch['last_name'].to(device)
                    )
                
                loss = criterion(outputs, batch['gender'].to(device))
                val_loss += loss.item()
                
                probs = torch.sigmoid(outputs)
                preds = (probs >= 0.5).long()
                
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(batch['gender'].cpu().numpy())
        
        # Calculate metrics
        from sklearn.metrics import precision_recall_fscore_support
        _, _, f1, _ = precision_recall_fscore_support(val_targets, val_preds, average='binary')
        
        # Update history
        history['train_loss'].append(train_loss / len(train_loader))
        history['val_loss'].append(val_loss / len(val_loader))
        history['val_f1'].append(f1)
        
        # Update scheduler
        if scheduler:
            if hasattr(scheduler, 'step'):
                if hasattr(args, 'round') and args.round >= 3:
                    scheduler.step(epoch)  # Custom scheduler
                else:
                    scheduler.step()
            
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss/len(train_loader):.4f}, "
              f"Val Loss: {val_loss/len(val_loader):.4f}, Val F1: {f1:.4f}")
        
        # Save best model
        if f1 > best_val_f1:
            best_val_f1 = f1
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'best_f1': best_val_f1,
                'vocab_size': model.vocab_size if hasattr(model, 'vocab_size') else len(preprocessor.char_to_idx)
            }
            experiment.save_model_checkpoint(checkpoint)
        
        # Early stopping
        if early_stopping and early_stopping(model, f1):
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    experiment.log_training_history(history)
    return history

def main():
    parser = argparse.ArgumentParser(description="Unified gender prediction training")
    
    # Basic parameters
    parser.add_argument('--round', type=int, default=0, choices=[0,1,2,3],
                       help='Training round (0=basic, 1=enhanced training, 2=enhanced arch, 3=v3)')
    parser.add_argument('--data_file', required=True, help='Training data CSV file')
    parser.add_argument('--save_dir', default='.', help='Directory to save results')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--note', default='', help='Experiment note')
    
    # Model architecture  
    parser.add_argument('--embedding_dim', type=int, default=16, help='Embedding dimension')
    parser.add_argument('--hidden_size', type=int, default=64, help='Hidden size')
    parser.add_argument('--n_layers', type=int, default=1, help='Number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--dual_input', action='store_true', help='Use dual input (enhanced models)')
    parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads (V3)')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--early_stop', type=int, default=0, help='Early stopping patience (0=disabled)')
    parser.add_argument('--gradient_clip', type=float, default=0, help='Gradient clipping (0=disabled)')
    
    # Loss and optimization
    parser.add_argument('--loss', choices=['bce', 'focal'], default='bce', help='Loss function')
    parser.add_argument('--alpha', type=float, default=0.5, help='Focal loss alpha')
    parser.add_argument('--gamma', type=float, default=2.0, help='Focal loss gamma')  
    parser.add_argument('--pos_weight', type=float, default=1.0, help='BCE positive class weight')
    parser.add_argument('--label_smooth', type=float, default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--balanced_sampler', action='store_true', help='Use balanced batch sampler')
    parser.add_argument('--auto_weight', action='store_true', help='Auto weight for focal loss (V3)')
    
    # Advanced features (V3)
    parser.add_argument('--advanced_preprocessing', action='store_true', 
                       help='Use improved preprocessor (V3)')
    parser.add_argument('--augment_prob', type=float, default=0.0, 
                       help='Data augmentation probability (V3)')
    
    args = parser.parse_args()
    
    # Initialize experiment manager
    experiment = ExperimentManager(args, base_dir=args.save_dir)
    print(f"Experiment ID: {experiment.experiment_id}")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Setup model and data
    model, preprocessor, train_dataset, val_dataset, test_dataset = setup_model_and_data(args)
    
    # Save preprocessor
    preprocessor.save(experiment.preprocessor_path)
    
    # Setup training components
    criterion, train_loader = setup_training_components(args, train_dataset)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Setup optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Setup scheduler
    scheduler = None
    if args.round >= 3:
        scheduler = CosineAnnealingWarmupScheduler(
            optimizer, warmup_epochs=3, max_epochs=args.epochs
        )
    
    # Setup early stopping
    early_stopping = EarlyStopping(patience=args.early_stop) if args.early_stop > 0 else None
    
    # Train model
    print("Starting training...")
    history = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        args.epochs, device, experiment, early_stopping
    )
    
    # Evaluate on test set
    print("Evaluating on test set...")
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    from gender_predict.evaluation import ModelEvaluator
    evaluator = ModelEvaluator(model, preprocessor, device)
    test_results = evaluator.evaluate_dataset(test_dataset)
    
    experiment.log_test_metrics(test_results)
    experiment.save_confusion_matrix(test_results['targets'], test_results['predictions'])
    
    # Generate report
    report_path = experiment.generate_report()
    
    print(f"\\n✅ Training completed!")
    print(f"   Experiment ID: {experiment.experiment_id}")
    print(f"   Test Accuracy: {test_results['accuracy']:.4f}")
    print(f"   Test F1: {test_results['f1']:.4f}")
    print(f"   Report: {report_path}")

if __name__ == "__main__":
    main()
'''
    
    create_file_with_content('scripts/train_model.py', content)

def create_evaluate_script():
    """Crea script di evaluation unificato."""
    
    content = '''#!/usr/bin/env python3
"""
Unified evaluation script for gender prediction models.
"""

import argparse
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import torch

from gender_predict.evaluation import ModelEvaluator
from gender_predict.data import NameGenderDataset

def main():
    parser = argparse.ArgumentParser(description="Evaluate gender prediction models")
    
    parser.add_argument('--model', required=True, help='Path to model checkpoint')
    parser.add_argument('--preprocessor', required=True, help='Path to preprocessor')
    parser.add_argument('--test_data', required=True, help='Path to test dataset CSV')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--device', default='cuda', help='Device to use')
    parser.add_argument('--output_dir', default='evaluation_results', help='Output directory')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load evaluator
    evaluator = ModelEvaluator.from_checkpoint(
        args.model, args.preprocessor, args.device
    )
    
    # Load test dataset
    test_df = pd.read_csv(args.test_data)
    test_dataset = NameGenderDataset(test_df, evaluator.preprocessor, mode='test')
    
    # Evaluate
    results = evaluator.evaluate_dataset(test_dataset, args.batch_size)
    
    # Bias analysis
    bias_results = evaluator.detailed_bias_analysis(
        results['targets'], results['predictions']
    )
    
    # Save results
    import json
    output_file = os.path.join(args.output_dir, 'evaluation_results.json')
    with open(output_file, 'w') as f:
        serializable_results = {
            'accuracy': float(results['accuracy']),
            'precision': float(results['precision']),
            'recall': float(results['recall']),
            'f1': float(results['f1']),
            'bias_ratio': float(bias_results['bias_ratio']),
            'male_error_rate': float(bias_results['male_error_rate']),
            'female_error_rate': float(bias_results['female_error_rate'])
        }
        json.dump(serializable_results, f, indent=2)
    
    print(f"✅ Evaluation completed. Results saved to {output_file}")
    print(f"   Accuracy: {results['accuracy']:.4f}")
    print(f"   F1 Score: {results['f1']:.4f}")
    print(f"   Bias Ratio: {bias_results['bias_ratio']:.4f}")

if __name__ == "__main__":
    main()
'''
    
    create_file_with_content('scripts/evaluate_model.py', content)

def create_experiment_tools_script():
    """Crea script per experiment comparison."""
    
    content = '''#!/usr/bin/env python3
"""
Experiment analysis and comparison tools.
"""

import argparse
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from gender_predict.experiments import (
    compare_experiments, compare_bias_metrics, generate_full_report, compare_learning_curves
)

def main():
    parser = argparse.ArgumentParser(description="Experiment analysis tools")
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare experiments')
    compare_parser.add_argument('--base_dir', default='.', help='Base directory')
    compare_parser.add_argument('--metric', default='test_accuracy', help='Metric to compare')
    compare_parser.add_argument('--round', type=int, help='Filter by round')
    compare_parser.add_argument('--output', help='Output file for plot')
    
    # Bias command
    bias_parser = subparsers.add_parser('bias', help='Compare bias metrics')
    bias_parser.add_argument('--base_dir', default='.', help='Base directory')
    bias_parser.add_argument('--round', type=int, help='Filter by round')
    bias_parser.add_argument('--output', help='Output file for plot')
    
    # Report command
    report_parser = subparsers.add_parser('report', help='Generate reports')
    report_parser.add_argument('--base_dir', default='.', help='Base directory')
    report_parser.add_argument('--output', help='Output file for report')
    
    args = parser.parse_args()
    
    if args.command == 'compare':
        filter_dict = {}
        if args.round is not None:
            filter_dict['round'] = args.round
        
        compare_experiments(
            base_dir=args.base_dir,
            filter_dict=filter_dict,
            metric=args.metric,
            save_path=args.output
        )
    
    elif args.command == 'bias':
        filter_dict = {}
        if args.round is not None:
            filter_dict['round'] = args.round
        
        compare_bias_metrics(
            base_dir=args.base_dir,
            filter_dict=filter_dict,
            save_path=args.output
        )
    
    elif args.command == 'report':
        generate_full_report(
            base_dir=args.base_dir,
            output_path=args.output
        )
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
'''
    
    create_file_with_content('scripts/experiment_tools.py', content)

def create_setup_py():
    """Crea setup.py per il package."""
    
    content = '''"""
Setup script for gender-predict package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="gender-predict",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Deep learning models for gender prediction from names",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/gender-predict",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest",
            "pytest-cov",
            "black",
            "flake8",
            "mypy",
        ],
    },
    scripts=[
        "scripts/train_model.py",
        "scripts/evaluate_model.py", 
        "scripts/experiment_tools.py",
        "scripts/prepare_data.py",
    ],
    entry_points={
        "console_scripts": [
            "gender-predict-train=scripts.train_model:main",
            "gender-predict-eval=scripts.evaluate_model:main",
            "gender-predict-tools=scripts.experiment_tools:main",
        ],
    },
)
'''
    
    create_file_with_content('setup.py', content)

def make_scripts_executable():
    """Rende gli script eseguibili."""
    import stat
    
    scripts = [
        'scripts/train_model.py',
        'scripts/evaluate_model.py',
        'scripts/experiment_tools.py',
        'scripts/prepare_data.py'
    ]
    
    for script in scripts:
        if os.path.exists(script):
            # Add execute permission
            st = os.stat(script)
            os.chmod(script, st.st_mode | stat.S_IEXEC)

def main():
    """Crea tutti gli script unificati."""
    
    print("🚀 STEP 4: Creazione script unificati")
    print("=" * 50)
    
    # Assicurati che scripts/ esista
    os.makedirs('scripts', exist_ok=True)
    
    print("\\n📝 Creando script unificati...")
    create_train_script()
    create_evaluate_script()
    create_experiment_tools_script()
    
    print("\\n📦 Creando setup.py...")
    create_setup_py()
    
    print("\\n🔧 Rendendo script eseguibili...")
    make_scripts_executable()
    
    print("\\n✅ STEP 4 COMPLETATO!")
    print("\\nFile creati:")
    print("  • scripts/train_model.py - Script di training unificato")
    print("  • scripts/evaluate_model.py - Script di evaluation unificato") 
    print("  • scripts/experiment_tools.py - Tools per confronto esperimenti")
    print("  • setup.py - Package setup")
    
    print("\\n🎯 Il package è pronto!")
    print("\\nPer testare:")
    print("  pip install -e .  # Installa in development mode")
    print("  python scripts/train_model.py --help")

if __name__ == "__main__":
    main()
