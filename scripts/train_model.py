#!/usr/bin/env python3
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
import joblib
import time
import copy
from tqdm import tqdm
# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import matplotlib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

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

BACKEND = "Agg"  # default sicuro
if os.environ.get("DISPLAY") and os.environ.get("XDG_SESSION_TYPE") == "wayland":
    try:
        import PySide6             # o PySide6
        BACKEND = "Qt6Agg"
    except ImportError:
        # se Qt non c'è, resta su Agg
        pass

matplotlib.use(BACKEND)
import matplotlib.pyplot as plt

def set_all_seeds(seed: int):
    """Set all random seeds for reproducibility."""
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # For deterministic behavior (might be slower)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"🎲 All seeds set to {seed}")

def make_json_serializable(obj):
    """Convert numpy types to Python types for JSON serialization."""
    import numpy as np
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

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

    return model, preprocessor, train_dataset, val_dataset, test_dataset, feature_extractor

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
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory
        )

    return criterion, train_loader

def set_embedding_trainable(model, trainable=True):
    """
    Congela o scongela i layer di embedding del modello.

    Args:
        model: Il modello
        trainable: True per rendere trainable, False per congelare
    """
    # Identifica i layer di embedding
    embedding_layers = []

    # Cerca tutti i layer di embedding nel modello
    for name, module in model.named_modules():
        if isinstance(module, nn.Embedding):
            embedding_layers.append((name, module))

    # Imposta requires_grad
    for name, layer in embedding_layers:
        for param in layer.parameters():
            param.requires_grad = trainable

    # Log dello stato
    status = "trainable" if trainable else "frozen"
    print(f"🧊 Embedding layers {status}: {[name for name, _ in embedding_layers]}")

    return len(embedding_layers)

def evaluate_with_tta(model, test_dataset, preprocessor, augmenter, device,
                      n_aug=5, experiment=None):
    """
    Valuta il modello usando Test Time Augmentation.

    Returns:
        Dict con metriche TTA
    """
    from gender_predict.evaluation.tta import TestTimeAugmentation
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    print("\n🔮 Evaluating with Test Time Augmentation...")

    # Crea TTA evaluator
    tta_evaluator = TestTimeAugmentation(
        model, preprocessor, augmenter, device
    )

    # Valuta con TTA
    tta_results = tta_evaluator.predict_dataset(
        test_dataset,
        n_aug=n_aug,
        show_progress=True
    )

    # Calcola metriche
    tta_acc = accuracy_score(tta_results['true_labels'], tta_results['predictions'])
    tta_precision, tta_recall, tta_f1, _ = precision_recall_fscore_support(
        tta_results['true_labels'],
        tta_results['predictions'],
        average='binary'
    )

    # Prepara risultati
    tta_metrics = {
        'tta_accuracy': float(tta_acc),
        'tta_precision': float(tta_precision),
        'tta_recall': float(tta_recall),
        'tta_f1': float(tta_f1),
        'tta_n_aug': n_aug,
        'tta_avg_confidence': float(np.mean(tta_results['confidences']))
    }

    return tta_metrics, tta_results

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
                num_epochs, device, experiment, early_stopping=None, gradient_clip=0, is_v3=False, freeze_epochs=0, use_mixup=False, mixup_alpha=0.2):
    """Enhanced training loop with comprehensive logging and analysis."""

    model.to(device)
    best_val_f1 = 0

    # Gestione freeze epochs
    embeddings_frozen = False
    if freeze_epochs > 0:
        print(f"\n❄️  Freezing embedding layers for first {freeze_epochs} epochs")
        num_frozen = set_embedding_trainable(model, trainable=False)
        embeddings_frozen = True

        # Ricrea l'optimizer per escludere i parametri frozen
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = optim.AdamW(trainable_params, lr=optimizer.param_groups[0]['lr'],
                               weight_decay=1e-5)

    # Complete history tracking like the original
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'val_precision': [],
        'val_recall': [],
        'val_f1': [],
        'learning_rate': []
    }

    # Progress bar per le epoche
    epoch_pbar = tqdm(range(num_epochs), desc="Training Epochs")

    for epoch in epoch_pbar:
        start_time = time.time()

        # Scongela embeddings dopo freeze_epochs
        if embeddings_frozen and epoch == freeze_epochs:
            print(f"\n🔥 Unfreezing embedding layers at epoch {epoch+1}")
            set_embedding_trainable(model, trainable=True)
            embeddings_frozen = False

            # Ricrea optimizer con tutti i parametri
            optimizer = optim.AdamW(model.parameters(), lr=scheduler.optimizer.param_groups[0]['lr'],
                                   weight_decay=1e-5)

            # Riassegna optimizer allo scheduler se presente
            if scheduler:
                scheduler.optimizer = optimizer

        # Update learning rate first (for V3 scheduler)
        current_lr = scheduler.step(epoch) if scheduler else optimizer.param_groups[0]['lr']
        history['learning_rate'].append(current_lr)

        # Training phase
        model.train()
        train_loss = 0.0
        train_preds = []
        train_targets = []

        # Progress bar per i batch di training
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False)

        for batch in train_pbar:
            optimizer.zero_grad()
            if use_mixup and epoch >= freeze_epochs:
                from gender_predict.training.schedulers import mixup_data

                # Get targets
                targets = batch['gender'].float().to(device)

                # Per testo, usiamo mixup solo sui target
                _, targets_a, targets_b, lam, _ = mixup_data(
                    batch['first_name'].to(device),  # Non viene usato per interpolazione
                    targets,
                    alpha=mixup_alpha,
                    is_embedding_idx=True  # Importante!
                )

                # Forward pass normale (NO mixup sugli input)
                if is_v3:
                    outputs = model(
                        batch['first_name'].to(device),
                        batch['last_name'].to(device),
                        batch['first_suffix'].to(device),
                        batch['last_suffix'].to(device),
                        batch['phonetic_features'].to(device)
                    )
                else:
                    outputs = model(
                        batch['first_name'].to(device),
                        batch['last_name'].to(device)
                    )

                # Mixup loss
                loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
            else:
                # Training normale (codice esistente)
                if is_v3:
                    outputs = model(
                        batch['first_name'].to(device),
                        batch['last_name'].to(device),
                        batch['first_suffix'].to(device),
                        batch['last_suffix'].to(device),
                        batch['phonetic_features'].to(device)
                    )
                else:
                    outputs = model(
                        batch['first_name'].to(device),
                        batch['last_name'].to(device)
                    )
                loss = criterion(outputs, batch['gender'].to(device))

            loss.backward()

            # Gradient clipping
            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

            optimizer.step()

            train_loss += loss.item() * batch['first_name'].size(0)

            # Collect predictions for training accuracy
            probs = torch.sigmoid(outputs)
            preds = (probs >= 0.5).long()

            train_preds.extend(preds.cpu().numpy())
            train_targets.extend(batch['gender'].cpu().numpy())

            # Update progress bar with loss and lr
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}', 'lr': f'{current_lr:.6f}'})

        train_loss /= len(train_loader.dataset)
        train_acc = accuracy_score(train_targets, train_preds)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []
        val_probs  = []
        val_labels = []

        # Progress bar per validation
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", leave=False)

        with torch.no_grad():
            for batch in val_pbar:
                if 'first_suffix' in batch:  # V3 model
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
                val_loss += loss.item() * batch['first_name'].size(0)

                probs = torch.sigmoid(outputs)
                preds = (probs >= 0.5).long()

                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(batch['gender'].cpu().numpy())
                val_probs.extend(probs.squeeze().cpu().numpy())
                val_labels.extend(batch['gender'].cpu().numpy())

                # Update validation progress bar
                val_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        val_loss /= len(val_loader.dataset)
        val_acc = accuracy_score(val_targets, val_preds)

        # Calculate detailed metrics
        precision, recall, f1, _ = precision_recall_fscore_support(val_targets, val_preds, average='binary')

        # Update complete history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['val_precision'].append(precision)
        history['val_recall'].append(recall)
        history['val_f1'].append(f1)

        # Calculate epoch time
        time_elapsed = time.time() - start_time

        # Enhanced epoch summary
        print(f"\nEpoch {epoch+1}/{num_epochs} | Time: {time_elapsed:.2f}s | LR: {current_lr:.6f}")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        print(f"Val Precision: {precision:.4f} | Val Recall: {recall:.4f} | Val F1: {f1:.4f}")

        # Update epoch progress bar
        epoch_pbar.set_postfix({
            'Train Loss': f'{train_loss:.4f}',
            'Val Loss': f'{val_loss:.4f}',
            'Val F1': f'{f1:.4f}',
            'LR': f'{current_lr:.6f}'
        })

        if epoch == num_epochs - 1:            # solo all'ultima epoca
            import pandas as pd, os
            df_val = pd.DataFrame({
                "prob":  val_probs,
                "label": val_labels
            })
            save_dir = os.path.join(experiment.dir, "analysis")
            os.makedirs(save_dir, exist_ok=True)
            csv_path = os.path.join(save_dir, "val_probs_labels.csv")
            df_val.to_csv(csv_path, index=False)
            print(f"📝 Validation probabilities salvate in {csv_path}")

        # Save best model
        if f1 > best_val_f1:
            best_val_f1 = f1
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'best_f1': best_val_f1,
                'vocab_size': getattr(model, 'vocab_size', 100),
                'embedding_dim': getattr(model, 'embedding_dim', 16),
                'hidden_size': getattr(model, 'hidden_size', 64),
                'n_layers': getattr(model, 'n_layers', 1),
                'dual_input': getattr(model, 'dual_input', True)
            }

            # Add V3-specific parameters
            if is_v3:
                checkpoint.update({
                    'suffix_vocab_size': getattr(model, 'suffix_vocab_size', 50),
                    'num_attention_heads': getattr(model, 'num_attention_heads', 4)
                })

            experiment.save_model_checkpoint(checkpoint)
            print(f"💾 Nuovo miglior modello salvato! (F1: {best_val_f1:.4f})")

        # Periodic bias analysis every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
            print("\nAnalisi del bias sul set di validazione:")
            try:
                # Try to import plot_confusion_matrix from utils
                from gender_predict.utils.common import plot_confusion_matrix
                cm_path = experiment.get_plot_path(f'confusion_matrix_epoch_{epoch+1}')
                plot_confusion_matrix(val_targets, val_preds, output_file=cm_path)
                print(f"Confusion matrix salvata: {cm_path}")
            except ImportError:
                # Fallback: just save via experiment manager
                experiment.save_confusion_matrix(val_targets, val_preds)
                print("Confusion matrix salvata tramite ExperimentManager")

        # Early stopping
        if early_stopping and early_stopping(model, f1):
            print(f"⏰ Early stopping triggered after epoch {epoch+1}")
            break

        print("-" * 60)

    epoch_pbar.close()
    experiment.log_training_history(history)

    print(f"\n🎯 Training completato! Miglior F1: {best_val_f1:.4f}")
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
    parser.add_argument('--use_mixup', action='store_true',
                        help='Use mixup data augmentation')
    parser.add_argument('--mixup_alpha', type=float, default=0.2,
                        help='Mixup interpolation coefficient')
    parser.add_argument('--find_lr', action='store_true',
                        help='Run learning rate finder before training')
    parser.add_argument('--lr_finder_iters', type=int, default=100,
                        help='Number of iterations for LR finder')
    # DataLoader optimization parameters
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers (use 0 for Windows, 8 for Linux with good CPU)')
    parser.add_argument('--pin_memory', action='store_true',
                        help='Pin memory for faster GPU transfer (recommended for CUDA)')

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

    # Test Time Augmentation parameters
    parser.add_argument('--use_tta', action='store_true',
                        help='Use Test Time Augmentation for evaluation')
    parser.add_argument('--tta_n_aug', type=int, default=5,
                        help='Number of augmentations for TTA')
    parser.add_argument('--tta_strategy', choices=['basic', 'smart'],
                        default='basic')
    parser.add_argument('--tta_min_aug', type=int, default=3)
    parser.add_argument('--tta_max_aug', type=int, default=10)
    parser.add_argument('--tta_std',     type=float, default=0.15)

    # Advanced scheduler parameters
    parser.add_argument('--min_lr', type=float, default=1e-7, help='Minimum learning rate for scheduler')
    parser.add_argument('--warmup_epochs', type=int, default=3, help='Warmup epochs for scheduler')

    # Advanced training parameters
    parser.add_argument('--freeze_epochs', type=int, default=0, help='Epochs to freeze embedding layers')

    # Error analysis
    parser.add_argument('--enable_error_analysis', action='store_true', help='Enable error analysis during training')

    args = parser.parse_args()

    set_all_seeds(args.seed)

    # Initialize experiment manager
    experiment = ExperimentManager(args, base_dir=args.save_dir)
    print(f"Experiment ID: {experiment.experiment_id}")

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Setup model and data
    model, preprocessor, train_dataset, val_dataset, test_dataset, feature_extractor = setup_model_and_data(args)
    # Save preprocessor
    preprocessor.save(experiment.preprocessor_path)
    if feature_extractor is not None:
        joblib.dump(
            feature_extractor,
            os.path.join(experiment.experiment_dir, "feature_extractor.pkl")
        )
        print(f"💾 feature_extractor salvato in {experiment.experiment_dir}")



    # Setup training components
    criterion, train_loader = setup_training_components(args, train_dataset)
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory
    )

    # Setup optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)

    # Setup scheduler
    scheduler = None
    if args.round >= 3:
        scheduler = CosineAnnealingWarmupScheduler(
            optimizer,
            warmup_epochs=args.warmup_epochs,
            max_epochs=args.epochs,
            min_lr=args.min_lr
        )

    # Setup early stopping
    early_stopping = EarlyStopping(
        patience=args.early_stop,
        min_delta=0.001,          # ← AGGIUNTO: threshold più stringente
        restore_best_weights=True
    ) if args.early_stop > 0 else None

        # Learning Rate Finder (opzionale)
    if args.find_lr:
        print("\n🔍 Running Learning Rate Finder...")
        from gender_predict.utils.lr_finder import find_optimal_lr

        # Crea una copia del modello per non alterare i pesi
        model_copy = copy.deepcopy(model).to(device)

        # Trova LR ottimale
        optimal_lr = find_optimal_lr(
            model_copy,
            train_loader,
            criterion,
            device,
            start_lr=1e-7,
            end_lr=10,
            num_iter=args.lr_finder_iters
        )

        # Salva plot
        lr_plot_path = experiment.get_plot_path('lr_finder')
        plt.savefig(lr_plot_path)
        plt.close()

        print(f"\n💡 Suggested learning rate: {optimal_lr:.2e}")
        print(f"   Current --lr setting: {args.lr:.2e}")

        # Chiedi se usare il nuovo LR
        response = input("\nUse suggested learning rate? (y/n): ").lower()
        if response == 'y':
            args.lr = optimal_lr / 10  # Usa 1/10 del suggerito per sicurezza
            print(f"✅ Using new learning rate: {args.lr:.2e}")

            # Ricrea optimizer con nuovo LR
            optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)

    # Train model
    print("Starting training...")
    history = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        args.epochs, device, experiment, early_stopping,
        gradient_clip=args.gradient_clip,
        is_v3=(args.round >= 3),
        freeze_epochs=args.freeze_epochs,
        use_mixup=args.use_mixup,
        mixup_alpha=args.mixup_alpha
    )


    print(f"\n🔄 Loading best model checkpoint...")
    checkpoint = torch.load(experiment.model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    best_f1_loaded = checkpoint.get('best_f1', 'unknown')
    print(f"✅ Using model with F1: {best_f1_loaded}")

    # Evaluate on test set
    print("Evaluating on test set...")
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory
    )

    from gender_predict.evaluation import ModelEvaluator
    evaluator = ModelEvaluator(model, preprocessor, device)
    test_results = evaluator.evaluate_dataset(test_dataset)

    # Filtra solo le metriche scalari per il log
    scalar_metrics = {
        'accuracy': float(test_results['accuracy']),
        'precision': float(test_results['precision']),
        'recall': float(test_results['recall']),
        'f1': float(test_results['f1'])
    }
    experiment.log_test_metrics(scalar_metrics)
    experiment.save_confusion_matrix(test_results['targets'], test_results['predictions'])

    # Dopo la valutazione standard
    if args.use_tta:
        from gender_predict.evaluation.tta import TestTimeAugmentation, SmartTTA
        from gender_predict.data import NameAugmenter
        import pickle

        print("\n🔮  Test-Time Augmentation attiva")

        augmenter = NameAugmenter(augment_prob=0.0)  # trasformazioni label-safe

        # carica il feature_extractor se serve (round 3)
        feature_path = os.path.join(experiment.dir, "feature_extractor.pkl")
        feature_extractor = None
        if os.path.exists(feature_path):
            with open(feature_path, "rb") as f:
                feature_extractor = pickle.load(f)

        # costrutto basic vs smart
        if args.tta_strategy == "smart":
            tta = SmartTTA(
                model, preprocessor, feature_extractor, device,
                min_aug=args.tta_min_aug,
                max_aug=args.tta_max_aug,
                std_thresh=args.tta_std
            )
            print(f"🛠 Smart TTA | min={args.tta_min_aug}, "
                f"max={args.tta_max_aug}, std={args.tta_std}")
        else:
            tta = TestTimeAugmentation(
                model, preprocessor, feature_extractor, device,
                n_aug=args.tta_n_aug
            )
            print(f"🛠 Basic TTA | n_aug={args.tta_n_aug}")

        # → valutazione con TTA
        tta_metrics, _ = evaluate_with_tta(
            model, test_dataset, preprocessor, augmenter,
            device, experiment=experiment, feature_extractor=feature_extractor
        )
        scalar_metrics.update(tta_metrics)

    else:
        print("ℹ️  TTA disabilitata")


    # Error Analysis (se abilitata)
    ERROR_ANALYSIS_AVAILABLE = False
    if args.enable_error_analysis:
        # Prima prova il path standard
        try:
            from gender_predict.evaluation.error_analysis import add_error_analysis_to_training
            ERROR_ANALYSIS_AVAILABLE = True
        except ImportError:
            # Prova percorso alternativo (per compatibilità con vecchi script)
            try:
                import sys
                sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                from error_analysis_tool import add_error_analysis_to_training
                ERROR_ANALYSIS_AVAILABLE = True
            except ImportError:
                print("⚠️  Error analysis tool non disponibile. Funzionalità disabilitata.")

        if ERROR_ANALYSIS_AVAILABLE:
            print("\n🔍 === ANALISI DETTAGLIATA DEGLI ERRORI ===")
            try:
                # Esegui analisi completa
                predictions_df, analysis_results, error_summary = add_error_analysis_to_training(
                    experiment,
                    model,
                    test_dataset,
                    preprocessor,
                    device
                )

                # Stampa insights principali
                print(f"\n💡 KEY INSIGHTS:")
                print(f"   • {error_summary['total_errors']} errori su {analysis_results['total_samples']} campioni ({error_summary['error_rate']:.2%})")
                print(f"   • Accuratezza: {error_summary['accuracy']:.2%}")
                print(f"   • Errori ad alta confidenza: {error_summary['high_confidence_errors']} ({error_summary['high_confidence_error_rate']:.1%})")

                if error_summary['most_problematic_names']:
                    print(f"   • Nomi più problematici: {', '.join(error_summary['most_problematic_names'])}")

                print(f"\n📁 File generati:")
                print(f"   • {experiment.logs_dir}/error_analysis.csv")
                print(f"   • {experiment.logs_dir}/error_analysis_results.json")
                print(f"   • {experiment.logs_dir}/error_summary.json")
                print(f"   • {experiment.plots_dir}/error_analysis.png")

            except Exception as e:
                print(f"⚠️  Errore durante l'analisi: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("❌ Modulo error analysis non trovato in nessun path.")
            print("   Paths provati:")
            print("   - gender_predict.evaluation.error_analysis")
            print("   - error_analysis_tool (parent directory)")
    else:
        print("ℹ️  Analisi errori disabilitata (usa --enable_error_analysis per abilitarla)")

    # Generate report
    report_path = experiment.generate_report()

    print(f"\n✅ Training completed!")
    print(f"   Experiment ID: {experiment.experiment_id}")
    print(f"   Test Accuracy: {test_results['accuracy']:.4f}")
    print(f"   Test F1: {test_results['f1']:.4f}")
    print(f"   Report: {report_path}")

if __name__ == "__main__":
    main()
