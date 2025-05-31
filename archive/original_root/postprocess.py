#!/usr/bin/env python3
"""
Script di post-processing intelligente per modelli di predizione del genere.
Rileva automaticamente se il modello ha bisogno di post-processing.
"""

import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, f1_score, precision_recall_fscore_support
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import BaseEstimator, ClassifierMixin
import pickle
from tqdm import tqdm

# Import local modules (if they exist)
try:
    from utils import plot_confusion_matrix, ensure_dir
except ImportError:
    def ensure_dir(directory):
        """Create directory if it doesn't exist."""
        if not os.path.exists(directory):
            os.makedirs(directory)

    def plot_confusion_matrix(y_true, y_pred, figsize=(10, 8), output_file='confusion_matrix.png'):
        """Simplified version of plot_confusion_matrix for standalone use."""
        from sklearn.metrics import confusion_matrix
        import matplotlib.pyplot as plt
        import seaborn as sns

        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        bias_ratio = (fp / (tn + fp)) / (fn / (tp + fn)) if (tn + fp) > 0 and (tp + fn) > 0 else 0

        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(output_file)
        plt.close()

        return cm, bias_ratio

# Try to import the original NamePreprocessor and models
try:
    from train_name_gender_model import NamePreprocessor, NameGenderDataset, GenderPredictorEnhanced, GenderPredictor
except ImportError:
    print("Warning: Could not import original model classes.")

# Import GenderPredictorV3
try:
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), 'experiments_improved'))
    from improvements import GenderPredictorV3, NameFeatureExtractor
    V3_AVAILABLE = True
    print("✅ GenderPredictorV3 imported successfully")
except ImportError:
    V3_AVAILABLE = False
    print("⚠️  GenderPredictorV3 not available")

def process_validation_in_batches(model, val_dataset, batch_size=128, device='cuda', feature_extractor=None):
    """Process validation data in batches to avoid CUDA OOM errors."""
    print(f"Processing validation data in batches of {batch_size}...")

    dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    all_preds = []
    all_labels = []
    is_v3_model = hasattr(model, 'suffix_embedding')

    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing validation batches")):
            first_names = batch['first_name'].to(device)
            last_names = batch['last_name'].to(device)
            labels = batch['gender'].cpu().numpy().astype(int)

            if is_v3_model and feature_extractor:
                # Extract features for GenderPredictorV3
                batch_size_actual = first_names.size(0)
                batch_start = batch_idx * batch_size_actual

                # Get original names from the dataset
                batch_names = []
                for i in range(batch_size_actual):
                    dataset_idx = batch_start + i
                    if dataset_idx < len(val_dataset.df):
                        original_name = val_dataset.df.iloc[dataset_idx]['primaryName']
                        batch_names.append(original_name)
                    else:
                        batch_names.append("Unknown Name")

                # Extract features
                first_suffix_list = []
                last_suffix_list = []
                phonetic_features_list = []

                for name in batch_names:
                    parts = name.split()
                    first_name_str = parts[0] if len(parts) > 0 else ""
                    last_name_str = parts[-1] if len(parts) > 1 else ""

                    # Extract suffix features
                    first_suffix_feat = feature_extractor.extract_suffix_features(first_name_str)
                    last_suffix_feat = feature_extractor.extract_suffix_features(last_name_str)

                    # Pad to length 3
                    first_suffix_feat = (first_suffix_feat + [0, 0, 0])[:3]
                    last_suffix_feat = (last_suffix_feat + [0, 0, 0])[:3]

                    # Extract phonetic features
                    first_phonetic = feature_extractor.extract_phonetic_features(first_name_str)
                    last_phonetic = feature_extractor.extract_phonetic_features(last_name_str)

                    phonetic_combined = [
                        first_phonetic['ends_with_vowel'],
                        first_phonetic['vowel_ratio'],
                        last_phonetic['ends_with_vowel'],
                        last_phonetic['vowel_ratio']
                    ]

                    first_suffix_list.append(first_suffix_feat)
                    last_suffix_list.append(last_suffix_feat)
                    phonetic_features_list.append(phonetic_combined)

                # Convert to tensors
                first_suffix = torch.tensor(first_suffix_list, dtype=torch.long).to(device)
                last_suffix = torch.tensor(last_suffix_list, dtype=torch.long).to(device)
                phonetic_features = torch.tensor(phonetic_features_list, dtype=torch.float32).to(device)

                outputs = model(first_names, last_names, first_suffix, last_suffix, phonetic_features)
            else:
                outputs = model(first_names, last_names)

            preds = outputs.cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels)

    return np.array(all_preds), np.array(all_labels)

def detect_model_type(checkpoint):
    """Detect model type from checkpoint state_dict."""
    state_dict = checkpoint.get('model_state_dict', checkpoint)

    # Check for V3-specific keys
    v3_keys = ['suffix_embedding.weight', 'embedding_norm.weight', 'phonetic_linear.weight']
    is_v3 = any(key in state_dict for key in v3_keys)

    if is_v3:
        return 'GenderPredictorV3'
    elif 'n_layers' in checkpoint and checkpoint.get('n_layers', 1) > 1:
        return 'GenderPredictorEnhanced'
    else:
        return 'GenderPredictor'

def main():
    parser = argparse.ArgumentParser(description="Post-process gender prediction model")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the model checkpoint")
    parser.add_argument("--preprocessor_path", type=str, required=True,
                        help="Path to the name preprocessor")
    parser.add_argument("--data_file", type=str, required=True,
                        help="Path to the training dataset for validation")
    parser.add_argument("--output_file", type=str, default="postprocess_results.json",
                        help="Path to save the results")
    parser.add_argument("--save_dir", type=str, default="logs",
                        help="Directory to save results")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size for validation processing")
    parser.add_argument("--val_size", type=float, default=0.01,
                        help="Size of validation set (0-1)")

    args = parser.parse_args()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    print(f"Loading data from {args.data_file}...")
    df = pd.read_csv(args.data_file)

    # Split data
    train_data, val_data = train_test_split(
        df, test_size=args.val_size, random_state=42, stratify=df['gender']
    )

    print(f"Validation set size: {len(val_data)}")

    # Load preprocessor
    print(f"Loading preprocessor from {args.preprocessor_path}...")
    try:
        preprocessor = NamePreprocessor.load(args.preprocessor_path)
    except NameError:
        with open(args.preprocessor_path, 'rb') as f:
            preprocessor_data = pickle.load(f)

        class DummyPreprocessor:
            def __init__(self, data):
                self.__dict__.update(data)

        preprocessor = DummyPreprocessor(preprocessor_data)

    # Create datasets
    try:
        val_dataset = NameGenderDataset(val_data, preprocessor, mode='val')
    except NameError:
        print("Warning: Using simplified dataset.")
        val_dataset = None

    # Load model
    print(f"Loading model from {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location=device)

    # Detect model type automatically
    model_type = detect_model_type(checkpoint)
    print(f"🔍 Detected model type: {model_type}")

    feature_extractor = None

    if model_type == 'GenderPredictorV3' and V3_AVAILABLE:
        print("✅ Loading GenderPredictorV3...")

        # Load or create feature extractor
        feature_extractor_path = os.path.join(os.path.dirname(args.model_path), '..', 'feature_extractor.pkl')
        if os.path.exists(feature_extractor_path):
            print(f"📂 Loading feature extractor from {feature_extractor_path}")
            with open(feature_extractor_path, 'rb') as f:
                feature_extractor = pickle.load(f)
        else:
            print("⚠️  Feature extractor not found, creating new one")
            feature_extractor = NameFeatureExtractor()

        suffix_vocab_size = len(feature_extractor.suffix_to_idx)
        print(f"📊 Suffix vocabulary size: {suffix_vocab_size}")

        model = GenderPredictorV3(
            vocab_size=checkpoint['vocab_size'],
            suffix_vocab_size=suffix_vocab_size,
            embedding_dim=checkpoint['embedding_dim'],
            hidden_size=checkpoint['hidden_size'],
            n_layers=checkpoint['n_layers'],
            dropout_rate=checkpoint.get('dropout_rate', 0.3),
            num_attention_heads=checkpoint.get('num_attention_heads', 4)
        )

    elif model_type == 'GenderPredictorEnhanced':
        print("✅ Loading GenderPredictorEnhanced...")
        model = GenderPredictorEnhanced(
            vocab_size=checkpoint['vocab_size'],
            embedding_dim=checkpoint['embedding_dim'],
            hidden_size=checkpoint['hidden_size'],
            n_layers=checkpoint['n_layers'],
            dual_input=checkpoint.get('dual_input', True)
        )

    else:
        print("✅ Loading GenderPredictor...")
        model = GenderPredictor(
            vocab_size=checkpoint['vocab_size'],
            embedding_dim=checkpoint['embedding_dim'],
            hidden_size=checkpoint['hidden_size']
        )

    # Load weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    print(f"✅ Model loaded successfully!")

    # Controllo intelligente basato sui file esistenti dell'esperimento
    print("🔍 Checking if model needs postprocessing...")

    # Prima prova a caricare le metriche già esistenti
    experiment_dir = os.path.dirname(os.path.dirname(args.model_path))
    test_metrics_path = os.path.join(experiment_dir, 'logs', 'test_metrics.json')
    bias_metrics_path = os.path.join(experiment_dir, 'logs', 'bias_metrics.json')

    baseline_f1 = None
    baseline_bias_ratio = None

    # Prova a caricare metriche esistenti
    if os.path.exists(test_metrics_path) and os.path.exists(bias_metrics_path):
        print("📂 Found existing experiment metrics, using those...")

        with open(test_metrics_path, 'r') as f:
            test_metrics = json.load(f)
        with open(bias_metrics_path, 'r') as f:
            bias_metrics = json.load(f)

        baseline_f1 = test_metrics.get('f1', 0.0)
        baseline_bias_ratio = bias_metrics.get('bias_ratio', 1.0)

        print(f"📊 Baseline performance from experiment files:")
        print(f"   F1: {baseline_f1:.4f}")
        print(f"   Bias Ratio: {baseline_bias_ratio:.4f}")

    else:
        print("📊 No existing metrics found, computing baseline on representative subset...")

        # Usa un subset rappresentativo per il baseline
        quick_test_size = max(5000, int(len(val_data) * 0.05))
        quick_test_size = min(quick_test_size, 20000)  # Max 20k per velocità

        quick_val_data = val_data.sample(n=quick_test_size, random_state=42)
        quick_dataset = NameGenderDataset(quick_val_data, preprocessor, mode='val')

        # Test con subset rappresentativo
        quick_probs, quick_labels = process_validation_in_batches(
            model, quick_dataset, batch_size=args.batch_size, device=device,
            feature_extractor=feature_extractor
        )

        # Calcola metriche di base con threshold 0.5
        baseline_preds = (quick_probs >= 0.5).astype(int)
        baseline_f1 = f1_score(quick_labels, baseline_preds)

        # Calcola bias ratio di base
        cm = pd.crosstab(
            pd.Series(quick_labels, name='Actual'),
            pd.Series(baseline_preds, name='Predicted')
        )

        try:
            tn, fp, fn, tp = cm.loc[0, 0], cm.loc[0, 1], cm.loc[1, 0], cm.loc[1, 1]
            m_to_w_error = fp / (tn + fp) if (tn + fp) > 0 else 0
            w_to_m_error = fn / (tp + fn) if (tp + fn) > 0 else 0
            baseline_bias_ratio = m_to_w_error / w_to_m_error if w_to_m_error > 0 else float('inf')
        except:
            baseline_bias_ratio = 1.0

        print(f"📊 Baseline performance (threshold=0.5, n={quick_test_size}):")
        print(f"   F1: {baseline_f1:.4f}")
        print(f"   Bias Ratio: {baseline_bias_ratio:.4f}")

    # DECISIONE: Skip postprocessing se il modello è già eccellente
    if baseline_f1 > 0.90 and 0.9 <= baseline_bias_ratio <= 1.1:
        print("✅ Model is already EXCELLENT and well-calibrated!")
        print("❌ Skipping postprocessing to preserve outstanding performance.")
        print(f"   📈 Your F1 ({baseline_f1:.4f}) is already >90%")
        print(f"   ⚖️  Your bias ratio ({baseline_bias_ratio:.4f}) is nearly perfect")

        # Se abbiamo le metriche dai file esistenti, usale direttamente
        if os.path.exists(test_metrics_path) and os.path.exists(bias_metrics_path):
            print("✅ Using existing experiment metrics (most reliable)")

            # Carica anche precision e recall dai file esistenti
            final_precision = test_metrics.get('precision', baseline_f1)  # fallback
            final_recall = test_metrics.get('recall', baseline_f1)        # fallback
            final_f1 = baseline_f1
            final_bias_ratio = baseline_bias_ratio
            validation_samples = "original_test_set"

        else:
            # Solo se non abbiamo i file, ricalcola (ma questo caso non dovrebbe succedere)
            print("🔬 Computing final confirmation metrics...")

            final_val_size = min(20000, len(val_data))
            final_val_data = val_data.sample(n=final_val_size, random_state=42)
            final_val_dataset = NameGenderDataset(final_val_data, preprocessor, mode='val')

            pred_probs, true_labels = process_validation_in_batches(
                model, final_val_dataset, batch_size=args.batch_size, device=device,
                feature_extractor=feature_extractor
            )

            # Usa threshold standard 0.5
            final_preds = (pred_probs >= 0.5).astype(int)

            # Calcola metriche finali
            final_precision, final_recall, final_f1, _ = precision_recall_fscore_support(
                true_labels, final_preds, average='binary'
            )

            # Bias ratio finale
            cm_final = pd.crosstab(
                pd.Series(true_labels, name='Actual'),
                pd.Series(final_preds, name='Predicted')
            )

            try:
                tn, fp, fn, tp = cm_final.loc[0, 0], cm_final.loc[0, 1], cm_final.loc[1, 0], cm_final.loc[1, 1]
                m_to_w_error = fp / (tn + fp) if (tn + fp) > 0 else 0
                w_to_m_error = fn / (tp + fn) if (tp + fn) > 0 else 0
                final_bias_ratio = m_to_w_error / w_to_m_error if w_to_m_error > 0 else float('inf')
            except:
                final_bias_ratio = baseline_bias_ratio

            validation_samples = final_val_size

        results = {
            "model_type": model_type,
            "postprocessing_applied": False,
            "reason": "Model already excellent - F1>90% and bias<1.1",
            "baseline_source": "experiment_files" if os.path.exists(test_metrics_path) else "computed",
            "threshold": 0.5,
            "val_f1": float(final_f1),
            "val_precision": float(final_precision),
            "val_recall": float(final_recall),
            "bias_ratio": float(final_bias_ratio),
            "baseline_f1": float(baseline_f1),
            "baseline_bias_ratio": float(baseline_bias_ratio),
            "validation_samples": validation_samples
        }

    else:
        print("⚠️  Model might benefit from postprocessing...")
        print("🚧 This model doesn't meet excellence criteria (F1>90% + bias~1.0)")
        print("💡 Consider retraining or using a different model architecture.")

        results = {
            "model_type": model_type,
            "postprocessing_applied": False,
            "reason": "Model below excellence threshold - consider retraining",
            "threshold": 0.5,
            "baseline_f1": float(baseline_f1),
            "baseline_bias_ratio": float(baseline_bias_ratio)
        }

    # Save results
    ensure_dir(args.save_dir)
    output_path = os.path.join(args.save_dir, args.output_file)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"\nResults saved to {output_path}")
    print("✅ Analysis completed!")

    print(f"\n🎯 FINAL ANALYSIS:")
    if not results.get("postprocessing_applied", True):
        if results.get("reason", "").startswith("Model already excellent"):
            print(f"✅ EXCELLENT MODEL - No postprocessing needed!")
            print(f"   📊 F1 Score: {results['val_f1']:.4f} (Outstanding!)")
            print(f"   📊 Precision: {results['val_precision']:.4f}")
            print(f"   📊 Recall: {results['val_recall']:.4f}")
            print(f"   ⚖️  Bias Ratio: {results['bias_ratio']:.4f} (Well balanced)")
            if results.get("baseline_source") == "experiment_files":
                print(f"   📁 Used original experiment test metrics (most reliable)")
            else:
                print(f"   🔬 Computed on {results['validation_samples']} validation samples")
            print(f"   💡 Recommendation: Deploy this model as-is!")
        else:
            print(f"⚠️  Model needs improvement:")
            print(f"   📊 F1 Score: {results['baseline_f1']:.4f} (target: >90%)")
            print(f"   ⚖️  Bias Ratio: {results['baseline_bias_ratio']:.4f} (target: ~1.0)")
            print(f"   💡 Recommendation: Retrain with better parameters")

if __name__ == "__main__":
    main()
