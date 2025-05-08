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
from tqdm import tqdm  # Aggiunto per le progress bars

# Import local modules (if they exist)
try:
    from utils import plot_confusion_matrix, ensure_dir
except ImportError:
    # Simplified implementations for standalone use
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

        # Calculate bias ratio
        bias_ratio = (fp / (tn + fp)) / (fn / (tp + fn)) if (tn + fp) > 0 and (tp + fn) > 0 else 0

        # Create a simple confusion matrix plot
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
    print("Warning: Could not import original model classes. Using simplified wrapper.")
    # This will not execute the code but allows the script to load from pickled files

class TorchModelWrapper(BaseEstimator, ClassifierMixin):
    """
    Wrapper for PyTorch models to use with scikit-learn's CalibratedClassifierCV.

    This wrapper allows PyTorch models to be used with scikit-learn's probability
    calibration methods like Platt scaling.
    """
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.classes_ = np.array([0, 1])  # Binary classification

    def fit(self, X, y):
        """
        Dummy fit method (model is already trained).

        Args:
            X: Feature data (first_name, last_name tuples)
            y: Target labels
        """
        return self

    def predict_proba(self, X):
        """
        Predict probabilities using the wrapped PyTorch model.

        Args:
            X: Feature data (first_name, last_name tuples)

        Returns:
            Array with probabilities for both classes [p(y=0), p(y=1)]
        """
        self.model.eval()
        with torch.no_grad():
            # X is a tuple of (first_name_batch, last_name_batch)
            first_name_batch, last_name_batch = X

            # Move to device
            first_name_batch = first_name_batch.to(self.device)
            last_name_batch = last_name_batch.to(self.device)

            # Get model predictions
            preds = self.model(first_name_batch, last_name_batch).cpu().numpy()

            # Return probabilities for both classes
            return np.vstack([1 - preds, preds]).T

    def predict(self, X):
        """
        Predict class labels using the wrapped PyTorch model.

        Args:
            X: Feature data (first_name, last_name tuples)

        Returns:
            Predicted class labels (0 or 1)
        """
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

def grid_search_threshold(y_true, y_scores, min_thresh=0.40, max_thresh=0.60, step=0.005):
    """
    Find optimal classification threshold that maximizes F1 score.

    Args:
        y_true: True binary labels
        y_scores: Predicted probabilities
        min_thresh: Minimum threshold to consider
        max_thresh: Maximum threshold to consider
        step: Step size for threshold grid

    Returns:
        best_threshold: Threshold that maximizes F1 score
        metrics: Dictionary with metrics for the best threshold
    """
    print("Performing threshold grid search...")

    thresholds = np.arange(min_thresh, max_thresh + step, step)
    best_f1 = 0
    best_threshold = 0.5  # Default threshold
    best_metrics = {}

    results = []

    for threshold in thresholds:
        # Apply threshold
        y_pred = (y_scores >= threshold).astype(int)

        # Calculate metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='binary'
        )

        # Create confusion matrix
        cm = pd.crosstab(
            pd.Series(y_true, name='Actual'),
            pd.Series(y_pred, name='Predicted')
        )

        # Calculate bias metrics
        try:
            tn = cm.loc[0, 0]
            fp = cm.loc[0, 1]
            fn = cm.loc[1, 0]
            tp = cm.loc[1, 1]

            # Bias ratio (ratio of error rates)
            m_to_w_error = fp / (tn + fp) if (tn + fp) > 0 else 0
            w_to_m_error = fn / (tp + fn) if (tp + fn) > 0 else 0
            bias_ratio = m_to_w_error / w_to_m_error if w_to_m_error > 0 else float('inf')
        except KeyError:
            # Handle case where confusion matrix is missing a class
            bias_ratio = float('inf')

        results.append({
            'threshold': threshold,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'bias_ratio': bias_ratio
        })

        # Update best threshold if F1 improves
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_metrics = {
                'val_precision': precision,
                'val_recall': recall,
                'val_f1': f1,
                'bias_ratio': bias_ratio
            }

    # Convert results to DataFrame for analysis
    results_df = pd.DataFrame(results)

    # Plot results
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.plot(results_df['threshold'], results_df['precision'], label='Precision')
    plt.plot(results_df['threshold'], results_df['recall'], label='Recall')
    plt.plot(results_df['threshold'], results_df['f1'], label='F1', linewidth=2)
    plt.axvline(x=best_threshold, color='r', linestyle='--', label=f'Best Threshold: {best_threshold:.3f}')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Precision, Recall, and F1 vs. Threshold')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(results_df['threshold'], results_df['bias_ratio'], label='Bias Ratio')
    plt.axhline(y=1.0, color='g', linestyle='--', label='Ideal Ratio (1.0)')
    plt.axvline(x=best_threshold, color='r', linestyle='--', label=f'Best Threshold: {best_threshold:.3f}')
    plt.xlabel('Threshold')
    plt.ylabel('Bias Ratio (M→W / W→M)')
    plt.title('Bias Ratio vs. Threshold')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('threshold_analysis.png')

    print(f"Best threshold: {best_threshold:.3f}")
    print(f"Metrics at best threshold:")
    for k, v in best_metrics.items():
        print(f"  {k}: {v:.3f}")

    return best_threshold, best_metrics, results_df

def calibrate_probabilities(model_wrapper, X_train, y_train, method='sigmoid'):
    """
    Calibrate model probabilities using Platt scaling.

    Args:
        model_wrapper: Wrapped PyTorch model
        X_train: Training features tuple (first_name, last_name)
        y_train: Training labels
        method: Calibration method ('sigmoid' or 'isotonic')

    Returns:
        Calibrated classifier
    """
    print(f"Calibrating probabilities using {method} method...")

    # Initialize calibrated classifier
    calibrated_model = CalibratedClassifierCV(
        base_estimator=model_wrapper,
        method=method,
        cv='prefit'  # Use the pretrained model without cross-validation
    )

    # Fit calibration on validation data
    calibrated_model.fit(X_train, y_train)

    return calibrated_model

def apply_equalized_odds_postprocessing(y_true, y_pred, sensitive_attr, threshold_adjustment=True):
    """
    Apply a simplified version of equalized odds post-processing.

    This is a simplified implementation that adjusts thresholds for different groups
    to achieve more equal false positive and false negative rates.

    Args:
        y_true: True labels
        y_pred: Predicted probabilities
        sensitive_attr: Sensitive attribute (usually same as y_true for gender prediction)
        threshold_adjustment: Whether to adjust thresholds or not

    Returns:
        y_fair: Fair predictions after post-processing
        thresholds: Dictionary of thresholds for each group
    """
    print("Applying equalized odds post-processing...")

    if not threshold_adjustment:
        return (y_pred >= 0.5).astype(int), {'all': 0.5}

    # Group indices
    group_0_idx = (sensitive_attr == 0)
    group_1_idx = (sensitive_attr == 1)

    # Find optimal thresholds for each group
    _, metrics_0, df_0 = grid_search_threshold(
        y_true[group_0_idx], y_pred[group_0_idx],
        min_thresh=0.3, max_thresh=0.7, step=0.01
    )

    _, metrics_1, df_1 = grid_search_threshold(
        y_true[group_1_idx], y_pred[group_1_idx],
        min_thresh=0.3, max_thresh=0.7, step=0.01
    )

    # Find thresholds that minimize disparity in error rates
    # (simplified approach - in practice, would use more sophisticated methods)

    # Calculate disparity metric for all threshold combinations
    best_disparity = float('inf')
    best_thresholds = (0.5, 0.5)

    for t0 in df_0['threshold'].values:
        for t1 in df_1['threshold'].values:
            # Apply thresholds
            y_pred_0 = (y_pred[group_0_idx] >= t0).astype(int)
            y_pred_1 = (y_pred[group_1_idx] >= t1).astype(int)

            # Calculate error rates
            error_0 = 1 - (y_pred_0 == y_true[group_0_idx]).mean()
            error_1 = 1 - (y_pred_1 == y_true[group_1_idx]).mean()

            # Calculate disparity
            disparity = abs(error_0 - error_1)

            if disparity < best_disparity:
                best_disparity = disparity
                best_thresholds = (t0, t1)

    # Apply the best thresholds
    y_fair = np.zeros_like(y_pred, dtype=int)
    y_fair[group_0_idx] = (y_pred[group_0_idx] >= best_thresholds[0]).astype(int)
    y_fair[group_1_idx] = (y_pred[group_1_idx] >= best_thresholds[1]).astype(int)

    thresholds = {
        'group_0': best_thresholds[0],
        'group_1': best_thresholds[1]
    }

    print(f"Equalized odds thresholds: {thresholds}")
    print(f"Disparity reduction: {best_disparity:.4f}")

    return y_fair, thresholds

def process_validation_in_batches(model, val_dataset, batch_size=128, device='cuda'):
    """
    Process validation data in batches to avoid CUDA OOM errors.

    Args:
        model: PyTorch model
        val_dataset: Validation dataset
        batch_size: Batch size to use
        device: Device to run model on

    Returns:
        Predictions and true labels
    """
    print(f"Processing validation data in batches of {batch_size}...")

    # Create DataLoader to handle batching
    dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Lists to store results
    all_preds = []
    all_labels = []

    # Process batches
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Processing validation batches"):
            # Get data from batch
            first_names = batch['first_name'].to(device)
            last_names = batch['last_name'].to(device)
            labels = batch['gender'].cpu().numpy().astype(int)

            # Run model
            outputs = model(first_names, last_names)
            preds = outputs.cpu().numpy()

            # Store results
            all_preds.extend(preds)
            all_labels.extend(labels)

    return np.array(all_preds), np.array(all_labels)

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Post-process gender prediction model")
    parser.add_argument("--model_path", type=str, default="models/round2_best.pth",
                        help="Path to the model checkpoint")
    parser.add_argument("--preprocessor_path", type=str, default="name_preprocessor.pkl",
                        help="Path to the name preprocessor")
    parser.add_argument("--data_file", type=str, default="training_dataset.csv",
                        help="Path to the training dataset for validation")
    parser.add_argument("--output_file", type=str, default="results_round3.json",
                        help="Path to save the results")
    parser.add_argument("--save_dir", type=str, default="logs",
                        help="Directory per salvare i risultati")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size for validation processing")
    parser.add_argument("--val_size", type=float, default=0.1,
                        help="Size of validation set (0-1)")
    parser.add_argument("--apply_calibration", action="store_true",
                        help="Apply Platt scaling for probability calibration")
    parser.add_argument("--apply_equalized_odds", action="store_true",
                        help="Apply equalized odds post-processing")

    args = parser.parse_args()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    print(f"Loading data from {args.data_file}...")
    df = pd.read_csv(args.data_file)

    # Split data
    # Ottieni solo una piccola parte del dataset per la validazione
    train_data, val_data = train_test_split(
        df, test_size=args.val_size, random_state=42, stratify=df['gender']
    )

    print(f"Validation set size: {len(val_data)}")

    # Load preprocessor
    print(f"Loading preprocessor from {args.preprocessor_path}...")
    try:
        preprocessor = NamePreprocessor.load(args.preprocessor_path)
    except NameError:
        # Fallback if NamePreprocessor is not imported
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
        # Simplified dataset creation if NameGenderDataset is not imported
        print("Warning: Using simplified dataset. Post-processing might be limited.")
        val_dataset = None

    # Load model
    print(f"Loading model from {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location=device)

    # Determine model type
    if 'n_layers' in checkpoint and checkpoint.get('n_layers', 1) > 1:
        model = GenderPredictorEnhanced(
            vocab_size=checkpoint['vocab_size'],
            embedding_dim=checkpoint['embedding_dim'],
            hidden_size=checkpoint['hidden_size'],
            n_layers=checkpoint['n_layers'],
            dual_input=checkpoint.get('dual_input', True)
        )
    else:
        model = GenderPredictor(
            vocab_size=checkpoint['vocab_size'],
            embedding_dim=checkpoint['embedding_dim'],
            hidden_size=checkpoint['hidden_size']
        )

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # Process validation data
    if val_dataset is not None:
        print("Processing validation data...")
        # Modifica principale: processare i dati in batch
        pred_probs, true_labels = process_validation_in_batches(
            model, val_dataset, batch_size=args.batch_size, device=device
        )

        # Find optimal threshold
        best_threshold, threshold_metrics, _ = grid_search_threshold(
            true_labels, pred_probs, min_thresh=0.40, max_thresh=0.60, step=0.005
        )

        # Create results dictionary
        results = {
            "best_threshold": float(best_threshold),
            "val_f1": float(threshold_metrics['val_f1']),
            "val_precision": float(threshold_metrics['val_precision']),
            "val_recall": float(threshold_metrics['val_recall']),
            "bias_ratio": float(threshold_metrics['bias_ratio'])
        }

        # Apply calibration if requested
        if args.apply_calibration:
            print("Applying probability calibration...")

            # Create model wrapper
            model_wrapper = TorchModelWrapper(model, device)

            # Prepare data for calibration
            # Importante: per la calibrazione dovremmo usare un subset più piccolo
            # per evitare problemi di memoria
            calibration_size = min(10000, len(true_labels))
            indices = np.random.choice(len(true_labels), calibration_size, replace=False)

            # Crea mini-batch per la calibrazione
            first_names_cal = torch.stack([val_dataset[i]['first_name'] for i in indices])
            last_names_cal = torch.stack([val_dataset[i]['last_name'] for i in indices])
            labels_cal = true_labels[indices]

            X_train = (first_names_cal, last_names_cal)
            y_train = labels_cal

            # Calibrate model
            calibrated_model = calibrate_probabilities(model_wrapper, X_train, y_train)

            # Get calibrated predictions (process in batches)
            calibrated_probs = []
            batch_size = 512

            for i in range(0, len(true_labels), batch_size):
                end_idx = min(i + batch_size, len(true_labels))
                batch_indices = list(range(i, end_idx))

                batch_first_names = torch.stack([val_dataset[j]['first_name'] for j in batch_indices])
                batch_last_names = torch.stack([val_dataset[j]['last_name'] for j in batch_indices])

                X_batch = (batch_first_names, batch_last_names)
                batch_probs = calibrated_model.predict_proba(X_batch)[:, 1]
                calibrated_probs.extend(batch_probs)

            calibrated_probs = np.array(calibrated_probs)

            # Find optimal threshold for calibrated probabilities
            cal_threshold, cal_metrics, _ = grid_search_threshold(
                true_labels, calibrated_probs, min_thresh=0.40, max_thresh=0.60, step=0.005
            )

            # Update results
            results.update({
                "calibrated_threshold": float(cal_threshold),
                "calibrated_f1": float(cal_metrics['val_f1']),
                "calibrated_precision": float(cal_metrics['val_precision']),
                "calibrated_recall": float(cal_metrics['val_recall']),
                "calibrated_bias_ratio": float(cal_metrics['bias_ratio'])
            })

            # Save calibrated model
            with open('calibrated_model.pkl', 'wb') as f:
                pickle.dump(calibrated_model, f)

            print(f"Calibrated model saved to calibrated_model.pkl")

            # Use calibrated probabilities for further processing
            pred_probs = calibrated_probs

        # Apply equalized odds if requested
        if args.apply_equalized_odds:
            print("Applying equalized odds...")

            # Use gender as sensitive attribute (simplified approach)
            sensitive_attr = true_labels

            # Apply post-processing
            fair_preds, fair_thresholds = apply_equalized_odds_postprocessing(
                true_labels, pred_probs, sensitive_attr
            )

            # Evaluate fair predictions
            fair_precision, fair_recall, fair_f1, _ = precision_recall_fscore_support(
                true_labels, fair_preds, average='binary'
            )

            # Calculate bias ratio for fair predictions
            cm, bias_ratio = plot_confusion_matrix(
                true_labels, fair_preds, output_file='fair_confusion_matrix.png'
            )

            # Update results
            results.update({
                "fair_thresholds": fair_thresholds,
                "fair_f1": float(fair_f1),
                "fair_precision": float(fair_precision),
                "fair_recall": float(fair_recall),
                "fair_bias_ratio": float(bias_ratio)
            })

        # Assicurati che la directory esista
        try:
            from utils import ensure_dir
            ensure_dir(args.save_dir)
        except ImportError:
            if not os.path.exists(args.save_dir):
                os.makedirs(args.save_dir)

        # Salva immagini nella directory corretta
        threshold_analysis_path = os.path.join(args.save_dir, 'threshold_analysis.png')
        confusion_path = os.path.join(args.save_dir, 'best_threshold_confusion.png')

        # Plot confusion matrix for best threshold
        best_preds = (pred_probs >= best_threshold).astype(int)
        plot_confusion_matrix(true_labels, best_preds, output_file=confusion_path)

        # Save results
        output_path = os.path.join(args.save_dir, args.output_file)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=4)

        print(f"Results saved to {output_path}")
        print("Post-processing completed!")
        print("\nResults summary:")
        for k, v in results.items():
            print(f"  {k}: {v}")
    else:
        print("Error: Could not create validation dataset. Post-processing aborted.")

if __name__ == "__main__":
    main()
