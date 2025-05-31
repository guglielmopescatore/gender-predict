"""
Common utility functions.

This module contains utility functions used across the gender prediction project,
including early stopping, visualization helpers, and file operations.
"""

import torch
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

class EarlyStopping:
    """
    Early stopping implementation to prevent overfitting.

    Tracks validation performance and stops training when performance stops improving
    for a specified number of epochs.

    Args:
        patience: Number of epochs with no improvement after which training will stop
        min_delta: Minimum change in monitored value to qualify as improvement
        restore_best_weights: Whether to restore the model weights from the epoch with the best value
    """
    def __init__(self, patience=7, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_model = None
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, model, val_score):
        """
        Check if training should be stopped.

        Args:
            model: The current model
            val_score: Current validation score (higher is better)

        Returns:
            True if training should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = val_score
            self.save_checkpoint(model)
            return False

        if val_score > self.best_score + self.min_delta:
            # Score improved
            self.best_score = val_score
            self.counter = 0
            self.save_checkpoint(model)
        else:
            # Score did not improve
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.restore_best_weights:
                    model.load_state_dict(self.best_model)
                return True

        return False

    def save_checkpoint(self, model):
        """Save the weights of the best model."""
        self.best_model = model.state_dict().copy()

def ensure_dir(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_metrics_to_csv(metrics, filepath):
    """Save training metrics to CSV file."""
    df = pd.DataFrame(metrics)
    df.to_csv(filepath, index=False)

def plot_confusion_matrix(y_true, y_pred, figsize=(10, 8), output_file='confusion_matrix.png'):
    """
    Visualize and analyze the confusion matrix for the gender prediction.

    Args:
        y_true: True values (0 for 'M', 1 for 'W')
        y_pred: Predicted values (0 for 'M', 1 for 'W')
        figsize: Size of the figure
        output_file: Path to save the figure
    """
    # Calcola la matrice di confusione
    cm = confusion_matrix(y_true, y_pred)

    # Crea labels per la visualizzazione
    labels = ['Maschio (M)', 'Femmina (W)']

    # Calcola metriche specifiche per genere
    tn, fp, fn, tp = cm.ravel()

    # Calcola tassi specifici per genere
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    m_precision = tn / (tn + fn) if (tn + fn) > 0 else 0  # Quanti dei predetti maschi sono effettivamente maschi
    w_precision = tp / (tp + fp) if (tp + fp) > 0 else 0  # Quante delle predette femmine sono effettivamente femmine
    m_recall = tn / (tn + fp) if (tn + fp) > 0 else 0     # Quanti maschi reali sono stati identificati correttamente
    w_recall = tp / (tp + fn) if (tp + fn) > 0 else 0     # Quante femmine reali sono state identificate correttamente

    m_f1 = 2 * (m_precision * m_recall) / (m_precision + m_recall) if (m_precision + m_recall) > 0 else 0
    w_f1 = 2 * (w_precision * w_recall) / (w_precision + w_recall) if (w_precision + w_recall) > 0 else 0

    # Calcola il bias
    bias_ratio = (fp / (tn + fp)) / (fn / (tp + fn)) if (tn + fp) > 0 and (tp + fn) > 0 else 0

    # Crea la figura
    plt.figure(figsize=figsize)

    # Visualizza la matrice di confusione
    plt.subplot(1, 2, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Matrice di Confusione')
    plt.ylabel('Valore Reale')
    plt.xlabel('Valore Predetto')

    # Visualizza le metriche specifiche per genere
    plt.subplot(1, 2, 2)
    metrics_data = {
        'Metrica': ['Precisione', 'Recall', 'F1-Score'],
        'Maschio (M)': [m_precision, m_recall, m_f1],
        'Femmina (W)': [w_precision, w_recall, w_f1]
    }
    metrics_df = pd.DataFrame(metrics_data)
    metrics_df = metrics_df.set_index('Metrica')

    sns.heatmap(metrics_df, annot=True, fmt='.3f', cmap='Greens', cbar=False)
    plt.title('Metriche per Genere')

    # Aggiungi il riepilogo delle statistiche come testo
    plt.figtext(0.5, 0.01,
                f"Accuratezza Globale: {accuracy:.3f} | "
                f"Bias Ratio (M→W : W→M): {bias_ratio:.3f}\n"
                f"Errore M→W: {fp} ({fp/(tn+fp):.1%} dei maschi) | "
                f"Errore W→M: {fn} ({fn/(tp+fn):.1%} delle femmine)",
                ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(output_file)

    print(f"\n--- Analisi del Bias di Genere ---")
    print(f"Matrice di Confusione:\n{cm}")
    print(f"\nAccuratezza Globale: {accuracy:.3f}")
    print(f"\nMetriche per Genere:")
    print(f"  Maschi (M):")
    print(f"    Precisione: {m_precision:.3f} (quanti dei predetti maschi sono effettivamente maschi)")
    print(f"    Recall: {m_recall:.3f} (quanti maschi reali sono stati identificati correttamente)")
    print(f"    F1-Score: {m_f1:.3f}")
    print(f"  Femmine (W):")
    print(f"    Precisione: {w_precision:.3f} (quante delle predette femmine sono effettivamente femmine)")
    print(f"    Recall: {w_recall:.3f} (quante femmine reali sono state identificate correttamente)")
    print(f"    F1-Score: {w_f1:.3f}")

    print(f"\nAnalisi del Bias:")
    print(f"  Errore M→W: {fp} nomi maschili classificati come femminili ({fp/(tn+fp):.1%} dei maschi)")
    print(f"  Errore W→M: {fn} nomi femminili classificati come maschili ({fn/(tp+fn):.1%} delle femmine)")
    print(f"  Bias Ratio (M→W : W→M): {bias_ratio:.3f}")

    if bias_ratio > 1.1:
        print(f"  → Il modello tende a classificare erroneamente i nomi maschili come femminili con maggiore frequenza")
    elif bias_ratio < 0.9:
        print(f"  → Il modello tende a classificare erroneamente i nomi femminili come maschili con maggiore frequenza")
    else:
        print(f"  → Il modello mostra un bias equilibrato tra i generi")

    return cm, accuracy, (m_precision, w_precision), (m_recall, w_recall), (m_f1, w_f1), bias_ratio
