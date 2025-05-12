#!/usr/bin/env python3
"""
Script specifico per valutare modelli GenderPredictorEnhanced (Round 2).
Supporta modelli multi-layer e funzionalità avanzate.

Utilizzo:
  python evaluate_enhanced_model.py --model models/round2_best.pth --preprocessor name_preprocessor.pkl --test comparison_dataset.csv
"""

import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import time
from tqdm import tqdm
import json

# Imposta il backend di matplotlib in modo che non richieda un display
import matplotlib
matplotlib.use('Agg')  # Usa il backend 'Agg' che non richiede display

class NamePreprocessor:
    """Classe per il preprocessing dei nomi."""

    def __init__(self, max_name_length=20, max_surname_length=20):
        """
        Inizializza il preprocessore.

        Args:
            max_name_length: Lunghezza massima per i nomi
            max_surname_length: Lunghezza massima per i cognomi
        """
        self.max_name_length = max_name_length
        self.max_surname_length = max_surname_length
        self.char_to_idx = {'<PAD>': 0}  # Inizializza con padding token

        # Aggiungi caratteri alfabetici (a-z)
        for i, c in enumerate("abcdefghijklmnopqrstuvwxyz"):
            self.char_to_idx[c] = i + 1

        # Aggiungi alcuni caratteri speciali comuni nei nomi
        for i, c in enumerate("'-áàâäãåçéèêëíìîïñóòôöõúùûüýÿ"):
            self.char_to_idx[c] = len(self.char_to_idx)

        self.idx_to_char = {v: k for k, v in self.char_to_idx.items()}
        self.vocab_size = len(self.char_to_idx)

    def clean_name(self, name):
        """Normalizza un nome."""
        if not isinstance(name, str) or pd.isna(name):
            return ""
        # Converti in minuscolo e rimuovi spazi iniziali/finali
        name = name.lower().strip()
        # Rimuovi caratteri non alfabetici (tranne quelli speciali)
        name = ''.join(c for c in name if c.isalpha() or c in self.char_to_idx)
        return name

    def split_full_name(self, full_name):
        """Divide un nome completo in nome e cognome."""
        if not isinstance(full_name, str) or pd.isna(full_name):
            return "", ""

        parts = full_name.split()
        if len(parts) == 1:
            return parts[0], ""  # Solo nome, nessun cognome
        elif len(parts) == 2:
            return parts[0], parts[1]  # Caso normale: nome e cognome
        else:
            # Casi complessi: prendiamo il primo come nome, il resto come cognome
            return parts[0], " ".join(parts[1:])

    def name_to_indices(self, name, max_length):
        """Converte un nome in una sequenza di indici."""
        name = self.clean_name(name)
        # Converti ogni carattere in indice
        indices = [self.char_to_idx.get(c, 0) for c in name[:max_length]]
        # Padding
        indices = indices + [0] * (max_length - len(indices))
        return indices

    def preprocess_name(self, full_name):
        """Preprocessa un nome completo."""
        first_name, last_name = self.split_full_name(full_name)

        first_name_indices = self.name_to_indices(first_name, self.max_name_length)
        last_name_indices = self.name_to_indices(last_name, self.max_surname_length)

        return {
            'first_name': first_name_indices,
            'last_name': last_name_indices
        }

    @classmethod
    def load(cls, path):
        """Carica un preprocessore da disco."""
        with open(path, 'rb') as f:
            data = pickle.load(f)

        preprocessor = cls(
            max_name_length=data['max_name_length'],
            max_surname_length=data['max_surname_length']
        )

        preprocessor.char_to_idx = data['char_to_idx']
        preprocessor.idx_to_char = data['idx_to_char']
        preprocessor.vocab_size = data['vocab_size']

        return preprocessor

class NameGenderDataset(Dataset):
    """Dataset per l'addestramento del modello di predizione del genere."""

    def __init__(self, dataframe, preprocessor, mode='test', name_column='primaryName', gender_column='gender'):
        """
        Inizializza il dataset.

        Args:
            dataframe: DataFrame pandas con i dati
            preprocessor: Istanza di NamePreprocessor
            mode: 'test' o 'predict'
            name_column: Nome della colonna contenente i nomi
            gender_column: Nome della colonna contenente il genere
        """
        self.df = dataframe
        self.preprocessor = preprocessor
        self.mode = mode
        self.name_column = name_column
        self.gender_column = gender_column

        # Mappa genere a indice
        self.gender_to_idx = {'M': 0, 'W': 1}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Estrai nome e cognome
        name_data = self.preprocessor.preprocess_name(row[self.name_column])

        # Converti in tensori
        first_name_tensor = torch.tensor(name_data['first_name'], dtype=torch.long)
        last_name_tensor = torch.tensor(name_data['last_name'], dtype=torch.long)

        # Prepara l'output
        result = {
            'first_name': first_name_tensor,
            'last_name': last_name_tensor
        }

        # Aggiungi il genere per il test
        if self.gender_column in row and self.mode != 'predict':
            gender_idx = self.gender_to_idx.get(row[self.gender_column], 0)
            result['gender'] = torch.tensor(gender_idx, dtype=torch.float)

        # Aggiungi l'ID se presente
        if 'nconst' in row:
            result['id'] = row['nconst']

        return result

class AttentionLayer(nn.Module):
    """Semplice layer di attenzione per BiLSTM."""

    def __init__(self, hidden_size):
        """
        Inizializza il layer di attenzione.

        Args:
            hidden_size: Dimensione dello stato nascosto della LSTM
        """
        super(AttentionLayer, self).__init__()
        self.attention = nn.Linear(hidden_size * 2, 1)

    def forward(self, lstm_output):
        """
        Forward pass del layer di attenzione.

        Args:
            lstm_output: Output della BiLSTM [batch, seq_len, hidden_size*2]

        Returns:
            Context vector [batch, hidden_size*2]
        """
        # Calcola i pesi di attenzione
        attention_weights = torch.softmax(self.attention(lstm_output), dim=1)

        # Calcola il vettore di contesto pesato
        context_vector = torch.sum(attention_weights * lstm_output, dim=1)

        return context_vector

class GenderPredictorEnhanced(nn.Module):
    """Enhanced BiLSTM model with improved capacity and architecture."""

    def __init__(self, vocab_size, embedding_dim=16, hidden_size=80, n_layers=2,
                dropout_rate=0.3, dual_input=True):
        """
        Initialize the enhanced model.

        Args:
            vocab_size: Size of character vocabulary
            embedding_dim: Dimension of character embeddings
            hidden_size: Hidden size of LSTM layers
            n_layers: Number of LSTM layers
            dropout_rate: Dropout rate
            dual_input: Whether to use separate encoders for first and last name
        """
        super(GenderPredictorEnhanced, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dual_input = dual_input

        # Shared character embedding
        self.char_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # First name LSTM
        self.firstname_lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate if n_layers > 1 else 0
        )

        # Layer normalization between LSTM layers
        self.firstname_norm = nn.LayerNorm(hidden_size * 2)

        # Last name LSTM (only if dual_input)
        if dual_input:
            self.lastname_lstm = nn.LSTM(
                input_size=embedding_dim,
                hidden_size=hidden_size,
                num_layers=n_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout_rate if n_layers > 1 else 0
            )

            # Layer normalization for last name
            self.lastname_norm = nn.LayerNorm(hidden_size * 2)

        # Attention layers
        self.firstname_attention = AttentionLayer(hidden_size)
        if dual_input:
            self.lastname_attention = AttentionLayer(hidden_size)
            # Output dimension will be doubled if using dual input
            output_dim = hidden_size * 4
        else:
            output_dim = hidden_size * 2

        # Output layers
        self.fc1 = nn.Linear(output_dim, hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, first_name, last_name):
        """
        Forward pass of the model.

        Args:
            first_name: First name indices [batch, seq_len]
            last_name: Last name indices [batch, seq_len]

        Returns:
            Gender prediction probability
        """
        # First name processing
        first_emb = self.char_embedding(first_name)
        first_lstm_out, _ = self.firstname_lstm(first_emb)

        # Apply layer normalization for multi-layer LSTM
        if self.n_layers > 1:
            first_lstm_out = self.firstname_norm(first_lstm_out)

        first_att = self.firstname_attention(first_lstm_out)

        if self.dual_input:
            # Last name processing
            last_emb = self.char_embedding(last_name)
            last_lstm_out, _ = self.lastname_lstm(last_emb)

            # Apply layer normalization
            if self.n_layers > 1:
                last_lstm_out = self.lastname_norm(last_lstm_out)

            last_att = self.lastname_attention(last_lstm_out)

            # Concatenate features
            combined = torch.cat((first_att, last_att), dim=1)
        else:
            combined = first_att

        # Output layers
        x = self.fc1(combined)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x.squeeze()

def plot_confusion_matrix(y_true, y_pred, figsize=(10, 8), output_file='confusion_matrix.png', model_name=""):
    """
    Visualizza e analizza la matrice di confusione per la predizione del genere.

    Args:
        y_true: Valori veri (0 per 'M', 1 per 'W')
        y_pred: Valori predetti (0 per 'M', 1 per 'W')
        figsize: Dimensione della figura
        output_file: Percorso dove salvare la figura
        model_name: Nome del modello da visualizzare nel titolo
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

    title_suffix = f" - {model_name}" if model_name else ""

    # Visualizza la matrice di confusione
    plt.subplot(1, 2, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(f'Matrice di Confusione{title_suffix}')
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
    plt.title(f'Metriche per Genere{title_suffix}')

    # Aggiungi il riepilogo delle statistiche come testo
    plt.figtext(0.5, 0.01,
                f"Accuratezza Globale: {accuracy:.3f} | "
                f"Bias Ratio (M→W : W→M): {bias_ratio:.3f}\n"
                f"Errore M→W: {fp} ({fp/(tn+fp):.1%} dei maschi) | "
                f"Errore W→M: {fn} ({fn/(tp+fn):.1%} delle femmine)",
                ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(output_file)
    plt.close()  # Chiudi il plot esplicitamente

    print(f"\n--- Analisi del Bias di Genere{title_suffix} ---")
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

    # Raccogli tutte le metriche in un dizionario
    metrics = {
        "accuracy": float(accuracy),
        "male_precision": float(m_precision),
        "male_recall": float(m_recall),
        "male_f1": float(m_f1),
        "female_precision": float(w_precision),
        "female_recall": float(w_recall),
        "female_f1": float(w_f1),
        "bias_ratio": float(bias_ratio),
        "error_m_to_w_percent": float(fp/(tn+fp)) if (tn+fp) > 0 else 0,
        "error_w_to_m_percent": float(fn/(tp+fn)) if (tp+fn) > 0 else 0,
        "confusion_matrix": {
            "true_negative": int(tn),  # Maschi predetti correttamente
            "false_positive": int(fp),  # Maschi predetti come femmine
            "false_negative": int(fn),  # Femmine predette come maschi
            "true_positive": int(tp)   # Femmine predette correttamente
        }
    }

    return metrics

def load_trained_model(model_path, preprocessor_path, device='cuda', weights_only=True):
    """
    Carica un modello addestrato dal disco.

    Args:
        model_path: Percorso al file del modello
        preprocessor_path: Percorso al file del preprocessore
        device: Dispositivo su cui caricare il modello
        weights_only: Se caricare solo i pesi (consigliato per sicurezza)

    Returns:
        Modello caricato e preprocessore
    """
    # Carica il preprocessore
    preprocessor = NamePreprocessor.load(preprocessor_path)

    # Carica il checkpoint del modello
    checkpoint = torch.load(model_path, map_location=device, weights_only=weights_only)

    # Estrai i parametri del modello
    vocab_size = checkpoint.get('vocab_size', preprocessor.vocab_size)
    embedding_dim = checkpoint.get('embedding_dim', 16)
    hidden_size = checkpoint.get('hidden_size', 80)
    n_layers = checkpoint.get('n_layers', 2)
    dual_input = checkpoint.get('dual_input', True)

    print(f"Caricando modello con parametri:")
    print(f"  vocab_size: {vocab_size}")
    print(f"  embedding_dim: {embedding_dim}")
    print(f"  hidden_size: {hidden_size}")
    print(f"  n_layers: {n_layers}")
    print(f"  dual_input: {dual_input}")

    # Crea un'istanza del modello
    model = GenderPredictorEnhanced(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
        n_layers=n_layers,
        dual_input=dual_input
    )

    # Carica i pesi
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    return model, preprocessor

def evaluate_model(model, test_loader, device='cuda', model_name=""):
    """
    Valuta il modello sul test set.

    Args:
        model: Modello addestrato
        test_loader: DataLoader per il test set
        device: Dispositivo su cui eseguire la valutazione
        model_name: Nome del modello per i report

    Returns:
        Dizionario con le metriche di valutazione
    """
    model.eval()
    test_preds = []
    test_targets = []

    print(f"Valutazione del modello{' ' + model_name if model_name else ''}...")

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            first_name = batch['first_name'].to(device)
            last_name = batch['last_name'].to(device)

            # Verifica se ci sono etichette per calcolare le metriche
            if 'gender' in batch:
                gender = batch['gender'].to(device)

                # Get raw logits
                logits = model(first_name, last_name)

                # Convert to probabilities and predictions
                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).long()

                test_preds.extend(preds.cpu().numpy())
                test_targets.extend(gender.cpu().numpy())

    # Se non ci sono etichette, non calcolare le metriche
    if not test_targets:
        print("Nessuna etichetta trovata nel dataset di test. Impossibile calcolare le metriche.")
        return None

    # Calcola le metriche sul test set
    test_acc = accuracy_score(test_targets, test_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        test_targets, test_preds, average='binary')

    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")
    print(f"Test F1: {f1:.4f}")

    # Genera nome file per la matrice di confusione
    if model_name:
        output_file = f'confusion_matrix_{model_name.replace(" ", "_").lower()}.png'
    else:
        output_file = 'confusion_matrix_enhanced_test.png'

    # Analisi approfondita del bias sul test set
    print(f"\nAnalisi dettagliata del bias sul test set:")
    metrics = plot_confusion_matrix(
        test_targets,
        test_preds,
        output_file=output_file,
        model_name=model_name
    )

    # Aggiungi le metriche principali
    metrics.update({
        "accuracy": float(test_acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1)
    })

    return metrics

def predict_with_model(model, preprocessor, test_df, output_file=None, device='cuda',
                       batch_size=128, name_column='primaryName'):
    """
    Esegue predizioni con il modello e salva i risultati.

    Args:
        model: Modello addestrato
        preprocessor: Preprocessore dei nomi
        test_df: DataFrame con i dati di test
        output_file: File dove salvare i risultati
        device: Dispositivo su cui eseguire l'inferenza
        batch_size: Dimensione del batch
        name_column: Nome della colonna contenente i nomi

    Returns:
        DataFrame con le predizioni
    """
    # Crea un dataset per la predizione
    dataset = NameGenderDataset(test_df, preprocessor, mode='predict', name_column=name_column)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Inizializza liste per i risultati
    all_probs = []
    all_preds = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            first_name = batch['first_name'].to(device)
            last_name = batch['last_name'].to(device)

            # Get raw logits
            logits = model(first_name, last_name)

            # Convert to probabilities and predictions
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).long()

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    # Crea il DataFrame dei risultati
    results_df = test_df.copy()
    results_df['gender_pred'] = ['W' if p == 1 else 'M' for p in all_preds]
    results_df['prob_female'] = all_probs

    # Salva i risultati se richiesto
    if output_file:
        results_df.to_csv(output_file, index=False)
        print(f"Risultati salvati in {output_file}")

    return results_df

def main():
    # Configura argparse per gestire i parametri da linea di comando
    parser = argparse.ArgumentParser(description='Valuta un modello GenderPredictorEnhanced su un dataset di test')

    # Parametri obbligatori
    parser.add_argument('--model', type=str, required=True,
                        help='Percorso al file del modello (.pth)')
    parser.add_argument('--preprocessor', type=str, required=True,
                        help='Percorso al file del preprocessore (.pkl)')
    parser.add_argument('--test', type=str, required=True,
                        help='Percorso al file CSV del dataset di test')

    # Parametri opzionali
    parser.add_argument('--output', type=str, default=None,
                        help='Percorso al file CSV dove salvare i risultati')
    parser.add_argument('--name-column', type=str, default='primaryName',
                        help='Nome della colonna contenente i nomi (default: primaryName)')
    parser.add_argument('--gender-column', type=str, default='gender',
                        help='Nome della colonna contenente il genere (default: gender)')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Dimensione del batch (default: 128)')
    parser.add_argument('--cpu', action='store_true',
                        help='Forza l\'uso della CPU anche se è disponibile la GPU')
    parser.add_argument('--model-name', type=str, default="Enhanced Model",
                        help='Nome del modello per i report')
    parser.add_argument('--metrics-file', type=str, default=None,
                        help='Percorso dove salvare le metriche in formato JSON')

    args = parser.parse_args()

    # Determina il dispositivo
    device = torch.device("cpu" if args.cpu else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        # Carica il modello e il preprocessore
        model, preprocessor = load_trained_model(args.model, args.preprocessor, device)
        print(f"Modello caricato: {args.model}")
        print(f"Preprocessore caricato: {args.preprocessor}")

        # Carica il dataset di test
        test_df = pd.read_csv(args.test)
        print(f"Dataset di test caricato: {args.test} ({len(test_df)} record)")

        # Verifica la distribuzione di genere se disponibile
        if args.gender_column in test_df.columns:
            print("\nDistribuzione di genere nel dataset di test:")
            gender_counts = test_df[args.gender_column].value_counts()
            for gender, count in gender_counts.items():
                percentage = (count / len(test_df)) * 100
                print(f"  {gender}: {count} ({percentage:.2f}%)")

        # Crea il dataset
        test_dataset = NameGenderDataset(
            test_df,
            preprocessor,
            mode='test',
            name_column=args.name_column,
            gender_column=args.gender_column
        )

        # Crea il DataLoader
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        # Valuta il modello
        metrics = evaluate_model(model, test_loader, device, model_name=args.model_name)

        # Salva le metriche se richiesto
        if args.metrics_file and metrics:
            with open(args.metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2)
            print(f"Metriche salvate in {args.metrics_file}")

        # Esegui predizioni e salva i risultati se richiesto
        if args.output:
            results_df = predict_with_model(
                model,
                preprocessor,
                test_df,
                args.output,
                device,
                args.batch_size,
                args.name_column
            )

            # Mostra le prime 5 predizioni
            print("\nPrime 5 predizioni:")
            print(results_df[[args.name_column, 'gender_pred', 'prob_female']].head())

    except Exception as e:
        print(f"Errore durante la valutazione del modello: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    main()
