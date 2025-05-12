import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import re
import time
from tqdm import tqdm

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

    def save(self, path):
        """Salva il preprocessore su disco."""
        with open(path, 'wb') as f:
            pickle.dump({
                'max_name_length': self.max_name_length,
                'max_surname_length': self.max_surname_length,
                'char_to_idx': self.char_to_idx,
                'idx_to_char': self.idx_to_char,
                'vocab_size': self.vocab_size
            }, f)

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

    def __init__(self, dataframe, preprocessor, mode='train'):
        """
        Inizializza il dataset.

        Args:
            dataframe: DataFrame pandas con i dati
            preprocessor: Istanza di NamePreprocessor
            mode: 'train', 'val' o 'test'
        """
        self.df = dataframe
        self.preprocessor = preprocessor
        self.mode = mode

        # Mappa genere a indice
        self.gender_to_idx = {'M': 0, 'W': 1}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Estrai nome e cognome
        name_data = self.preprocessor.preprocess_name(row['primaryName'])

        # Converti in tensori
        first_name_tensor = torch.tensor(name_data['first_name'], dtype=torch.long)
        last_name_tensor = torch.tensor(name_data['last_name'], dtype=torch.long)

        # Prepara l'output
        result = {
            'first_name': first_name_tensor,
            'last_name': last_name_tensor
        }

        # Aggiungi il genere per training/validation
        if 'gender' in row and self.mode != 'predict':
            gender_idx = self.gender_to_idx.get(row['gender'], 0)  # Default a 'M' se mancante
            result['gender'] = torch.tensor(gender_idx, dtype=torch.float)

        # Aggiungi l'ID per il test/predizione
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

class GenderPredictor(nn.Module):
    """Modello BiLSTM con attenzione per la predizione del genere dai nomi."""

    def __init__(self, vocab_size, embedding_dim=16, hidden_size=64, dropout_rate=0.2):
        """
        Inizializza il modello.

        Args:
            vocab_size: Dimensione del vocabolario dei caratteri
            embedding_dim: Dimensione dell'embedding dei caratteri
            hidden_size: Dimensione dello stato nascosto della LSTM
            dropout_rate: Tasso di dropout
        """
        super(GenderPredictor, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size

        # Layer di embedding condiviso per caratteri
        self.char_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # BiLSTM per il nome
        self.firstname_lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True
        )

        # BiLSTM per il cognome
        self.lastname_lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True
        )

        # Layer di attenzione
        self.firstname_attention = AttentionLayer(hidden_size)
        self.lastname_attention = AttentionLayer(hidden_size)

        # Layer di output
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 4, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, first_name, last_name):
        """
        Forward pass del modello.

        Args:
            first_name: Tensor dei nomi [batch, max_name_length]
            last_name: Tensor dei cognomi [batch, max_surname_length]

        Returns:
            Probabilità di genere femminile [batch, 1]
        """
        # Embedding dei caratteri
        first_name_emb = self.char_embedding(first_name)
        last_name_emb = self.char_embedding(last_name)

        # LSTM per nome e cognome
        first_name_lstm_out, _ = self.firstname_lstm(first_name_emb)
        last_name_lstm_out, _ = self.lastname_lstm(last_name_emb)

        # Applicazione dell'attenzione
        first_name_att = self.firstname_attention(first_name_lstm_out)
        last_name_att = self.lastname_attention(last_name_lstm_out)

        # Concatenazione delle feature
        combined = torch.cat((first_name_att, last_name_att), dim=1)

        # Output finale
        output = self.fc(combined)

        return output.squeeze()

class EarlyStopping:
    """Implementazione di Early Stopping per evitare overfitting."""

    def __init__(self, patience=5, min_delta=0, restore_best_weights=True):
        """
        Inizializza l'oggetto Early Stopping.

        Args:
            patience: Numero di epoche con peggioramento prima di fermare il training
            min_delta: Cambiamento minimo da considerare come miglioramento
            restore_best_weights: Se ripristinare i pesi migliori quando si ferma l'addestramento
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_model = None
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, model, val_score):
        """
        Controlla se fermare l'addestramento.

        Args:
            model: Il modello corrente
            val_score: Punteggio di validazione corrente

        Returns:
            True se fermare l'addestramento, False altrimenti
        """
        if self.best_score is None:
            self.best_score = val_score
            self.save_checkpoint(model)
            return False

        if val_score > self.best_score + self.min_delta:
            self.best_score = val_score
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.restore_best_weights:
                    model.load_state_dict(self.best_model)
                return True

        return False

    def save_checkpoint(self, model):
        """Salva i pesi migliori del modello."""
        self.best_model = model.state_dict().copy()

def plot_confusion_matrix(y_true, y_pred, figsize=(10, 8), output_file='confusion_matrix.png'):
    """
    Visualizza e analizza la matrice di confusione per la predizione del genere.

    Args:
        y_true: Valori veri (0 per 'M', 1 per 'W')
        y_pred: Valori predetti (0 per 'M', 1 per 'W')
        figsize: Dimensione della figura
        output_file: Percorso dove salvare la figura
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

def train_model(model, train_loader, val_loader, criterion, optimizer,
                num_epochs=20, device='cuda', patience=5,
                model_save_path='gender_predictor_model.pth'):
    """
    Addestra il modello di predizione del genere.

    Args:
        model: Istanza di GenderPredictor
        train_loader: DataLoader per i dati di training
        val_loader: DataLoader per i dati di validazione
        criterion: Funzione di loss
        optimizer: Ottimizzatore
        num_epochs: Numero massimo di epoche
        device: Dispositivo su cui eseguire il training ('cuda' o 'cpu')
        patience: Epoche di attesa per early stopping
        model_save_path: Percorso dove salvare il modello

    Returns:
        Dizionario con la storia del training
    """
    model.to(device)

    # Setup per early stopping
    early_stopping = EarlyStopping(patience=patience)

    # Storia del training
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'val_precision': [],
        'val_recall': [],
        'val_f1': []
    }

    # Training loop
    for epoch in range(num_epochs):
        start_time = time.time()

        # Training
        model.train()
        train_loss = 0.0
        train_preds = []
        train_targets = []

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            first_name = batch['first_name'].to(device)
            last_name = batch['last_name'].to(device)
            gender = batch['gender'].to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(first_name, last_name)
            loss = criterion(outputs, gender)

            # Backward pass
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * first_name.size(0)

            # Salva predizioni e target per calcolare le metriche
            train_preds.extend((outputs > 0.5).cpu().detach().numpy().astype(int))
            train_targets.extend(gender.cpu().detach().numpy().astype(int))

        train_loss /= len(train_loader.dataset)
        train_acc = accuracy_score(train_targets, train_preds)

        # Validation
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                first_name = batch['first_name'].to(device)
                last_name = batch['last_name'].to(device)
                gender = batch['gender'].to(device)

                outputs = model(first_name, last_name)
                loss = criterion(outputs, gender)

                val_loss += loss.item() * first_name.size(0)

                val_preds.extend((outputs > 0.5).cpu().numpy().astype(int))
                val_targets.extend(gender.cpu().numpy().astype(int))

        val_loss /= len(val_loader.dataset)
        val_acc = accuracy_score(val_targets, val_preds)

        # Calcola precision, recall, F1
        precision, recall, f1, _ = precision_recall_fscore_support(
            val_targets, val_preds, average='binary')

        # Aggiorna la storia del training
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['val_precision'].append(precision)
        history['val_recall'].append(recall)
        history['val_f1'].append(f1)

        # Tempo trascorso
        time_elapsed = time.time() - start_time

        # Stampa i risultati
        print(f"Epoch {epoch+1}/{num_epochs} | Time: {time_elapsed:.2f}s")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        print(f"Val Precision: {precision:.4f} | Val Recall: {recall:.4f} | Val F1: {f1:.4f}")

        # Analisi del bias sul set di validazione
        if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
            print("\nAnalisi del bias sul set di validazione:")
            plot_confusion_matrix(
                val_targets,
                val_preds,
                output_file=f'confusion_matrix_epoch_{epoch+1}.png'
            )

        print("-" * 60)

        # Early stopping
        if early_stopping(model, val_acc):
            print(f"Early stopping triggered after epoch {epoch+1}")
            break

    # Salva il modello finale
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab_size': model.vocab_size,
        'embedding_dim': model.embedding_dim,
        'hidden_size': model.hidden_size
    }, model_save_path)

    print(f"Model saved to {model_save_path}")

    return history

def predict_gender(model, preprocessor, names, device='cuda', batch_size=64):
    """
    Predice il genere per una lista di nomi.

    Args:
        model: Modello addestrato
        preprocessor: Preprocessore dei nomi
        names: Lista di nomi da predire
        device: Dispositivo su cui eseguire l'inferenza
        batch_size: Dimensione del batch

    Returns:
        DataFrame con nomi, predizioni e probabilità
    """
    model.to(device)
    model.eval()

    # Crea un DataFrame con i nomi
    df_pred = pd.DataFrame({'primaryName': names})

    # Crea un dataset
    dataset = NameGenderDataset(df_pred, preprocessor, mode='predict')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Inizializza liste per i risultati
    all_probs = []
    all_preds = []

    # Esegui le predizioni
    with torch.no_grad():
        for batch in dataloader:
            first_name = batch['first_name'].to(device)
            last_name = batch['last_name'].to(device)

            outputs = model(first_name, last_name)
            probs = outputs.cpu().numpy()
            preds = (probs > 0.5).astype(int)

            all_probs.extend(probs)
            all_preds.extend(preds)

    # Crea il DataFrame dei risultati
    results = pd.DataFrame({
        'name': names,
        'gender_pred': ['W' if p == 1 else 'M' for p in all_preds],
        'prob_female': all_probs
    })

    return results

def visualize_training_history(history):
    """
    Visualizza la storia dell'addestramento.

    Args:
        history: Dizionario con la storia dell'addestramento
    """
    plt.figure(figsize=(15, 10))

    # Plot dell'accuratezza
    plt.subplot(2, 2, 1)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot della loss
    plt.subplot(2, 2, 2)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot di precision e recall
    plt.subplot(2, 2, 3)
    plt.plot(history['val_precision'], label='Precision')
    plt.plot(history['val_recall'], label='Recall')
    plt.plot(history['val_f1'], label='F1 Score')
    plt.title('Validation Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

def load_trained_model(model_path, preprocessor_path, device='cuda'):
    """
    Carica un modello addestrato dal disco.

    Args:
        model_path: Percorso al file del modello
        preprocessor_path: Percorso al file del preprocessore
        device: Dispositivo su cui caricare il modello

    Returns:
        Modello caricato e preprocessore
    """
    # Carica il preprocessore
    preprocessor = NamePreprocessor.load(preprocessor_path)

    # Carica il checkpoint del modello
    checkpoint = torch.load(model_path, map_location=device)

    # Crea un'istanza del modello
    model = GenderPredictor(
        vocab_size=checkpoint['vocab_size'],
        embedding_dim=checkpoint['embedding_dim'],
        hidden_size=checkpoint['hidden_size']
    )

    # Carica i pesi
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return model, preprocessor

def main():
    """Funzione principale per l'esecuzione dell'addestramento."""
    # Parametri
    data_file = "training_dataset.csv"
    batch_size = 128
    num_epochs = 30
    learning_rate = 0.001
    patience = 5
    test_size = 0.1
    val_size = 0.1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Crea un dizionario di parametri per l'ExperimentManager
    experiment_params = {
        "round": 0,
        "data_file": data_file,
        "batch_size": batch_size,
        "epochs": num_epochs,
        "learning_rate": learning_rate,
        "patience": patience,
        "test_size": test_size,
        "val_size": val_size,
        "device": str(device),
        "model_type": "GenderPredictor"
    }

    # Converti il dizionario in un oggetto Namespace
    import argparse
    args = argparse.Namespace(**experiment_params)

    # Inizializza l'ExperimentManager con i parametri
    from experiment_manager import ExperimentManager
    experiment = ExperimentManager(args)  # Corretto, passa l'oggetto Namespace
    print(f"Experiment ID: {experiment.experiment_id}")
    print(f"Experiment directory: {experiment.experiment_dir}")

    print(f"Using device: {device}")

    # Carica i dati
    print(f"Loading data from {data_file}...")
    df = pd.read_csv(data_file)
    print(f"Loaded {len(df)} records")

    # Estrai 10.000 nomi per il test comparativo (stratificato per genere)
    comparison_test_set, training_data = train_test_split(
        df, test_size=len(df)-10000,
        random_state=42,
        stratify=df['gender']
    )

    # Salva il set di test comparativo dentro la directory dell'esperimento
    comparison_test_path = os.path.join(experiment.logs_dir, "comparison_test_set_10k.csv")
    comparison_test_set.to_csv(comparison_test_path, index=False)
    print(f"Extracted 10,000 names for comparison testing and saved to {comparison_test_path}")

    # Divisione train/val/test dai dati di training
    train_val_df, test_df = train_test_split(training_data, test_size=test_size, random_state=42, stratify=training_data['gender'])
    train_df, val_df = train_test_split(train_val_df, test_size=val_size/(1-test_size), random_state=42, stratify=train_val_df['gender'])

    print(f"Train set: {len(train_df)} records")
    print(f"Validation set: {len(val_df)} records")
    print(f"Test set: {len(test_df)} records")
    print(f"Comparison test set: {len(comparison_test_set)} records")

    # Verifica la distribuzione di genere
    print("\nDistribuzione di genere:")
    for dataset_name, dataset in [('Train', train_df), ('Validation', val_df), ('Test', test_df), ('Comparison', comparison_test_set)]:
        gender_counts = dataset['gender'].value_counts()
        print(f"  {dataset_name}:")
        for gender, count in gender_counts.items():
            percentage = (count / len(dataset)) * 100
            print(f"    {gender}: {count} ({percentage:.2f}%)")

    # Crea il preprocessore
    preprocessor = NamePreprocessor()
    preprocessor.save(experiment.preprocessor_path)

    # Crea i dataset
    train_dataset = NameGenderDataset(train_df, preprocessor, mode='train')
    val_dataset = NameGenderDataset(val_df, preprocessor, mode='val')
    test_dataset = NameGenderDataset(test_df, preprocessor, mode='test')

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Crea il modello
    model = GenderPredictor(vocab_size=preprocessor.vocab_size)

    # Loss e optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Definisci una funzione di callback per salvare checkpoints intermedi
    def save_checkpoint_callback(model, epoch, metrics):
        """Callback per salvare checkpoint durante il training."""
        # Salva un checkpoint ogni 5 epoche o all'ultima epoca
        if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'vocab_size': model.vocab_size,
                'embedding_dim': model.embedding_dim,
                'hidden_size': model.hidden_size
            }
            # Aggiungi le metriche
            for key, value in metrics.items():
                if isinstance(value, list) and len(value) > 0:
                    checkpoint[key] = value[-1]  # Prendi l'ultimo valore
            experiment.save_model_checkpoint(checkpoint, epoch=epoch+1)

    # Modifica la funzione train_model per accettare il callback
    def train_model_with_callback(model, train_loader, val_loader, criterion, optimizer,
                      num_epochs=20, device='cuda', patience=5,
                      model_save_path='gender_predictor_model.pth', checkpoint_callback=None):
        model.to(device)

        # Setup per early stopping
        early_stopping = EarlyStopping(patience=patience)

        # Storia del training
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'val_precision': [],
            'val_recall': [],
            'val_f1': []
        }

        # Training loop
        for epoch in range(num_epochs):
            start_time = time.time()

            # Training
            model.train()
            train_loss = 0.0
            train_preds = []
            train_targets = []

            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
                first_name = batch['first_name'].to(device)
                last_name = batch['last_name'].to(device)
                gender = batch['gender'].to(device)

                # Forward pass
                optimizer.zero_grad()
                outputs = model(first_name, last_name)
                loss = criterion(outputs, gender)

                # Backward pass
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * first_name.size(0)

                # Salva predizioni e target per calcolare le metriche
                train_preds.extend((outputs > 0.5).cpu().detach().numpy().astype(int))
                train_targets.extend(gender.cpu().detach().numpy().astype(int))

            train_loss /= len(train_loader.dataset)
            train_acc = accuracy_score(train_targets, train_preds)

            # Validation
            model.eval()
            val_loss = 0.0
            val_preds = []
            val_targets = []

            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                    first_name = batch['first_name'].to(device)
                    last_name = batch['last_name'].to(device)
                    gender = batch['gender'].to(device)

                    outputs = model(first_name, last_name)
                    loss = criterion(outputs, gender)

                    val_loss += loss.item() * first_name.size(0)

                    val_preds.extend((outputs > 0.5).cpu().numpy().astype(int))
                    val_targets.extend(gender.cpu().numpy().astype(int))

            val_loss /= len(val_loader.dataset)
            val_acc = accuracy_score(val_targets, val_preds)

            # Calcola precision, recall, F1
            precision, recall, f1, _ = precision_recall_fscore_support(
                val_targets, val_preds, average='binary')

            # Aggiorna la storia del training
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            history['val_precision'].append(precision)
            history['val_recall'].append(recall)
            history['val_f1'].append(f1)

            # Tempo trascorso
            time_elapsed = time.time() - start_time

            # Stampa i risultati
            print(f"Epoch {epoch+1}/{num_epochs} | Time: {time_elapsed:.2f}s")
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            print(f"Val Precision: {precision:.4f} | Val Recall: {recall:.4f} | Val F1: {f1:.4f}")

            # Chiama il callback se fornito
            if checkpoint_callback is not None:
                checkpoint_callback(model, epoch, history)

            # Analisi del bias sul set di validazione
            if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
                print("\nAnalisi del bias sul set di validazione:")
                confusion_path = os.path.join(experiment.plots_dir, f"confusion_matrix_epoch_{epoch+1}.png")
                plot_confusion_matrix(
                    val_targets,
                    val_preds,
                    output_file=confusion_path
                )

            print("-" * 60)

            # Early stopping
            if early_stopping(model, val_acc):
                print(f"Early stopping triggered after epoch {epoch+1}")
                break

        # Salva il modello finale
        torch.save({
            'model_state_dict': model.state_dict(),
            'vocab_size': model.vocab_size,
            'embedding_dim': model.embedding_dim,
            'hidden_size': model.hidden_size
        }, model_save_path)

        print(f"Model saved to {model_save_path}")

        return history

    # Addestra il modello
    print("Starting training...")
    history = train_model_with_callback(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        num_epochs=num_epochs,
        device=device,
        patience=patience,
        model_save_path=experiment.model_path,
        checkpoint_callback=save_checkpoint_callback
    )

    # Salva la storia del training tramite ExperimentManager
    experiment.log_training_history(history)

    # Visualizza e salva il grafico della storia di training
    experiment.plot_training_history(history)

    # Valuta il modello sul test set
    print("Evaluating on test set...")
    model, preprocessor = load_trained_model(experiment.model_path, experiment.preprocessor_path, device)

    test_preds = []
    test_targets = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            first_name = batch['first_name'].to(device)
            last_name = batch['last_name'].to(device)
            gender = batch['gender'].to(device)

            outputs = model(first_name, last_name)

            test_preds.extend((outputs > 0.5).cpu().numpy().astype(int))
            test_targets.extend(gender.cpu().numpy().astype(int))

    # Calcola le metriche sul test set
    test_acc = accuracy_score(test_targets, test_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(test_targets, test_preds, average='binary')

    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")
    print(f"Test F1: {f1:.4f}")

    # Salva le metriche del test
    test_metrics = {
        'accuracy': float(test_acc),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1)
    }
    experiment.log_test_metrics(test_metrics)

    # Analisi approfondita del bias sul test set
    print("\nAnalisi dettagliata del bias sul test set:")
    experiment.save_confusion_matrix(test_targets, test_preds, labels=["Male", "Female"])

    # Esempio di inferenza
    sample_names = [
        "John Smith",
        "Maria Garcia",
        "David Johnson",
        "Emma Wilson",
        "Alessandro Rossi",
        "Francesca Bianchi",
        "Yuki Tanaka",
        "Mei Chen"
    ]

    print("\nSample predictions:")
    results = predict_gender(model, preprocessor, sample_names, device)
    print(results)

    # Salva le predizioni di esempio
    results_path = os.path.join(experiment.logs_dir, "sample_predictions.csv")
    results.to_csv(results_path, index=False)

    # Genera il report dell'esperimento
    report_path = experiment.generate_report()

    print(f"Experiment {experiment.experiment_id} completed successfully!")
    print(f"Model saved to {experiment.model_path}")
    print(f"Report generated at {report_path}")
    print(f"Run 'python experiment_tools.py list' to see all experiments")
    print(f"Run 'python experiment_tools.py report --full' to generate a complete report")

if __name__ == "__main__":
    main()
