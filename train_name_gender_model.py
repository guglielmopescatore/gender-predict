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
import random
import time
import argparse
from tqdm import tqdm

# Import custom modules (they will be dynamically imported based on the round)
# These imports are moved to the main function

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
        )

    def forward(self, first_name, last_name):
        """
        Forward pass del modello.

        Args:
            first_name: Tensor dei nomi [batch, max_name_length]
            last_name: Tensor dei cognomi [batch, max_surname_length]

        Returns:
            Logit del genere femminile [batch, 1]
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
        combined = torch.cat((first_name_att, last_name_att), dim=1)  # [B, hidden*4]
        logits   = self.fc(combined)                   # [B,1]
        return logits.squeeze(1)                       # logit

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

    # Import early stopping based on args.round
    try:
        from utils import EarlyStopping
        early_stopping = EarlyStopping(patience=patience, min_delta=0.001)
    except ImportError:
        # Fallback to the original implementation
        class EarlyStopping:
            def __init__(self, patience=5, min_delta=0.001, restore_best_weights=True):
                self.patience = patience
                self.min_delta = min_delta
                self.restore_best_weights = restore_best_weights
                self.best_model = None
                self.best_score = None
                self.counter = 0
                self.early_stop = False

            def __call__(self, model, val_score):
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
                self.best_model = model.state_dict().copy()

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

        # --------------------- TRAIN ---------------------------
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            first_name = batch['first_name'].to(device)
            last_name  = batch['last_name'].to(device)
            gender     = batch['gender'].to(device)

            optimizer.zero_grad()

            logits = model(first_name, last_name)          # ora sono *logit*
            loss   = criterion(logits, gender)            # FocalLoss su logit
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * first_name.size(0)

            # ---- da logit → prob → pred
            probs = torch.sigmoid(logits)                 # ∈ [0,1]
            preds = (probs >= 0.5).long()                 # 0/1

            train_preds.extend(preds.cpu().numpy())
            train_targets.extend(gender.cpu().numpy())

        train_loss /= len(train_loader.dataset)
        train_acc   = accuracy_score(train_targets, train_preds)

        # --------------------- VALIDATION ----------------------
        model.eval()
        val_loss, val_preds, val_targets = 0.0, [], []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                first_name = batch['first_name'].to(device)
                last_name  = batch['last_name'].to(device)
                gender     = batch['gender'].to(device)

                logits = model(first_name, last_name)      # logit
                loss   = criterion(logits, gender)

                val_loss += loss.item() * first_name.size(0)

                probs = torch.sigmoid(logits)              # prob
                preds = (probs >= 0.5).long()              # pred

                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(gender.cpu().numpy())

        val_loss /= len(val_loader.dataset)
        val_acc   = accuracy_score(val_targets, val_preds)


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
            try:
                from utils import plot_confusion_matrix
                plot_confusion_matrix(
                    val_targets,
                    val_preds,
                    output_file=f'confusion_matrix_epoch_{epoch+1}.png'
                )
            except ImportError:
                # Fallback to the original implementation
                plot_confusion_matrix(
                    val_targets,
                    val_preds,
                    output_file=f'confusion_matrix_epoch_{epoch+1}.png'
                )

        print("-" * 60)

        # Early stopping
        if early_stopping(model, f1):
            print(f"Early stopping triggered after epoch {epoch+1}")
            break

    # Salva il modello finale
    if isinstance(model, GenderPredictorEnhanced):
        torch.save({
            'model_state_dict': model.state_dict(),
            'vocab_size': model.vocab_size,
            'embedding_dim': model.embedding_dim,
            'hidden_size': model.hidden_size,
            'n_layers': model.n_layers,
            'dual_input': model.dual_input
        }, model_save_path)
    else:
        torch.save({
            'model_state_dict': model.state_dict(),
            'vocab_size': model.vocab_size,
            'embedding_dim': model.embedding_dim,
            'hidden_size': model.hidden_size
        }, model_save_path)

    print(f"Model saved to {model_save_path}")

    return history

def train_model_with_freezing(model, train_loader, val_loader, criterion, optimizer,
                              num_epochs=20, device='cuda', patience=5,
                              model_save_path='model.pth', freeze_epochs=4):
    """
    Train model with freezing embedding and first LSTM layer for initial epochs.

    Args:
        model: Model instance
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        criterion: Loss function
        optimizer: Optimizer
        num_epochs: Maximum number of epochs
        device: Device to run training on
        patience: Early stopping patience
        model_save_path: Path to save the model
        freeze_epochs: Number of epochs to freeze embedding and first LSTM layer

    Returns:
        Training history
    """

    model.to(device)

    # Setup early stopping
    try:
        from utils import EarlyStopping
        early_stopping = EarlyStopping(patience=patience, min_delta=0.001)
    except ImportError:
        # Fallback to the original implementation
        class EarlyStopping:
            def __init__(self, patience=5, min_delta=0.001, restore_best_weights=True):
                self.patience = patience
                self.min_delta = min_delta
                self.restore_best_weights = restore_best_weights
                self.best_model = None
                self.best_score = None
                self.counter = 0
                self.early_stop = False

            def __call__(self, model, val_score):
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
                self.best_model = model.state_dict().copy()

        early_stopping = EarlyStopping(patience=patience)

    # Initialize history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'val_precision': [],
        'val_recall': [],
        'val_f1': []
    }

    for epoch in range(num_epochs):
        start_time = time.time()

        # Freeze/unfreeze layers based on current epoch
        if epoch < freeze_epochs:
            print(f"Epoch {epoch+1}/{num_epochs}: Freezing embedding and first LSTM layer")
            # Freeze embedding
            for param in model.char_embedding.parameters():
                param.requires_grad = False

            # Freeze first LSTM
            for param in model.firstname_lstm.parameters():
                param.requires_grad = False

            if hasattr(model, 'lastname_lstm'):
                for param in model.lastname_lstm.parameters():
                    param.requires_grad = False
        else:
            # Unfreeze all layers if this is the first epoch after freezing
            if epoch == freeze_epochs:
                print(f"Epoch {epoch+1}/{num_epochs}: Unfreezing all layers")
                for param in model.parameters():
                    param.requires_grad = True

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

            # Handle different model output formats
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            loss = criterion(outputs, gender)

            # Backward pass
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * first_name.size(0)

            # Save predictions and targets
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
                last_name  = batch['last_name'].to(device)
                gender     = batch['gender'].to(device)

                logits = model(first_name, last_name)     # ← ora sono logit

                # Se il modello restituisce tuple, prendi il primo elemento
                if isinstance(logits, tuple):
                    logits = logits[0]

                loss = criterion(logits, gender)          # FocalLoss su logit

                val_loss += loss.item() * first_name.size(0)

                # ---------  NUOVO: da logit → prob → pred  -----------------
                probs = torch.sigmoid(logits)             # prob ∈ [0,1]
                preds = (probs >= 0.5).long()             # soglia 0.5
                # ------------------------------------------------------------

                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(gender.cpu().numpy())

        val_loss /= len(val_loader.dataset)
        val_acc   = accuracy_score(val_targets, val_preds)


        # Calculate precision, recall, F1
        precision, recall, f1, _ = precision_recall_fscore_support(
            val_targets, val_preds, average='binary')

        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['val_precision'].append(precision)
        history['val_recall'].append(recall)
        history['val_f1'].append(f1)

        # Time elapsed
        time_elapsed = time.time() - start_time

        # Print results
        print(f"Epoch {epoch+1}/{num_epochs} | Time: {time_elapsed:.2f}s")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        print(f"Val Precision: {precision:.4f} | Val Recall: {recall:.4f} | Val F1: {f1:.4f}")

        # Analyze bias on validation set periodically
        if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
            print("\nBias analysis on validation set:")
            try:
                from utils import plot_confusion_matrix
                plot_confusion_matrix(
                    val_targets,
                    val_preds,
                    output_file=f'confusion_matrix_epoch_{epoch+1}.png'
                )
            except ImportError:
                plot_confusion_matrix(
                    val_targets,
                    val_preds,
                    output_file=f'confusion_matrix_epoch_{epoch+1}.png'
                )

        print("-" * 60)

        # Early stopping
        if early_stopping(model, f1):
            print(f"Early stopping triggered after epoch {epoch+1}")
            break

    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab_size': model.vocab_size,
        'embedding_dim': model.embedding_dim,
        'hidden_size': model.hidden_size,
        'n_layers': model.n_layers if hasattr(model, 'n_layers') else 1,
        'dual_input': model.dual_input if hasattr(model, 'dual_input') else False
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

def visualize_training_history(history, save_path='training_history.png'):
    """
    Visualizza la storia dell'addestramento.

    Args:
        history: Dizionario con la storia dell'addestramento
        save_path: Percorso dove salvare il grafico
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
    plt.savefig(save_path)
    plt.close()

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

    # Check if it's an enhanced model
    if 'n_layers' in checkpoint and checkpoint['n_layers'] > 1:
        # Create an enhanced model instance
        model = GenderPredictorEnhanced(
            vocab_size=checkpoint['vocab_size'],
            embedding_dim=checkpoint['embedding_dim'],
            hidden_size=checkpoint['hidden_size'],
            n_layers=checkpoint['n_layers'],
            dual_input=checkpoint.get('dual_input', True)
        )
    else:
        # Create a standard model instance
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

def set_all_seeds(seed):
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Set deterministic behavior for CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"All seeds set to {seed}")

def parse_args():
    """Parse command‑line arguments for training/testing the gender predictor."""
    parser = argparse.ArgumentParser(
        description="Train and evaluate the BiLSTM gender‑prediction model")

    # --- esperimenti / logging ------------------------------------------------
    parser.add_argument("--round", type=int, default=0, choices=[0, 1, 2],
                        help="0=baseline, 1=training tricks, 2=capacity boost")
    parser.add_argument("--save_dir", type=str, default=".",
                        help="Where to save logs and model checkpoints")

    parser.add_argument("--epochs", type=int, default=20,
                        help="Numero massimo di epoche")

    # --- riproducibilità & dati ----------------------------------------------
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--data_file", type=str,
                        default="training_dataset.csv",
                        help="CSV with first_name,last_name,gender")

    # --- loss function / round 1 ---------------------------------------------
    parser.add_argument("--loss", type=str, default="bce",
                        choices=["bce", "focal"],
                        help="Loss function to use")
    parser.add_argument("--alpha", type=float, default=0.7,
                        help="Focal‑Loss α (weight for class 1 = female)")
    parser.add_argument("--gamma", type=float, default=2.0,
                        help="Focal‑Loss γ (focusing parameter)")
    parser.add_argument("--pos_weight", type=float, default=1.0,
                        help="Positive‑class weight for BCEWithLogitsLoss; 1.0 = unweighted")
    parser.add_argument("--label_smooth", type=float, default=0.0,
                        help="Label‑smoothing ε")
    parser.add_argument("--balanced_sampler", action="store_true",
                        help="Use a balanced batch sampler (round 1)")
    parser.add_argument("--early_stop", type=int, default=5,
                        help="Early‑stopping patience (0 = off)")

    # --- architecture / round 2 ---------------------------------------------
    parser.add_argument("--n_layers", type=int, default=1,
                        help="Number of BiLSTM layers")
    parser.add_argument("--hidden_size", type=int, default=64,
                        help="Hidden size of BiLSTM layers")
    parser.add_argument("--dual_input", action="store_true",
                        help="Separate encoders for first and last name (round 2)")
    parser.add_argument("--freeze_epochs", type=int, default=0,
                        help="Epochs to freeze embedding + first LSTM layer")

    return parser.parse_args()

def build_loss(args, device=None):
    """Costruisce la loss function in base ai parametri specificati."""
    if args.round >= 1 and args.loss == "focal":
        try:
            # Importa correttamente le classi dal modulo losses
            from losses import FocalLoss, LabelSmoothing

            # Crea prima la FocalLoss
            criterion = FocalLoss(
                gamma=args.gamma,
                alpha=args.alpha,
                reduction="mean"
            )
            print(f"Using FocalLoss with gamma={args.gamma}, alpha={args.alpha}")

            # Poi applica LabelSmoothing se necessario
            if args.label_smooth > 0.0:
                criterion = LabelSmoothing(
                    base_loss=criterion,
                    epsilon=args.label_smooth
                )
                print(f"Applying label smoothing with epsilon={args.label_smooth}")

            return criterion
        except ImportError as e:
            print(f"Error importing loss functions: {e}")
            print("Falling back to BCEWithLogitsLoss")
            # Usa BCEWithLogitsLoss con pos_weight se necessario
            if args.pos_weight != 1.0:
                # Crea il peso e lo sposta sul device corretto se specificato
                weight = torch.tensor(args.pos_weight)
                if device is not None:
                    weight = weight.to(device)
                print(f"Using BCEWithLogitsLoss with pos_weight={args.pos_weight}")
                return nn.BCEWithLogitsLoss(pos_weight=weight)
            else:
                return nn.BCEWithLogitsLoss()
    else:
        # Round 0 o loss = BCE
        # Controlla se usare pos_weight
        if args.pos_weight != 1.0:
            # Crea il peso e lo sposta sul device corretto se specificato
            weight = torch.tensor(args.pos_weight)
            if device is not None:
                weight = weight.to(device)
            print(f"Using BCEWithLogitsLoss with pos_weight={args.pos_weight}")
            return nn.BCEWithLogitsLoss(pos_weight=weight)
        else:
            print("Using standard BCEWithLogitsLoss")
            return nn.BCEWithLogitsLoss()

def main():
    """Funzione principale per l'esecuzione dell'addestramento."""
    # Parse arguments
    args = parse_args()

    # Set all seeds for reproducibility
    set_all_seeds(args.seed)

    # Create directories
    models_dir = os.path.join(args.save_dir, "models")
    logs_dir = os.path.join(args.save_dir, "logs")

    # Ensure directories exist
    try:
        from utils import ensure_dir
        ensure_dir(models_dir)
        ensure_dir(logs_dir)
    except ImportError:
        # Fallback implementation
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir)

    # Paths for saving
    model_path = os.path.join(models_dir, f"round{args.round}_best.pth")
    metrics_path = os.path.join(logs_dir, f"round{args.round}_metrics.csv")
    confusion_path = os.path.join(logs_dir, f"round{args.round}_confusion.png")
    history_path = os.path.join(logs_dir, f"round{args.round}_history.png")

    # Parametri
    preprocessor_path = "name_preprocessor.pkl"
    batch_size = 128
    num_epochs = args.epochs
    learning_rate = 0.001
    test_size = 0.1
    val_size = 0.1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Crea la funzione di loss passando il device
    criterion = build_loss(args, device)

    print(f"Using device: {device}")
    print(f"Running Round {args.round} training...")


    # Carica i dati
    print(f"Loading data from {args.data_file}...")
    training_data = pd.read_csv(args.data_file)
    print(f"Loaded {len(training_data)} records")

    # Divisione train/val/test dai dati di training
    train_val_df, test_df = train_test_split(
        training_data,
        test_size=test_size,
        random_state=args.seed,
        stratify=training_data['gender']
    )

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_size/(1-test_size),
        random_state=args.seed,
        stratify=train_val_df['gender']
    )

    print(f"Train set: {len(train_df)} records")
    print(f"Validation set: {len(val_df)} records")
    print(f"Test set: {len(test_df)} records")

    # Verifica la distribuzione di genere
    print("\nDistribuzione di genere:")
    for dataset_name, dataset in [('Train', train_df), ('Validation', val_df), ('Test', test_df)]:
        gender_counts = dataset['gender'].value_counts()
        print(f"  {dataset_name}:")
        for gender, count in gender_counts.items():
            percentage = (count / len(dataset)) * 100
            print(f"    {gender}: {count} ({percentage:.2f}%)")

    # Crea il preprocessore
    preprocessor = NamePreprocessor()
    preprocessor.save(preprocessor_path)

    # Crea i dataset
    train_dataset = NameGenderDataset(train_df, preprocessor, mode='train')
    val_dataset = NameGenderDataset(val_df, preprocessor, mode='val')
    test_dataset = NameGenderDataset(test_df, preprocessor, mode='test')

    # DataLoader with appropriate sampler
    if args.round >= 1 and args.balanced_sampler:
        try:
            from sampler import BalancedBatchSampler
            print("Using BalancedBatchSampler...")
            train_sampler = BalancedBatchSampler(train_dataset, batch_size)
            train_loader = DataLoader(train_dataset, batch_sampler=train_sampler)
        except ImportError:
            print("BalancedBatchSampler not found, using standard DataLoader with shuffle=True")
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Create model based on round
    if args.round <= 1:
        # Round 0 and 1 use the original model architecture
        model = GenderPredictor(vocab_size=preprocessor.vocab_size)
        print("Using original GenderPredictor model")
    else:
        # Round 2 uses enhanced model
        dual_input = args.dual_input
        model = GenderPredictorEnhanced(
            vocab_size=preprocessor.vocab_size,
            hidden_size=args.hidden_size,
            n_layers=args.n_layers,
            dual_input=dual_input
        )
        print(f"Using enhanced model with {args.n_layers} layers, hidden size {args.hidden_size}, "
              f"dual_input={dual_input}")

    # Setup optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    print(f"Starting training for Round {args.round} with {num_epochs} epochs...")  # Modificato

    # Training function with freezing for Round 2
    if args.round == 2 and args.freeze_epochs > 0:
        print(f"Using freezing for first {args.freeze_epochs} epochs")
        history = train_model_with_freezing(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            num_epochs=num_epochs,
            device=device,
            patience=args.early_stop,
            model_save_path=model_path,
            freeze_epochs=args.freeze_epochs
        )
    else:
        history = train_model(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            num_epochs=num_epochs,
            device=device,
            patience=args.early_stop,
            model_save_path=model_path
        )

    # Save metrics to CSV
    try:
        from utils import save_metrics_to_csv
        save_metrics_to_csv(history, metrics_path)
    except ImportError:
        # Convert history to DataFrame and save
        history_df = pd.DataFrame(history)
        history_df.to_csv(metrics_path, index=False)

    # Visualize training history
    visualize_training_history(history, save_path=history_path)

    # Evaluate on test set
    print("Evaluating on test set...")
    model, preprocessor = load_trained_model(model_path, preprocessor_path, device)

    test_preds = []
    test_targets = []
    all_probs = []
    all_preds = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            first_name = batch['first_name'].to(device)
            last_name  = batch['last_name'].to(device)
            gender     = batch['gender'].to(device)

            logits = model(first_name, last_name)      # logit

            probs = torch.sigmoid(logits)              # prob ∈ [0,1]
            preds = (probs >= 0.5).long()              # pred 0/1

            test_preds.extend(preds.cpu().numpy())
            test_targets.extend(gender.cpu().numpy())

            all_probs.extend(probs.cpu().numpy())      # se ti servono le prob
            all_preds.extend(preds.cpu().numpy())


    # Calcola le metriche sul test set
    test_acc = accuracy_score(test_targets, test_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(test_targets, test_preds, average='binary')

    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")
    print(f"Test F1: {f1:.4f}")

    # Analisi approfondita del bias sul test set
    print("\nDetailed bias analysis on test set:")
    try:
        from utils import plot_confusion_matrix
        plot_confusion_matrix(test_targets, test_preds, output_file=confusion_path)
    except ImportError:
        plot_confusion_matrix(test_targets, test_preds, output_file=confusion_path)

    # Save test metrics
    test_metrics = {
        'accuracy': float(test_acc),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1)
    }

    with open(os.path.join(logs_dir, f"round{args.round}_test_metrics.json"), 'w') as f:
        import json
        json.dump(test_metrics, f, indent=4)

    print(f"Round {args.round} completed successfully!")

if __name__ == "__main__":
    main()
