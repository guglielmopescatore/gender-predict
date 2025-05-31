"""
Base gender prediction models.

This module contains the core model architectures for gender prediction:
- GenderPredictor: Basic BiLSTM model with attention
- GenderPredictorEnhanced: Enhanced version with improved capacity
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import AttentionLayer

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

