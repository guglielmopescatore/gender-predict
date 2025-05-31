"""
Advanced gender prediction model (V3).

This module contains the most advanced model architecture with:
- Multi-head attention
- Feature engineering
- Advanced linguistic features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .layers import ImprovedAttentionLayer

class GenderPredictorV3(nn.Module):
    """
    Versione migliorata del modello con feature engineering avanzata e architettura ottimizzata.
    """
    
    def __init__(self, vocab_size, suffix_vocab_size, embedding_dim=32, hidden_size=128, 
                 n_layers=2, dropout_rate=0.3, num_attention_heads=4):
        super(GenderPredictorV3, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        
        # Embeddings
        self.char_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.suffix_embedding = nn.Embedding(suffix_vocab_size, embedding_dim // 2)
        
        # Batch normalization per embeddings
        self.embedding_norm = nn.BatchNorm1d(embedding_dim)
        
        # Feature fonetiche
        self.phonetic_linear = nn.Linear(4, embedding_dim // 4)
        
        # LSTM con variational dropout
        self.firstname_lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate if n_layers > 1 else 0
        )
        
        self.lastname_lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate if n_layers > 1 else 0
        )
        
        # Layer normalization
        self.lstm_norm = nn.LayerNorm(hidden_size * 2)
        
        # Multi-head attention
        self.firstname_attention = ImprovedAttentionLayer(hidden_size, num_attention_heads, dropout_rate)
        self.lastname_attention = ImprovedAttentionLayer(hidden_size, num_attention_heads, dropout_rate)
        
        # Dimensione input per i layer finali
        # first_context + last_context: hidden_size * 4
        # first_suffix + last_suffix: (embedding_dim // 2) * 3 * 2
        # phonetic: embedding_dim // 4
        feature_dim = hidden_size * 4 + (embedding_dim // 2) * 3 * 2 + embedding_dim // 4
        
        # Deep output network con skip connections
        self.fc1 = nn.Linear(feature_dim, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.bn2 = nn.BatchNorm1d(hidden_size // 2)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.fc3 = nn.Linear(hidden_size // 2 + hidden_size, 1)  # Skip connection
        
    def forward(self, first_name, last_name, first_suffix, last_suffix, phonetic_features):
        batch_size = first_name.size(0)
        
        # Character embeddings
        first_emb = self.char_embedding(first_name)
        last_emb = self.char_embedding(last_name)
        
        # Normalizza embeddings (richiede reshape per batch norm)
        first_emb_flat = first_emb.view(-1, self.embedding_dim)
        first_emb = self.embedding_norm(first_emb_flat).view(batch_size, -1, self.embedding_dim)
        
        last_emb_flat = last_emb.view(-1, self.embedding_dim)
        last_emb = self.embedding_norm(last_emb_flat).view(batch_size, -1, self.embedding_dim)
        
        # LSTM processing
        first_lstm_out, _ = self.firstname_lstm(first_emb)
        last_lstm_out, _ = self.lastname_lstm(last_emb)
        
        # Layer norm
        first_lstm_out = self.lstm_norm(first_lstm_out)
        last_lstm_out = self.lstm_norm(last_lstm_out)
        
        # Attention
        first_context = self.firstname_attention(first_lstm_out)
        last_context = self.lastname_attention(last_lstm_out)
        
        # Suffix embeddings
        first_suffix_emb = self.suffix_embedding(first_suffix).view(batch_size, -1)
        last_suffix_emb = self.suffix_embedding(last_suffix).view(batch_size, -1)
        
        # Phonetic features
        phonetic_encoded = self.phonetic_linear(phonetic_features)
        
        # Concatena tutte le features
        combined = torch.cat([
            first_context, 
            last_context, 
            first_suffix_emb, 
            last_suffix_emb,
            phonetic_encoded
        ], dim=1)
        
        # Deep network con skip connections
        x1 = self.fc1(combined)
        x1 = self.bn1(x1)
        x1 = F.relu(x1)
        x1 = self.dropout1(x1)
        
        x2 = self.fc2(x1)
        x2 = self.bn2(x2)
        x2 = F.relu(x2)
        x2 = self.dropout2(x2)
        
        # Skip connection
        x_final = torch.cat([x1, x2], dim=1)
        logits = self.fc3(x_final)
        
        return logits.squeeze()



