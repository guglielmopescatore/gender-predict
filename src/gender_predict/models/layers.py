"""
Attention layers for neural network models.

This module contains various attention mechanisms used in the gender prediction models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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


class ImprovedAttentionLayer(nn.Module):
    """
    Multi-head attention layer migliorato per catturare pattern più complessi.
    """
    def __init__(self, hidden_size, num_heads=4, dropout=0.1):
        super(ImprovedAttentionLayer, self).__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // num_heads
        
        assert hidden_size % num_heads == 0, "hidden_size deve essere divisibile per num_heads"
        
        self.query = nn.Linear(hidden_size * 2, hidden_size)
        self.key = nn.Linear(hidden_size * 2, hidden_size)
        self.value = nn.Linear(hidden_size * 2, hidden_size)
        
        self.dropout = nn.Dropout(dropout)
        self.output_linear = nn.Linear(hidden_size, hidden_size * 2)
        
    def forward(self, lstm_output):
        batch_size, seq_len, _ = lstm_output.shape
        
        # Calcola Q, K, V
        Q = self.query(lstm_output).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(lstm_output).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(lstm_output).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Calcola attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Applica attention
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        
        # Output projection
        output = self.output_linear(context)
        
        # Global pooling (mean over sequence)
        return output.mean(dim=1)



