"""
Single-string variant of the V3 model (experimental).

GenderPredictorV3 encodes the first word of the name and the rest with two separate BiLSTMs, which builds into the
model the assumption "first word = given name". GenderPredictorV3S reads the whole normalised name as one character
sequence with one BiLSTM and the same attention and output layers, so that the positions that matter are learned.

The forward signature is the one of V3, so that datasets, training loop and inference need no change: ``first_name``
carries the whole name (the preprocessor is built with ``single_string=True``) and ``last_name`` is ignored.
With ``aux_features=False`` the suffix and phonetic features, which depend on the first-word/rest split, are not used.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import ImprovedAttentionLayer


class GenderPredictorV3S(nn.Module):
    model_type = "v3s"

    def __init__(self, vocab_size, suffix_vocab_size, embedding_dim=32, hidden_size=128, n_layers=2,
                 dropout_rate=0.3, num_attention_heads=4, aux_features=True):
        super().__init__()
        self.vocab_size, self.suffix_vocab_size, self.embedding_dim = vocab_size, suffix_vocab_size, embedding_dim
        self.hidden_size, self.n_layers, self.dropout_rate = hidden_size, n_layers, dropout_rate
        self.num_attention_heads, self.aux_features = num_attention_heads, aux_features

        self.char_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embedding_norm = nn.BatchNorm1d(embedding_dim)
        self.name_lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=n_layers, batch_first=True,
                                 bidirectional=True, dropout=dropout_rate if n_layers > 1 else 0)
        self.lstm_norm = nn.LayerNorm(hidden_size * 2)
        self.name_attention = ImprovedAttentionLayer(hidden_size, num_attention_heads, dropout_rate)

        feature_dim = hidden_size * 2
        if aux_features:
            self.suffix_embedding = nn.Embedding(suffix_vocab_size, embedding_dim // 2)
            self.phonetic_linear = nn.Linear(4, embedding_dim // 4)
            feature_dim += (embedding_dim // 2) * 3 * 2 + embedding_dim // 4

        self.fc1 = nn.Linear(feature_dim, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.bn2 = nn.BatchNorm1d(hidden_size // 2)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(hidden_size // 2 + hidden_size, 1)

    def forward(self, first_name, last_name, first_suffix, last_suffix, phonetic_features):
        batch_size = first_name.size(0)
        emb = self.char_embedding(first_name)
        emb = self.embedding_norm(emb.view(-1, self.embedding_dim)).view(batch_size, -1, self.embedding_dim)
        out, _ = self.name_lstm(emb)
        parts = [self.name_attention(self.lstm_norm(out))]
        if self.aux_features:
            parts += [self.suffix_embedding(first_suffix).view(batch_size, -1), self.suffix_embedding(last_suffix).view(batch_size, -1),
                      self.phonetic_linear(phonetic_features)]
        x1 = self.dropout1(F.relu(self.bn1(self.fc1(torch.cat(parts, dim=1)))))
        x2 = self.dropout2(F.relu(self.bn2(self.fc2(x1))))
        return self.fc3(torch.cat([x1, x2], dim=1)).squeeze()
