import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import re
from typing import List, Tuple, Dict
import random


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


class NameFeatureExtractor:
    """
    Estrae feature linguistiche avanzate dai nomi per migliorare la predizione.
    """
    
    # Suffissi comuni per genere in diverse lingue
    FEMALE_SUFFIXES = [
        # Italiano
        'a', 'ella', 'etta', 'ina', 'uccia',
        # Spagnolo
        'ez', 'ita', 'uela',
        # Francese
        'ette', 'elle', 'ine',
        # Russo
        'ova', 'eva', 'ina', 'aya',
        # Inglese
        'een', 'lyn', 'leigh',
        # Tedesco
        'chen', 'lein',
    ]
    
    MALE_SUFFIXES = [
        # Italiano
        'o', 'i', 'ino', 'ello', 'uccio',
        # Spagnolo
        'os', 'ez', 'ito',
        # Francese
        'eau', 'ot', 'in',
        # Russo
        'ov', 'ev', 'sky', 'ich',
        # Inglese
        'son', 'ton', 'ley',
        # Tedesco
        'mann', 'stein',
    ]
    
    def __init__(self, max_ngram=3):
        self.max_ngram = max_ngram
        self.suffix_to_idx = self._build_suffix_index()
        
    def _build_suffix_index(self):
        """Costruisce un indice dei suffissi."""
        suffix_to_idx = {'<UNK>': 0}
        idx = 1
        
        for suffix in self.FEMALE_SUFFIXES + self.MALE_SUFFIXES:
            if suffix not in suffix_to_idx:
                suffix_to_idx[suffix] = idx
                idx += 1
                
        return suffix_to_idx
    
    def extract_suffix_features(self, name: str) -> List[int]:
        """Estrae feature basate sui suffissi del nome."""
        name = name.lower().strip()
        
        # Gestisci nomi vuoti
        if not name:
            return [0]  # Unknown suffix
        
        features = []
        
        # Controlla suffissi di lunghezza variabile (2-5 caratteri)
        for length in range(2, min(6, len(name) + 1)):
            suffix = name[-length:]
            if suffix in self.suffix_to_idx:
                features.append(self.suffix_to_idx[suffix])
            
        # Padding o troncamento a lunghezza fissa
        if not features:
            features = [0]  # Unknown suffix
            
        return features[:3]  # Massimo 3 suffissi
    
    def extract_phonetic_features(self, name: str) -> Dict[str, float]:
        """Estrae feature fonetiche dal nome."""
        name = name.lower()
        
        # Gestisci nomi vuoti
        if not name:
            return {
                'ends_with_vowel': 0.0,
                'vowel_ratio': 0.0,
                'has_double_consonant': 0.0,
                'length_normalized': 0.0,
            }
        
        features = {
            'ends_with_vowel': 1.0 if name[-1] in 'aeiou' else 0.0,
            'vowel_ratio': sum(1 for c in name if c in 'aeiou') / max(len(name), 1),
            'has_double_consonant': 1.0 if any(name[i] == name[i+1] and name[i] not in 'aeiou' 
                                              for i in range(len(name)-1)) else 0.0,
            'length_normalized': min(len(name) / 20.0, 1.0),  # Normalizzato a [0, 1]
        }
        
        return features
    
    def extract_ngram_features(self, name: str, n: int = 3) -> List[str]:
        """Estrae n-grammi di caratteri."""
        name = f"^{name.lower()}$"  # Aggiungi marcatori di inizio/fine
        ngrams = []
        
        for i in range(len(name) - n + 1):
            ngrams.append(name[i:i+n])
            
        return ngrams


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


class NameAugmenter:
    """
    Data augmentation per nomi per migliorare la robustezza del modello.
    """
    
    def __init__(self, augment_prob=0.15):
        self.augment_prob = augment_prob
        
        # Mapping per errori di battitura comuni
        self.typo_map = {
            'a': ['q', 's'],
            'e': ['w', 'r', '3'],
            'i': ['u', 'o', '1', 'l'],
            'o': ['i', 'p', '0'],
            'u': ['y', 'i'],
            's': ['z', 'a'],
            'c': ['k', 'x'],
            'k': ['c'],
            'y': ['i'],
            'v': ['b'],
            'b': ['v'],
            'n': ['m'],
            'm': ['n'],
        }
        
    def augment(self, name: str) -> str:
        """Applica augmentation casuale al nome."""
        if random.random() > self.augment_prob:
            return name
            
        augmentation_type = random.choice(['typo', 'case', 'duplicate', 'swap'])
        
        if augmentation_type == 'typo':
            return self._add_typo(name)
        elif augmentation_type == 'case':
            return self._random_case(name)
        elif augmentation_type == 'duplicate':
            return self._duplicate_char(name)
        elif augmentation_type == 'swap':
            return self._swap_chars(name)
            
        return name
    
    def _add_typo(self, name: str) -> str:
        """Simula errori di battitura."""
        if len(name) < 2:
            return name
            
        name_list = list(name.lower())
        idx = random.randint(0, len(name_list) - 1)
        char = name_list[idx]
        
        if char in self.typo_map:
            name_list[idx] = random.choice(self.typo_map[char])
            
        return ''.join(name_list)
    
    def _random_case(self, name: str) -> str:
        """Cambia casualmente maiuscole/minuscole."""
        return ''.join(c.upper() if random.random() > 0.5 else c.lower() for c in name)
    
    def _duplicate_char(self, name: str) -> str:
        """Duplica casualmente un carattere."""
        if len(name) < 2:
            return name
            
        idx = random.randint(0, len(name) - 1)
        return name[:idx] + name[idx] + name[idx:]
    
    def _swap_chars(self, name: str) -> str:
        """Scambia due caratteri adiacenti."""
        if len(name) < 2:
            return name
            
        idx = random.randint(0, len(name) - 2)
        name_list = list(name)
        name_list[idx], name_list[idx + 1] = name_list[idx + 1], name_list[idx]
        
        return ''.join(name_list)


class CosineAnnealingWarmupScheduler:
    """
    Learning rate scheduler con warmup e cosine annealing.
    """
    
    def __init__(self, optimizer, warmup_epochs, max_epochs, min_lr=1e-6, max_lr=1e-3):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.min_lr = min_lr
        self.max_lr = max_lr
        
    def step(self, epoch):
        if epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.max_lr * (epoch + 1) / self.warmup_epochs
        else:
            # Cosine annealing
            progress = (epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.max_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
            
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            
        return lr


def mixup_data(x, y, alpha=0.2):
    """
    Implementa mixup augmentation per migliorare la generalizzazione.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
        
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


class FocalLossImproved(nn.Module):
    """
    Versione migliorata della Focal Loss con adaptive weighting.
    """
    
    def __init__(self, alpha=None, gamma=2.0, reduction='mean', auto_weight=True):
        super(FocalLossImproved, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.auto_weight = auto_weight
        self.eps = 1e-7
        
    def forward(self, inputs, targets, sample_weights=None):
        """
        inputs: logits dal modello
        targets: target binari (0 o 1)
        sample_weights: pesi per campione (opzionale)
        """
        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # Calcola pt
        p_t = p * targets + (1 - p) * (1 - targets)
        
        # Calcola alpha_t
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        else:
            # Auto-weight basato sulla distribuzione del batch
            if self.auto_weight:
                pos_ratio = targets.mean()
                alpha_t = (1 - pos_ratio) * targets + pos_ratio * (1 - targets)
            else:
                alpha_t = 1.0
                
        # Focal loss
        focal_weight = (1 - p_t + self.eps) ** self.gamma
        focal_loss = alpha_t * focal_weight * ce_loss
        
        # Applica sample weights se forniti
        if sample_weights is not None:
            focal_loss = focal_loss * sample_weights
            
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
