"""
Fix per il bug critico in GenderPredictor che causa doppio sigmoid.
Questo patch va applicato al file train_name_gender_model.py
"""

def get_fixed_gender_predictor_forward():
    """
    Restituisce il metodo forward corretto per GenderPredictor.
    
    Il bug originale aveva nn.Sigmoid() nell'ultima layer, ma poi si usava
    BCEWithLogitsLoss che applica internamente sigmoid, causando un doppio sigmoid.
    """
    
    forward_code = '''
    def forward(self, first_name, last_name):
        """
        Forward pass del modello.

        Args:
            first_name: Tensor dei nomi [batch, max_name_length]
            last_name: Tensor dei cognomi [batch, max_surname_length]

        Returns:
            Logit del genere femminile [batch]
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
        
        # Output finale - NOTA: Rimosso nn.Sigmoid() qui
        # BCEWithLogitsLoss si aspetta logit, non probabilità
        logits = self.fc(combined)                   # [B,1]
        return logits.squeeze(1)                     # [B] logit
'''
    
    return forward_code


def get_fixed_gender_predictor_init():
    """
    Restituisce il metodo __init__ corretto per GenderPredictor.
    Rimuove nn.Sigmoid() dal Sequential.
    """
    
    init_code = '''
        # Layer di output - CORRETTO: senza Sigmoid
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 4, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, 1),
            # nn.Sigmoid() RIMOSSO - BCEWithLogitsLoss vuole logit
        )
'''
    
    return init_code


# Patch da applicare al file originale
PATCH_INSTRUCTIONS = """
Per applicare il fix al file train_name_gender_model.py:

1. Nella classe GenderPredictor, sostituire il Sequential nel metodo __init__:

   DA:
   ```python
   self.fc = nn.Sequential(
       nn.Linear(hidden_size * 4, hidden_size),
       nn.ReLU(),
       nn.Dropout(dropout_rate),
       nn.Linear(hidden_size, 1),
       nn.Sigmoid()
   )
   ```

   A:
   ```python
   self.fc = nn.Sequential(
       nn.Linear(hidden_size * 4, hidden_size),
       nn.ReLU(),
       nn.Dropout(dropout_rate),
       nn.Linear(hidden_size, 1),
   )
   ```

2. Nel metodo forward, assicurarsi che restituisca logit:

   ```python
   def forward(self, first_name, last_name):
       # ... codice esistente ...
       
       # Output finale
       logits = self.fc(combined)
       return logits.squeeze(1)  # Restituisce logit, non probabilità
   ```

3. In tutti i punti dove si usano le predizioni durante training/evaluation,
   applicare sigmoid esplicitamente:

   ```python
   # Durante il training
   logits = model(first_name, last_name)
   loss = criterion(logits, gender)  # BCEWithLogitsLoss vuole logit
   
   # Per le predizioni
   probs = torch.sigmoid(logits)
   preds = (probs >= 0.5).long()
   ```

Questo fix è già implementato in GenderPredictorEnhanced, quindi il bug
riguarda solo il modello base GenderPredictor.
"""

if __name__ == "__main__":
    print("Fix per GenderPredictor")
    print("=" * 50)
    print(PATCH_INSTRUCTIONS)
    print("\nCodice corretto per __init__:")
    print(get_fixed_gender_predictor_init())
    print("\nCodice corretto per forward:")
    print(get_fixed_gender_predictor_forward())
