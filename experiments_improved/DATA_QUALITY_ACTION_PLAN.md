# 📊 Piano d'Azione per Migliorare la Qualità dei Dati

## 🎯 Obiettivo: Superare 93% Accuracy tramite Data Quality

### 1. **Analisi del Dataset** (1-2 ore)

```bash
# Analizza la qualità attuale del dataset
python data_cleaning_pipeline.py --input training_dataset.csv --analyze-only --show-ambiguous 100

# Genera report dettagliato
python data_cleaning_pipeline.py --input training_dataset.csv --analyze-only > dataset_quality_report.txt
```

Questo ti mostrerà:
- Quanti nomi hanno numeri o caratteri speciali
- Quanti hanno diacritici
- Quanti sono sospettosamente corti/lunghi
- Quanti potrebbero avere ordine nome-cognome ambiguo

### 2. **Pulizia del Dataset** (2-3 ore di processing)

```bash
# Crea dataset pulito rimuovendo record problematici
python data_cleaning_pipeline.py \
    --input training_dataset.csv \
    --output training_dataset_clean.csv \
    --show-ambiguous 50

# Versione che mantiene diacritici (se vuoi testare entrambe)
python data_cleaning_pipeline.py \
    --input training_dataset.csv \
    --output training_dataset_clean_diacritics.csv \
    --keep-diacritics
```

### 3. **Test A/B con Dataset Pulito**

#### Esperimento A: Dataset pulito senza diacritici
```bash
cd experiments_improved
python train_improved_model_v2.py \
    --data_file ../training_dataset_clean.csv \
    --round 3 \
    --epochs 20 \
    --focal_alpha 0.48 \
    --note "clean_no_diacritics"
```

#### Esperimento B: Preprocessore migliorato
```python
# Modifica train_improved_model_v2.py per usare ImprovedNamePreprocessor
# invece di NamePreprocessor
```

### 4. **Metriche di Confronto**

Dopo i test, confronta:
```bash
python experiment_tools.py compare --metric test_f1
python experiment_tools.py bias
```

## 🔍 Problemi Identificati e Soluzioni

### Problema 1: **Middle Names**
- **Attuale**: "John Lee Oswald" → first="John", last="Lee Oswald"
- **Soluzione**: Gestione intelligente che identifica "Lee" come middle name

### Problema 2: **Ordine Nome-Cognome**
- **Rischio**: "Enrico Martina" (M) vs "Martina Enrico" (F)
- **Soluzione**: 
  - Validation score basato su pattern di genere
  - Flag record ambigui nel dataset

### Problema 3: **Diacritici**
- **Issue**: Inconsistenza tra "José" e "Jose"
- **Soluzione**: Normalizzazione opzionale (testare entrambe)

### Problema 4: **Nomi con Trattini**
- **Issue**: "Jean-Pierre" vs "Jean Pierre" vs "Jeanpierre"
- **Soluzioni da testare**:
  - Keep: mantieni come "jean-pierre"
  - Space: converti in "jean pierre"
  - Remove: converti in "jeanpierre"

### Problema 5: **Record Spuri IMDB**
- **Esempi**: Nomi con numeri, caratteri speciali, troppo corti/lunghi
- **Soluzione**: Rimozione automatica basata su pattern

## 📈 Risultati Attesi

Con dataset pulito:
- **Riduzione rumore**: -5-10% dei record ma +0.5-1.0% accuracy
- **Consistenza**: Meno varianza nelle predizioni
- **Bias**: Potrebbe migliorare ulteriormente (già ottimo a 0.993)

## 🚀 Quick Start

```bash
# 1. Copia gli script nella tua directory
cp data_cleaning_pipeline.py ~/gender-predict/
cp improved_preprocessor.py ~/gender-predict/experiments_improved/

# 2. Analizza
cd ~/gender-predict
python data_cleaning_pipeline.py --analyze-only

# 3. Pulisci
python data_cleaning_pipeline.py

# 4. Re-train
cd experiments_improved
./launch_improved_training.sh  # modifica per usare training_dataset_clean.csv
```

## 💡 Suggerimenti Avanzati

1. **Validazione Incrociata Nome-Cognome**:
   - Crea un piccolo dataset manuale di nomi ambigui
   - Testa il modello specificamente su questi casi

2. **Analisi degli Errori**:
   ```python
   # Trova pattern comuni negli errori
   errors = df[df['predicted'] != df['actual']]
   # Analizza se sono concentrati su nomi specifici
   ```

3. **Dataset Augmentation Mirata**:
   - Genera varianti dei nomi più problematici
   - Aggiungi al training set

4. **Ensemble con Preprocessing Diversi**:
   - Modello 1: con diacritici
   - Modello 2: senza diacritici
   - Modello 3: trattini → spazi
   - Media delle predizioni

## ⚠️ Attenzione

- Il cleaning potrebbe rimuovere 5-10% dei dati
- Alcuni nomi "strani" potrebbero essere legittimi (es. nomi artistici)
- Testare sempre su validation set non pulito per verificare robustezza

## 📊 Metriche di Successo

- [ ] Dataset cleaning report generato
- [ ] Almeno 2 esperimenti con preprocessing diversi
- [ ] Accuracy > 92.5%
- [ ] Bias ratio rimane < 1.1
- [ ] Analisi errori su nomi ambigui

---

**Il data cleaning è probabilmente l'ultima grande opportunità per miglioramenti significativi!**