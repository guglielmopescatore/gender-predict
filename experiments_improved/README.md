# 🚀 Experiments Improved

Questa directory contiene i miglioramenti al modello di predizione del genere per superare il 92% di accuratezza.

## 📁 Struttura

```
experiments_improved/
├── __init__.py                  # Package init
├── improvements.py              # Nuove classi e architetture
├── train_improved_model_v2.py   # Script di training integrato
├── test_improvements.py         # Test suite
├── launch_improved_training.sh  # Script di lancio
└── README.md                    # Questa documentazione
```

## 🚀 Quick Start

### 1. Test dei componenti
```bash
cd experiments_improved
python test_improvements.py
```

### 2. Training
```bash
# Dalla directory experiments_improved
./launch_improved_training.sh

# O dalla root del progetto
cd experiments_improved && ./launch_improved_training.sh
```

### 3. Training personalizzato
```bash
cd experiments_improved
python train_improved_model_v2.py \
    --data_file ../training_dataset.csv \
    --save_dir .. \
    --hidden_size 128 \
    --focal_alpha 0.52
```

## 🔧 Dipendenze

Questo modulo dipende dai file nella directory principale:
- `train_name_gender_model.py` → NamePreprocessor
- `experiment_manager.py` → Sistema di logging
- `utils.py` → EarlyStopping, plot_confusion_matrix
- `sampler.py` → BalancedBatchSampler (opzionale)

## 📊 Parametri Consigliati

Per dataset sbilanciato (62.32% M / 37.68% F) su RTX 4090:

```bash
--batch_size 512
--hidden_size 128
--focal_alpha 0.52  # Equivalente a pos_weight=1.02
--n_layers 2
--num_heads 4
```

## 🎯 Risultati Attesi

- Accuracy: 93-94% (da 92%)
- F1 Score: 91-92% (da 90%)
- Training time: ~45-60 min su RTX 4090
- VRAM usage: ~12-15GB

## 📝 Note

- I modelli vengono salvati in `../experiments/` tramite ExperimentManager
- I log sono completamente integrati con il sistema esistente
- Tutti i path sono relativi alla directory `experiments_improved/`