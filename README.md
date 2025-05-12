# Gender Prediction from Names

Un modello di machine learning basato su reti neurali BiLSTM con attenzione per la predizione del genere a partire dal nome e cognome di una persona.

## Descrizione del Progetto

Questo progetto implementa un sistema avanzato per predire il genere (M/W) di una persona basandosi sul suo nome e cognome. Utilizza architetture di deep learning (BiLSTM con meccanismi di attenzione) e include tecniche per migliorare l'equità e ridurre il bias nei risultati.

Il progetto è strutturato in "rounds" di addestramento progressivamente più sofisticati:
- **Round 0**: Modello base
- **Round 1**: Miglioramenti al training (focal loss, label smoothing, campionamento bilanciato)
- **Round 2**: Potenziamento dell'architettura (più layer, dimensione nascosta aumentata)
- **Round 3**: Post-processing per migliorare l'equità e calibrare le probabilità

## Funzionalità Principali

- Preprocessing dei nomi con gestione di caratteri speciali
- Architettura BiLSTM con meccanismo di attenzione
- Tecniche avanzate di training (focal loss, campionamento bilanciato)
- **Analisi approfondita del bias di genere** con metriche specifiche e visualizzazioni
- **Sistema di gestione degli esperimenti** con tracciamento automatico dei parametri e risultati
- **Strumenti di confronto** per comparare diversi modelli in termini di performance e bias
- **Report HTML interattivi** per l'analisi dettagliata dei risultati
- Post-processing per migliorare l'equità della predizione
- Calibrazione delle probabilità

## Struttura del Progetto

- `train_name_gender_model.py`: Script principale per l'addestramento del modello
- `losses.py`: Implementazioni di funzioni di loss specializzate (FocalLoss, LabelSmoothing)
- `postprocess.py`: Metodi di post-processing per calibrazione e riduzione del bias
- `prepare_comparison_dataset.py`: Preparazione del dataset di confronto
- `requirements.txt`: Dipendenze del progetto
- `sampler.py`: Implementazione di un batch sampler bilanciato per gestire dataset sbilanciati
- `utils.py`: Funzioni di utilità (early stopping, visualizzazione, ecc.)
- **`experiment_manager.py`**: Sistema di gestione degli esperimenti con organizzazione strutturata
- **`experiment_tools.py`**: Strumenti avanzati per l'analisi e il confronto di esperimenti
- `evaluate_gender_model.py`: Valutazione dei modelli Round 1 su dataset esterni
- `evaluate_enhanced_model.py`: Valutazione dei modelli Round 2 su dataset esterni

## Installazione

```bash
git clone https://github.com/yourusername/gender-prediction-from-names.git
cd gender-prediction-from-names
pip install -r requirements.txt
```

## Dataset

I dataset per il training e i modelli pre-addestrati sono disponibili a [questo link](https://liveunibo-my.sharepoint.com/:f:/g/personal/guglielmo_pescatore_unibo_it/Enai-Uyg75BAlSVi23p9xYcBWyKq6mM1wNUJhmxf3LHHRg?e=lfZfNm).

Il dataset dovrebbe contenere almeno le seguenti colonne:
- `primaryName`: Nome completo della persona
- `gender`: Genere ('M' o 'W')
- `nconst` (opzionale): Identificatore unico

## Utilizzo

### Addestramento del Modello Base (Round 0)

```bash
python train_name_gender_model.py --round 0 --data_file training_dataset.csv --epochs 20
```

### Addestramento con Tecniche Avanzate (Round 1)

```bash
python train_name_gender_model.py --round 1 --data_file training_dataset.csv --loss focal --alpha 0.7 --gamma 2.0 --label_smooth 0.05 --balanced_sampler --epochs 25
```

### Potenziamento dell'Architettura (Round 2)

```bash
python train_name_gender_model.py --round 2 --data_file training_dataset.csv --n_layers 2 --hidden_size 80 --dual_input --freeze_epochs 4 --epochs 30
```

### Post-processing e Calibrazione (Round 3)

```bash
python postprocess.py --model_path models/round2_best.pth --preprocessor_path name_preprocessor.pkl --data_file training_dataset.csv --apply_calibration --apply_equalized_odds
```

### Preparazione Dataset di Confronto

```bash
python prepare_comparison_dataset.py --data_file imdb_actors_actresses.csv --comparison_size 30000
```

### Analisi e Confronto degli Esperimenti

```bash
# Elencare tutti gli esperimenti
python experiment_tools.py list

# Confrontare metriche di performance
python experiment_tools.py compare --metric test_f1

# Analizzare il bias di genere
python experiment_tools.py bias

# Confrontare la distribuzione degli errori di genere
python experiment_tools.py bias --experiments exp1_id exp2_id

# Generare heatmap del bias
python experiment_tools.py bias --heatmap

# Confrontare curve di apprendimento
python experiment_tools.py curves --experiments exp1_id exp2_id

# Generare report completo
python experiment_tools.py report --full
```

## Parametri di Configurazione

### Parametri Generali
- `--round`: Modalità di addestramento (0=base, 1=tecniche avanzate, 2=architettura potenziata)
- `--save_dir`: Directory per salvare log e checkpoint
- `--epochs`: Numero massimo di epoche
- `--seed`: Seed per la riproducibilità
- `--data_file`: Percorso al file CSV con i dati

### Parametri per il Round 1
- `--loss`: Funzione di loss ("bce" o "focal")
- `--alpha`: Alpha per FocalLoss (peso per la classe femminile)
- `--gamma`: Gamma per FocalLoss (parametro di focalizzazione)
- `--pos_weight`: Peso per la classe positiva in BCEWithLogitsLoss
- `--label_smooth`: Epsilon per il label smoothing
- `--balanced_sampler`: Usa un batch sampler bilanciato
- `--early_stop`: Patience per l'early stopping (0 = disattivato)

### Parametri per il Round 2
- `--n_layers`: Numero di layer BiLSTM
- `--hidden_size`: Dimensione hidden dei layer BiLSTM
- `--dual_input`: Usa encoder separati per nome e cognome
- `--freeze_epochs`: Epoche per cui congelare l'embedding e il primo layer LSTM

## Architettura del Modello

Il progetto implementa due modelli principali:

1. **GenderPredictor**: Modello base BiLSTM con attenzione
   - Embedding di caratteri condiviso
   - BiLSTM separati per nome e cognome
   - Layer di attenzione
   - Fully connected layer per la predizione

2. **GenderPredictorEnhanced**: Architettura potenziata
   - Supporto per multiple layer BiLSTM
   - Layer normalization
   - Architettura più profonda e con maggiore capacità
   - Opzione per modalità dual/single input

## Tecniche di Training Avanzate

- **FocalLoss**: Migliora l'apprendimento focalizzandosi sugli esempi difficili da classificare
- **Label Smoothing**: Riduce l'overfitting prevenendo confidenze troppo elevate
- **Balanced Batch Sampler**: Garantisce rappresentatività bilanciata delle classi
- **Layer Freezing**: Congela gli embedding e i primi layer durante le prime epoche
- **Early Stopping**: Previene l'overfitting interrompendo l'addestramento quando la performance smette di migliorare

## Sistema di Gestione degli Esperimenti

Il progetto include un sistema avanzato per gestire, tracciare e confrontare gli esperimenti:

- **Organizzazione strutturata**: Ogni esperimento ha la propria directory con sottocartelle per modelli, log e visualizzazioni
- **ID Esperimenti**: Gli ID degli esperimenti includono automaticamente i parametri chiave (round, loss, architettura)
- **Tracking automatico**: Tutti i parametri e risultati vengono salvati in file JSON facilmente accessibili
- **Log centralizzato**: Un file CSV centrale tiene traccia di tutti gli esperimenti e delle loro performance
- **Reports HTML**: Per ogni esperimento viene generato un report dettagliato con metriche e visualizzazioni

## Analisi del Bias di Genere

Il progetto include strumenti avanzati per analizzare il bias del modello:

- **Bias Ratio**: Rapporto tra i tassi di errore M→W e W→M (ideale: 1.0)
- **Equality of Opportunity**: Differenza nei tassi di recall tra generi (ideale: 0)
- **Predictive Equality**: Differenza nei tassi di errore tra generi (ideale: 0)
- **Metriche Specifiche per Genere**: Precision, recall, F1 e tassi di errore separati per M e W
- **Visualizzazioni di Bias**: Dashboard dedicata con 4 grafici per un'analisi dettagliata
- **Dashboard Comparative**: Strumenti per confrontare visivamente il bias tra diversi modelli
- **Heatmap di Bias**: Rappresentazione visiva delle metriche di bias per confronti rapidi

## Post-processing per l'Equità

- **Threshold Optimization**: Ottimizzazione della soglia per massimizzare metriche come F1
- **Probability Calibration**: Calibrazione delle probabilità tramite Platt scaling
- **Equalized Odds**: Applicazione di soglie diverse per gruppi diversi per equalizzare i tassi di errore

## Risultati

I risultati degli esperimenti vengono salvati in una struttura organizzata:
- `experiments/[experiment_id]/models/`: Modelli addestrati e checkpoint
- `experiments/[experiment_id]/logs/`: Metriche, storia di training e dati
- `experiments/[experiment_id]/plots/`: Visualizzazioni, matrici di confusione e analisi del bias
- `experiments/[experiment_id]/report.html`: Report completo dell'esperimento

## Licenza

[Inserire informazione sulla licenza]

## Contributi

Contributi, segnalazioni di bug e richieste di funzionalità sono benvenuti. Sentiti libero di aprire una issue o una pull request.

## Contatti

[Inserire informazioni di contatto]

---

Progetto sviluppato per [inserire scopo/organizzazione]
