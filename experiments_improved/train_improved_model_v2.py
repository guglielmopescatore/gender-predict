#!/usr/bin/env python3
"""
Script di training migliorato per il modello di predizione del genere.
Integrato con ExperimentManager e ottimizzato per RTX 4090.
Target: >93% accuracy con dataset sbilanciato 62.32% M / 37.68% F
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os
import argparse
from tqdm import tqdm
import time

# Aggiungi il parent directory al path per importare i moduli dalla root
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import dei moduli originali dal progetto esistente (nella root)
from train_name_gender_model import (
    NamePreprocessor, 
    NameGenderDataset,
    set_all_seeds
)
from experiment_manager import ExperimentManager
from utils import plot_confusion_matrix
from sampler import BalancedBatchSampler

# Import dei nuovi moduli (nella stessa directory)
from improvements import (
    GenderPredictorV3, 
    NameFeatureExtractor, 
    NameAugmenter,
    CosineAnnealingWarmupScheduler,
    FocalLossImproved
)

# Import per error analysis (aggiungi dopo gli altri import)
from error_analysis_tool import ErrorAnalyzer, add_error_analysis_to_training

# Import condizionale per error analysis
ERROR_ANALYSIS_AVAILABLE = False
try:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from error_analysis_tool import ErrorAnalyzer, add_error_analysis_to_training
    ERROR_ANALYSIS_AVAILABLE = True
except ImportError:
    print("⚠️  Error analysis tool non disponibile. Funzionalità disabilitata.")


class ImprovedNameGenderDataset(NameGenderDataset):
    """
    Estende il dataset originale con feature engineering e augmentation.
    """
    
    def __init__(self, dataframe, preprocessor, feature_extractor=None, mode='train', 
                 augmenter=None, augment_prob=0.0):
        super().__init__(dataframe, preprocessor, mode)
        self.feature_extractor = feature_extractor
        self.augmenter = augmenter
        self.augment_prob = augment_prob if mode == 'train' else 0.0
        
    def __getitem__(self, idx):
        # Ottieni i dati base dalla classe padre
        base_data = super().__getitem__(idx)
        
        # Se non abbiamo feature extractor, ritorna i dati base
        if not self.feature_extractor:
            return base_data
        
        # Estrai il nome completo
        row = self.df.iloc[idx]
        full_name = row['primaryName']
        
        # Applica augmentation se in training
        if self.augmenter and np.random.random() < self.augment_prob:
            full_name = self.augmenter.augment(full_name)
            # Ri-preprocessa il nome augmentato
            name_data = self.preprocessor.preprocess_name(full_name)
            base_data['first_name'] = torch.tensor(name_data['first_name'], dtype=torch.long)
            base_data['last_name'] = torch.tensor(name_data['last_name'], dtype=torch.long)
        
        # Estrai features linguistiche
        first_name, last_name = self.preprocessor.split_full_name(full_name)
        
        first_suffix = self.feature_extractor.extract_suffix_features(first_name)
        last_suffix = self.feature_extractor.extract_suffix_features(last_name)
        
        # Estrai features fonetiche
        phonetic_first = self.feature_extractor.extract_phonetic_features(first_name)
        phonetic_last = self.feature_extractor.extract_phonetic_features(last_name)
        
        # Combina features fonetiche
        phonetic_features = [
            phonetic_first['ends_with_vowel'],
            phonetic_first['vowel_ratio'],
            phonetic_last['ends_with_vowel'],
            phonetic_last['vowel_ratio']
        ]
        
        # Padding per suffix features
        first_suffix = first_suffix + [0] * (3 - len(first_suffix))
        last_suffix = last_suffix + [0] * (3 - len(last_suffix))
        
        # Aggiungi le nuove features
        base_data['first_suffix'] = torch.tensor(first_suffix[:3], dtype=torch.long)
        base_data['last_suffix'] = torch.tensor(last_suffix[:3], dtype=torch.long)
        base_data['phonetic_features'] = torch.tensor(phonetic_features, dtype=torch.float32)
        
        return base_data


def train_improved_model(model, train_loader, val_loader, criterion, optimizer, 
                         scheduler, num_epochs, device, experiment, patience=7, 
                         gradient_clip=1.0):
    """
    Training loop migliorato integrato con ExperimentManager.
    """
    model.to(device)
    
    # Import EarlyStopping dal modulo utils
    from utils import EarlyStopping
    early_stopping = EarlyStopping(patience=patience, min_delta=0.001)
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'val_precision': [],
        'val_recall': [],
        'val_f1': [],
        'learning_rate': []
    }
    
    best_val_f1 = 0
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # Aggiorna learning rate
        current_lr = scheduler.step(epoch)
        history['learning_rate'].append(current_lr)
        
        # Training
        model.train()
        train_loss = 0.0
        train_preds = []
        train_targets = []
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        
        for batch in progress_bar:
            # Carica dati
            first_name = batch['first_name'].to(device)
            last_name = batch['last_name'].to(device)
            first_suffix = batch['first_suffix'].to(device)
            last_suffix = batch['last_suffix'].to(device)
            phonetic_features = batch['phonetic_features'].to(device)
            gender = batch['gender'].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            logits = model(first_name, last_name, first_suffix, last_suffix, phonetic_features)
            loss = criterion(logits, gender)
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            
            optimizer.step()
            
            train_loss += loss.item() * first_name.size(0)
            
            # Predizioni
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).long()
            
            train_preds.extend(preds.cpu().numpy())
            train_targets.extend(gender.cpu().numpy())
            
            # Aggiorna progress bar
            progress_bar.set_postfix({'loss': loss.item(), 'lr': current_lr})
        
        train_loss /= len(train_loader.dataset)
        train_acc = accuracy_score(train_targets, train_preds)
        
        # Validation
        model.eval()
        val_loss, val_preds, val_targets = 0.0, [], []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                first_name = batch['first_name'].to(device)
                last_name = batch['last_name'].to(device)
                first_suffix = batch['first_suffix'].to(device)
                last_suffix = batch['last_suffix'].to(device)
                phonetic_features = batch['phonetic_features'].to(device)
                gender = batch['gender'].to(device)
                
                logits = model(first_name, last_name, first_suffix, last_suffix, phonetic_features)
                loss = criterion(logits, gender)
                
                val_loss += loss.item() * first_name.size(0)
                
                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).long()
                
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(gender.cpu().numpy())
        
        val_loss /= len(val_loader.dataset)
        val_acc = accuracy_score(val_targets, val_preds)
        
        # Calcola metriche
        precision, recall, f1, _ = precision_recall_fscore_support(
            val_targets, val_preds, average='binary')
        
        # Aggiorna storia
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['val_precision'].append(precision)
        history['val_recall'].append(recall)
        history['val_f1'].append(f1)
        
        # Tempo trascorso
        time_elapsed = time.time() - start_time
        
        # Stampa risultati
        print(f"\nEpoch {epoch+1}/{num_epochs} | Time: {time_elapsed:.2f}s | LR: {current_lr:.6f}")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        print(f"Val Precision: {precision:.4f} | Val Recall: {recall:.4f} | Val F1: {f1:.4f}")
        
        # Salva il miglior modello tramite ExperimentManager
        if f1 > best_val_f1:
            best_val_f1 = f1
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'vocab_size': model.vocab_size,
                'embedding_dim': model.embedding_dim,
                'hidden_size': model.hidden_size,
                'n_layers': model.n_layers,
                'epoch': epoch,
                'best_f1': best_val_f1,
                'suffix_vocab_size': len(feature_extractor.suffix_to_idx) if 'feature_extractor' in locals() else 100
            }
            experiment.save_model_checkpoint(checkpoint)
            print(f"💾 Nuovo miglior modello salvato! (F1: {best_val_f1:.4f})")
        
        # Analisi del bias periodica
        if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
            print("\nAnalisi del bias sul set di validazione:")
            cm_path = experiment.get_plot_path('confusion_matrix_epoch_{}'.format(epoch+1))
            plot_confusion_matrix(val_targets, val_preds, output_file=cm_path)
        
        # Early stopping
        if early_stopping(model, f1):
            print(f"Early stopping triggered after epoch {epoch+1}")
            break
        
        print("-" * 60)
    
    # Salva la storia del training tramite ExperimentManager
    experiment.log_training_history(history)
    
    print(f"\n🎯 Training completato! Miglior F1: {best_val_f1:.4f}")
    
    return history


def main():
    parser = argparse.ArgumentParser(description="Training migliorato per superare 92%% accuracy")
    
    # Parametri base compatibili con ExperimentManager
    parser.add_argument("--round", type=int, default=3,
                        help="Round number for ExperimentManager")
    parser.add_argument("--data_file", type=str, default="training_dataset.csv",
                        help="Dataset di training")
    parser.add_argument("--save_dir", type=str, default=".",
                        help="Directory base per ExperimentManager")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Numero di epoche")
    parser.add_argument("--early_stop", type=int, default=5,
                        help="Early stopping patience")
    parser.add_argument("--note", type=str, default="",
                        help="Nota descrittiva per l'esperimento")

    # Parametri architettura (basati sui tuoi migliori risultati)
    parser.add_argument("--embedding_dim", type=int, default=32,
                        help="Dimensione embeddings")
    parser.add_argument("--hidden_size", type=int, default=128,  # Aumentato per RTX 4090
                        help="Hidden size LSTM")
    parser.add_argument("--n_layers", type=int, default=2,
                        help="Numero layer LSTM")
    parser.add_argument("--num_heads", type=int, default=4,
                        help="Numero di attention heads")
    parser.add_argument("--dropout", type=float, default=0.3,
                        help="Dropout rate")
    parser.add_argument("--dual_input", action="store_true", default=True,
                        help="Usa encoder separati (come nei tuoi migliori risultati)")
    
    # Parametri training ottimizzati per RTX 4090
    parser.add_argument("--batch_size", type=int, default=512,  # Aumentato per RTX 4090
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate massimo")
    parser.add_argument("--min_lr", type=float, default=1e-6,
                        help="Learning rate minimo")
    parser.add_argument("--warmup_epochs", type=int, default=3,
                        help="Epoche di warmup")
    parser.add_argument("--gradient_clip", type=float, default=1.0,
                        help="Gradient clipping value")
    parser.add_argument("--freeze_epochs", type=int, default=4,
                        help="Freeze embedding epochs (come nei tuoi risultati)")
    
    # Parametri augmentation
    parser.add_argument("--augment_prob", type=float, default=0.15,
                        help="Probabilità di augmentation")
    
    # Loss parameters per dataset sbilanciato (62.32% M, 37.68% F)
    parser.add_argument("--loss", type=str, default="focal", choices=["bce", "focal"],
                        help="Loss function")
    parser.add_argument("--focal_gamma", type=float, default=2.0,
                        help="Gamma per focal loss")
    parser.add_argument("--focal_alpha", type=float, default=0.62,
                        help="Alpha per focal loss (0.62 per bilanciare 62%%M/38%%F)")
    parser.add_argument("--pos_weight", type=float, default=1.0,
                        help="Positive weight for BCE loss")
    parser.add_argument("--label_smooth", type=float, default=0.0,
                        help="Label smoothing")
    parser.add_argument("--balanced_sampler", action="store_true",
                        help="Usa balanced batch sampler")

    # Error analysis
    parser.add_argument("--enable_error_analysis", action="store_true",
                    help="Abilita analisi dettagliata degli errori sul test set")

    
    args = parser.parse_args()

    # Compatibilità con ExperimentManager che si aspetta 'alpha' e 'gamma'
    args.alpha = args.focal_alpha
    args.gamma = args.focal_gamma
    
    # Inizializza ExperimentManager
    # Aggiungi la nota all'experiment se specificata
    if args.note:
        original_note = getattr(args, 'note', '')
        if hasattr(args, 'round'):
            args.note = f"r{args.round}_{original_note}" if original_note else f"r{args.round}"
        else:
            args.note = original_note

    # Inizializza ExperimentManager
    experiment = ExperimentManager(args, base_dir=args.save_dir)
    print(f"Experiment ID: {experiment.experiment_id}")
    print(f"Experiment directory: {experiment.experiment_dir}")
    
    # Set seeds
    set_all_seeds(args.seed)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Load data
    print(f"Loading data from {args.data_file}...")
    df = pd.read_csv(args.data_file)
    print(f"Loaded {len(df)} records")

    # Verifica distribuzione
    gender_dist = df['gender'].value_counts(normalize=True)
    print(f"Gender distribution: M={gender_dist.get('M', 0):.2%}, F={gender_dist.get('W', 0):.2%}")
    
    # Split data
    train_val_df, test_df = train_test_split(
        df, test_size=0.1, random_state=args.seed, stratify=df['gender']
    )
    train_df, val_df = train_test_split(
        train_val_df, test_size=0.1, random_state=args.seed, stratify=train_val_df['gender']
    )
    
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Create preprocessors
    preprocessor = NamePreprocessor()
    feature_extractor = NameFeatureExtractor()
    augmenter = NameAugmenter(augment_prob=args.augment_prob)
    
    # Salva preprocessor e feature extractor
    preprocessor.save(experiment.preprocessor_path)
    
    import pickle
    with open(os.path.join(experiment.experiment_dir, 'feature_extractor.pkl'), 'wb') as f:
        pickle.dump(feature_extractor, f)
    
    # Calculate suffix vocabulary size
    suffix_vocab_size = len(feature_extractor.suffix_to_idx)
    print(f"Suffix vocabulary size: {suffix_vocab_size}")
    
    # Create datasets
    train_dataset = ImprovedNameGenderDataset(
        train_df, preprocessor, feature_extractor, 
        mode='train', augmenter=augmenter, augment_prob=args.augment_prob
    )
    val_dataset = ImprovedNameGenderDataset(
        val_df, preprocessor, feature_extractor, mode='val'
    )
    test_dataset = ImprovedNameGenderDataset(
        test_df, preprocessor, feature_extractor, mode='test'
    )
    
    # Create dataloaders
    if args.balanced_sampler:
        print("Using BalancedBatchSampler...")
        train_sampler = BalancedBatchSampler(train_dataset, args.batch_size)
        train_loader = DataLoader(train_dataset, batch_sampler=train_sampler,
                                  num_workers=8, pin_memory=True)
    else:
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=8, pin_memory=True
        )
    
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=8, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=8, pin_memory=True
    )
    
    # Create model
    model = GenderPredictorV3(
        vocab_size=preprocessor.vocab_size,
        suffix_vocab_size=suffix_vocab_size,
        embedding_dim=args.embedding_dim,
        hidden_size=args.hidden_size,
        n_layers=args.n_layers,
        dropout_rate=args.dropout,
        num_attention_heads=args.num_heads
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create loss
    if args.loss == "focal":
        # Per dataset 62.32% M / 37.68% F
        # alpha = peso per classe F (minoritaria)
        criterion = FocalLossImproved(
            alpha=args.focal_alpha,  # 0.62 bilancia il dataset
            gamma=args.focal_gamma,
            auto_weight=False  # Usiamo alpha manuale
        )
        print(f"Using FocalLoss with alpha={args.focal_alpha}, gamma={args.focal_gamma}")
    else:
        # BCE loss
        pos_weight = torch.tensor(args.pos_weight).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print(f"Using BCEWithLogitsLoss with pos_weight={args.pos_weight}")
    
    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=1e-5,
        betas=(0.9, 0.999)
    )
    
    # Create scheduler
    scheduler = CosineAnnealingWarmupScheduler(
        optimizer, 
        warmup_epochs=args.warmup_epochs,
        max_epochs=args.epochs,
        min_lr=args.min_lr,
        max_lr=args.lr
    )
    
    # Opzionale: implementa freeze epochs come nei tuoi migliori risultati
    if args.freeze_epochs > 0:
        print(f"Freezing embeddings for first {args.freeze_epochs} epochs")
        # Questo richiederebbe modifiche al training loop
    
    # Train model
    history = train_improved_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        num_epochs=args.epochs,
        device=device,
        experiment=experiment,
        patience=args.early_stop,
        gradient_clip=args.gradient_clip
    )
    
    # Evaluate on test set
    print("\n📊 Evaluating on test set...")
    model.eval()
    test_preds = []
    test_targets = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            first_name = batch['first_name'].to(device)
            last_name = batch['last_name'].to(device)
            first_suffix = batch['first_suffix'].to(device)
            last_suffix = batch['last_suffix'].to(device)
            phonetic_features = batch['phonetic_features'].to(device)
            gender = batch['gender'].to(device)

            logits = model(first_name, last_name, first_suffix, last_suffix, phonetic_features)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).long()

            test_preds.extend(preds.cpu().numpy())
            test_targets.extend(gender.cpu().numpy())

    # Calculate final metrics
    test_acc = accuracy_score(test_targets, test_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        test_targets, test_preds, average='binary'
    )

    print(f"\n🎯 RISULTATI FINALI SUL TEST SET:")
    print(f"Accuracy: {test_acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Salva metriche tramite ExperimentManager
    test_metrics = {
        'accuracy': float(test_acc),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1)
    }
    experiment.log_test_metrics(test_metrics)

    # CONDITIONAL ERROR ANALYSIS
    if args.enable_error_analysis and ERROR_ANALYSIS_AVAILABLE:
        print("\n🔍 === ANALISI DETTAGLIATA DEGLI ERRORI ===")
        try:
            predictions_df, analysis_results = add_error_analysis_to_training(
                experiment, model, test_dataset, preprocessor, device
            )

            # Aggiungi i risultati principali dell'analisi ai log
            error_summary = {
                'total_errors': analysis_results['total_errors'],
                'error_rate': analysis_results['error_rate'],
                'most_problematic_names': list(analysis_results['most_common_error_names'].keys())[:5],
                'avg_error_name_length': analysis_results['length_analysis']['error_stats']['mean'],
                'avg_correct_name_length': analysis_results['length_analysis']['correct_stats']['mean'],
                'high_confidence_error_rate': analysis_results['confidence_analysis']['high_confidence_error_rate']
            }

            # Salva summary tramite experiment manager (aggiungere il metodo)
            if hasattr(experiment, 'log_error_summary'):
                experiment.log_error_summary(error_summary)

            print(f"\n💡 KEY INSIGHTS:")
            print(f"   • {analysis_results['total_errors']} errori su {analysis_results['total_samples']} campioni")
            if analysis_results['most_common_error_names']:
                top_error_name = list(analysis_results['most_common_error_names'].keys())[0]
                print(f"   • Nome più problematico: {top_error_name}")
            print(f"   • Lunghezza media errori: {error_summary['avg_error_name_length']:.1f} caratteri")
            print(f"   • Errori ad alta confidenza: {error_summary['high_confidence_error_rate']:.1%}")

            print(f"\n📁 File generati:")
            print(f"   • Dataset errori: {experiment.logs_dir}/error_analysis.csv")
            print(f"   • Analisi completa: {experiment.logs_dir}/error_analysis_results.json")
            print(f"   • Visualizzazioni: {experiment.plots_dir}/error_analysis.png")

        except Exception as e:
            print(f"⚠️  Errore nell'analisi degli errori: {e}")
            print("   Continuando con il salvataggio standard...")
            import traceback
            traceback.print_exc()

    elif args.enable_error_analysis and not ERROR_ANALYSIS_AVAILABLE:
        print("⚠️  Analisi errori richiesta ma error_analysis_tool non disponibile")
        print("   Assicurati che error_analysis_tool.py sia nella directory parent")

    elif not args.enable_error_analysis:
        print("ℹ️  Analisi errori disabilitata (usa --enable_error_analysis per abilitarla)")

    # Analisi del bias (sempre abilitata)
    experiment.save_confusion_matrix(test_targets, test_preds, labels=["Male", "Female"])
    
    # Analisi del bias
    experiment.save_confusion_matrix(test_targets, test_preds, labels=["Male", "Female"])
    
    # Genera report
    report_path = experiment.generate_report()
    
    print(f"\n✅ Training completato con successo!")
    print(f"   Experiment ID: {experiment.experiment_id}")
    print(f"   Model saved to: {experiment.model_path}")
    print(f"   Report: {report_path}")
    print(f"\nPer confrontare con altri esperimenti:")
    print(f"   python experiment_tools.py compare --metric test_f1")
    print(f"   python experiment_tools.py bias --experiments {experiment.experiment_id}")


if __name__ == "__main__":
    main()
