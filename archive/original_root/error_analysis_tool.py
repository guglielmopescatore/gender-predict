#!/usr/bin/env python3
"""
Strumento avanzato per l'analisi degli errori del modello di predizione del genere.
Versione corretta che gestisce correttamente diversi tipi di dataset e modelli.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from collections import Counter
import os
from typing import Dict, List, Tuple
import torch
import inspect


class ErrorAnalyzer:
    """
    Analizzatore avanzato degli errori per il modello di predizione del genere.
    """

    def __init__(self, experiment_manager=None):
        """
        Args:
            experiment_manager: Istanza di ExperimentManager per salvare i risultati
        """
        self.experiment_manager = experiment_manager
        self.errors_df = None
        self.correct_df = None

    def generate_error_dataset(self, model, dataset, preprocessor, device='cuda',
                              batch_size=256, save_path=None):
        """
        Genera un dataset completo con predizioni e errori per l'analisi.
        Gestisce automaticamente diversi tipi di modelli e dataset.

        Args:
            model: Modello addestrato
            dataset: Dataset di test
            preprocessor: Preprocessore per i nomi
            device: Device per l'inferenza
            batch_size: Dimensione del batch
            save_path: Percorso dove salvare il dataset di errori

        Returns:
            DataFrame con tutte le predizioni e flag di errore
        """
        from torch.utils.data import DataLoader
        from tqdm import tqdm

        print("🔍 Generazione dataset errori...")

        # Crea DataLoader
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        # Liste per raccogliere risultati
        all_names = []
        all_true_labels = []
        all_pred_labels = []
        all_probabilities = []
        all_ids = []

        # Rileva il tipo di modello automaticamente
        model_type = self._detect_model_type(model)
        print(f"   Tipo modello rilevato: {model_type}")

        model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Predizioni")):
                batch_size_actual = len(batch['first_name'])

                # Estrai nomi dal dataset originale
                start_idx = batch_idx * batch_size
                end_idx = start_idx + batch_size_actual

                if hasattr(dataset, 'df'):
                    batch_names = dataset.df.iloc[start_idx:end_idx]['primaryName'].tolist()
                else:
                    batch_names = [f"sample_{i}" for i in range(start_idx, end_idx)]

                all_names.extend(batch_names)

                # Predizioni del modello - gestione automatica del tipo
                logits = self._predict_with_model(model, batch, device, model_type)

                # Converti logits in probabilità e predizioni
                probs = torch.sigmoid(logits).cpu().numpy()
                preds = (probs >= 0.5).astype(int)

                # Raccogli risultati
                all_probabilities.extend(probs)
                all_pred_labels.extend(preds)
                all_true_labels.extend(batch['gender'].cpu().numpy().astype(int))

                # ID se disponibili
                if 'id' in batch:
                    all_ids.extend(batch['id'])
                else:
                    all_ids.extend([None] * len(preds))

        # Crea DataFrame risultati
        results_df = pd.DataFrame({
            'primaryName': all_names,
            'true_gender': ['W' if label == 1 else 'M' for label in all_true_labels],
            'pred_gender': ['W' if label == 1 else 'M' for label in all_pred_labels],
            'probability_female': all_probabilities,
            'true_label': all_true_labels,
            'pred_label': all_pred_labels,
            'is_error': [true != pred for true, pred in zip(all_true_labels, all_pred_labels)],
            'error_type': [self._get_error_type(true, pred) for true, pred in zip(all_true_labels, all_pred_labels)],
            'confidence': [abs(prob - 0.5) for prob in all_probabilities],
            'id': all_ids
        })

        # Aggiungi feature linguistiche per analisi
        results_df = self._add_linguistic_features(results_df)

        # Salva se richiesto
        if save_path:
            results_df.to_csv(save_path, index=False)
            print(f"💾 Dataset errori salvato in: {save_path}")

        # Salva tramite ExperimentManager se disponibile
        if self.experiment_manager:
            error_path = os.path.join(self.experiment_manager.logs_dir, "error_analysis.csv")
            results_df.to_csv(error_path, index=False)
            print(f"💾 Dataset errori salvato in: {error_path}")

        return results_df

    def _detect_model_type(self, model):
        """Rileva automaticamente il tipo di modello dalla signature."""
        try:
            sig = inspect.signature(model.forward)
            param_names = list(sig.parameters.keys())

            if len(param_names) >= 5:
                return "GenderPredictorV3"
            else:
                return "GenderPredictor"
        except:
            return "Unknown"

    def _predict_with_model(self, model, batch, device, model_type):
        """
        Esegue predizioni gestendo automaticamente diversi tipi di modello.
        """
        first_name = batch['first_name'].to(device)
        last_name = batch['last_name'].to(device)

        if model_type == "GenderPredictorV3":
            # Modello avanzato - richiede feature aggiuntive
            first_suffix = batch.get('first_suffix')
            last_suffix = batch.get('last_suffix')
            phonetic_features = batch.get('phonetic_features')

            # Se non disponibili nel batch, crea valori di default
            if first_suffix is None:
                first_suffix = torch.zeros(len(first_name), 3, dtype=torch.long)
            if last_suffix is None:
                last_suffix = torch.zeros(len(first_name), 3, dtype=torch.long)
            if phonetic_features is None:
                phonetic_features = torch.zeros(len(first_name), 4, dtype=torch.float32)

            first_suffix = first_suffix.to(device)
            last_suffix = last_suffix.to(device)
            phonetic_features = phonetic_features.to(device)

            logits = model(first_name, last_name, first_suffix, last_suffix, phonetic_features)

        else:
            # Modello standard
            logits = model(first_name, last_name)

        return logits

    def _get_error_type(self, true_label, pred_label):
        """Determina il tipo di errore."""
        if true_label == pred_label:
            return 'correct'
        elif true_label == 0 and pred_label == 1:
            return 'M_to_W'  # Maschio predetto come femmina
        else:
            return 'W_to_M'  # Femmina predetta come maschio

    def _add_linguistic_features(self, df):
        """Aggiunge feature linguistiche per l'analisi."""
        print("📊 Aggiunta feature linguistiche...")

        df['name_length'] = df['primaryName'].str.len()
        df['num_words'] = df['primaryName'].str.split().str.len()
        df['first_name'] = df['primaryName'].str.split().str[0]
        df['has_space'] = df['primaryName'].str.contains(' ')
        df['has_hyphen'] = df['primaryName'].str.contains('-')
        df['has_apostrophe'] = df['primaryName'].str.contains("'")
        df['ends_with_vowel'] = df['first_name'].str.lower().str.endswith(('a', 'e', 'i', 'o', 'u'))
        df['starts_with_vowel'] = df['first_name'].str.lower().str.startswith(('a', 'e', 'i', 'o', 'u'))

        # Suffissi comuni
        df['ends_with_a'] = df['first_name'].str.lower().str.endswith('a')
        df['ends_with_o'] = df['first_name'].str.lower().str.endswith('o')
        df['ends_with_e'] = df['first_name'].str.lower().str.endswith('e')
        df['ends_with_consonant'] = ~df['ends_with_vowel']

        # Lunghezza del primo nome
        df['first_name_length'] = df['first_name'].str.len()

        return df

    def analyze_errors(self, predictions_df=None, save_plots=True):
        """
        Analizza gli errori in dettaglio.

        Args:
            predictions_df: DataFrame con le predizioni (se None, usa quello generato)
            save_plots: Se salvare i grafici

        Returns:
            Dict con risultati dell'analisi
        """
        if predictions_df is not None:
            df = predictions_df
        else:
            df = self.errors_df

        if df is None:
            raise ValueError("Nessun dataset di predizioni disponibile. Usa generate_error_dataset prima.")

        print("🔍 Analisi dettagliata degli errori...")

        # Separa errori e predizioni corrette
        errors = df[df['is_error']].copy()
        correct = df[~df['is_error']].copy()

        self.errors_df = errors
        self.correct_df = correct

        # Statistiche base
        analysis_results = {
            'total_samples': len(df),
            'total_errors': len(errors),
            'error_rate': len(errors) / len(df),
            'accuracy': len(correct) / len(df)
        }

        print(f"\n📊 STATISTICHE GENERALI:")
        print(f"   Campioni totali: {analysis_results['total_samples']:,}")
        print(f"   Errori totali: {analysis_results['total_errors']:,}")
        print(f"   Tasso di errore: {analysis_results['error_rate']:.3%}")
        print(f"   Accuratezza: {analysis_results['accuracy']:.3%}")

        # Analisi per tipo di errore
        error_types = errors['error_type'].value_counts()
        analysis_results['error_types'] = error_types.to_dict()

        print(f"\n📊 ERRORI PER TIPO:")
        for error_type, count in error_types.items():
            percentage = count / len(errors) * 100
            print(f"   {error_type}: {count:,} ({percentage:.1f}%)")

        # Analisi nomi più problematici
        most_common_errors = errors['primaryName'].value_counts().head(20)
        analysis_results['most_common_error_names'] = most_common_errors.to_dict()

        print(f"\n📊 NOMI CON PIÙ ERRORI:")
        for name, count in most_common_errors.head(10).items():
            error_examples = errors[errors['primaryName'] == name]
            error_type = error_examples['error_type'].iloc[0]
            true_gender = error_examples['true_gender'].iloc[0]
            pred_gender = error_examples['pred_gender'].iloc[0]
            print(f"   {name}: {count} errori ({true_gender}→{pred_gender})")

        # Analisi per lunghezza nome
        length_analysis = self._analyze_by_feature(errors, correct, 'name_length', 'Lunghezza Nome')
        analysis_results['length_analysis'] = length_analysis

        # Analisi per numero di parole
        words_analysis = self._analyze_by_feature(errors, correct, 'num_words', 'Numero Parole')
        analysis_results['words_analysis'] = words_analysis

        # Analisi per suffissi
        suffix_analysis = self._analyze_suffixes(errors, correct)
        analysis_results['suffix_analysis'] = suffix_analysis

        # Analisi per confidenza
        confidence_analysis = self._analyze_confidence(errors, correct)
        analysis_results['confidence_analysis'] = confidence_analysis

        # Genera visualizzazioni
        if save_plots:
            self._create_error_visualizations(errors, correct, analysis_results)

        # Salva analisi
        if self.experiment_manager:
            analysis_path = os.path.join(self.experiment_manager.logs_dir, "error_analysis_results.json")
            import json
            with open(analysis_path, 'w') as f:
                # Converti numpy types per JSON
                json_results = self._convert_for_json(analysis_results)
                json.dump(json_results, f, indent=2)
            print(f"💾 Analisi salvata in: {analysis_path}")

        return analysis_results

    def _analyze_by_feature(self, errors, correct, feature, feature_name):
        """Analizza errori per una feature specifica."""
        print(f"\n📊 ANALISI PER {feature_name.upper()}:")

        # Statistiche descrittive
        error_stats = errors[feature].describe()
        correct_stats = correct[feature].describe()

        print(f"   Errori - {feature_name}:")
        print(f"     Media: {error_stats['mean']:.2f}")
        print(f"     Mediana: {error_stats['50%']:.2f}")
        print(f"     Min-Max: {error_stats['min']:.0f}-{error_stats['max']:.0f}")

        print(f"   Corretti - {feature_name}:")
        print(f"     Media: {correct_stats['mean']:.2f}")
        print(f"     Mediana: {correct_stats['50%']:.2f}")
        print(f"     Min-Max: {correct_stats['min']:.0f}-{correct_stats['max']:.0f}")

        # Test statistico semplice
        difference = error_stats['mean'] - correct_stats['mean']
        print(f"   Differenza media: {difference:.2f}")

        return {
            'error_stats': error_stats.to_dict(),
            'correct_stats': correct_stats.to_dict(),
            'mean_difference': difference
        }

    def _analyze_suffixes(self, errors, correct):
        """Analizza pattern dei suffissi negli errori."""
        print(f"\n📊 ANALISI SUFFISSI:")

        suffix_features = ['ends_with_a', 'ends_with_o', 'ends_with_e', 'ends_with_vowel']

        results = {}
        for feature in suffix_features:
            error_rate = errors[feature].mean()
            correct_rate = correct[feature].mean()

            print(f"   {feature}:")
            print(f"     Errori: {error_rate:.3f}")
            print(f"     Corretti: {correct_rate:.3f}")
            print(f"     Differenza: {error_rate - correct_rate:.3f}")

            results[feature] = {
                'error_rate': error_rate,
                'correct_rate': correct_rate,
                'difference': error_rate - correct_rate
            }

        return results

    def _analyze_confidence(self, errors, correct):
        """Analizza la confidenza del modello negli errori."""
        print(f"\n📊 ANALISI CONFIDENZA:")

        error_conf = errors['confidence'].describe()
        correct_conf = correct['confidence'].describe()

        print(f"   Errori - Confidenza:")
        print(f"     Media: {error_conf['mean']:.3f}")
        print(f"     Mediana: {error_conf['50%']:.3f}")

        print(f"   Corretti - Confidenza:")
        print(f"     Media: {correct_conf['mean']:.3f}")
        print(f"     Mediana: {correct_conf['50%']:.3f}")

        # Errori ad alta confidenza (più preoccupanti)
        high_conf_errors = errors[errors['confidence'] > 0.3]
        print(f"   Errori ad alta confidenza (>0.3): {len(high_conf_errors)} ({len(high_conf_errors)/len(errors):.1%})")

        return {
            'error_confidence': error_conf.to_dict(),
            'correct_confidence': correct_conf.to_dict(),
            'high_confidence_errors': len(high_conf_errors),
            'high_confidence_error_rate': len(high_conf_errors) / len(errors)
        }

    def _create_error_visualizations(self, errors, correct, analysis_results):
        """Crea visualizzazioni per l'analisi degli errori."""
        print("📊 Generazione visualizzazioni...")

        # Figura multi-pannello
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # 1. Distribuzione lunghezza nomi
        axes[0, 0].hist(errors['name_length'], bins=30, alpha=0.7, label='Errori', color='red')
        axes[0, 0].hist(correct['name_length'], bins=30, alpha=0.7, label='Corretti', color='green')
        axes[0, 0].set_xlabel('Lunghezza Nome')
        axes[0, 0].set_ylabel('Frequenza')
        axes[0, 0].set_title('Distribuzione Lunghezza Nomi')
        axes[0, 0].legend()

        # 2. Distribuzione confidenza
        axes[0, 1].hist(errors['confidence'], bins=30, alpha=0.7, label='Errori', color='red')
        axes[0, 1].hist(correct['confidence'], bins=30, alpha=0.7, label='Corretti', color='green')
        axes[0, 1].set_xlabel('Confidenza')
        axes[0, 1].set_ylabel('Frequenza')
        axes[0, 1].set_title('Distribuzione Confidenza')
        axes[0, 1].legend()

        # 3. Errori per tipo
        error_types = errors['error_type'].value_counts()
        axes[0, 2].bar(error_types.index, error_types.values, color=['blue', 'orange'])
        axes[0, 2].set_xlabel('Tipo Errore')
        axes[0, 2].set_ylabel('Conteggio')
        axes[0, 2].set_title('Errori per Tipo')

        # 4. Boxplot lunghezza per tipo errore
        error_data = [errors[errors['error_type'] == 'M_to_W']['name_length'].values,
                     errors[errors['error_type'] == 'W_to_M']['name_length'].values]
        axes[1, 0].boxplot(error_data, labels=['M→W', 'W→M'])
        axes[1, 0].set_ylabel('Lunghezza Nome')
        axes[1, 0].set_title('Lunghezza Nome per Tipo Errore')

        # 5. Heatmap suffissi
        suffix_data = []
        suffix_labels = ['ends_with_a', 'ends_with_o', 'ends_with_e', 'ends_with_vowel']

        for error_type in ['M_to_W', 'W_to_M']:
            error_subset = errors[errors['error_type'] == error_type]
            suffix_rates = [error_subset[feature].mean() for feature in suffix_labels]
            suffix_data.append(suffix_rates)

        im = axes[1, 1].imshow(suffix_data, cmap='RdYlBu', aspect='auto')
        axes[1, 1].set_xticks(range(len(suffix_labels)))
        axes[1, 1].set_xticklabels(suffix_labels, rotation=45)
        axes[1, 1].set_yticks(range(2))
        axes[1, 1].set_yticklabels(['M→W', 'W→M'])
        axes[1, 1].set_title('Pattern Suffissi per Tipo Errore')

        # Aggiungi colorbar
        plt.colorbar(im, ax=axes[1, 1])

        # 6. Top errori
        top_errors = errors['primaryName'].value_counts().head(10)
        axes[1, 2].barh(range(len(top_errors)), top_errors.values)
        axes[1, 2].set_yticks(range(len(top_errors)))
        axes[1, 2].set_yticklabels(top_errors.index)
        axes[1, 2].set_xlabel('Numero Errori')
        axes[1, 2].set_title('Top 10 Nomi con Più Errori')

        plt.tight_layout()

        # Salva visualizzazione
        if self.experiment_manager:
            plot_path = os.path.join(self.experiment_manager.plots_dir, "error_analysis.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"📊 Visualizzazioni salvate in: {plot_path}")
        else:
            plt.savefig("error_analysis.png", dpi=300, bbox_inches='tight')

        plt.close()

    def _convert_for_json(self, obj):
        """Converte oggetti numpy/pandas per JSON."""
        if isinstance(obj, dict):
            return {key: self._convert_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        else:
            return obj

    def find_similar_error_patterns(self, min_frequency=3):
        """Trova pattern simili negli errori per identificare problemi sistematici."""
        if self.errors_df is None:
            raise ValueError("Esegui analyze_errors prima di cercare pattern")

        print(f"\n🔍 RICERCA PATTERN SIMILI (min frequency: {min_frequency}):")

        errors = self.errors_df

        # Pattern per primo nome
        first_names = errors['first_name'].value_counts()
        frequent_first_names = first_names[first_names >= min_frequency]

        if len(frequent_first_names) > 0:
            print(f"\nPrimi nomi con errori frequenti:")
            for name, count in frequent_first_names.head(10).items():
                name_errors = errors[errors['first_name'] == name]
                error_types = name_errors['error_type'].value_counts()
                print(f"   {name}: {count} errori - {dict(error_types)}")

        # Pattern per suffissi
        print(f"\nPattern suffissi problematici:")
        for length in [2, 3, 4]:
            suffixes = errors['first_name'].str.lower().str[-length:].value_counts()
            frequent_suffixes = suffixes[suffixes >= min_frequency]

            if len(frequent_suffixes) > 0:
                print(f"   Suffissi di {length} caratteri:")
                for suffix, count in frequent_suffixes.head(5).items():
                    print(f"     -{suffix}: {count} errori")


# Funzione per integrare nel train_improved_model_v2.py
def add_error_analysis_to_training(experiment, model, test_dataset, preprocessor, device):
    """
    Funzione da aggiungere al training script per generare automaticamente l'analisi errori.

    Args:
        experiment: Istanza ExperimentManager
        model: Modello addestrato
        test_dataset: Dataset di test
        preprocessor: Preprocessore
        device: Device

    Returns:
        predictions_df, analysis_results
    """
    print("\n🔍 Avvio analisi errori automatica...")

    # Crea analyzer
    analyzer = ErrorAnalyzer(experiment)

    # Genera dataset errori
    predictions_df = analyzer.generate_error_dataset(
        model, test_dataset, preprocessor, device
    )

    # Analizza errori
    analysis_results = analyzer.analyze_errors(predictions_df)

    # Cerca pattern
    analyzer.find_similar_error_patterns()

    print("✅ Analisi errori completata!")

    return predictions_df, analysis_results


if __name__ == "__main__":
    # Script standalone per analizzare file di predizioni esistenti
    import argparse

    parser = argparse.ArgumentParser(description="Analizza errori del modello")
    parser.add_argument("predictions_file", help="File CSV con predizioni")
    parser.add_argument("--output_dir", default=".", help="Directory output")

    args = parser.parse_args()

    # Carica predizioni
    df = pd.read_csv(args.predictions_file)

    # Crea analyzer
    analyzer = ErrorAnalyzer()

    # Analizza
    results = analyzer.analyze_errors(df)

    print(f"\n✅ Analisi completata! Risultati in {args.output_dir}")
