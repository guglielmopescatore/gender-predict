import os
import json
import time
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import torch
class ExperimentManager:
    """Classe per gestire esperimenti di machine learning."""

    def __init__(self, args, base_dir=".", auto_create=True):
        """
        Inizializza il manager degli esperimenti.

        Args:
            args: Argomenti di training (Namespace o dict)
            base_dir: Directory base per il salvataggio
            auto_create: Se True, crea automaticamente le directory e inizializza i file
        """
        self.args = args
        self.base_dir = base_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_id = self._generate_experiment_id()

        # Crea la struttura delle directory
        self.experiments_dir = os.path.join(base_dir, "experiments")
        self.experiment_dir = os.path.join(self.experiments_dir, self.experiment_id)
        self.models_dir = os.path.join(self.experiment_dir, "models")
        self.logs_dir = os.path.join(self.experiment_dir, "logs")
        self.plots_dir = os.path.join(self.experiment_dir, "plots")
        self.bias_dir = os.path.join(self.plots_dir, "bias_analysis")  # Directory specifica per le analisi di bias

        # Percorsi per i file principali
        self.model_path = os.path.join(self.models_dir, "model.pth")
        self.preprocessor_path = os.path.join(self.experiment_dir, "preprocessor.pkl")
        self.metrics_path = os.path.join(self.logs_dir, "metrics.csv")
        self.train_history_path = os.path.join(self.logs_dir, "train_history.csv")
        self.test_metrics_path = os.path.join(self.logs_dir, "test_metrics.json")
        self.bias_metrics_path = os.path.join(self.logs_dir, "bias_metrics.json")
        self.confusion_path = os.path.join(self.plots_dir, "confusion_matrix.png")
        self.history_plot_path = os.path.join(self.plots_dir, "training_history.png")
        self.bias_analysis_path = os.path.join(self.bias_dir, "gender_bias_analysis.png")

        # Inizializza se richiesto
        if auto_create:
            self._ensure_dirs()
            self._save_params()
            self._init_experiment_log()

    def _generate_experiment_id(self):
        """Genera un ID univoco per l'esperimento basato sui parametri principali."""
        # ID base con timestamp e round
        base_id = f"{self.timestamp}_r{self.args.round}"

        # Aggiungi parametri di loss function per round ≥ 1
        if self.args.round >= 1:
            if hasattr(self.args, 'loss') and self.args.loss == "focal":
                base_id += f"_focal_a{self.args.alpha}_g{self.args.gamma}"
            else:
                base_id += f"_bce"

            if hasattr(self.args, 'pos_weight') and self.args.pos_weight != 1.0:
                base_id += f"_pw{self.args.pos_weight}"

            if hasattr(self.args, 'label_smooth') and self.args.label_smooth > 0:
                base_id += f"_ls{self.args.label_smooth}"

            if hasattr(self.args, 'balanced_sampler') and self.args.balanced_sampler:
                base_id += "_bs"

        # Aggiungi parametri architetturali per round ≥ 2
        if self.args.round >= 2:
            base_id += f"_h{self.args.hidden_size}_l{self.args.n_layers}"

            if hasattr(self.args, 'dual_input') and self.args.dual_input:
                base_id += "_dual"

            if hasattr(self.args, 'freeze_epochs') and self.args.freeze_epochs > 0:
                base_id += f"_frz{self.args.freeze_epochs}"

        return base_id

    def _ensure_dirs(self):
        """Crea le directory necessarie se non esistono."""
        for directory in [self.experiments_dir, self.experiment_dir,
                          self.models_dir, self.logs_dir, self.plots_dir, self.bias_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
                print(f"Created directory: {directory}")

    def _save_params(self):
        """Salva i parametri dell'esperimento in un file JSON."""
        params_path = os.path.join(self.experiment_dir, "parameters.json")

        # Estrai i parametri come dizionario
        if hasattr(self.args, '__dict__'):
            params = vars(self.args)
        else:
            params = self.args

        with open(params_path, 'w') as f:
            json.dump(params, f, indent=4)

        print(f"Parameters saved to {params_path}")

    def _init_experiment_log(self):
        """Inizializza o aggiorna il file di log degli esperimenti."""
        experiments_log_path = os.path.join(self.experiments_dir, "experiments_log.csv")
        if os.path.exists(experiments_log_path):
            log_df = pd.read_csv(experiments_log_path)
            idx = log_df[log_df['experiment_id'] == self.experiment_id].index

            if len(idx) > 0:
                # Aggiungi le metriche di errore al log
                for key, value in error_summary.items():
                    if isinstance(value, (int, float)):
                        log_df.loc[idx[0], f"error_{key}"] = value
                    elif isinstance(value, list):
                        log_df.loc[idx[0], f"error_{key}"] = str(value)

                log_df.to_csv(experiments_log_path, index=False)

        # Parametri principali per il log
        log_entry = {
            "timestamp": self.timestamp,
            "experiment_id": self.experiment_id,
            "experiment_dir": self.experiment_dir,
            "round": self.args.round,
            "loss": getattr(self.args, "loss", "bce")
        }

        # Aggiungi altri parametri rilevanti
        if hasattr(self.args, '__dict__'):
            for key, value in vars(self.args).items():
                if key not in log_entry:
                    log_entry[key] = value
        else:
            for key, value in self.args.items():
                if key not in log_entry:
                    log_entry[key] = value

        # Aggiungi al log esistente o crea un nuovo log
        if os.path.exists(experiments_log_path):
            log_df = pd.read_csv(experiments_log_path)
            log_df = pd.concat([log_df, pd.DataFrame([log_entry])], ignore_index=True)
        else:
            log_df = pd.DataFrame([log_entry])

        log_df.to_csv(experiments_log_path, index=False)
        print(f"Experiment added to log at {experiments_log_path}")

    def log_training_history(self, history):
        """
        Salva la storia dell'addestramento.

        Args:
            history: Dizionario con le metriche di training
        """
        # Converti in DataFrame se è un dizionario
        if isinstance(history, dict):
            history_df = pd.DataFrame(history)
        else:
            history_df = history

        history_df.to_csv(self.train_history_path, index=False)
        print(f"Training history saved to {self.train_history_path}")

        # Aggiorna anche il plot della storia di training
        self.plot_training_history(history_df)

    def log_error_summary(self, error_summary):
        """
        Salva un summary dell'analisi degli errori.

        Args:
            error_summary: Dict con il summary degli errori
        """
        import json

        error_summary_path = os.path.join(self.logs_dir, "error_summary.json")

        with open(error_summary_path, 'w') as f:
            json.dump(error_summary, f, indent=2)

        print(f"💾 Error summary saved to {error_summary_path}")

        # Aggiorna anche il log degli esperimenti
        experiments_log_path = os.path.join(self.experiments_dir, "experiments_log.csv")
        if os.path.exists(experiments_log_path):
            import pandas as pd
            log_df = pd.read_csv(experiments_log_path)
            idx = log_df[log_df['experiment_id'] == self.experiment_id].index

            if len(idx) > 0:
                # Aggiungi le metriche di errore al log
                for key, value in error_summary.items():
                    if isinstance(value, (int, float)):
                        log_df.loc[idx[0], f"error_{key}"] = value
                    elif isinstance(value, list):
                        log_df.loc[idx[0], f"error_{key}"] = str(value)

                log_df.to_csv(experiments_log_path, index=False)

    def log_test_metrics(self, metrics):
        """
        Salva le metriche del test set.

        Args:
            metrics: Dizionario con le metriche
        """
        with open(self.test_metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)

        print(f"Test metrics saved to {self.test_metrics_path}")

        # Aggiorna anche il log degli esperimenti con le metriche di test
        experiments_log_path = os.path.join(self.experiments_dir, "experiments_log.csv")
        if os.path.exists(experiments_log_path):
            log_df = pd.read_csv(experiments_log_path)

            # Trova la riga corrispondente all'esperimento attuale
            idx = log_df[log_df['experiment_id'] == self.experiment_id].index

            if len(idx) > 0:
                for metric_name, metric_value in metrics.items():
                    log_df.loc[idx[0], f"test_{metric_name}"] = metric_value

                log_df.to_csv(experiments_log_path, index=False)

    def analyze_gender_bias(self, y_true, y_pred, output_file=None, figsize=(15, 12)):
        """
        Analizza in dettaglio il bias di genere e salva le metriche.

        Args:
            y_true: Etichette vere (0 per 'M', 1 per 'W')
            y_pred: Valori predetti (0 per 'M', 1 per 'W')
            output_file: Percorso dove salvare il grafico di analisi
            figsize: Dimensione della figura

        Returns:
            Dizionario con le metriche di bias calcolate
        """
        # Calcola la matrice di confusione
        cm = confusion_matrix(y_true, y_pred)

        # Estrai i valori della matrice di confusione
        tn, fp, fn, tp = cm.ravel()

        # Calcola metriche specifiche per genere
        # Maschi (M)
        m_precision = tn / (tn + fn) if (tn + fn) > 0 else 0  # Quanti dei predetti maschi sono effettivamente maschi
        m_recall = tn / (tn + fp) if (tn + fp) > 0 else 0     # Quanti maschi reali sono stati identificati correttamente
        m_f1 = 2 * (m_precision * m_recall) / (m_precision + m_recall) if (m_precision + m_recall) > 0 else 0
        m_error_rate = fp / (tn + fp) if (tn + fp) > 0 else 0  # Tasso di errore M→W

        # Femmine (W)
        w_precision = tp / (tp + fp) if (tp + fp) > 0 else 0  # Quante delle predette femmine sono effettivamente femmine
        w_recall = tp / (tp + fn) if (tp + fn) > 0 else 0     # Quante femmine reali sono state identificate correttamente
        w_f1 = 2 * (w_precision * w_recall) / (w_precision + w_recall) if (w_precision + w_recall) > 0 else 0
        w_error_rate = fn / (tp + fn) if (tp + fn) > 0 else 0  # Tasso di errore W→M

        # Calcola il bias ratio
        bias_ratio = (fp / (tn + fp)) / (fn / (tp + fn)) if (tn + fp) > 0 and (tp + fn) > 0 and fn > 0 else float('inf')

        # Metriche di disparità
        equality_of_opportunity = m_recall - w_recall  # Differenza nei tassi di recall
        predictive_equality = m_error_rate - w_error_rate  # Differenza nei tassi di errore

        # Calcola accuratezza globale
        accuracy = (tp + tn) / (tp + tn + fp + fn)

        # Determina la direzione del bias
        if bias_ratio > 1.1:
            bias_direction = "Bias verso W"  # Tende a classificare erroneamente M come W
        elif bias_ratio < 0.9:
            bias_direction = "Bias verso M"  # Tende a classificare erroneamente W come M
        else:
            bias_direction = "Bias bilanciato"

        # Crea un dizionario con tutte le metriche di bias
        bias_metrics = {
            # Metriche generali
            'accuracy': float(accuracy),
            'bias_ratio': float(bias_ratio) if bias_ratio != float('inf') else 999.0,
            'bias_direction': bias_direction,

            # Metriche specifiche per genere
            'm_precision': float(m_precision),
            'm_recall': float(m_recall),
            'm_f1': float(m_f1),
            'm_error_rate': float(m_error_rate),

            'w_precision': float(w_precision),
            'w_recall': float(w_recall),
            'w_f1': float(w_f1),
            'w_error_rate': float(w_error_rate),

            # Metriche di disparità
            'equality_of_opportunity': float(equality_of_opportunity),
            'predictive_equality': float(predictive_equality),

            # Valori di confusione matrix per riferimento
            'true_negative': int(tn),  # M→M
            'false_positive': int(fp),  # M→W
            'false_negative': int(fn),  # W→M
            'true_positive': int(tp),   # W→W

            # Statistiche di popolazione
            'male_count': int(tn + fp),
            'female_count': int(fn + tp)
        }

        # Salva le metriche di bias
        with open(self.bias_metrics_path, 'w') as f:
            json.dump(bias_metrics, f, indent=4)

        # Aggiorna anche il log generale degli esperimenti
        experiments_log_path = os.path.join(self.experiments_dir, "experiments_log.csv")
        if os.path.exists(experiments_log_path):
            log_df = pd.read_csv(experiments_log_path)

            idx = log_df[log_df['experiment_id'] == self.experiment_id].index
            if len(idx) > 0:
                # Aggiungi le principali metriche di bias al log
                bias_columns = ['bias_ratio', 'm_precision', 'm_recall', 'w_precision', 'w_recall',
                               'equality_of_opportunity', 'predictive_equality']

                for col in bias_columns:
                    log_df.loc[idx[0], f"bias_{col}"] = bias_metrics[col]

                log_df.to_csv(experiments_log_path, index=False)

        # Genera una visualizzazione dettagliata del bias
        if output_file is None:
            output_file = self.bias_analysis_path

        self._plot_gender_bias_analysis(bias_metrics, cm, output_file, figsize)

        return bias_metrics

    def _plot_gender_bias_analysis(self, bias_metrics, cm, output_file, figsize=(15, 12)):
        """
        Crea una visualizzazione dettagliata dell'analisi del bias di genere.

        Args:
            bias_metrics: Dizionario con le metriche di bias
            cm: Matrice di confusione
            output_file: Percorso dove salvare il grafico
            figsize: Dimensione della figura
        """
        plt.figure(figsize=figsize)

        # 1. Matrice di confusione
        plt.subplot(2, 2, 1)
        labels = ['Maschio (M)', 'Femmina (W)']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.title('Matrice di Confusione')
        plt.ylabel('Valore Reale')
        plt.xlabel('Valore Predetto')

        # 2. Metriche per genere
        plt.subplot(2, 2, 2)
        metrics_data = {
            'Metrica': ['Precisione', 'Recall', 'F1-Score', 'Tasso di Errore'],
            'Maschio (M)': [bias_metrics['m_precision'], bias_metrics['m_recall'],
                           bias_metrics['m_f1'], bias_metrics['m_error_rate']],
            'Femmina (W)': [bias_metrics['w_precision'], bias_metrics['w_recall'],
                           bias_metrics['w_f1'], bias_metrics['w_error_rate']]
        }
        metrics_df = pd.DataFrame(metrics_data).set_index('Metrica')

        # Aggiungi un heatmap per le metriche
        ax = sns.heatmap(metrics_df, annot=True, fmt='.3f', cmap='Greens', cbar=False)
        plt.title('Metriche Specifiche per Genere')

        # 3. Visualizzazione del Bias Ratio
        plt.subplot(2, 2, 3)
        bias_ratio = bias_metrics['bias_ratio']

        # Crea una scala di colori per il bias
        bias_range = np.linspace(0, 2, 100)
        colors = []
        for x in bias_range:
            if x < 0.9:
                colors.append('blue')  # Bias verso M
            elif x > 1.1:
                colors.append('red')   # Bias verso W
            else:
                colors.append('green') # Bilanciato

        plt.bar(['Bias Ratio'], [bias_ratio], color='purple')
        plt.axhline(y=1.0, color='green', linestyle='--', alpha=0.7, label='Perfetto bilanciamento')
        plt.axhline(y=0.9, color='blue', linestyle=':', alpha=0.7, label='Soglia bias verso M')
        plt.axhline(y=1.1, color='red', linestyle=':', alpha=0.7, label='Soglia bias verso W')

        plt.ylim(0, min(3, bias_ratio*1.5))  # Limita l'asse y per casi di bias estremo
        plt.title(f"Rapporto di Bias: {bias_ratio:.2f}\nDirezione: {bias_metrics['bias_direction']}")
        plt.legend(loc='upper right')

        # 4. Errori di classificazione per genere
        plt.subplot(2, 2, 4)
        error_data = {
            'Tipo di Errore': ['M→W (Maschi classificati come Femmine)',
                              'W→M (Femmine classificate come Maschi)'],
            'Conteggio': [bias_metrics['false_positive'], bias_metrics['false_negative']],
            'Percentuale': [bias_metrics['m_error_rate']*100, bias_metrics['w_error_rate']*100]
        }
        error_df = pd.DataFrame(error_data)

        # Usa barplot con due serie: conteggio e percentuale
        ax = plt.gca()
        bars = sns.barplot(x='Tipo di Errore', y='Conteggio', hue='Tipo di Errore', data=error_df, palette='Set1', alpha=0.7, ax=ax, legend=False)

        # Aggiungi etichette con percentuali sopra le barre
        for i, p in enumerate(bars.patches):
            percentage = error_df.iloc[i]['Percentuale']
            bars.annotate(f'{percentage:.1f}%',
                        (p.get_x() + p.get_width()/2., p.get_height()),
                        ha='center', va='center',
                        fontsize=11, color='black',
                        xytext=(0, 10), textcoords='offset points')

        plt.title('Distribuzione degli Errori di Classificazione')
        plt.ylabel('Numero di Errori')

        # Statistiche riassuntive
        plt.figtext(0.5, 0.01,
                   f"Accuratezza Globale: {bias_metrics['accuracy']:.3f} | "
                   f"Equality of Opportunity: {bias_metrics['equality_of_opportunity']:.3f} | "
                   f"Predictive Equality: {bias_metrics['predictive_equality']:.3f}",
                   ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

        plt.suptitle('Analisi del Bias di Genere', fontsize=16, y=0.98)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(output_file)
        plt.close()

        print(f"Bias analysis plot saved to {output_file}")

    def save_confusion_matrix(self, y_true, y_pred, labels=None, output_file=None):
        """
        Calcola e salva la matrice di confusione.

        Args:
            y_true: Etichette vere
            y_pred: Predizioni
            labels: Etichette opzionali per gli assi
            output_file: Percorso file di output opzionale

        Returns:
            Matrice di confusione calcolata
        """
        # Calcola la matrice di confusione
        cm = confusion_matrix(y_true, y_pred)

        # Etichette predefinite
        if labels is None:
            labels = ["Male", "Female"]

        # Visualizza la matrice
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')

        # Salva il grafico
        output_file = output_file or self.confusion_path
        plt.savefig(output_file)
        plt.close()

        print(f"Confusion matrix saved to {output_file}")

        # Automaticamente esegui anche un'analisi completa del bias di genere
        self.analyze_gender_bias(y_true, y_pred)

        return cm

    def plot_training_history(self, history=None, output_file=None):
        """
        Visualizza e salva la storia dell'addestramento.

        Args:
            history: DataFrame o dizionario con la storia dell'addestramento (opzionale)
            output_file: Percorso file di output opzionale
        """
        # Se history non è fornito, carica dal file
        if history is None:
            if os.path.exists(self.train_history_path):
                history = pd.read_csv(self.train_history_path)
            else:
                print(f"Training history file not found: {self.train_history_path}")
                return

        # Converti in DataFrame se è un dizionario
        if isinstance(history, dict):
            history = pd.DataFrame(history)

        # Crea il grafico
        plt.figure(figsize=(15, 10))

        # Plot dell'accuratezza
        plt.subplot(2, 2, 1)
        if 'train_acc' in history.columns:
            plt.plot(history['train_acc'], label='Train Accuracy')
        if 'val_acc' in history.columns:
            plt.plot(history['val_acc'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        # Plot della loss
        plt.subplot(2, 2, 2)
        if 'train_loss' in history.columns:
            plt.plot(history['train_loss'], label='Train Loss')
        if 'val_loss' in history.columns:
            plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        # Plot di precision e recall
        plt.subplot(2, 2, 3)
        if 'val_precision' in history.columns:
            plt.plot(history['val_precision'], label='Precision')
        if 'val_recall' in history.columns:
            plt.plot(history['val_recall'], label='Recall')
        if 'val_f1' in history.columns:
            plt.plot(history['val_f1'], label='F1 Score')
        plt.title('Validation Metrics')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.legend()

        # Aggiungi informazioni sull'esperimento
        plt.suptitle(f"Training History - Experiment {self.experiment_id}", fontsize=16)

        # Aggiungi griglia a tutti i subplot
        for i in range(1, 4):
            plt.subplot(2, 2, i)
            plt.grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Aggiusta per il suptitle

        # Salva il grafico
        output_file = output_file or self.history_plot_path
        plt.savefig(output_file)
        plt.close()

        print(f"Training history plot saved to {output_file}")

    def save_model_checkpoint(self, model_state, epoch=None, metrics=None):
        """
        Salva un checkpoint del modello.

        Args:
            model_state: Stato del modello o dizionario completo del checkpoint
            epoch: Numero dell'epoca (opzionale)
            metrics: Metriche da salvare con il checkpoint (opzionale)
        """
        if epoch is not None:
            checkpoint_path = self.get_model_path(f"epoch_{epoch}")
        else:
            checkpoint_path = self.model_path

        # Se model_state è già un dizionario di checkpoint completo
        if isinstance(model_state, dict) and 'model_state_dict' in model_state:
            checkpoint = model_state
        else:
            # Altrimenti, crea un nuovo dizionario di checkpoint
            checkpoint = {
                'model_state_dict': model_state,
                'epoch': epoch
            }

            # Aggiungi metriche se fornite
            if metrics is not None:
                for key, value in metrics.items():
                    checkpoint[key] = value

        # Salva il checkpoint
        torch.save(checkpoint, checkpoint_path)
        print(f"Model checkpoint saved to {checkpoint_path}")

    def get_model_path(self, suffix=None):
        """
        Ottieni il percorso del file del modello.

        Args:
            suffix: Suffisso opzionale per il file (es. 'epoch_10')

        Returns:
            Percorso completo del file del modello
        """
        if suffix:
            base, ext = os.path.splitext(self.model_path)
            return f"{base}_{suffix}{ext}"
        return self.model_path

    def get_plot_path(self, name):
        """
        Ottieni il percorso per un grafico specifico.

        Args:
            name: Nome del grafico

        Returns:
            Percorso completo del file del grafico
        """
        return os.path.join(self.plots_dir, f"{name}.png")

    def log_error_summary(self, error_summary):
        """
        Salva un summary dell'analisi degli errori.

        Args:
            error_summary: Dict con il summary degli errori
        """
        error_summary_path = os.path.join(self.logs_dir, "error_summary.json")

        with open(error_summary_path, 'w') as f:
            json.dump(error_summary, f, indent=2)

        print(f"Error summary saved to {error_summary_path}")

    def generate_report(self):
        """
        Genera un report HTML sintetico dell'esperimento.

        Returns:
            Percorso del file HTML generato
        """
        report_path = os.path.join(self.experiment_dir, "report.html")

        # Raccogli i dati
        parameters = {}
        if os.path.exists(os.path.join(self.experiment_dir, "parameters.json")):
            with open(os.path.join(self.experiment_dir, "parameters.json"), 'r') as f:
                parameters = json.load(f)

        test_metrics = {}
        if os.path.exists(self.test_metrics_path):
            with open(self.test_metrics_path, 'r') as f:
                test_metrics = json.load(f)

        bias_metrics = {}
        if os.path.exists(self.bias_metrics_path):
            with open(self.bias_metrics_path, 'r') as f:
                bias_metrics = json.load(f)

        # Genera il report HTML
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Experiment Report - {self.experiment_id}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #333; }}
                .parameter {{ font-family: monospace; }}
                .metrics {{ color: #0066cc; font-weight: bold; }}
                .bias-balanced {{ color: green; font-weight: bold; }}
                .bias-male {{ color: blue; font-weight: bold; }}
                .bias-female {{ color: red; font-weight: bold; }}
                .card {{ border: 1px solid #ddd; padding: 15px; margin-bottom: 20px; border-radius: 5px; }}
                .container {{ display: flex; flex-wrap: wrap; }}
                .chart {{ margin: 10px; flex: 1; min-width: 300px; }}
                .metrics-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
                img {{ max-width: 100%; height: auto; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>Experiment Report</h1>
            <p><strong>ID:</strong> {self.experiment_id}</p>
            <p><strong>Directory:</strong> {self.experiment_dir}</p>
            <p><strong>Date:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>

            <div class="card">
                <h2>Parameters</h2>
                <ul>
        """

        # Aggiungi parametri
        for key, value in parameters.items():
            html += f"<li><span class='parameter'>{key}</span>: {value}</li>"

        html += """
                </ul>
            </div>

            <div class="card">
                <h2>Test Metrics</h2>
                <div class="metrics-grid">
                    <div>
                        <h3>General Metrics</h3>
                        <table>
                            <tr><th>Metric</th><th>Value</th></tr>
        """

        # Aggiungi metriche generali
        if test_metrics:
            for key in ['accuracy', 'precision', 'recall', 'f1']:
                if key in test_metrics:
                    html += f"<tr><td>{key.capitalize()}</td><td class='metrics'>{test_metrics[key]:.4f}</td></tr>"

        html += """
                        </table>
                    </div>
        """

        # Aggiungi metriche di bias
        if bias_metrics:
            # Determina la classe CSS per il bias direction
            bias_class = "bias-balanced"
            if bias_metrics.get('bias_direction') == "Bias verso M":
                bias_class = "bias-male"
            elif bias_metrics.get('bias_direction') == "Bias verso W":
                bias_class = "bias-female"

            html += f"""
                    <div>
                        <h3>Gender Bias Metrics</h3>
                        <table>
                            <tr><th>Metric</th><th>Value</th></tr>
                            <tr><td>Bias Ratio (M→W:W→M)</td><td class='{bias_class}'>{bias_metrics.get('bias_ratio', 'N/A'):.4f}</td></tr>
                            <tr><td>Bias Direction</td><td class='{bias_class}'>{bias_metrics.get('bias_direction', 'N/A')}</td></tr>
                            <tr><td>M→W Error Rate</td><td>{bias_metrics.get('m_error_rate', 'N/A'):.4f}</td></tr>
                            <tr><td>W→M Error Rate</td><td>{bias_metrics.get('w_error_rate', 'N/A'):.4f}</td></tr>
                            <tr><td>Equality of Opportunity</td><td>{bias_metrics.get('equality_of_opportunity', 'N/A'):.4f}</td></tr>
                            <tr><td>Predictive Equality</td><td>{bias_metrics.get('predictive_equality', 'N/A'):.4f}</td></tr>
                        </table>
                    </div>
            """

            # Aggiungi altre metriche specifiche per genere
            html += """
                </div>

                <h3>Metrics by Gender</h3>
                <table>
                    <tr>
                        <th>Metric</th>
                        <th>Male (M)</th>
                        <th>Female (W)</th>
                    </tr>
            """

            for metric in ['precision', 'recall', 'f1']:
                html += f"""
                    <tr>
                        <td>{metric.capitalize()}</td>
                        <td>{bias_metrics.get(f'm_{metric}', 'N/A'):.4f}</td>
                        <td>{bias_metrics.get(f'w_{metric}', 'N/A'):.4f}</td>
                    </tr>
                """

            html += """
                </table>
            """

        html += """
            </div>

            <div class="card">
                <h2>Visualizations</h2>
                <div class="container">
        """

        # Aggiungi visualizzazioni
        if os.path.exists(self.bias_analysis_path):
            rel_path = os.path.relpath(self.bias_analysis_path, self.experiment_dir)
            html += f"""
                    <div class="chart">
                        <h3>Gender Bias Analysis</h3>
                        <img src="{rel_path}" alt="Gender Bias Analysis">
                    </div>
            """

        if os.path.exists(self.confusion_path):
            rel_path = os.path.relpath(self.confusion_path, self.experiment_dir)
            html += f"""
                    <div class="chart">
                        <h3>Confusion Matrix</h3>
                        <img src="{rel_path}" alt="Confusion Matrix">
                    </div>
            """

        if os.path.exists(self.history_plot_path):
            rel_path = os.path.relpath(self.history_plot_path, self.experiment_dir)
            html += f"""
                    <div class="chart">
                        <h3>Training History</h3>
                        <img src="{rel_path}" alt="Training History">
                    </div>
            """

        html += """
                </div>
            </div>
        </body>
        </html>
        """

        # Salva il report
        with open(report_path, 'w') as f:
            f.write(html)

        print(f"Report generated at {report_path}")
        return report_path

    @classmethod
    def from_experiment_id(cls, experiment_id, base_dir="."):
        """
        Carica un esperimento esistente.

        Args:
            experiment_id: ID dell'esperimento
            base_dir: Directory base degli esperimenti

        Returns:
            Istanza di ExperimentManager per l'esperimento esistente
        """
        # Costruisci il percorso dell'esperimento
        experiment_dir = os.path.join(base_dir, "experiments", experiment_id)

        if not os.path.exists(experiment_dir):
            raise ValueError(f"Experiment {experiment_id} not found in {base_dir}/experiments")

        # Carica i parametri
        params_path = os.path.join(experiment_dir, "parameters.json")
        if os.path.exists(params_path):
            with open(params_path, 'r') as f:
                params = json.load(f)
        else:
            # Se i parametri non esistono, crea un oggetto vuoto
            params = {"round": 0}

        # Crea un'istanza di ExperimentManager con auto_create=False
        import argparse
        args = argparse.Namespace(**params)
        experiment = cls(args, base_dir, auto_create=False)

        # Sovrascrivi l'ID e i percorsi
        experiment.experiment_id = experiment_id
        experiment.experiment_dir = experiment_dir
        experiment.models_dir = os.path.join(experiment_dir, "models")
        experiment.logs_dir = os.path.join(experiment_dir, "logs")
        experiment.plots_dir = os.path.join(experiment_dir, "plots")
        experiment.bias_dir = os.path.join(experiment.plots_dir, "bias_analysis")

        # Ripristina i percorsi per i file principali
        experiment.model_path = os.path.join(experiment.models_dir, "model.pth")
        experiment.preprocessor_path = os.path.join(experiment.experiment_dir, "preprocessor.pkl")
        experiment.metrics_path = os.path.join(experiment.logs_dir, "metrics.csv")
        experiment.train_history_path = os.path.join(experiment.logs_dir, "train_history.csv")
        experiment.test_metrics_path = os.path.join(experiment.logs_dir, "test_metrics.json")
        experiment.bias_metrics_path = os.path.join(experiment.logs_dir, "bias_metrics.json")
        experiment.confusion_path = os.path.join(experiment.plots_dir, "confusion_matrix.png")
        experiment.history_plot_path = os.path.join(experiment.plots_dir, "training_history.png")
        experiment.bias_analysis_path = os.path.join(experiment.bias_dir, "gender_bias_analysis.png")

        return experiment
