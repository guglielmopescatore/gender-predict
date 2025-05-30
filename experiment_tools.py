import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import argparse
from datetime import datetime
import glob
import torch
from sklearn.manifold import TSNE

def compare_experiments(base_dir=".", filter_dict=None, metric="test_accuracy", group_by=None, save_path=None, show=True):
    """
    Confronta diversi esperimenti in base a una metrica.

    Args:
        base_dir: Directory base degli esperimenti
        filter_dict: Dizionario con filtri (es. {"round": 2})
        metric: Metrica da confrontare
        group_by: Parametro per raggruppare
        save_path: Percorso dove salvare il grafico
        show: Se mostrare il grafico

    Returns:
        DataFrame filtrato con gli esperimenti
    """
    # Carica il log degli esperimenti
    log_path = os.path.join(base_dir, "experiments", "experiments_log.csv")
    if not os.path.exists(log_path):
        print(f"Experiments log not found at {log_path}")
        return None

    df = pd.read_csv(log_path)

    # Filtra gli esperimenti
    if filter_dict:
        for key, value in filter_dict.items():
            if key in df.columns:
                df = df[df[key] == value]

    if len(df) == 0:
        print("No experiments match the filter criteria")
        return None

    # Verifica se la metrica è presente
    if metric not in df.columns:
        print(f"Metric '{metric}' not found in experiments log")
        return df

    # Crea il grafico
    plt.figure(figsize=(12, 6))

    if group_by and group_by in df.columns:
        # Raggruppa gli esperimenti
        df_grouped = df.groupby(group_by)[metric].mean().reset_index()

        # Crea un grafico a barre raggruppato
        palette = sns.color_palette("husl", len(df_grouped))
        ax = sns.barplot(x=group_by, y=metric, hue=group_by, data=df_grouped, palette=palette, legend=False)

        # Aggiungi testo con il valore
        for i, v in enumerate(df_grouped[metric]):
            ax.text(i, v + 0.01, f"{v:.4f}", ha='center')

        # Aggiungi il numero di esperimenti per gruppo
        group_counts = df.groupby(group_by).size()
        plt.xlabel(f"{group_by} (Number of experiments)")
        plt.xticks(range(len(group_counts)), [f"{g} (n={n})" for g, n in zip(group_counts.index, group_counts)])

    else:
        # Crea un grafico a barre semplice
        ax = sns.barplot(x='experiment_id', y=metric, data=df)
        plt.xticks(rotation=45, ha='right')

        # Aggiungi testo con il valore
        for i, v in enumerate(df[metric]):
            ax.text(i, v + 0.01, f"{v:.4f}", ha='center')

    plt.title(f"Comparison of {metric} across experiments")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return df

def compare_bias_metrics(base_dir=".", filter_dict=None, save_path=None, show=True):
    """
    Confronta esperimenti in base alle loro metriche di bias di genere.

    Args:
        base_dir: Directory base degli esperimenti
        filter_dict: Dizionario con filtri (es. {"round": 2})
        save_path: Percorso dove salvare il grafico
        show: Se mostrare il grafico

    Returns:
        DataFrame filtrato con gli esperimenti
    """
    # Carica il log degli esperimenti
    log_path = os.path.join(base_dir, "experiments", "experiments_log.csv")
    if not os.path.exists(log_path):
        print(f"Experiments log not found at {log_path}")
        return None

    df = pd.read_csv(log_path)

    # Verifica la presenza delle colonne di bias
    bias_columns = [col for col in df.columns if col.startswith('bias_')]
    if not bias_columns:
        print("No bias metrics found in experiments log. Make sure you've run experiments with the updated ExperimentManager")
        return df

    # Filtra gli esperimenti
    if filter_dict:
        for key, value in filter_dict.items():
            if key in df.columns:
                df = df[df[key] == value]

    if len(df) == 0:
        print("No experiments match the filter criteria")
        return None

    # Crea un grafico multi-pannello per confrontare diverse metriche di bias
    plt.figure(figsize=(15, 10))

    # 1. Bias Ratio
    plt.subplot(2, 2, 1)
    if 'bias_bias_ratio' in df.columns:
        # Usa barplot con colorazione condizionale
        colors = []
        for ratio in df['bias_bias_ratio']:
            if ratio < 0.9:
                colors.append('blue')  # Bias verso M
            elif ratio > 1.1:
                colors.append('red')   # Bias verso W
            else:
                colors.append('green') # Bilanciato

        bars = plt.bar(df['experiment_id'], df['bias_bias_ratio'], color=colors)

        # Aggiungi linee di riferimento
        plt.axhline(y=1.0, color='green', linestyle='--', alpha=0.7, label='Perfect Balance')
        plt.axhline(y=0.9, color='blue', linestyle=':', alpha=0.7, label='M Bias Threshold')
        plt.axhline(y=1.1, color='red', linestyle=':', alpha=0.7, label='W Bias Threshold')

        plt.title('Bias Ratio by Experiment')
        plt.ylabel('Bias Ratio (M→W : W→M)')
        plt.legend()
    else:
        plt.text(0.5, 0.5, 'No bias ratio data available', ha='center', va='center')

    plt.xticks(rotation=45, ha='right')

    # 2. Error Rates by Gender
    plt.subplot(2, 2, 2)
    if 'm_error_rate' in bias_columns and 'w_error_rate' in bias_columns:
        # Crea un DataFrame ristrutturato per seaborn
        error_df = pd.melt(df,
                           id_vars=['experiment_id'],
                           value_vars=['bias_m_error_rate', 'bias_w_error_rate'],
                           var_name='error_type', value_name='error_rate')

        # Rinomina i tipi di errore per la leggibilità
        error_df['error_type'] = error_df['error_type'].replace({
            'bias_m_error_rate': 'M→W Error',
            'bias_w_error_rate': 'W→M Error'
        })

        # Crea un grafico a barre raggruppate
        sns.barplot(x='experiment_id', y='error_rate', hue='error_type', data=error_df, palette=['red', 'blue'])
        plt.title('Error Rates by Gender')
        plt.ylabel('Error Rate')
        plt.legend(title='Error Type')
    else:
        plt.text(0.5, 0.5, 'No error rate data available', ha='center', va='center')

    plt.xticks(rotation=45, ha='right')

    # 3. Fairness Metrics
    plt.subplot(2, 2, 3)
    if 'bias_equality_of_opportunity' in df.columns and 'bias_predictive_equality' in df.columns:
        # Crea un DataFrame ristrutturato per seaborn
        fairness_df = pd.melt(df,
                              id_vars=['experiment_id'],
                              value_vars=['bias_equality_of_opportunity', 'bias_predictive_equality'],
                              var_name='fairness_metric', value_name='value')

        # Rinomina le metriche per la leggibilità
        fairness_df['fairness_metric'] = fairness_df['fairness_metric'].replace({
            'bias_equality_of_opportunity': 'Equality of Opportunity',
            'bias_predictive_equality': 'Predictive Equality'
        })

        # Colora le barre in base al valore assoluto (0 è perfetto)
        colors = []
        for val in fairness_df['value']:
            if abs(val) < 0.05:
                colors.append('green')  # Molto equo
            elif abs(val) < 0.1:
                colors.append('orange') # Moderatamente equo
            else:
                colors.append('red')    # Non equo

        # Crea il grafico
        sns.barplot(x='experiment_id', y='value', hue='fairness_metric', data=fairness_df)
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.title('Fairness Metrics (closer to 0 is better)')
        plt.ylabel('Disparity Value')
        plt.legend(title='Fairness Metric')
    else:
        plt.text(0.5, 0.5, 'No fairness metrics available', ha='center', va='center')

    plt.xticks(rotation=45, ha='right')

    # 4. F1 Scores by Gender
    plt.subplot(2, 2, 4)
    if 'bias_m_f1' in df.columns and 'bias_w_f1' in df.columns:
        # Crea un DataFrame ristrutturato per seaborn
        f1_df = pd.melt(df,
                        id_vars=['experiment_id'],
                        value_vars=['bias_m_f1', 'bias_w_f1', 'test_f1'],
                        var_name='f1_type', value_name='f1_score')

        # Rinomina i tipi di F1 per la leggibilità
        f1_df['f1_type'] = f1_df['f1_type'].replace({
            'bias_m_f1': 'Male F1',
            'bias_w_f1': 'Female F1',
            'test_f1': 'Overall F1'
        })

        # Crea un grafico a barre raggruppate
        sns.barplot(x='experiment_id', y='f1_score', hue='f1_type', data=f1_df,
                   palette=['blue', 'red', 'purple'])
        plt.title('F1 Scores by Gender')
        plt.ylabel('F1 Score')
        plt.legend(title='F1 Type')
    else:
        plt.text(0.5, 0.5, 'No F1 score data by gender available', ha='center', va='center')

    plt.xticks(rotation=45, ha='right')

    # Completa il grafico
    plt.suptitle('Gender Bias Analysis Across Experiments', fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        plt.savefig(save_path)
        print(f"Bias comparison plot saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return df

def compare_learning_curves(experiment_ids, base_dir=".", metrics=None, save_path=None, show=True):
    """
    Confronta le curve di apprendimento di diversi esperimenti.

    Args:
        experiment_ids: Lista di ID degli esperimenti
        base_dir: Directory base degli esperimenti
        metrics: Lista di metriche da visualizzare
        save_path: Percorso dove salvare il grafico
        show: Se mostrare il grafico
    """
    if not metrics:
        metrics = ['val_acc', 'val_loss', 'val_f1']

    # Crea una figura per ogni metrica
    for metric in metrics:
        plt.figure(figsize=(12, 6))

        for exp_id in experiment_ids:
            # Costruisci il percorso del file di training history
            history_path = os.path.join(base_dir, "experiments", exp_id, "logs", "train_history.csv")

            if os.path.exists(history_path):
                # Carica la storia di training
                history_df = pd.read_csv(history_path)

                if metric in history_df.columns:
                    # Plotta la metrica
                    plt.plot(history_df[metric], label=exp_id)
                else:
                    print(f"Metric '{metric}' not found in history for {exp_id}")
            else:
                print(f"History file not found for experiment {exp_id}")

        plt.title(f"Learning Curves - {metric}")
        plt.xlabel("Epoch")
        plt.ylabel(metric)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)

        if save_path:
            # Crea un nome file per ogni metrica
            metric_save_path = save_path.replace('.png', f'_{metric}.png')
            plt.savefig(metric_save_path)
            print(f"Plot saved to {metric_save_path}")

        if show:
            plt.show()
        else:
            plt.close()

def generate_bias_heatmap(base_dir=".", filter_dict=None, save_path=None, show=True):
    """
    Genera una heatmap che confronta diverse metriche di bias tra esperimenti.

    Args:
        base_dir: Directory base degli esperimenti
        filter_dict: Dizionario con filtri (es. {"round": 2})
        save_path: Percorso dove salvare il grafico
        show: Se mostrare il grafico
    """
    # Carica il log degli esperimenti
    log_path = os.path.join(base_dir, "experiments", "experiments_log.csv")
    if not os.path.exists(log_path):
        print(f"Experiments log not found at {log_path}")
        return None

    df = pd.read_csv(log_path)

    # Filtra gli esperimenti
    if filter_dict:
        for key, value in filter_dict.items():
            if key in df.columns:
                df = df[df[key] == value]

    if len(df) == 0:
        print("No experiments match the filter criteria")
        return None

    # Seleziona solo le colonne di bias e l'ID dell'esperimento
    bias_columns = [col for col in df.columns if col.startswith('bias_')]

    if not bias_columns:
        print("No bias metrics found in experiments log")
        return df

    # Rimuovi il prefisso 'bias_' per rendere le etichette più leggibili
    df_plot = df[['experiment_id'] + bias_columns].copy()
    df_plot.columns = [col.replace('bias_', '') if col != 'experiment_id' else col for col in df_plot.columns]

    # Imposta l'ID dell'esperimento come indice
    df_plot = df_plot.set_index('experiment_id')

    # Crea una heatmap
    plt.figure(figsize=(14, len(df) * 0.8))

    # Normalizza i dati per una migliore visualizzazione
    # Alcune metriche sono migliori quando vicine a 0, altre quando vicine a 1
    heatmap_data = df_plot.copy()

    # Per bias_ratio, la normalizzazione è rispetto a 1 (valore ideale)
    if 'bias_ratio' in heatmap_data.columns:
        heatmap_data['bias_ratio'] = heatmap_data['bias_ratio'].apply(lambda x: abs(1 - x))

    # Per metriche di disparità, 0 è il valore ideale
    for col in ['equality_of_opportunity', 'predictive_equality']:
        if col in heatmap_data.columns:
            heatmap_data[col] = heatmap_data[col].abs()

    # Crea la heatmap con i valori originali come annotazioni
    ax = sns.heatmap(heatmap_data, annot=df_plot, fmt='.3f', cmap='RdYlGn_r', linewidths=.5)

    plt.title('Gender Bias Metrics Across Experiments')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Bias heatmap saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return df

def compare_error_distributions(experiment_ids, base_dir=".", save_path=None, show=True):
    """
    Confronta le distribuzioni degli errori di genere tra gli esperimenti.

    Args:
        experiment_ids: Lista di ID degli esperimenti da confrontare
        base_dir: Directory base degli esperimenti
        save_path: Percorso dove salvare il grafico
        show: Se mostrare il grafico
    """
    plt.figure(figsize=(14, 10))

    # Dati per il grafico
    experiment_labels = []
    m_to_w_errors = []  # Errori M→W
    w_to_m_errors = []  # Errori W→M
    m_correct = []     # M correttamente classificati
    w_correct = []     # W correttamente classificati

    for exp_id in experiment_ids:
        # Costruisci il percorso al file delle metriche di bias
        bias_path = os.path.join(base_dir, "experiments", exp_id, "logs", "bias_metrics.json")

        if os.path.exists(bias_path):
            with open(bias_path, 'r') as f:
                bias_data = json.load(f)

            # Estrai i dati necessari
            tn = bias_data.get('true_negative', 0)   # M→M
            fp = bias_data.get('false_positive', 0)  # M→W
            fn = bias_data.get('false_negative', 0)  # W→M
            tp = bias_data.get('true_positive', 0)   # W→W

            # Aggiungi i dati alle liste
            experiment_labels.append(exp_id)
            m_to_w_errors.append(fp)
            w_to_m_errors.append(fn)
            m_correct.append(tn)
            w_correct.append(tp)
        else:
            print(f"Bias metrics file not found for experiment {exp_id}")

    if not experiment_labels:
        print("No valid experiments found with bias metrics")
        return

    # 1. Confronto dei conteggi raw degli errori
    plt.subplot(2, 2, 1)
    bar_width = 0.35
    index = np.arange(len(experiment_labels))

    plt.bar(index, m_to_w_errors, bar_width, label='M→W Errors', color='red', alpha=0.7)
    plt.bar(index + bar_width, w_to_m_errors, bar_width, label='W→M Errors', color='blue', alpha=0.7)

    plt.xlabel('Experiment')
    plt.ylabel('Number of Errors')
    plt.title('Raw Error Counts by Gender and Experiment')
    plt.xticks(index + bar_width/2, experiment_labels, rotation=45, ha='right')
    plt.legend()

    # 2. Tassi di errore (percentuale degli esempi classificati erroneamente)
    plt.subplot(2, 2, 2)

    # Calcola i tassi di errore
    m_error_rates = [m_err / (m_err + m_cor) * 100 if (m_err + m_cor) > 0 else 0
                     for m_err, m_cor in zip(m_to_w_errors, m_correct)]
    w_error_rates = [w_err / (w_err + w_cor) * 100 if (w_err + w_cor) > 0 else 0
                     for w_err, w_cor in zip(w_to_m_errors, w_correct)]

    plt.bar(index, m_error_rates, bar_width, label='M→W Error Rate (%)', color='red', alpha=0.7)
    plt.bar(index + bar_width, w_error_rates, bar_width, label='W→M Error Rate (%)', color='blue', alpha=0.7)

    plt.xlabel('Experiment')
    plt.ylabel('Error Rate (%)')
    plt.title('Error Rates by Gender and Experiment')
    plt.xticks(index + bar_width/2, experiment_labels, rotation=45, ha='right')
    plt.legend()

    # 3. Bias Ratio (M→W / W→M)
    plt.subplot(2, 2, 3)

    # Calcola i bias ratio
    bias_ratios = []
    for m_rate, w_rate in zip(m_error_rates, w_error_rates):
        if w_rate > 0:
            bias_ratios.append(m_rate / w_rate)
        else:
            bias_ratios.append(float('inf'))

    # Definisci i colori in base al valore
    colors = []
    for ratio in bias_ratios:
        if ratio < 0.9:
            colors.append('blue')      # Bias verso M
        elif ratio > 1.1:
            colors.append('red')       # Bias verso W
        else:
            colors.append('green')     # Equilibrato

    plt.bar(index, bias_ratios, color=colors)
    plt.axhline(y=1.0, color='green', linestyle='--', label='Perfect Balance')
    plt.axhline(y=0.9, color='blue', linestyle=':', label='M Bias Threshold')
    plt.axhline(y=1.1, color='red', linestyle=':', label='W Bias Threshold')

    plt.xlabel('Experiment')
    plt.ylabel('Bias Ratio (M→W : W→M)')
    plt.title('Gender Bias Ratio by Experiment')
    plt.xticks(index, experiment_labels, rotation=45, ha='right')
    plt.legend()

    # 4. Confronto delle accuratezze per genere
    plt.subplot(2, 2, 4)

    # Calcola le accuratezze per genere
    m_accuracies = [m_cor / (m_err + m_cor) * 100 if (m_err + m_cor) > 0 else 0
                   for m_err, m_cor in zip(m_to_w_errors, m_correct)]
    w_accuracies = [w_cor / (w_err + w_cor) * 100 if (w_err + w_cor) > 0 else 0
                   for w_err, w_cor in zip(w_to_m_errors, w_correct)]

    plt.bar(index, m_accuracies, bar_width, label='Male Accuracy (%)', color='blue', alpha=0.7)
    plt.bar(index + bar_width, w_accuracies, bar_width, label='Female Accuracy (%)', color='red', alpha=0.7)

    plt.xlabel('Experiment')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy by Gender and Experiment')
    plt.xticks(index + bar_width/2, experiment_labels, rotation=45, ha='right')
    plt.legend()

    plt.suptitle('Gender Bias Error Analysis Across Experiments', fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        plt.savefig(save_path)
        print(f"Error distribution plot saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

def generate_full_report(base_dir=".", output_path=None):
    """
    Genera un report HTML completo di tutti gli esperimenti.

    Args:
        base_dir: Directory base degli esperimenti
        output_path: Percorso del file HTML di output

    Returns:
        Percorso del file HTML generato
    """
    # Percorso del log degli esperimenti
    log_path = os.path.join(base_dir, "experiments", "experiments_log.csv")
    if not os.path.exists(log_path):
        print(f"Experiments log not found at {log_path}")
        return None

    # Carica il log degli esperimenti
    df = pd.read_csv(log_path)

    # Percorso del report
    if output_path is None:
        output_path = os.path.join(base_dir, "experiments", "full_report.html")

    # Verifica se ci sono metriche di bias
    has_bias_metrics = any(col.startswith('bias_') for col in df.columns)

    # Genera grafici comparativi per il report
    plots_dir = os.path.join(base_dir, "experiments", "plots")
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    # Plot standard di accuratezza e F1
    comparison_plot_path = os.path.join(plots_dir, "accuracy_comparison.png")
    compare_experiments(base_dir, metric='test_accuracy', save_path=comparison_plot_path, show=False)

    f1_plot_path = os.path.join(plots_dir, "f1_comparison.png")
    compare_experiments(base_dir, metric='test_f1', save_path=f1_plot_path, show=False)

    # Plot di bias se disponibili
    bias_plot_path = None
    bias_heatmap_path = None
    if has_bias_metrics:
        bias_plot_path = os.path.join(plots_dir, "bias_comparison.png")
        compare_bias_metrics(base_dir, save_path=bias_plot_path, show=False)

        bias_heatmap_path = os.path.join(plots_dir, "bias_heatmap.png")
        generate_bias_heatmap(base_dir, save_path=bias_heatmap_path, show=False)

    # Calcola statistiche complessive
    stats = {}

    # Statistiche per round
    if 'round' in df.columns:
        stats['rounds'] = df['round'].value_counts().to_dict()

    # Statistiche per le metriche
    for col in df.columns:
        if (col.startswith('test_') or col.startswith('bias_')) and df[col].dtype != 'object':
            if pd.notna(df[col]).any():  # Verifica che ci siano valori non NaN
                best_idx = df[col].idxmax()
                best_exp = df.loc[best_idx, 'experiment_id']
                stats[f'best_{col}'] = {
                    'value': df.loc[best_idx, col],
                    'experiment': best_exp
                }

    # Prepara il report HTML
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Experiments Full Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2, h3 {{ color: #333; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ padding: 8px; text-align: left; border: 1px solid #ddd; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .metrics {{ color: #0066cc; font-weight: bold; }}
            .bias-balanced {{ color: green; font-weight: bold; }}
            .bias-male {{ color: blue; font-weight: bold; }}
            .bias-female {{ color: red; font-weight: bold; }}
            .experiment-card {{ border: 1px solid #ddd; padding: 15px; margin-bottom: 20px; border-radius: 5px; }}
            .parameter {{ font-family: monospace; }}
            .container {{ display: flex; flex-wrap: wrap; }}
            .chart {{ margin: 10px; flex: 1; min-width: 300px; }}
            img {{ max-width: 100%; height: auto; }}
            .summary {{ background-color: #f9f9f9; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
            .highlight {{ background-color: #fff3cd; }}
            .tabs {{ display: flex; margin-bottom: 15px; }}
            .tab {{ padding: 10px 15px; margin-right: 5px; border: 1px solid #ddd; border-radius: 5px 5px 0 0; cursor: pointer; }}
            .tab.active {{ background-color: #f0f0f0; border-bottom: none; }}
            .tab-content {{ display: none; padding: 15px; border: 1px solid #ddd; }}
            .tab-content.active {{ display: block; }}
        </style>
        <script>
            function openTab(evt, tabName) {{
                var i, tabcontent, tablinks;
                tabcontent = document.getElementsByClassName("tab-content");
                for (i = 0; i < tabcontent.length; i++) {{
                    tabcontent[i].style.display = "none";
                }}
                tablinks = document.getElementsByClassName("tab");
                for (i = 0; i < tablinks.length; i++) {{
                    tablinks[i].className = tablinks[i].className.replace(" active", "");
                }}
                document.getElementById(tabName).style.display = "block";
                evt.currentTarget.className += " active";
            }}

            window.onload = function() {{
                document.getElementsByClassName("tab")[0].click();
            }};
        </script>
    </head>
    <body>
        <h1>Gender Prediction Model - Complete Experiments Report</h1>
        <p>Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>

        <div class="summary">
            <h2>Summary Statistics</h2>
            <p><strong>Total experiments:</strong> {len(df)}</p>
    """

    # Aggiungi statistiche per round
    if 'rounds' in stats:
        html += "<p><strong>Experiments by round:</strong></p><ul>"
        for round_num, count in stats['rounds'].items():
            html += f"<li>Round {round_num}: {count} experiments</li>"
        html += "</ul>"

    # Aggiungi informazioni sulle migliori performance
    html += "<p><strong>Best performing experiments:</strong></p><ul>"
    accuracy_metrics = [metric for metric in stats.keys() if metric.startswith('best_test_')]
    for metric in accuracy_metrics:
        clean_metric = metric.replace('best_test_', '')
        html += f"<li>{clean_metric}: {stats[metric]['value']:.4f} (Experiment: {stats[metric]['experiment']})</li>"
    html += "</ul>"

    # Aggiungi informazioni sul bias se disponibili
    if has_bias_metrics:
        html += "<p><strong>Bias metrics highlights:</strong></p><ul>"
        bias_metrics = [metric for metric in stats.keys() if metric.startswith('best_bias_')]
        for metric in bias_metrics:
            clean_metric = metric.replace('best_bias_', '')
            html += f"<li>{clean_metric}: {stats[metric]['value']:.4f} (Experiment: {stats[metric]['experiment']})</li>"
        html += "</ul>"

    html += """
        </div>

        <div class="tabs">
            <button class="tab" onclick="openTab(event, 'AllExperiments')">All Experiments</button>
            <button class="tab" onclick="openTab(event, 'PerformanceComparison')">Performance Comparison</button>
    """

    if has_bias_metrics:
        html += """
            <button class="tab" onclick="openTab(event, 'BiasAnalysis')">Bias Analysis</button>
        """

    html += """
        </div>

        <div id="AllExperiments" class="tab-content">
            <h2>All Experiments</h2>
            <table>
                <tr>
                    <th>ID</th>
                    <th>Round</th>
                    <th>Architecture</th>
                    <th>Loss Function</th>
                    <th>Test Accuracy</th>
                    <th>Test F1</th>
    """

    if has_bias_metrics:
        html += """
                    <th>Bias Ratio</th>
                    <th>Bias Direction</th>
        """

    html += """
                    <th>Actions</th>
                </tr>
    """

    # Trova i migliori esperimenti per ogni metrica
    best_exps = {}
    for col in df.columns:
        if (col.startswith('test_') or col.startswith('bias_')) and df[col].dtype != 'object':
            if pd.notna(df[col]).any():  # Verifica che ci siano valori non NaN
                best_exps[col] = df.loc[df[col].idxmax(), 'experiment_id']

    # Aggiungi una riga per ogni esperimento
    for _, row in df.iterrows():
        exp_id = row['experiment_id']

        # Determina l'architettura
        if 'n_layers' in row and 'hidden_size' in row:
            layers = row['n_layers']
            hidden = row['hidden_size']
            dual = "dual" if 'dual_input' in row and row['dual_input'] else "single"
            arch = f"{layers}L-{hidden}H-{dual}"
        else:
            arch = "Basic"

        # Determina la loss function
        if 'loss' in row:
            loss_fn = row['loss']
            if loss_fn == 'focal' and 'alpha' in row and 'gamma' in row:
                loss_fn = f"Focal(α={row['alpha']}, γ={row['gamma']})"
            elif loss_fn == 'bce' and 'pos_weight' in row and row['pos_weight'] != 1.0:
                loss_fn = f"BCE(pos_weight={row['pos_weight']})"
        else:
            loss_fn = "BCE"

        # Ottieni le metriche principali
        acc = row.get('test_accuracy', "-")
        f1 = row.get('test_f1', "-")

        if isinstance(acc, float):
            acc_str = f"{acc:.4f}"
            # Aggiungi classe highlight se è il migliore
            acc_highlight = ' class="highlight"' if best_exps.get('test_accuracy') == exp_id else ''
        else:
            acc_str = acc
            acc_highlight = ''

        if isinstance(f1, float):
            f1_str = f"{f1:.4f}"
            # Aggiungi classe highlight se è il migliore
            f1_highlight = ' class="highlight"' if best_exps.get('test_f1') == exp_id else ''
        else:
            f1_str = f1
            f1_highlight = ''

        # Bias metrics se disponibili
        bias_ratio = row.get('bias_bias_ratio', "-")
        bias_direction = row.get('bias_direction', "-")

        if isinstance(bias_ratio, float):
            bias_str = f"{bias_ratio:.4f}"

            # Determina la classe CSS per il bias
            if bias_ratio < 0.9:
                bias_class = "bias-male"
            elif bias_ratio > 1.1:
                bias_class = "bias-female"
            else:
                bias_class = "bias-balanced"
        else:
            bias_str = bias_ratio
            bias_class = ""

        # Percorso al report individuale
        exp_report = os.path.join("experiments", exp_id, "report.html")
        exp_dir = os.path.join("experiments", exp_id)

        html += f"""
            <tr>
                <td>{exp_id}</td>
                <td>{row.get('round', '-')}</td>
                <td>{arch}</td>
                <td>{loss_fn}</td>
                <td{acc_highlight} class="metrics">{acc_str}</td>
                <td{f1_highlight} class="metrics">{f1_str}</td>
        """

        if has_bias_metrics:
            html += f"""
                <td class="{bias_class}">{bias_str}</td>
                <td class="{bias_class}">{bias_direction}</td>
            """

        html += f"""
                <td>
                    <a href="{exp_dir}" target="_blank">View Files</a>
                    {f'| <a href="{exp_report}" target="_blank">Report</a>' if os.path.exists(os.path.join(base_dir, exp_report)) else ''}
                </td>
            </tr>
        """

    html += """
            </table>
        </div>

        <div id="PerformanceComparison" class="tab-content">
            <h2>Performance Comparison</h2>
            <div class="container">
    """

    # Aggiungi grafici comparativi di performance
    html += f"""
                <div class="chart">
                    <h3>Test Accuracy Comparison</h3>
                    <img src="{os.path.relpath(comparison_plot_path, base_dir)}" alt="Accuracy Comparison">
                </div>

                <div class="chart">
                    <h3>F1 Score Comparison</h3>
                    <img src="{os.path.relpath(f1_plot_path, base_dir)}" alt="F1 Score Comparison">
                </div>
    """

    html += """
            </div>
        </div>
    """

    # Aggiungi tab di analisi del bias se disponibile
    if has_bias_metrics and bias_plot_path and bias_heatmap_path:
        html += f"""
        <div id="BiasAnalysis" class="tab-content">
            <h2>Bias Analysis</h2>

            <div class="container">
                <div class="chart">
                    <h3>Gender Bias Comparison</h3>
                    <img src="{os.path.relpath(bias_plot_path, base_dir)}" alt="Bias Comparison">
                </div>

                <div class="chart">
                    <h3>Bias Metrics Heatmap</h3>
                    <img src="{os.path.relpath(bias_heatmap_path, base_dir)}" alt="Bias Heatmap">
                </div>
            </div>

            <h3>Understanding Gender Bias Metrics</h3>
            <p>
                <strong>Bias Ratio:</strong> Rapporto tra il tasso di errore M→W e il tasso di errore W→M.
                Un valore di 1.0 indica un bias perfettamente bilanciato, mentre valori > 1.1 indicano
                un bias verso W (il modello erroneamente classifica gli uomini come donne con più frequenza),
                e valori < 0.9 indicano un bias verso M (il modello erroneamente classifica le donne come
                uomini con più frequenza).
            </p>
            <p>
                <strong>Equality of Opportunity:</strong> Misura la differenza nei tassi di recall tra i due generi.
                Un valore vicino a 0 indica che il modello ha simile capacità di riconoscere correttamente
                entrambi i generi.
            </p>
            <p>
                <strong>Predictive Equality:</strong> Misura la differenza nei tassi di errore tra i due generi.
                Un valore vicino a 0 indica che il modello commette errori a tassi simili per entrambi i generi.
            </p>
        </div>
        """

    html += """
    </body>
    </html>
    """

    # Salva il report
    with open(output_path, 'w') as f:
        f.write(html)

    print(f"Full report generated at {output_path}")
    return output_path

def generate_individual_reports(base_dir="."):
    """
    Genera report individuali per tutti gli esperimenti.

    Args:
        base_dir: Directory base degli esperimenti

    Returns:
        Lista dei percorsi dei report generati
    """
    from experiment_manager import ExperimentManager

    # Carica il log degli esperimenti
    log_path = os.path.join(base_dir, "experiments", "experiments_log.csv")
    if not os.path.exists(log_path):
        print(f"Experiments log not found at {log_path}")
        return []

    df = pd.read_csv(log_path)

    reports = []
    for exp_id in df['experiment_id']:
        try:
            # Carica l'esperimento
            experiment = ExperimentManager.from_experiment_id(exp_id, base_dir)

            # Genera il report
            report_path = experiment.generate_report()
            reports.append(report_path)

            print(f"Generated report for experiment {exp_id}")
        except Exception as e:
            print(f"Error generating report for experiment {exp_id}: {e}")

    return reports

def main():
    """Funzione principale."""
    parser = argparse.ArgumentParser(description="Experiment analysis and reporting tools")

    # Sottocomandi
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')

    # Comando 'list'
    list_parser = subparsers.add_parser('list', help='List all experiments')
    list_parser.add_argument('--base_dir', default='.', help='Base directory for experiments')
    list_parser.add_argument('--detail', action='store_true', help='Show detailed information')

    # Comando 'compare'
    compare_parser = subparsers.add_parser('compare', help='Compare experiments')
    compare_parser.add_argument('--base_dir', default='.', help='Base directory for experiments')
    compare_parser.add_argument('--metric', default='test_accuracy', help='Metric to compare')
    compare_parser.add_argument('--group_by', help='Group experiments by parameter')
    compare_parser.add_argument('--round', type=int, help='Filter by round')
    compare_parser.add_argument('--output', help='Output file for the plot')

    # Comando 'bias'
    bias_parser = subparsers.add_parser('bias', help='Compare gender bias metrics')
    bias_parser.add_argument('--base_dir', default='.', help='Base directory for experiments')
    bias_parser.add_argument('--round', type=int, help='Filter by round')
    bias_parser.add_argument('--output', help='Output file for the plot')
    bias_parser.add_argument('--heatmap', action='store_true', help='Generate bias heatmap')
    bias_parser.add_argument('--experiments', nargs='+', help='Specific experiment IDs to compare error distributions')

    # Comando 'curves'
    curves_parser = subparsers.add_parser('curves', help='Compare learning curves')
    curves_parser.add_argument('--base_dir', default='.', help='Base directory for experiments')
    curves_parser.add_argument('--experiments', nargs='+', required=True, help='Experiment IDs to compare')
    curves_parser.add_argument('--metrics', nargs='+', default=['val_acc', 'val_loss', 'val_f1'],
                             help='Metrics to visualize')
    curves_parser.add_argument('--output', help='Output file for the plot')

    # Comando 'report'
    report_parser = subparsers.add_parser('report', help='Generate reports')
    report_parser.add_argument('--base_dir', default='.', help='Base directory for experiments')
    report_parser.add_argument('--full', action='store_true', help='Generate full report with all experiments')
    report_parser.add_argument('--individual', action='store_true', help='Generate individual reports for each experiment')
    report_parser.add_argument('--output', help='Output file for the report')

    args = parser.parse_args()

    # Esegui il comando richiesto
    if args.command == 'list':
        df = pd.read_csv(os.path.join(args.base_dir, "experiments", "experiments_log.csv"))
        if args.detail:
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', 1000)
        print(df)

    elif args.command == 'compare':
        filter_dict = {}
        if args.round is not None:
            filter_dict['round'] = args.round

        df = compare_experiments(
            base_dir=args.base_dir,
            filter_dict=filter_dict,
            metric=args.metric,
            group_by=args.group_by,
            save_path=args.output
        )

    elif args.command == 'bias':
        filter_dict = {}
        if args.round is not None:
            filter_dict['round'] = args.round

        if args.heatmap:
            generate_bias_heatmap(
                base_dir=args.base_dir,
                filter_dict=filter_dict,
                save_path=args.output
            )
        elif args.experiments:
            compare_error_distributions(
                experiment_ids=args.experiments,
                base_dir=args.base_dir,
                save_path=args.output
            )
        else:
            compare_bias_metrics(
                base_dir=args.base_dir,
                filter_dict=filter_dict,
                save_path=args.output
            )

    elif args.command == 'curves':
        compare_learning_curves(
            experiment_ids=args.experiments,
            base_dir=args.base_dir,
            metrics=args.metrics,
            save_path=args.output
        )

    elif args.command == 'report':
        if args.full:
            generate_full_report(
                base_dir=args.base_dir,
                output_path=args.output
            )

        if args.individual:
            generate_individual_reports(args.base_dir)

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
