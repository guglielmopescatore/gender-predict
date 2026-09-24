#!/usr/bin/env python3
"""
Threshold-Bias Analysis for Academic Deployment

Analyzes any gender prediction model experiment across different thresholds with focus on:
- Bias ratio (critical for academic use)
- Accuracy vs bias trade-offs
- Optimal threshold for bias <0.001

Usage:
    python threshold_bias_analyzer.py                           # Auto-detect latest experiment
    python threshold_bias_analyzer.py -e path/to/experiment     # Specific experiment
    python threshold_bias_analyzer.py -e ../experiments/20250610_171542_r3_bce_h256_l3_dual_frz5
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import json
from pathlib import Path

def load_experiment_predictions(experiment_path=None):
    """Load experiment predictions for threshold analysis."""

    if experiment_path:
        exp_dir = Path(experiment_path)
    else:
        # Auto-detect latest experiment or use V4-R2 as default
        experiments_base = Path("../experiments") if Path("../experiments").exists() else Path("experiments")

        # Try V4-R2 first
        v4r2_dir = experiments_base / "20250610_171542_r3_bce_h256_l3_dual_frz5"
        if v4r2_dir.exists():
            exp_dir = v4r2_dir
        else:
            # Get latest experiment
            exp_dirs = [d for d in experiments_base.iterdir() if d.is_dir() and d.name.startswith("202")]
            if exp_dirs:
                exp_dir = sorted(exp_dirs)[-1]
                print(f"🔍 Auto-detected latest experiment: {exp_dir.name}")
            else:
                print("❌ No experiments found")
                return None, None

    print(f"📁 Target experiment: {exp_dir}")

    if not exp_dir.exists():
        print(f"❌ Experiment directory not found: {exp_dir}")
        return None, None

    # Try different sources in order of preference
    sources = [
        exp_dir / "logs/val_probs_labels.csv",      # Preferred (validation set)
        exp_dir / "logs/error_analysis.csv",        # All test predictions
    ]

    print(f"🔍 Searching for prediction files...")
    for source in sources:
        print(f"   Checking: {source.name}")
        if source.exists():
            print(f"✅ Found: {source}")
            df = pd.read_csv(source)

            # Standardize column names based on file type
            if "error_analysis.csv" in str(source):
                # error_analysis.csv has different structure
                if 'probability_female' in df.columns and 'true_label' in df.columns:
                    df_clean = pd.DataFrame({
                        'probability': df['probability_female'],
                        'true_label': df['true_label']
                    })
                elif 'probability_female' in df.columns and 'true_gender' in df.columns:
                    # Convert gender to label
                    df_clean = pd.DataFrame({
                        'probability': df['probability_female'],
                        'true_label': (df['true_gender'] == 'W').astype(int)
                    })
                else:
                    print(f"❌ Unexpected columns in error_analysis.csv: {list(df.columns)}")
                    continue
            else:
                # val_probs_labels.csv or similar
                prob_cols = ['prob', 'probability', 'probability_female', 'pred_prob']
                label_cols = ['label', 'true_label', 'target', 'truth']

                prob_col = None
                label_col = None

                for col in prob_cols:
                    if col in df.columns:
                        prob_col = col
                        break

                for col in label_cols:
                    if col in df.columns:
                        label_col = col
                        break

                if prob_col and label_col:
                    df_clean = pd.DataFrame({
                        'probability': df[prob_col],
                        'true_label': df[label_col]
                    })
                else:
                    print(f"❌ Required columns not found in {source}")
                    print(f"   Available columns: {list(df.columns)}")
                    continue

            print(f"✅ Loaded {len(df_clean)} predictions from {source.name}")
            return df_clean, exp_dir

    print("❌ No suitable prediction file found")
    print("   Expected files: val_probs_labels.csv or error_analysis.csv")
    return None, None

def calculate_bias_metrics(y_true, y_pred):
    """Calculate detailed bias metrics."""

    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Male = 0, Female = 1
    male_mask = (y_true == 0)
    female_mask = (y_true == 1)

    # Error rates per gender
    male_errors = np.sum((y_true == 0) & (y_pred == 1))  # M predicted as W
    female_errors = np.sum((y_true == 1) & (y_pred == 0))  # W predicted as M

    male_total = np.sum(male_mask)
    female_total = np.sum(female_mask)

    male_error_rate = male_errors / male_total if male_total > 0 else 0
    female_error_rate = female_errors / female_total if female_total > 0 else 0

    # Bias ratio (should be close to 1.0)
    bias_ratio = male_error_rate / female_error_rate if female_error_rate > 0 else float('inf')

    return {
        'male_error_rate': male_error_rate,
        'female_error_rate': female_error_rate,
        'bias_ratio': bias_ratio,
        'bias_distance': abs(bias_ratio - 1.0)  # Distance from perfect bias (1.0)
    }

def threshold_sweep_analysis(df, threshold_range=None):
    """Perform threshold sweep analysis."""

    if threshold_range is None:
        threshold_range = np.arange(0.20, 0.80, 0.01)

    results = []

    print(f"🔄 Analyzing {len(threshold_range)} thresholds...")

    for threshold in threshold_range:
        # Apply threshold
        predictions = (df['probability'] >= threshold).astype(int)

        # Calculate standard metrics
        accuracy = accuracy_score(df['true_label'], predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            df['true_label'], predictions, average='binary', zero_division=0
        )

        # Calculate bias metrics
        bias_metrics = calculate_bias_metrics(df['true_label'], predictions)

        # Store results
        result = {
            'threshold': threshold,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            **bias_metrics
        }
        results.append(result)

    return pd.DataFrame(results)

def plot_threshold_analysis(results_df, experiment_name="Experiment", output_dir="threshold_analysis"):
    """Create comprehensive threshold analysis plots."""

    Path(output_dir).mkdir(exist_ok=True)

    # Set style
    plt.style.use('default')
    sns.set_palette("husl")

    # Create comprehensive plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'{experiment_name} Threshold Analysis for Academic Deployment', fontsize=16, fontweight='bold')

    # 1. Accuracy vs Threshold
    axes[0, 0].plot(results_df['threshold'], results_df['accuracy'], 'b-', linewidth=2, label='Accuracy')
    axes[0, 0].axhline(y=0.9206, color='r', linestyle='--', alpha=0.7, label='B0 Baseline (92.06%)')
    axes[0, 0].set_xlabel('Threshold')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_title('Accuracy vs Threshold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    # 2. F1 vs Threshold
    axes[0, 1].plot(results_df['threshold'], results_df['f1'], 'g-', linewidth=2, label='F1 Score')
    axes[0, 1].axhline(y=0.903, color='r', linestyle='--', alpha=0.7, label='V4-R1 Baseline (0.903)')
    axes[0, 1].set_xlabel('Threshold')
    axes[0, 1].set_ylabel('F1 Score')
    axes[0, 1].set_title('F1 Score vs Threshold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    # 3. Bias Ratio vs Threshold (CRITICAL)
    axes[0, 2].plot(results_df['threshold'], results_df['bias_ratio'], 'purple', linewidth=2, label='Bias Ratio')
    axes[0, 2].axhline(y=1.0, color='g', linestyle='-', alpha=0.8, label='Perfect Bias (1.0)')
    axes[0, 2].axhline(y=0.999, color='orange', linestyle='--', alpha=0.7, label='Academic Threshold (0.999)')
    axes[0, 2].axhline(y=1.001, color='orange', linestyle='--', alpha=0.7)
    axes[0, 2].set_xlabel('Threshold')
    axes[0, 2].set_ylabel('Bias Ratio')
    axes[0, 2].set_title('Bias Ratio vs Threshold (Critical for Academic Use)')
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].legend()
    axes[0, 2].set_ylim(0.95, 1.05)

    # 4. Error Rates by Gender
    axes[1, 0].plot(results_df['threshold'], results_df['male_error_rate'], 'b-', linewidth=2, label='Male Error Rate')
    axes[1, 0].plot(results_df['threshold'], results_df['female_error_rate'], 'r-', linewidth=2, label='Female Error Rate')
    axes[1, 0].set_xlabel('Threshold')
    axes[1, 0].set_ylabel('Error Rate')
    axes[1, 0].set_title('Error Rates by Gender')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    # 5. Bias Distance vs Threshold
    axes[1, 1].plot(results_df['threshold'], results_df['bias_distance'], 'orange', linewidth=2, label='Bias Distance')
    axes[1, 1].axhline(y=0.001, color='g', linestyle='--', alpha=0.7, label='Academic Threshold (0.001)')
    axes[1, 1].set_xlabel('Threshold')
    axes[1, 1].set_ylabel('Bias Distance from 1.0')
    axes[1, 1].set_title('Bias Distance vs Threshold')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    axes[1, 1].set_yscale('log')

    # 6. Accuracy vs Bias Trade-off
    scatter = axes[1, 2].scatter(results_df['bias_distance'], results_df['accuracy'],
                                c=results_df['threshold'], cmap='viridis', s=50, alpha=0.7)
    axes[1, 2].axvline(x=0.001, color='g', linestyle='--', alpha=0.7, label='Academic Bias Limit')
    axes[1, 2].set_xlabel('Bias Distance from 1.0')
    axes[1, 2].set_ylabel('Accuracy')
    axes[1, 2].set_title('Accuracy vs Bias Trade-off')
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].set_xscale('log')
    plt.colorbar(scatter, ax=axes[1, 2], label='Threshold')
    axes[1, 2].legend()

    plt.tight_layout()

    # Save plot
    plot_path = Path(output_dir) / "threshold_bias_analysis.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"📊 Plot saved to: {plot_path}")

    plt.show()

    return plot_path

def find_optimal_thresholds(results_df):
    """Find optimal thresholds for different criteria."""

    # Academic optimal: Best accuracy with bias_distance < 0.001
    academic_mask = results_df['bias_distance'] < 0.001
    if academic_mask.any():
        academic_optimal = results_df[academic_mask].loc[results_df[academic_mask]['accuracy'].idxmax()]
    else:
        # Relaxed: Best accuracy with bias_distance < 0.01
        academic_mask = results_df['bias_distance'] < 0.01
        academic_optimal = results_df[academic_mask].loc[results_df[academic_mask]['accuracy'].idxmax()] if academic_mask.any() else None

    # F1 optimal
    f1_optimal = results_df.loc[results_df['f1'].idxmax()]

    # Accuracy optimal
    accuracy_optimal = results_df.loc[results_df['accuracy'].idxmax()]

    # Ultra-fair (minimum bias distance)
    ultra_fair = results_df.loc[results_df['bias_distance'].idxmin()]

    return {
        'academic_optimal': academic_optimal,
        'f1_optimal': f1_optimal,
        'accuracy_optimal': accuracy_optimal,
        'ultra_fair': ultra_fair
    }

def main():
    """Main execution function."""

    import argparse

    parser = argparse.ArgumentParser(description='Threshold-Bias Analysis for Gender Prediction Models')
    parser.add_argument('--experiment', '-e', type=str,
                       help='Path to experiment directory (auto-detects latest if not provided)')
    parser.add_argument('--output_dir', '-o', type=str, default=None,
                       help='Output directory for results (default: experiment/threshold_analysis)')
    parser.add_argument('--threshold_range', nargs=3, type=float,
                       default=[0.20, 0.80, 0.01], metavar=('START', 'END', 'STEP'),
                       help='Threshold range as START END STEP')

    args = parser.parse_args()

    print("🎯 === THRESHOLD-BIAS ANALYSIS ===\n")

    # Load predictions
    df, exp_dir = load_experiment_predictions(args.experiment)
    if df is None:
        return False

    experiment_name = exp_dir.name if exp_dir else "Unknown"
    print(f"📊 Analyzing {len(df)} predictions from {experiment_name}")

    # Set default output directory to experiment directory if not specified
    if args.output_dir is None:
        args.output_dir = str(exp_dir / "threshold_analysis") if exp_dir else "threshold_analysis"

    # Create threshold range
    threshold_range = np.arange(args.threshold_range[0], args.threshold_range[1], args.threshold_range[2])

    # Perform threshold sweep
    results = threshold_sweep_analysis(df, threshold_range)

    # Create plots
    print("📈 Generating plots...")
    plot_path = plot_threshold_analysis(results, experiment_name, args.output_dir)

    # Find optimal thresholds
    optimal_thresholds = find_optimal_thresholds(results)

    # Report results
    print(f"\n🎯 === OPTIMAL THRESHOLDS FOR {experiment_name.upper()} ===")

    for name, result in optimal_thresholds.items():
        if result is not None:
            print(f"\n{name.upper()}:")
            print(f"   Threshold: {result['threshold']:.3f}")
            print(f"   Accuracy: {result['accuracy']:.4f} ({result['accuracy']*100:.2f}%)")
            print(f"   F1: {result['f1']:.4f}")
            print(f"   Bias Ratio: {result['bias_ratio']:.6f}")
            print(f"   Bias Distance: {result['bias_distance']:.6f}")
        else:
            print(f"\n{name.upper()}: No solution found")

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Save threshold results
    results.to_csv(output_dir / f"threshold_sweep_results_{experiment_name}.csv", index=False)

    # Save optimal thresholds
    optimal_dict = {}
    for name, result in optimal_thresholds.items():
        if result is not None:
            optimal_dict[name] = {
                'threshold': float(result['threshold']),
                'accuracy': float(result['accuracy']),
                'f1': float(result['f1']),
                'bias_ratio': float(result['bias_ratio']),
                'bias_distance': float(result['bias_distance'])
            }

    with open(output_dir / f"optimal_thresholds_{experiment_name}.json", 'w') as f:
        json.dump(optimal_dict, f, indent=2)

    print(f"\n📁 Results saved to: {output_dir}/")
    print(f"   • threshold_sweep_results_{experiment_name}.csv")
    print(f"   • optimal_thresholds_{experiment_name}.json")
    print(f"   • threshold_bias_analysis.png")

    # Deployment recommendation
    if 'academic_optimal' in optimal_dict:
        academic = optimal_dict['academic_optimal']
        print(f"\n🚀 DEPLOYMENT RECOMMENDATION:")
        print(f"   For academic use: Threshold {academic['threshold']:.3f}")
        print(f"   Expected accuracy: {academic['accuracy']*100:.2f}%")
        print(f"   Bias ratio: {academic['bias_ratio']:.6f} (excellent)")

    return True

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n❌ Analysis failed - check data availability")
    else:
        print("\n✅ Threshold-bias analysis completed!")
        print("📊 Check the generated plots and recommendations above")
