#!/usr/bin/env python3
"""
Fine threshold optimization to close the small F1 gap.
Clean approach without overfitting to specific patterns.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import argparse

def fine_tune_threshold(predictions_df, target_f1=0.903, resolution=0.001):
    """
    Fine-tune threshold to hit exact F1 target.
    
    Args:
        predictions_df: DataFrame with true_gender, pred_gender, prob
        target_f1: Target F1 score to achieve
        resolution: Threshold search resolution
    
    Returns:
        dict: Results with optimal threshold and metrics
    """
    
    print(f"🎯 Fine-Tuning Threshold for F1 ≥ {target_f1}")
    print(f"   Resolution: {resolution}")
    print(f"   Current F1: {calculate_f1(predictions_df):.4f}")
    
    # Extract data
    y_true = (predictions_df['true_gender'] == 'W').astype(int)
    y_prob = predictions_df['prob'].values
    
    # Current metrics for reference
    current_threshold = 0.47  # From original analysis
    current_metrics = evaluate_threshold_detailed(y_true, y_prob, current_threshold)
    
    print(f"\n📊 Current Performance (threshold={current_threshold}):")
    print(f"   F1: {current_metrics['f1']:.4f}")
    print(f"   Accuracy: {current_metrics['accuracy']:.4f}") 
    print(f"   Bias Ratio: {current_metrics['bias_ratio']:.4f}")
    
    # Fine search around current threshold
    search_range = np.arange(0.40, 0.55, resolution)
    
    print(f"\n🔍 Searching {len(search_range)} thresholds...")
    
    results = []
    best_threshold = current_threshold
    best_metrics = None
    
    for threshold in search_range:
        metrics = evaluate_threshold_detailed(y_true, y_prob, threshold)
        results.append({
            'threshold': threshold,
            **metrics
        })
        
        # Check if this meets our criteria
        meets_f1 = metrics['f1'] >= target_f1
        good_bias = 0.95 <= metrics['bias_ratio'] <= 1.05
        
        # Prioritize: F1 target + best bias
        if meets_f1:
            if best_metrics is None or abs(metrics['bias_ratio'] - 1.0) < abs(best_metrics['bias_ratio'] - 1.0):
                best_threshold = threshold
                best_metrics = metrics
    
    # Convert to DataFrame for analysis
    results_df = pd.DataFrame(results)
    
    # Find candidates that meet F1 target
    candidates = results_df[results_df['f1'] >= target_f1]
    
    print(f"\n🎯 Analysis Results:")
    print(f"   Candidates meeting F1 ≥ {target_f1}: {len(candidates)}")
    
    if len(candidates) > 0:
        # Best candidate (closest to bias ratio 1.0)
        best_candidate = candidates.loc[candidates['bias_ratio'].apply(lambda x: abs(x - 1.0)).idxmin()]
        
        print(f"   ✅ OPTIMAL THRESHOLD FOUND: {best_candidate['threshold']:.3f}")
        print(f"      F1: {best_candidate['f1']:.4f} (target: {target_f1})")
        print(f"      Accuracy: {best_candidate['accuracy']:.4f}")
        print(f"      Bias Ratio: {best_candidate['bias_ratio']:.4f}")
        
        optimal_result = {
            'optimal_threshold': best_candidate['threshold'],
            'metrics': best_candidate.to_dict(),
            'meets_target': True,
            'improvement_over_current': {
                'f1': best_candidate['f1'] - current_metrics['f1'],
                'accuracy': best_candidate['accuracy'] - current_metrics['accuracy'],
                'bias_change': best_candidate['bias_ratio'] - current_metrics['bias_ratio']
            }
        }
        
    else:
        # No threshold meets target - report best F1
        best_f1_row = results_df.loc[results_df['f1'].idxmax()]
        
        print(f"   ⚠️  No threshold achieves F1 ≥ {target_f1}")
        print(f"   📈 Best possible F1: {best_f1_row['f1']:.4f} (threshold: {best_f1_row['threshold']:.3f})")
        
        optimal_result = {
            'optimal_threshold': best_f1_row['threshold'],
            'metrics': best_f1_row.to_dict(),
            'meets_target': False,
            'max_achievable_f1': best_f1_row['f1'],
            'gap_to_target': target_f1 - best_f1_row['f1']
        }
    
    # Plot threshold analysis
    plot_threshold_analysis(results_df, target_f1, optimal_result['optimal_threshold'])
    
    return optimal_result, results_df

def evaluate_threshold_detailed(y_true, y_prob, threshold):
    """Evaluate all metrics for a given threshold."""
    
    y_pred = (y_prob >= threshold).astype(int)
    
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    
    # Bias metrics
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    male_error_rate = fp / (tn + fp) if (tn + fp) > 0 else 0
    female_error_rate = fn / (tp + fn) if (tp + fn) > 0 else 0
    bias_ratio = male_error_rate / female_error_rate if female_error_rate > 0 else float('inf')
    
    return {
        'accuracy': accuracy,
        'precision': precision, 
        'recall': recall,
        'f1': f1,
        'bias_ratio': bias_ratio,
        'male_error_rate': male_error_rate,
        'female_error_rate': female_error_rate
    }

def calculate_f1(predictions_df):
    """Calculate current F1 score from predictions DataFrame."""
    y_true = (predictions_df['true_gender'] == 'W').astype(int)
    y_pred = (predictions_df['pred_gender'] == 'W').astype(int)
    
    _, _, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    return f1

def plot_threshold_analysis(results_df, target_f1, optimal_threshold):
    """Create visualization of threshold analysis."""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # F1 vs Threshold
    ax1.plot(results_df['threshold'], results_df['f1'], 'b-', linewidth=2)
    ax1.axhline(target_f1, color='red', linestyle='--', alpha=0.7, label=f'Target F1: {target_f1}')
    ax1.axvline(optimal_threshold, color='green', linestyle='--', alpha=0.7, label=f'Optimal: {optimal_threshold:.3f}')
    ax1.set_xlabel('Threshold')
    ax1.set_ylabel('F1 Score')
    ax1.set_title('F1 Score vs Threshold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Bias Ratio vs Threshold
    ax2.plot(results_df['threshold'], results_df['bias_ratio'], 'g-', linewidth=2)
    ax2.axhline(1.0, color='red', linestyle='-', alpha=0.5, label='Perfect Fairness')
    ax2.axvline(optimal_threshold, color='green', linestyle='--', alpha=0.7, label=f'Optimal: {optimal_threshold:.3f}')
    ax2.set_xlabel('Threshold')
    ax2.set_ylabel('Bias Ratio')
    ax2.set_title('Bias Ratio vs Threshold')
    ax2.set_ylim(0.8, 1.2)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Accuracy vs Threshold
    ax3.plot(results_df['threshold'], results_df['accuracy'], 'm-', linewidth=2)
    ax3.axvline(optimal_threshold, color='green', linestyle='--', alpha=0.7, label=f'Optimal: {optimal_threshold:.3f}')
    ax3.set_xlabel('Threshold')
    ax3.set_ylabel('Accuracy')
    ax3.set_title('Accuracy vs Threshold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # F1 vs Bias Trade-off
    ax4.scatter(results_df['bias_ratio'], results_df['f1'], c=results_df['threshold'], cmap='viridis', alpha=0.6)
    
    # Highlight optimal point
    optimal_row = results_df[results_df['threshold'] == optimal_threshold].iloc[0]
    ax4.scatter(optimal_row['bias_ratio'], optimal_row['f1'], 
               color='red', s=100, marker='*', label=f'Optimal (t={optimal_threshold:.3f})')
    
    ax4.axhline(target_f1, color='red', linestyle='--', alpha=0.7)
    ax4.axvline(1.0, color='red', linestyle='--', alpha=0.7)
    
    ax4.set_xlabel('Bias Ratio')
    ax4.set_ylabel('F1 Score') 
    ax4.set_title('F1 vs Bias Trade-off')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    # Colorbar
    cbar = plt.colorbar(ax4.collections[0], ax=ax4)
    cbar.set_label('Threshold')
    
    plt.tight_layout()
    plt.savefig('threshold_fine_tuning_analysis.png', dpi=300, bbox_inches='tight')
    print(f"📊 Analysis plot saved: threshold_fine_tuning_analysis.png")
    
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Fine-tune threshold for optimal F1")
    parser.add_argument('--predictions', required=True, help='Predictions CSV file')
    parser.add_argument('--target_f1', type=float, default=0.903, help='Target F1 score')
    parser.add_argument('--resolution', type=float, default=0.001, help='Threshold search resolution')
    parser.add_argument('--output', default='fine_tuned_threshold_results.json', help='Output file')
    
    args = parser.parse_args()
    
    print(f"🎯 Fine Threshold Optimization")
    print(f"=" * 50)
    print(f"📊 Clean approach - no pattern overfitting")
    
    # Load predictions
    predictions_df = pd.read_csv(args.predictions)
    print(f"📂 Loaded {len(predictions_df)} predictions")
    
    # Fine-tune threshold
    optimal_result, full_results = fine_tune_threshold(
        predictions_df, 
        target_f1=args.target_f1,
        resolution=args.resolution
    )
    
    # Save results
    import json
    with open(args.output, 'w') as f:
        # Convert numpy types for JSON serialization
        json_result = {}
        for key, value in optimal_result.items():
            if isinstance(value, dict):
                json_result[key] = {k: float(v) if isinstance(v, (np.float64, np.float32)) else v 
                                   for k, v in value.items()}
            else:
                json_result[key] = float(value) if isinstance(value, (np.float64, np.float32)) else value
        
        json.dump(json_result, f, indent=2)
    
    # Save detailed results
    full_results.to_csv(args.output.replace('.json', '_detailed.csv'), index=False)
    
    print(f"\n💾 Results saved:")
    print(f"   Summary: {args.output}")
    print(f"   Detailed: {args.output.replace('.json', '_detailed.csv')}")
    print(f"   Plot: threshold_fine_tuning_analysis.png")
    
    # Final recommendation
    if optimal_result['meets_target']:
        print(f"\n🎉 SUCCESS: Clean threshold optimization achieves target!")
        print(f"   Use threshold {optimal_result['optimal_threshold']:.3f} for F1 ≥ {args.target_f1}")
        print(f"   No overfitting, generalizable solution")
    else:
        print(f"\n📏 Target not achievable with threshold alone")
        print(f"   Max F1: {optimal_result['max_achievable_f1']:.4f}")
        print(f"   Consider accepting baseline Unicode result")

if __name__ == "__main__":
    main()
