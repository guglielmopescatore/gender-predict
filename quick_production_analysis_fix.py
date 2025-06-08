#!/usr/bin/env python3
"""
Quick fix for production analysis based on preprocessing results.

Based on the analysis, both preprocessing approaches are equivalent in quality.
Let's focus on the actual model performance analysis.
"""

import sys
import os
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from gender_predict.evaluation.evaluator import ModelEvaluator
from gender_predict.data import ImprovedNameGenderDataset, NameFeatureExtractor
import joblib

def quick_production_analysis():
    """Quick production analysis focusing on key insights."""

    print("🚀 QUICK PRODUCTION V3 ANALYSIS")
    print("=" * 40)

    # Model paths
    model_path = "./experiments/20250604_192834_r3_bce_h256_l3_dual_frz5/models/model.pth"
    preprocessor_path = "./experiments/20250604_192834_r3_bce_h256_l3_dual_frz5/preprocessor.pkl"
    feature_extractor_path = "./experiments/20250604_192834_r3_bce_h256_l3_dual_frz5/feature_extractor.pkl"
    test_data_path = "./data/processed/comparison_dataset_clean.csv"
    production_threshold = 0.48

    # Load model and data
    print("🔄 Loading model and data...")

    evaluator = ModelEvaluator.from_checkpoint(model_path, preprocessor_path, 'cuda')
    feature_extractor = joblib.load(feature_extractor_path)
    test_df = pd.read_csv(test_data_path)

    print(f"✅ Model loaded, testing on {len(test_df):,} samples")

    # Create dataset and evaluate
    test_dataset = ImprovedNameGenderDataset(
        test_df, evaluator.preprocessor, feature_extractor, mode='test'
    )

    print("🔄 Running evaluation...")
    results = evaluator.evaluate_dataset(test_dataset)

    # Calculate metrics with production threshold (0.48)
    production_predictions = (np.array(results['probabilities']) >= production_threshold).astype(int)

    prod_accuracy = accuracy_score(results['targets'], production_predictions)
    prod_precision, prod_recall, prod_f1, _ = precision_recall_fscore_support(
        results['targets'], production_predictions, average='binary'
    )

    # Standard threshold (0.5) for comparison
    std_accuracy = accuracy_score(results['targets'], results['predictions'])
    std_precision, std_recall, std_f1, _ = precision_recall_fscore_support(
        results['targets'], results['predictions'], average='binary'
    )

    # Bias analysis
    cm_prod = confusion_matrix(results['targets'], production_predictions)
    cm_std = confusion_matrix(results['targets'], results['predictions'])

    tn_p, fp_p, fn_p, tp_p = cm_prod.ravel()
    prod_bias_ratio = (fp_p / (tn_p + fp_p)) / (fn_p / (tp_p + fn_p)) if (tn_p + fp_p) > 0 and (tp_p + fn_p) > 0 else 1.0

    tn_s, fp_s, fn_s, tp_s = cm_std.ravel()
    std_bias_ratio = (fp_s / (tn_s + fp_s)) / (fn_s / (tp_s + fn_s)) if (tn_s + fp_s) > 0 and (tp_s + fn_s) > 0 else 1.0

    # Results
    print(f"\n📊 PRODUCTION PERFORMANCE ANALYSIS")
    print(f"=" * 50)
    print(f"Dataset: {test_data_path} ({len(test_df):,} samples)")
    print(f"")
    print(f"{'Metric':<15} {'Prod (0.48)':<12} {'Std (0.50)':<12} {'Difference'}")
    print(f"{'-'*55}")
    print(f"{'Accuracy':<15} {prod_accuracy:<12.4f} {std_accuracy:<12.4f} {prod_accuracy-std_accuracy:+.4f}")
    print(f"{'F1 Score':<15} {prod_f1:<12.4f} {std_f1:<12.4f} {prod_f1-std_f1:+.4f}")
    print(f"{'Precision':<15} {prod_precision:<12.4f} {std_precision:<12.4f} {prod_precision-std_precision:+.4f}")
    print(f"{'Recall':<15} {prod_recall:<12.4f} {std_recall:<12.4f} {prod_recall-std_recall:+.4f}")
    print(f"{'Bias Ratio':<15} {prod_bias_ratio:<12.4f} {std_bias_ratio:<12.4f} {prod_bias_ratio-std_bias_ratio:+.4f}")

    # V4 targets
    target_accuracy = 0.94
    accuracy_gap = target_accuracy - prod_accuracy

    print(f"\n🎯 V4 DEVELOPMENT TARGETS")
    print(f"=" * 40)
    print(f"Current Accuracy: {prod_accuracy:.4f} ({prod_accuracy*100:.1f}%)")
    print(f"Target Accuracy:  {target_accuracy:.4f} ({target_accuracy*100:.1f}%)")
    print(f"Required Gain:    {accuracy_gap:.4f} ({accuracy_gap*100:.1f} percentage points)")
    print(f"Current Error Rate: {(1-prod_accuracy)*100:.1f}%")
    print(f"Target Error Rate:  <6.0%")

    # Error analysis
    errors = production_predictions != results['targets']
    total_errors = errors.sum()

    print(f"\n🔍 ERROR ANALYSIS")
    print(f"=" * 30)
    print(f"Total errors: {total_errors:,} ({total_errors/len(test_df)*100:.1f}%)")

    # High confidence errors
    high_conf_errors = errors & (np.array([max(p, 1-p) for p in results['probabilities']]) > 0.8)
    print(f"High confidence errors: {high_conf_errors.sum():,} ({high_conf_errors.sum()/total_errors*100:.1f}% of errors)")

    # Error types
    m_to_w_errors = errors & (results['targets'] == 0) & (production_predictions == 1)
    w_to_m_errors = errors & (results['targets'] == 1) & (production_predictions == 0)

    print(f"Male → Female errors: {m_to_w_errors.sum():,} ({m_to_w_errors.sum()/total_errors*100:.1f}%)")
    print(f"Female → Male errors: {w_to_m_errors.sum():,} ({w_to_m_errors.sum()/total_errors*100:.1f}%)")

    # V4 Priorities
    print(f"\n💡 V4 DEVELOPMENT PRIORITIES")
    print(f"=" * 40)

    high_conf_error_rate = high_conf_errors.sum() / total_errors

    if high_conf_error_rate > 0.15:  # >15% of errors are high confidence
        print(f"🔴 Priority 1: UNCERTAINTY CALIBRATION")
        print(f"   Issue: {high_conf_errors.sum()} high-confidence errors ({high_conf_error_rate:.1%} of errors)")
        print(f"   Solution: Implement V4.4 uncertainty-aware training")
        print(f"   Potential gain: ~{high_conf_errors.sum()/len(test_df)*100:.1f}% accuracy improvement")

    print(f"\n🔵 Priority 2: CHARACTER SEQUENCE MODELING")
    print(f"   Issue: BiLSTM limitations in long-range dependencies")
    print(f"   Solution: Implement V4.1 character-level Transformer")
    print(f"   Potential gain: 1.0-1.5% accuracy improvement")

    # Preprocessing insight from previous analysis
    print(f"\n✅ PREPROCESSING STATUS")
    print(f"   Analysis result: Equivalent quality across all preprocessing approaches")
    print(f"   Recommendation: Standardize for consistency")
    print(f"   V4 Action: Focus on architecture improvements, not preprocessing fixes")

    # Final recommendation
    estimated_v4_gain = 0.015 + (high_conf_errors.sum() / len(test_df))  # Transformer + uncertainty
    estimated_v4_accuracy = prod_accuracy + estimated_v4_gain

    print(f"\n🚀 V4 ROADMAP")
    print(f"=" * 25)
    print(f"Phase 1: V4.4 Uncertainty (reduce high-confidence errors)")
    print(f"Phase 2: V4.1 Transformer (better character modeling)")
    print(f"Phase 3: V4.3 Cross-cultural (if needed for international names)")
    print(f"")
    print(f"Estimated V4 performance: {estimated_v4_accuracy:.4f} ({estimated_v4_accuracy*100:.1f}%)")
    print(f"Target achievement: {'✅ ACHIEVABLE' if estimated_v4_accuracy >= target_accuracy else '⚠️ CHALLENGING'}")

    # Save key results
    results_summary = {
        'current_production_accuracy': float(prod_accuracy),
        'current_production_f1': float(prod_f1),
        'target_accuracy': float(target_accuracy),
        'accuracy_gap': float(accuracy_gap),
        'total_errors': int(total_errors),
        'high_confidence_errors': int(high_conf_errors.sum()),
        'estimated_v4_accuracy': float(estimated_v4_accuracy),
        'v4_phases': ['V4.4_Uncertainty', 'V4.1_Transformer', 'V4.3_Cross_cultural'],
        'preprocessing_status': 'EQUIVALENT_QUALITY'
    }

    import json
    os.makedirs("./v4_analysis", exist_ok=True)
    with open("./v4_analysis/quick_production_analysis.json", 'w') as f:
        json.dump(results_summary, f, indent=2)

    print(f"\n💾 Results saved to: ./v4_analysis/quick_production_analysis.json")
    print(f"✅ Analysis complete! Ready for V4 development.")

if __name__ == "__main__":
    quick_production_analysis()
