#!/usr/bin/env python3
"""
Model Comparison Script

Compare different models (B0, V4-R1, V4-R2) across multiple thresholds
to evaluate accuracy, F1 score, and bias metrics.
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
import json
import time
from datetime import datetime

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

from final_predictor import FinalGenderPredictor

# Model configurations to test
MODEL_CONFIGS = {
    'B0_Baseline': {
        'model_path': 'experiments/20250603_192912_r3_bce_h256_l3_dual_frz5/models/model.pth',
        'preprocessor_path': 'experiments/20250603_192912_r3_bce_h256_l3_dual_frz5/preprocessor.pkl',
        'description': 'Original B0 baseline model (4 phonetic features)',
        'unicode_preprocessing': True,
        'enabled': True
    },
    'V4_R1_Enhanced': {
        'model_path': 'experiments/20250610_150408_r3_bce_h256_l3_dual_frz5/models/model.pth',
        'preprocessor_path': 'experiments/20250610_150408_r3_bce_h256_l3_dual_frz5/preprocessor.pkl',
        'description': 'V4-R1 with enhanced preprocessing (has split_full_name issues)',
        'unicode_preprocessing': True,
        'enabled': True
    },
    # 'V4_R2_Features': {
    #     'model_path': 'experiments/20250610_190342_r3_bce_h256_l3_dual_frz5/models/model.pth',
    #     'preprocessor_path': 'experiments/20250610_190342_r3_bce_h256_l3_dual_frz5/preprocessor.pkl',
    #     'description': 'V4-R2 with enhanced features (9 phonetic - incompatible)',
    #     'unicode_preprocessing': True,
    #     'enabled': False  # Disabled due to 9 vs 4 phonetic features mismatch
    # },
    'Best_V3_Model': {
        'model_path': 'models/best_v3_model/models/model.pth',
        'preprocessor_path': 'models/best_v3_model/preprocessor.pkl',
        'description': 'Best V3 model in production (4 phonetic features)',
        'unicode_preprocessing': True,
        'enabled': True
    }
}

# Additional available models (uncomment and test individually)
ADDITIONAL_MODELS = {
    'Experiment_171542': {
        'model_path': 'experiments/20250610_171542_r3_bce_h256_l3_dual_frz5/models/model.pth',
        'preprocessor_path': 'experiments/20250610_171542_r3_bce_h256_l3_dual_frz5/preprocessor.pkl',
        'description': 'Experiment 171542 (check compatibility first)',
        'unicode_preprocessing': True,
        'enabled': False
    }
}

# Test thresholds
THRESHOLDS = [0.40, 0.42, 0.44, 0.46, 0.48, 0.50, 0.52, 0.54, 0.56, 0.58, 0.60]

def load_test_data(test_file='data/processed/comparison_dataset_clean.csv'):
    """Load test dataset."""
    if not os.path.exists(test_file):
        # Fallback test files
        fallback_files = [
            'data/raw/test_no_nconst.csv',
            'data/raw/test_simple.csv',
            'data/raw/sample_test.csv'
        ]
        for f in fallback_files:
            if os.path.exists(f):
                test_file = f
                break
        else:
            raise FileNotFoundError(f"No test file found. Checked: {test_file}, {fallback_files}")

    df = pd.read_csv(test_file)
    print(f"📊 Loaded dataset: {len(df):,} samples")

    # Ensure we have the required columns
    if 'primaryName' not in df.columns:
        if 'name' in df.columns:
            df['primaryName'] = df['name']
        else:
            raise ValueError("Test file must have 'primaryName' or 'name' column")

    # Handle gender column
    if 'knownForTitles' not in df.columns:
        if 'gender' in df.columns:
            df['knownForTitles'] = df['gender']
        elif 'actual_gender' in df.columns:
            df['knownForTitles'] = df['actual_gender']
        else:
            print("⚠️ Warning: No gender column found, using 'unknown'")
            df['knownForTitles'] = 'unknown'

    # Show gender distribution
    gender_dist = df['knownForTitles'].value_counts()
    print("📈 Gender distribution:")
    for gender, count in gender_dist.items():
        print(f"   {gender}: {count:,} ({count/len(df)*100:.1f}%)")

    # Optional: Sample for faster testing (comment out for full dataset)
    # df = df.sample(n=5000, random_state=42)
    # print(f"🎯 Using sample of {len(df):,} for faster testing")

    return df

def calculate_metrics(predictions, actuals, threshold):
    """Calculate comprehensive metrics for given threshold."""

    # Convert predictions to binary using threshold
    pred_binary = (predictions >= threshold).astype(int)

    # Convert actuals to binary (more robust gender mapping)
    actual_binary = []
    for actual in actuals:
        if pd.isna(actual) or actual == 'unknown' or actual == '':
            continue
        if isinstance(actual, str):
            actual_clean = actual.upper().strip()
            # Map various female indicators to 1
            if actual_clean in ['W', 'F', 'FEMALE', 'WOMAN', '1', 'TRUE']:
                actual_binary.append(1)
            # Map various male indicators to 0
            elif actual_clean in ['M', 'MALE', 'MAN', '0', 'FALSE']:
                actual_binary.append(0)
            else:
                continue  # Skip unknown values
        else:
            actual_binary.append(int(actual))

    # Align arrays
    min_len = min(len(pred_binary), len(actual_binary))
    pred_binary = pred_binary[:min_len]
    actual_binary = np.array(actual_binary[:min_len])

    if len(actual_binary) == 0:
        return {
            'threshold': threshold,
            'accuracy': 0.0,
            'f1_score': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'bias_ratio': 1.0,
            'male_accuracy': 0.0,
            'female_accuracy': 0.0,
            'sample_size': 0
        }

    # Basic metrics
    correct = (pred_binary == actual_binary).sum()
    accuracy = correct / len(actual_binary)

    # Confusion matrix
    tp = ((pred_binary == 1) & (actual_binary == 1)).sum()  # True Positive (predicted F, actual F)
    fp = ((pred_binary == 1) & (actual_binary == 0)).sum()  # False Positive (predicted F, actual M)
    tn = ((pred_binary == 0) & (actual_binary == 0)).sum()  # True Negative (predicted M, actual M)
    fn = ((pred_binary == 0) & (actual_binary == 1)).sum()  # False Negative (predicted M, actual F)

    # F1, Precision, Recall
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # Gender-specific accuracy
    male_mask = actual_binary == 0
    female_mask = actual_binary == 1

    male_accuracy = (pred_binary[male_mask] == actual_binary[male_mask]).mean() if male_mask.sum() > 0 else 0.0
    female_accuracy = (pred_binary[female_mask] == actual_binary[female_mask]).mean() if female_mask.sum() > 0 else 0.0

    # Bias ratio
    bias_ratio = female_accuracy / male_accuracy if male_accuracy > 0 else 1.0

    return {
        'threshold': threshold,
        'accuracy': float(accuracy),
        'f1_score': float(f1_score),
        'precision': float(precision),
        'recall': float(recall),
        'bias_ratio': float(bias_ratio),
        'male_accuracy': float(male_accuracy),
        'female_accuracy': float(female_accuracy),
        'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn),
        'sample_size': len(actual_binary)
    }

def test_model(model_name, config, test_df, thresholds):
    """Test a single model across multiple thresholds."""
    print(f"\n🧪 Testing {model_name}")
    print(f"    Description: {config['description']}")
    print(f"    Model: {config['model_path']}")

    # Check if files exist
    if not os.path.exists(config['model_path']):
        print(f"    ❌ Model file not found: {config['model_path']}")
        return None

    if not os.path.exists(config['preprocessor_path']):
        print(f"    ❌ Preprocessor file not found: {config['preprocessor_path']}")
        return None

    try:
        # Load model
        model_config = {
            'model_path': config['model_path'],
            'preprocessor_path': config['preprocessor_path'],
            'optimal_threshold': 0.5,  # Will be varied
            'unicode_preprocessing': config.get('unicode_preprocessing', True),
            'expected_performance': {
                'f1_score': 0.9, 'accuracy': 0.9, 'bias_ratio': 1.0, 'bias_deviation': 0.0
            }
        }

        predictor = FinalGenderPredictor(model_config)
        predictor.load_model()

        print(f"    ✅ Model loaded successfully")

        # Get predictions (probabilities)
        print(f"    🔄 Generating predictions for {len(test_df):,} samples...")
        predictions = []
        start_time = time.time()

        for i, name in enumerate(test_df['primaryName']):
            if i % 1000 == 0 and i > 0:
                elapsed = time.time() - start_time
                rate = i / elapsed
                eta = (len(test_df) - i) / rate
                print(f"       Progress: {i:,}/{len(test_df):,} ({i/len(test_df)*100:.1f}%) | "
                      f"Rate: {rate:.1f} names/sec | ETA: {eta/60:.1f} min")

            try:
                result = predictor.predict_single(str(name))
                predictions.append(result['probability_female'])
            except Exception as e:
                if i < 10:  # Only show first 10 errors to avoid spam
                    print(f"       ⚠️ Error predicting '{name}': {e}")
                predictions.append(0.5)  # Neutral prediction

        print(f"    ✅ Predictions complete")

        # Test across thresholds
        print(f"    📊 Testing {len(thresholds)} thresholds...")
        results = []

        for threshold in thresholds:
            metrics = calculate_metrics(
                np.array(predictions),
                test_df['knownForTitles'].values,
                threshold
            )
            metrics['model'] = model_name
            results.append(metrics)

        return results

    except Exception as e:
        print(f"    ❌ Error testing model: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main comparison function."""
    print("🎯 Model Comparison Analysis")
    print("=" * 50)

    # Load test data
    print("📂 Loading test data...")
    try:
        test_df = load_test_data()

        # Option for quick test with smaller sample
        if len(test_df) > 10000:
            response = input(f"\n🤔 Dataset has {len(test_df):,} samples. Use smaller sample for quick test? (y/N): ")
            if response.lower() == 'y':
                sample_size = int(input("   Enter sample size (5000): ") or "5000")
                test_df = test_df.sample(n=min(sample_size, len(test_df)), random_state=42)
                print(f"🎯 Using sample of {len(test_df):,} for faster testing")

        print(f"✅ Using {len(test_df):,} test samples")

    except Exception as e:
        print(f"❌ Error loading test data: {e}")
        return

    # Test each model
    all_results = []

    for model_name, config in MODEL_CONFIGS.items():
        # Skip disabled models
        if not config.get('enabled', True):
            print(f"\n⏭️ Skipping {model_name} (disabled)")
            continue

        results = test_model(model_name, config, test_df, THRESHOLDS)
        if results:
            all_results.extend(results)

            # Save intermediate results
            if all_results:
                temp_df = pd.DataFrame(all_results)
                temp_file = f"temp_results_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                temp_df.to_csv(temp_file, index=False)
                print(f"    💾 Intermediate results saved to: {temp_file}")

        # Optional: add a small delay between models to let GPU cool down
        time.sleep(2)

    if not all_results:
        print("❌ No results generated")
        return

    # Create results DataFrame
    results_df = pd.DataFrame(all_results)

    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"model_comparison_{timestamp}.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\n💾 Detailed results saved to: {output_file}")

    # Summary analysis
    print("\n📊 SUMMARY ANALYSIS")
    print("=" * 50)

    for model in results_df['model'].unique():
        model_data = results_df[results_df['model'] == model]

        if len(model_data) == 0:
            continue

        # Find best threshold for different metrics
        best_f1_idx = model_data['f1_score'].idxmax()
        best_acc_idx = model_data['accuracy'].idxmax()
        best_bias_idx = model_data['bias_ratio'].apply(lambda x: abs(1.0 - x)).idxmin()

        best_f1 = model_data.loc[best_f1_idx]
        best_acc = model_data.loc[best_acc_idx]
        best_bias = model_data.loc[best_bias_idx]

        print(f"\n🔍 {model}")
        print(f"   Best F1:       {best_f1['f1_score']:.4f} @ threshold {best_f1['threshold']:.2f}")
        print(f"   Best Accuracy: {best_acc['accuracy']:.4f} @ threshold {best_acc['threshold']:.2f}")
        print(f"   Best Bias:     {best_bias['bias_ratio']:.4f} @ threshold {best_bias['threshold']:.2f}")
        print(f"   Sample size:   {best_f1['sample_size']}")

    # Optimal threshold comparison
    print(f"\n🎯 OPTIMAL THRESHOLD COMPARISON (F1 Score)")
    print("-" * 60)

    for model in results_df['model'].unique():
        model_data = results_df[results_df['model'] == model]
        if len(model_data) == 0:
            continue

        best_f1_row = model_data.loc[model_data['f1_score'].idxmax()]

        print(f"{model:20} | Threshold: {best_f1_row['threshold']:.2f} | "
              f"F1: {best_f1_row['f1_score']:.4f} | "
              f"Acc: {best_f1_row['accuracy']:.4f} | "
              f"Bias: {best_f1_row['bias_ratio']:.4f}")

    print(f"\n✅ Analysis complete! Check {output_file} for detailed results.")

if __name__ == "__main__":
    main()
