#!/usr/bin/env python3
"""
Production V3 Model Analysis for V4 Development
===============================================

Analyzes the current production model:
- Model: 20250604_192834_r3_bce_h256_l3_dual_frz5/models/model.pth
- Threshold: 0.48 (optimized for fairness)

This analysis will identify specific improvement targets for V4.
"""

import sys
import os
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import re
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from gender_predict.evaluation.evaluator import ModelEvaluator
from gender_predict.data import ImprovedNameGenderDataset, NameFeatureExtractor
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix


class ProductionV3Analyzer:
    """
    Analyzer specifically for the production V3 model to guide V4 development.
    """

    def __init__(self):
        # Production model configuration
        self.model_path = "./experiments/20250604_192834_r3_bce_h256_l3_dual_frz5/models/model.pth"
        self.preprocessor_path = "./experiments/20250604_192834_r3_bce_h256_l3_dual_frz5/preprocessor.pkl"
        self.feature_extractor_path = "./experiments/20250604_192834_r3_bce_h256_l3_dual_frz5/feature_extractor.pkl"
        self.production_threshold = 0.48  # Optimized threshold

        self.evaluator = None
        self.feature_extractor = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Results storage
        self.analysis_results = {}

    def verify_production_model(self):
        """Verify all production model files exist."""
        print("🔍 Verifying production model files...")

        files_to_check = {
            'Model': self.model_path,
            'Preprocessor': self.preprocessor_path,
            'Feature Extractor': self.feature_extractor_path
        }

        all_good = True
        for name, path in files_to_check.items():
            if os.path.exists(path):
                size_mb = os.path.getsize(path) / (1024 * 1024)
                print(f"   ✅ {name}: {path} ({size_mb:.1f} MB)")
            else:
                print(f"   ❌ {name}: {path} (NOT FOUND)")
                all_good = False

        if not all_good:
            print("❌ Missing required files. Please check the paths.")
            return False

        print(f"✅ All production model files verified!")
        return True

    def load_production_model(self):
        """Load the production V3 model."""
        print("🔄 Loading production V3 model...")

        try:
            # Load evaluator
            self.evaluator = ModelEvaluator.from_checkpoint(
                self.model_path, self.preprocessor_path, self.device
            )

            # Load feature extractor
            import joblib
            self.feature_extractor = joblib.load(self.feature_extractor_path)

            print(f"✅ Production model loaded successfully!")
            print(f"   🎯 Model: V3 BiLSTM + Multi-head attention")
            print(f"   🔧 Hidden size: 256, Layers: 3")
            print(f"   ⚖️  Production threshold: {self.production_threshold}")
            print(f"   🚀 Device: {self.device}")

            return True

        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False

    def find_test_dataset(self):
        """Find appropriate test dataset."""
        print("📊 Looking for test datasets...")

        potential_datasets = [
            "./data/test_set.csv",
            "./data/test.csv",
            "./data/imdb_names_test.csv",
            "./data/comparison_dataset.csv",
            "./data/validation_set.csv"
        ]

        for dataset_path in potential_datasets:
            if os.path.exists(dataset_path):
                try:
                    df = pd.read_csv(dataset_path)
                    print(f"   ✅ Found: {dataset_path} ({len(df):,} samples)")
                    print(f"      Columns: {', '.join(df.columns)}")

                    # Check for required columns
                    if 'primaryName' in df.columns and 'gender' in df.columns:
                        print(f"   🎯 Using dataset: {dataset_path}")
                        return dataset_path, df

                except Exception as e:
                    print(f"   ⚠️  Error reading {dataset_path}: {e}")

        print("❌ No suitable test dataset found!")
        print("📝 Please provide a CSV with 'primaryName' and 'gender' columns")
        return None, None

    def analyze_production_performance(self, test_df, dataset_path):
        """Analyze current production performance with 0.48 threshold AND preprocessing mismatch."""
        print(f"\n🔄 Analyzing production performance...")
        print(f"⚠️  CRITICAL: Testing both training and production preprocessing!")

        # Test with TRAINING preprocessing (what model was trained on)
        training_dataset = ImprovedNameGenderDataset(
            test_df, self.evaluator.preprocessor, self.feature_extractor, mode='test'
        )
        training_results = self.evaluator.evaluate_dataset(training_dataset)

        # Test with PRODUCTION preprocessing (what's actually used in production)
        from scripts.final_predictor import ProductionRobustPreprocessor
        production_preprocessor = ProductionRobustPreprocessor(self.evaluator.preprocessor)

        production_dataset = ImprovedNameGenderDataset(
            test_df, production_preprocessor, self.feature_extractor, mode='test'
        )
        production_results = self.evaluator.evaluate_dataset(production_dataset)

        print(f"\n🔍 PREPROCESSING MISMATCH ANALYSIS:")
        print(f"   Training preprocessing accuracy:   {training_results['accuracy']:.4f}")
        print(f"   Production preprocessing accuracy: {production_results['accuracy']:.4f}")
        print(f"   Accuracy loss due to mismatch:    {training_results['accuracy'] - production_results['accuracy']:.4f}")

        # Use production results for main analysis (what actually happens in production)
        results = production_results

        # Apply production threshold (0.48 instead of 0.5)
        production_predictions = (np.array(results['probabilities']) >= self.production_threshold).astype(int)

        # Calculate metrics with production threshold
        prod_accuracy = accuracy_score(results['targets'], production_predictions)
        prod_precision, prod_recall, prod_f1, _ = precision_recall_fscore_support(
            results['targets'], production_predictions, average='binary'
        )

        # Bias analysis
        cm_prod = confusion_matrix(results['targets'], production_predictions)
        cm_std = confusion_matrix(results['targets'], results['predictions'])

        # Production bias
        tn_p, fp_p, fn_p, tp_p = cm_prod.ravel()
        prod_bias_ratio = (fp_p / (tn_p + fp_p)) / (fn_p / (tp_p + fn_p)) if (tn_p + fp_p) > 0 and (tp_p + fn_p) > 0 else 1.0

        # Standard bias
        tn_s, fp_s, fn_s, tp_s = cm_std.ravel()
        std_bias_ratio = (fp_s / (tn_s + fp_s)) / (fn_s / (tp_s + fn_s)) if (tn_s + fp_s) > 0 and (tp_s + fn_s) > 0 else 1.0

        performance_comparison = {
            'production_threshold_0.48': {
                'accuracy': float(prod_accuracy),
                'precision': float(prod_precision),
                'recall': float(prod_recall),
                'f1_score': float(prod_f1),
                'bias_ratio': float(prod_bias_ratio),
                'confusion_matrix': cm_prod.tolist()
            },
            'standard_threshold_0.50': {
                'accuracy': float(std_accuracy),
                'precision': float(std_precision),
                'recall': float(std_recall),
                'f1_score': float(std_f1),
                'bias_ratio': float(std_bias_ratio),
                'confusion_matrix': cm_std.tolist()
            },
            'dataset_info': {
                'path': dataset_path,
                'total_samples': len(test_df),
                'male_samples': int((test_df['gender'] == 'M').sum()),
                'female_samples': int((test_df['gender'] == 'W').sum())
            }
        }

        # Print comparison
        print(f"\n📊 PRODUCTION PERFORMANCE ANALYSIS")
        print(f"=" * 60)
        print(f"Dataset: {dataset_path} ({len(test_df):,} samples)")
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
        print(f"Error Reduction Needed: {((1-prod_accuracy) - 0.06)*100:.1f}% → {((1-prod_accuracy) - 0.06)/(1-prod_accuracy)*100:.1f}% relative reduction")

        # Add preprocessing mismatch impact to results
        if hasattr(self, 'preprocessing_mismatch_impact'):
            print(f"\n⚠️  PREPROCESSING MISMATCH IMPACT:")
            mismatch_loss = self.preprocessing_mismatch_impact
            print(f"   Accuracy loss from preprocessing: {mismatch_loss:.4f} ({mismatch_loss*100:.1f}%)")
            print(f"   Potential quick gain fixing preprocessing: {mismatch_loss*100:.1f}%")
            print(f"   Remaining gap after preprocessing fix: {(accuracy_gap - mismatch_loss)*100:.1f}%")

        # Store results
        self.analysis_results['performance'] = performance_comparison

        # Create detailed predictions DataFrame for error analysis
        predictions_df = pd.DataFrame({
            'primaryName': test_df['primaryName'],
            'true_gender': test_df['gender'],
            'true_label': results['targets'],
            'probability_female': results['probabilities'],
            'pred_label_std': results['predictions'],  # 0.5 threshold
            'pred_label_prod': production_predictions,  # 0.48 threshold
            'pred_gender_std': ['W' if p == 1 else 'M' for p in results['predictions']],
            'pred_gender_prod': ['W' if p == 1 else 'M' for p in production_predictions],
            'is_error_std': [t != p for t, p in zip(results['targets'], results['predictions'])],
            'is_error_prod': [t != p for t, p in zip(results['targets'], production_predictions)],
            'confidence': [max(p, 1-p) for p in results['probabilities']]
        })

        return predictions_df

        # Bias analysis
        cm_prod = confusion_matrix(results['targets'], production_predictions)
        cm_std = confusion_matrix(results['targets'], results['predictions'])

        # Production bias
        tn_p, fp_p, fn_p, tp_p = cm_prod.ravel()
        prod_bias_ratio = (fp_p / (tn_p + fp_p)) / (fn_p / (tp_p + fn_p)) if (tn_p + fp_p) > 0 and (tp_p + fn_p) > 0 else 1.0

        # Standard bias
        tn_s, fp_s, fn_s, tp_s = cm_std.ravel()
        std_bias_ratio = (fp_s / (tn_s + fp_s)) / (fn_s / (tp_s + fn_s)) if (tn_s + fp_s) > 0 and (tp_s + fn_s) > 0 else 1.0

        performance_comparison = {
            'production_threshold_0.48': {
                'accuracy': float(prod_accuracy),
                'precision': float(prod_precision),
                'recall': float(prod_recall),
                'f1_score': float(prod_f1),
                'bias_ratio': float(prod_bias_ratio),
                'confusion_matrix': cm_prod.tolist()
            },
            'standard_threshold_0.50': {
                'accuracy': float(std_accuracy),
                'precision': float(std_precision),
                'recall': float(std_recall),
                'f1_score': float(std_f1),
                'bias_ratio': float(std_bias_ratio),
                'confusion_matrix': cm_std.tolist()
            },
            'dataset_info': {
                'path': dataset_path,
                'total_samples': len(test_df),
                'male_samples': int((test_df['gender'] == 'M').sum()),
                'female_samples': int((test_df['gender'] == 'W').sum())
            }
        }

        # Print comparison
        print(f"\n📊 PRODUCTION PERFORMANCE ANALYSIS")
        print(f"=" * 60)
        print(f"Dataset: {dataset_path} ({len(test_df):,} samples)")
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
        print(f"Error Reduction Needed: {((1-prod_accuracy) - 0.06)*100:.1f}% → {((1-prod_accuracy) - 0.06)/(1-prod_accuracy)*100:.1f}% relative reduction")

        # Add preprocessing mismatch impact to results
        if hasattr(self, 'preprocessing_mismatch_impact'):
            print(f"\n⚠️  PREPROCESSING MISMATCH IMPACT:")
            mismatch_loss = self.preprocessing_mismatch_impact
            print(f"   Accuracy loss from preprocessing: {mismatch_loss:.4f} ({mismatch_loss*100:.1f}%)")
            print(f"   Potential quick gain fixing preprocessing: {mismatch_loss*100:.1f}%")
            print(f"   Remaining gap after preprocessing fix: {(accuracy_gap - mismatch_loss)*100:.1f}%")

        # Store results
        self.analysis_results['performance'] = performance_comparison

        # Create detailed predictions DataFrame for error analysis
        predictions_df = pd.DataFrame({
            'primaryName': test_df['primaryName'],
            'true_gender': test_df['gender'],
            'true_label': results['targets'],
            'probability_female': results['probabilities'],
            'pred_label_std': results['predictions'],  # 0.5 threshold
            'pred_label_prod': production_predictions,  # 0.48 threshold
            'pred_gender_std': ['W' if p == 1 else 'M' for p in results['predictions']],
            'pred_gender_prod': ['W' if p == 1 else 'M' for p in production_predictions],
            'is_error_std': [t != p for t, p in zip(results['targets'], results['predictions'])],
            'is_error_prod': [t != p for t, p in zip(results['targets'], production_predictions)],
            'confidence': [max(p, 1-p) for p in results['probabilities']]
        })

        return predictions_df

    def analyze_error_patterns(self, predictions_df):
        """Deep analysis of error patterns in production model."""
        print(f"\n🔍 Analyzing production model error patterns...")

        # Focus on production threshold errors
        errors = predictions_df[predictions_df['is_error_prod']].copy()
        total_errors = len(errors)

        print(f"📊 Production Model Error Analysis ({total_errors:,} errors)")

        # 1. Error types analysis
        error_types = {}
        m_to_w_errors = errors[(errors['true_gender'] == 'M') & (errors['pred_gender_prod'] == 'W')]
        w_to_m_errors = errors[(errors['true_gender'] == 'W') & (errors['pred_gender_prod'] == 'M')]

        error_types['M_predicted_as_W'] = {
            'count': len(m_to_w_errors),
            'percentage': len(m_to_w_errors) / total_errors * 100,
            'examples': m_to_w_errors['primaryName'].head(10).tolist()
        }

        error_types['W_predicted_as_M'] = {
            'count': len(w_to_m_errors),
            'percentage': len(w_to_m_errors) / total_errors * 100,
            'examples': w_to_m_errors['primaryName'].head(10).tolist()
        }

        # 2. High confidence errors (most problematic)
        high_conf_errors = errors[errors['confidence'] > 0.8]
        error_types['high_confidence_errors'] = {
            'count': len(high_conf_errors),
            'percentage': len(high_conf_errors) / total_errors * 100,
            'examples': high_conf_errors['primaryName'].head(10).tolist(),
            'priority': 'CRITICAL'
        }

        # 3. Name length analysis
        errors['name_length'] = errors['primaryName'].str.len()
        length_dist = errors['name_length'].value_counts().sort_index()

        # 4. Unicode/international names
        errors['has_unicode'] = errors['primaryName'].str.contains(r'[^\x00-\x7F]')
        unicode_error_rate = errors['has_unicode'].mean()

        # 5. Complex names (multiple words, hyphens, apostrophes)
        errors['is_complex'] = (
            errors['primaryName'].str.contains(' ') |
            errors['primaryName'].str.contains('-') |
            errors['primaryName'].str.contains("'")
        )
        complex_error_rate = errors['is_complex'].mean()

        # 6. Suffix analysis
        suffix_errors = {}
        for i in range(1, 5):
            suffixes = errors['primaryName'].str[-i:].value_counts()
            suffix_errors[f'{i}_char_suffixes'] = suffixes.head(10).to_dict()

        error_analysis = {
            'total_errors': total_errors,
            'error_rate': total_errors / len(predictions_df),
            'error_types': error_types,
            'name_characteristics': {
                'mean_length': float(errors['name_length'].mean()),
                'unicode_error_rate': float(unicode_error_rate),
                'complex_name_error_rate': float(complex_error_rate),
                'length_distribution': length_dist.head(20).to_dict()
            },
            'suffix_patterns': suffix_errors
        }

        print(f"   📊 Error Breakdown:")
        print(f"      Male → Female: {error_types['M_predicted_as_W']['count']:,} ({error_types['M_predicted_as_W']['percentage']:.1f}%)")
        print(f"      Female → Male: {error_types['W_predicted_as_M']['count']:,} ({error_types['W_predicted_as_M']['percentage']:.1f}%)")
        print(f"      High Confidence: {error_types['high_confidence_errors']['count']:,} ({error_types['high_confidence_errors']['percentage']:.1f}%)")
        print(f"   🌍 International Names Error Rate: {unicode_error_rate:.1%}")
        print(f"   🔗 Complex Names Error Rate: {complex_error_rate:.1%}")
        print(f"   📏 Mean Error Name Length: {errors['name_length'].mean():.1f} chars")

        self.analysis_results['error_patterns'] = error_analysis
        return error_analysis

    def generate_v4_priorities(self):
        """Generate specific V4 development priorities based on production analysis."""
        print(f"\n💡 Generating V4 development priorities...")

        # Extract key metrics from analysis
        current_accuracy = self.analysis_results['performance']['production_threshold_0.48']['accuracy']
        error_patterns = self.analysis_results['error_patterns']

        # Calculate improvement potential
        accuracy_gap = 0.94 - current_accuracy
        total_errors = error_patterns['total_errors']
        high_conf_errors = error_patterns['error_types']['high_confidence_errors']['count']
        unicode_error_rate = error_patterns['name_characteristics']['unicode_error_rate']
        complex_error_rate = error_patterns['name_characteristics']['complex_name_error_rate']

        # Priority calculation
        priorities = []

        # Priority 1: High confidence errors (immediate impact)
        if high_conf_errors > total_errors * 0.1:  # >10% of errors are high confidence
            priorities.append({
                'priority': 1,
                'component': 'Uncertainty Calibration',
                'issue': f'{high_conf_errors} high-confidence errors ({high_conf_errors/total_errors*100:.1f}% of errors)',
                'solution': 'Implement uncertainty-aware training with evidential loss',
                'potential_gain': f'{high_conf_errors/len(predictions_df)*100:.1f}% accuracy improvement',
                'v4_architecture': 'V4.4 - Uncertainty-aware components'
            })

        # Priority 2: International names (if significant)
        if unicode_error_rate > 0.3:  # >30% of errors have Unicode
            priorities.append({
                'priority': 2,
                'component': 'Unicode Handling',
                'issue': f'{unicode_error_rate:.1%} of errors involve international names',
                'solution': 'Multi-lingual character embeddings + cultural context modeling',
                'potential_gain': f'{unicode_error_rate * error_patterns["error_rate"] * 100:.1f}% accuracy improvement',
                'v4_architecture': 'V4.3 - Cross-cultural features'
            })

        # Priority 3: Character sequence modeling
        priorities.append({
            'priority': 3,
            'component': 'Character Modeling',
            'issue': 'BiLSTM limitations in long-range dependencies',
            'solution': 'Character-level Transformer with position encoding',
            'potential_gain': 'Est. 1.0-1.5% accuracy improvement',
            'v4_architecture': 'V4.1 - Character Transformer'
        })

        # Priority 4: Complex names
        if complex_error_rate > 0.25:
            priorities.append({
                'priority': 4,
                'component': 'Hierarchical Processing',
                'issue': f'{complex_error_rate:.1%} of errors involve complex names',
                'solution': 'Multi-scale attention + name component modeling',
                'potential_gain': f'{complex_error_rate * error_patterns["error_rate"] * 100:.1f}% accuracy improvement',
                'v4_architecture': 'V4.2 - Hierarchical attention'
            })

        v4_roadmap = {
            'current_performance': {
                'accuracy': current_accuracy,
                'target': 0.94,
                'gap': accuracy_gap,
                'error_count': total_errors
            },
            'priorities': priorities,
            'recommended_sequence': [
                'Phase 1: Implement V4.4 (Uncertainty) - Address high-confidence errors',
                'Phase 2: Implement V4.3 (Cross-cultural) - International names',
                'Phase 3: Implement V4.1 (Transformer) - Core architecture upgrade',
                'Phase 4: Implement V4.2 (Hierarchical) - Complex name handling',
                'Phase 5: Ensemble best V4 variants'
            ],
            'success_metrics': {
                'tier_1': 'Reduce high-confidence errors by 50%',
                'tier_2': 'Improve international name accuracy by 3%',
                'tier_3': 'Achieve >94% overall accuracy',
                'tier_4': 'Maintain bias ratio 0.95-1.05'
            }
        }

        print(f"🎯 V4 DEVELOPMENT ROADMAP")
        print(f"=" * 50)
        for i, phase in enumerate(v4_roadmap['recommended_sequence'], 1):
            print(f"{i}. {phase}")

        print(f"\n🔑 TOP PRIORITIES:")
        for priority in priorities[:3]:  # Top 3
            print(f"   {priority['priority']}. {priority['component']}: {priority['issue']}")
            print(f"      → {priority['solution']}")
            print(f"      → Potential gain: {priority['potential_gain']}")

        self.analysis_results['v4_roadmap'] = v4_roadmap
        return v4_roadmap

    def save_analysis_results(self, output_dir="./v4_analysis"):
        """Save comprehensive analysis results."""
        os.makedirs(output_dir, exist_ok=True)

        # Save main analysis
        with open(f"{output_dir}/production_v3_analysis.json", 'w') as f:
            json.dump(self.analysis_results, f, indent=2, default=str)

        # Save detailed predictions if available
        if hasattr(self, 'predictions_df'):
            self.predictions_df.to_csv(f"{output_dir}/production_detailed_predictions.csv", index=False)

        print(f"\n💾 Analysis results saved to: {output_dir}/")
        print(f"   📁 production_v3_analysis.json")
        print(f"   📁 production_detailed_predictions.csv")

        return output_dir

    def analyze_preprocessing_mismatch_impact(self, test_df):
        """Analyze preprocessing mismatch with intelligent quality assessment."""
        print(f"\n🔍 Analyzing preprocessing mismatch impact with quality assessment...")

        # Test with training preprocessing
        training_dataset = ImprovedNameGenderDataset(
            test_df, self.evaluator.preprocessor, self.feature_extractor, mode='test'
        )
        training_results = self.evaluator.evaluate_dataset(training_dataset)

        # Test with production preprocessing
        try:
            import sys
            sys.path.append('./scripts')
            from final_predictor import ProductionRobustPreprocessor

            production_preprocessor = ProductionRobustPreprocessor(self.evaluator.preprocessor)
            production_dataset = ImprovedNameGenderDataset(
                test_df, production_preprocessor, self.feature_extractor, mode='test'
            )
            production_results = self.evaluator.evaluate_dataset(production_dataset)

            # Calculate performance impact
            accuracy_mismatch = training_results['accuracy'] - production_results['accuracy']
            f1_mismatch = training_results['f1'] - production_results['f1']

            # Intelligent assessment: determine if mismatch is beneficial or harmful
            mismatch_assessment = self._assess_preprocessing_mismatch_quality(
                test_df, training_results, production_results, accuracy_mismatch
            )

            mismatch_analysis = {
                'training_preprocessing': {
                    'accuracy': float(training_results['accuracy']),
                    'f1_score': float(training_results['f1']),
                    'precision': float(training_results['precision']),
                    'recall': float(training_results['recall'])
                },
                'production_preprocessing': {
                    'accuracy': float(production_results['accuracy']),
                    'f1_score': float(production_results['f1']),
                    'precision': float(production_results['precision']),
                    'recall': float(production_results['recall'])
                },
                'mismatch_impact': {
                    'accuracy_loss': float(accuracy_mismatch),
                    'f1_loss': float(f1_mismatch),
                    'relative_accuracy_loss': float(accuracy_mismatch / training_results['accuracy'] * 100),
                    'severity': mismatch_assessment['severity'],
                    'mismatch_type': mismatch_assessment['type'],
                    'recommended_action': mismatch_assessment['action']
                },
                'intelligent_assessment': mismatch_assessment
            }

            self.preprocessing_mismatch_impact = accuracy_mismatch

            # Print intelligent analysis
            print(f"📊 INTELLIGENT PREPROCESSING MISMATCH ANALYSIS:")
            print(f"   Training preprocessing:   Acc={training_results['accuracy']:.4f}, F1={training_results['f1']:.4f}")
            print(f"   Production preprocessing: Acc={production_results['accuracy']:.4f}, F1={production_results['f1']:.4f}")
            print(f"   Performance impact:      {accuracy_mismatch:+.4f} accuracy, {f1_mismatch:+.4f} F1")
            print(f"   Assessment:              {mismatch_assessment['type']} mismatch")
            print(f"   Recommended action:      {mismatch_assessment['action']}")
            print(f"   Severity:                {mismatch_assessment['severity']}")

            if mismatch_assessment['type'] == 'BENEFICIAL':
                print(f"   🟢 GOOD NEWS: Production preprocessing appears to be more robust!")
                print(f"      Consider retraining model with production preprocessing for better international support")
            elif mismatch_assessment['type'] == 'HARMFUL':
                print(f"   🔴 ISSUE: Production preprocessing is degrading performance")
                print(f"      Fix by aligning production to training preprocessing")
            else:
                print(f"   ⚪ NEUTRAL: Preprocessing approaches have similar quality")
                print(f"      Align for consistency")

            return mismatch_analysis, production_results

        except ImportError as e:
            print(f"⚠️  Could not import ProductionRobustPreprocessor: {e}")
            print(f"   Using training preprocessing only")
            self.preprocessing_mismatch_impact = 0.0
            return None, training_results

    def _assess_preprocessing_mismatch_quality(self, test_df, training_results, production_results, accuracy_diff):
        """Intelligently assess whether preprocessing mismatch is beneficial or harmful."""

        # Analyze specific name characteristics in the dataset
        unicode_names = test_df[test_df['primaryName'].str.contains(r'[^\x00-\x7F]', na=False)]
        complex_names = test_df[test_df['primaryName'].str.contains(r'[-\'\s]', na=False)]

        unicode_ratio = len(unicode_names) / len(test_df) if len(test_df) > 0 else 0
        complex_ratio = len(complex_names) / len(test_df) if len(test_df) > 0 else 0

        # Assessment logic
        assessment = {
            'type': 'NEUTRAL',
            'action': 'ALIGN_FOR_CONSISTENCY',
            'severity': 'LOW',
            'rationale': [],
            'unicode_ratio': unicode_ratio,
            'complex_ratio': complex_ratio
        }

        if accuracy_diff > 0.02:  # Training significantly better
            if unicode_ratio > 0.1:  # Dataset has substantial international names
                # This suggests production preprocessing might be TOO aggressive
                assessment.update({
                    'type': 'HARMFUL',
                    'action': 'ALIGN_TO_TRAINING',
                    'severity': 'HIGH',
                    'rationale': [
                        'Production preprocessing loses accuracy on international names',
                        'Training preprocessing better preserves important character information',
                        'Production preprocessing may be over-normalizing Unicode characters'
                    ]
                })
            else:
                # Mostly ASCII names, so training is just better
                assessment.update({
                    'type': 'HARMFUL',
                    'action': 'ALIGN_TO_TRAINING',
                    'severity': 'MEDIUM',
                    'rationale': [
                        'Training preprocessing performs better on this dataset',
                        'Production preprocessing appears to be degrading quality'
                    ]
                })

        elif accuracy_diff < -0.01:  # Production better
            if unicode_ratio > 0.05 or complex_ratio > 0.2:
                # Production better on complex/international names - this is good!
                assessment.update({
                    'type': 'BENEFICIAL',
                    'action': 'RETRAIN_WITH_PRODUCTION',
                    'severity': 'OPPORTUNITY',
                    'rationale': [
                        'Production preprocessing better handles international/complex names',
                        'Production includes robust Unicode normalization',
                        'Production includes encoding error recovery',
                        'Retraining with production preprocessing could improve robustness'
                    ]
                })
            else:
                # Production better even on simple names
                assessment.update({
                    'type': 'BENEFICIAL',
                    'action': 'RETRAIN_WITH_PRODUCTION',
                    'severity': 'OPPORTUNITY',
                    'rationale': [
                        'Production preprocessing is more robust overall',
                        'Consider adopting production preprocessing as standard'
                    ]
                })
        else:
            # Roughly equivalent performance
            if unicode_ratio > 0.1:
                assessment['rationale'].append('Similar performance but should test more thoroughly on international names')
            assessment['rationale'].append('Align preprocessing for consistency and reproducibility')

        return assessment

    def run_complete_analysis(self, test_dataset_path=None):
        """Run complete production model analysis with preprocessing mismatch detection."""
        print("🚀 PRODUCTION V3 MODEL ANALYSIS FOR V4 DEVELOPMENT")
        print("🔍 Including Preprocessing Mismatch Detection")
        print("=" * 65)

        # Verify and load model
        if not self.verify_production_model():
            return None

        if not self.load_production_model():
            return None

        # Find or use provided test dataset
        if test_dataset_path and os.path.exists(test_dataset_path):
            test_df = pd.read_csv(test_dataset_path)
            dataset_path = test_dataset_path
            print(f"✅ Using provided dataset: {test_dataset_path} ({len(test_df):,} samples)")
        else:
            dataset_path, test_df = self.find_test_dataset()
            if test_df is None:
                return None

        # NEW: Analyze preprocessing mismatch first
        mismatch_analysis, production_results = self.analyze_preprocessing_mismatch_impact(test_df)

        # Store mismatch analysis
        if mismatch_analysis:
            self.analysis_results['preprocessing_mismatch'] = mismatch_analysis

        # Run performance analysis (now using production preprocessing results)
        self.predictions_df = self.analyze_production_performance_with_mismatch(
            test_df, dataset_path, production_results
        )

        # Run error pattern analysis
        self.analyze_error_patterns(self.predictions_df)

        # Generate V4 development priorities (now considering preprocessing)
        self.generate_v4_priorities()

        # Save results
        output_dir = self.save_analysis_results()

        print(f"\n✅ PRODUCTION ANALYSIS COMPLETE!")
        print(f"🎯 Ready to begin V4 development with data-driven priorities")

        # Print preprocessing mismatch summary
        if hasattr(self, 'preprocessing_mismatch_impact') and abs(self.preprocessing_mismatch_impact) > 0.01:
            print(f"\n⚠️  CRITICAL FINDING: Preprocessing mismatch detected!")
            print(f"   Immediate accuracy gain possible: {abs(self.preprocessing_mismatch_impact)*100:.1f}%")
            print(f"   Action: Align production and training preprocessing")

        return self.analysis_results

    def analyze_production_performance_with_mismatch(self, test_df, dataset_path, results):
        """Analyze production performance using already computed results."""
        print(f"\n🔄 Analyzing production threshold performance...")

        # Apply production threshold (0.48 instead of 0.5)
        production_predictions = (np.array(results['probabilities']) >= self.production_threshold).astype(int)

        # Calculate metrics with production threshold
        prod_accuracy = accuracy_score(results['targets'], production_predictions)
        prod_precision, prod_recall, prod_f1, _ = precision_recall_fscore_support(
            results['targets'], production_predictions, average='binary'
        )

        # Standard threshold for comparison
        std_accuracy = accuracy_score(results['targets'], results['predictions'])
        std_precision, std_recall, std_f1, _ = precision_recall_fscore_support(
            results['targets'], results['predictions'], average='binary'
        )


def main():
    """Main execution function."""
    import argparse

    parser = argparse.ArgumentParser(description="Production V3 Model Analysis")
    parser.add_argument('--test_data', help='Path to test dataset (optional)')
    parser.add_argument('--output_dir', default='./v4_analysis', help='Output directory')

    args = parser.parse_args()

    # Create analyzer
    analyzer = ProductionV3Analyzer()

    # Run analysis
    results = analyzer.run_complete_analysis(args.test_data)

    if results:
        print(f"\n🚀 Next Steps:")
        print(f"   1. Review analysis results in: {args.output_dir}/")
        print(f"   2. Begin V4.4 (Uncertainty) implementation")
        print(f"   3. Set up V4 experiment infrastructure")
        print(f"   4. Start systematic V4 architecture development")


if __name__ == "__main__":
    main()
