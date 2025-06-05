#!/usr/bin/env python3
"""
Unified evaluation script for gender prediction models.
"""

import argparse
import sys
import os
import json
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import torch

from gender_predict.evaluation import ModelEvaluator
from gender_predict.data import NameGenderDataset

def main():
    parser = argparse.ArgumentParser(description="Evaluate gender prediction models")
    
    parser.add_argument('--model', required=True, help='Path to model checkpoint')
    parser.add_argument('--preprocessor', required=True, help='Path to preprocessor')
    parser.add_argument('--test_data', required=True, help='Path to test dataset CSV')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--device', default='cuda', help='Device to use')
    parser.add_argument('--output_dir', default='evaluation_results', help='Output directory')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load evaluator
    evaluator = ModelEvaluator.from_checkpoint(
        args.model, args.preprocessor, args.device
    )
    
    # Load test dataset
    test_df = pd.read_csv(args.test_data)

    # Check if we need advanced features for V3 model
    from gender_predict.data.datasets import NameGenderDataset, ImprovedNameGenderDataset
    checkpoint = torch.load(args.model, map_location='cpu')
    is_v3_model = 'suffix_vocab_size' in checkpoint

    if is_v3_model:
        # V3 model needs feature extractor
        from gender_predict.data.feature_extraction import NameFeatureExtractor
        feature_extractor = NameFeatureExtractor()
        test_dataset = ImprovedNameGenderDataset(
            test_df, evaluator.preprocessor,
            feature_extractor=feature_extractor,
            mode='test'
        )
        print("Using ImprovedNameGenderDataset for V3 model")
    else:
        # Standard model
        test_dataset = NameGenderDataset(test_df, evaluator.preprocessor, mode='test')
        print("Using standard NameGenderDataset")
    
    # Evaluate
    results = evaluator.evaluate_dataset(test_dataset, args.batch_size)
    
    # Bias analysis
    bias_results = evaluator.detailed_bias_analysis(
        results['targets'], results['predictions']
    )

    # 1. PRIMA: Salva detailed predictions
    predictions_data = []
    for i in range(len(results['targets'])):
        predictions_data.append({
            'true_gender': 'W' if results['targets'][i] == 1 else 'M',
            'pred_gender': 'W' if results['predictions'][i] == 1 else 'M',
            'prob': float(results['probabilities'][i])
        })

    predictions_df = pd.DataFrame(predictions_data)
    predictions_file = os.path.join(args.output_dir, 'detailed_predictions.csv')
    predictions_df.to_csv(predictions_file, index=False)

    # 2. POI: Crea serializable_results con tutti i dati
    # Convert numpy arrays to JSON serializable types
    confusion_matrix = bias_results['confusion_matrix']
    if hasattr(confusion_matrix, 'tolist'):
        confusion_matrix = confusion_matrix.tolist()

    serializable_results = {
        'accuracy': float(results['accuracy']),
        'precision': float(results['precision']),
        'recall': float(results['recall']),
        'f1': float(results['f1']),
        'bias_ratio': float(bias_results['bias_ratio']),
        'male_error_rate': float(bias_results['male_error_rate']),
        'female_error_rate': float(bias_results['female_error_rate']),
        'total_samples': int(len(results['targets'])),
        'male_samples': int((np.array(results['targets']) == 0).sum()),
        'female_samples': int((np.array(results['targets']) == 1).sum()),
        'confusion_matrix': confusion_matrix,
        'equality_of_opportunity': float(bias_results['equality_of_opportunity'])
    }

    # 3. INFINE: Salva JSON
    output_file = os.path.join(args.output_dir, 'evaluation_results.json')
    with open(output_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)

    print(f"✅ Evaluation completed. Results saved to {output_file}")
    print(f"💾 Detailed predictions saved to {predictions_file}")
    print(f"   Accuracy: {results['accuracy']:.4f}")
    print(f"   F1 Score: {results['f1']:.4f}")
    print(f"   Bias Ratio: {bias_results['bias_ratio']:.4f}")

if __name__ == "__main__":
    main()
