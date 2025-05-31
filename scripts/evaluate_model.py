#!/usr/bin/env python3
"""
Unified evaluation script for gender prediction models.
"""

import argparse
import sys
import os

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
    test_dataset = NameGenderDataset(test_df, evaluator.preprocessor, mode='test')
    
    # Evaluate
    results = evaluator.evaluate_dataset(test_dataset, args.batch_size)
    
    # Bias analysis
    bias_results = evaluator.detailed_bias_analysis(
        results['targets'], results['predictions']
    )
    
    # Save results
    import json
    output_file = os.path.join(args.output_dir, 'evaluation_results.json')
    with open(output_file, 'w') as f:
        serializable_results = {
            'accuracy': float(results['accuracy']),
            'precision': float(results['precision']),
            'recall': float(results['recall']),
            'f1': float(results['f1']),
            'bias_ratio': float(bias_results['bias_ratio']),
            'male_error_rate': float(bias_results['male_error_rate']),
            'female_error_rate': float(bias_results['female_error_rate'])
        }
        json.dump(serializable_results, f, indent=2)
    
    print(f"✅ Evaluation completed. Results saved to {output_file}")
    print(f"   Accuracy: {results['accuracy']:.4f}")
    print(f"   F1 Score: {results['f1']:.4f}")
    print(f"   Bias Ratio: {bias_results['bias_ratio']:.4f}")

if __name__ == "__main__":
    main()
