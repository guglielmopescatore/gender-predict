"""
Unified model evaluation system.

This module provides a unified interface for evaluating gender prediction models,
combining functionality from both enhanced and standard evaluation scripts.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import os
import argparse
from tqdm import tqdm

# Set matplotlib backend for headless environments
import matplotlib
matplotlib.use('Agg')

class ModelEvaluator:
    """
    Unified evaluator for all model types.
    
    Supports:
    - GenderPredictor (Round 0/1)
    - GenderPredictorEnhanced (Round 2) 
    - GenderPredictorV3 (Round 3)
    """
    
    def __init__(self, model, preprocessor, device='cuda'):
        self.model = model
        self.preprocessor = preprocessor
        self.device = device
        self.model.to(device)
        self.model.eval()
    
    def evaluate_dataset(self, dataset, batch_size=128):
        """
        Evaluate model on a dataset.
        
        Args:
            dataset: PyTorch Dataset
            batch_size: Batch size for evaluation
            
        Returns:
            dict: Evaluation metrics
        """
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        all_preds = []
        all_targets = []
        all_probs = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                # Handle different model input requirements
                if hasattr(batch, 'keys') and 'first_suffix' in batch:
                    # V3 model with advanced features
                    outputs = self.model(
                        batch['first_name'].to(self.device),
                        batch['last_name'].to(self.device),
                        batch['first_suffix'].to(self.device),
                        batch['last_suffix'].to(self.device),
                        batch['phonetic_features'].to(self.device)
                    )
                else:
                    # Standard models
                    outputs = self.model(
                        batch['first_name'].to(self.device),
                        batch['last_name'].to(self.device)
                    )
                
                # Convert outputs to probabilities
                probs = torch.sigmoid(outputs)
                preds = (probs >= 0.5).long()
                
                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(batch['gender'].cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_targets, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets, all_preds, average='binary'
        )
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'predictions': all_preds,
            'probabilities': all_probs,
            'targets': all_targets
        }
    
    def detailed_bias_analysis(self, targets, predictions):
        """
        Perform detailed bias analysis.
        
        Args:
            targets: True labels
            predictions: Model predictions
            
        Returns:
            dict: Bias metrics
        """
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(targets, predictions)
        tn, fp, fn, tp = cm.ravel()
        
        # Calculate bias metrics
        m_error_rate = fp / (tn + fp) if (tn + fp) > 0 else 0
        w_error_rate = fn / (tp + fn) if (tp + fn) > 0 else 0
        bias_ratio = m_error_rate / w_error_rate if w_error_rate > 0 else float('inf')
        
        return {
            'confusion_matrix': cm,
            'male_error_rate': m_error_rate,
            'female_error_rate': w_error_rate,
            'bias_ratio': bias_ratio,
            'equality_of_opportunity': abs(tn/(tn+fp) - tp/(tp+fn)) if (tn+fp) > 0 and (tp+fn) > 0 else 0
        }
    
    @classmethod
    def from_checkpoint(cls, checkpoint_path, preprocessor_path, device='cuda'):
        """
        Load evaluator from model checkpoint.

        Args:
            checkpoint_path: Path to model checkpoint
            preprocessor_path: Path to preprocessor pickle
            device: Device to load model on

        Returns:
            ModelEvaluator instance
        """
        # Load preprocessor - determine which class was used
        import pickle
        with open(preprocessor_path, 'rb') as f:
            preprocessor_data = pickle.load(f)

        # Check which preprocessor type based on saved data
        if 'normalize_diacritics' in preprocessor_data or 'handle_hyphens' in preprocessor_data:
            # ImprovedNamePreprocessor
            from ..data.improved_preprocessing import ImprovedNamePreprocessor
            preprocessor = ImprovedNamePreprocessor.load(preprocessor_path)
        else:
            # Standard NamePreprocessor
            from ..data.preprocessing import NamePreprocessor
            preprocessor = NamePreprocessor.load(preprocessor_path)

        # Load model checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Subito dopo aver caricato il checkpoint, aggiungi:
        print("Checkpoint keys:", checkpoint.keys())
        print("suffix_vocab_size in checkpoint:", checkpoint.get('suffix_vocab_size', 'NOT FOUND'))
        print("Actual suffix embedding shape:", checkpoint['model_state_dict']['suffix_embedding.weight'].shape)
        # Determine model type and create appropriate model
        if 'suffix_vocab_size' in checkpoint:
            # V3 model - use actual embedding size instead of metadata
            actual_suffix_vocab_size = checkpoint['model_state_dict']['suffix_embedding.weight'].shape[0]
            from ..models import GenderPredictorV3
            model = GenderPredictorV3(
                vocab_size=checkpoint['vocab_size'],
                suffix_vocab_size=actual_suffix_vocab_size,  # ← USA LA DIMENSIONE REALE
                embedding_dim=checkpoint.get('embedding_dim', 32),
                hidden_size=checkpoint.get('hidden_size', 128),
                n_layers=checkpoint.get('n_layers', 2)
            )
        elif 'n_layers' in checkpoint and checkpoint.get('n_layers', 1) > 1:
            # Enhanced model
            from ..models import GenderPredictorEnhanced
            model = GenderPredictorEnhanced(
                vocab_size=checkpoint['vocab_size'],
                embedding_dim=checkpoint.get('embedding_dim', 16),
                hidden_size=checkpoint.get('hidden_size', 80),
                n_layers=checkpoint.get('n_layers', 2)
            )
        else:
            # Base model
            from ..models import GenderPredictor
            model = GenderPredictor(
                vocab_size=checkpoint['vocab_size'],
                embedding_dim=checkpoint.get('embedding_dim', 16),
                hidden_size=checkpoint.get('hidden_size', 64)
            )

        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])

        return cls(model, preprocessor, device)

def main():
    """Command line interface for model evaluation."""
    parser = argparse.ArgumentParser(description='Evaluate gender prediction models')
    
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
    
    # Create dataset (this would need to be adapted based on model type)
    from ..data import NameGenderDataset
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
        # Convert numpy types for JSON serialization
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
