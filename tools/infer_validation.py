"""infer_validation.py
────────────────────────────────────────────────────────────────────────────
Run validation inference for an existing experiment and generate probability CSV.
"""
import argparse, json, os, sys, pickle
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Project imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gender_predict.models import create_model
from gender_predict.data import (
    NameGenderDataset, ImprovedNameGenderDataset,
    NameFeatureExtractor
)
# Import specifici per preprocessor
from gender_predict.data.preprocessing import NamePreprocessor
from gender_predict.data.improved_preprocessing import ImprovedNamePreprocessor

def parse_args():
    p = argparse.ArgumentParser(description="Run validation inference for a saved experiment")
    p.add_argument("--exp_dir", required=True, help="Path to the experiment folder")
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--out_csv", default=None, help="Path to output CSV (default logs/val_probs_labels.csv)")
    return p.parse_args()

def load_experiment_components(exp_dir: Path):
    """Load all necessary components from experiment."""
    # Load parameters
    params_path = exp_dir / "parameters.json"
    if not params_path.exists():
        sys.exit(f"❌ parameters.json not found in {exp_dir}")
    params = json.loads(params_path.read_text())

    # Load preprocessor usando il metodo load() della classe
    preprocessor_path = exp_dir / "preprocessor.pkl"
    if not preprocessor_path.exists():
        sys.exit(f"❌ preprocessor.pkl not found in {exp_dir}")

    # Determina quale preprocessor usare basandosi sui parametri
    if params.get("round", 0) >= 3 and params.get("advanced_preprocessing", False):
        from gender_predict.data.improved_preprocessing import ImprovedNamePreprocessor
        preprocessor = ImprovedNamePreprocessor.load(str(preprocessor_path))
    else:
        from gender_predict.data.preprocessing import NamePreprocessor
        preprocessor = NamePreprocessor.load(str(preprocessor_path))

    print(f"   Preprocessor type: {type(preprocessor).__name__}")
    print(f"   Vocab size: {preprocessor.vocab_size}")

    # Load feature extractor if V3
    feature_extractor = None
    if params.get("round", 0) >= 3:
        fe_path = exp_dir / "feature_extractor.pkl"
        if fe_path.exists():
            with open(fe_path, 'rb') as f:
                feature_extractor = pickle.load(f)
            print(f"   Feature extractor loaded")
        else:
            print("⚠️  feature_extractor.pkl not found for V3 model, creating new one")
            from gender_predict.data import NameFeatureExtractor
            feature_extractor = NameFeatureExtractor()

            # Se il modello è V3, il feature extractor è essenziale
            # Prova a ricostruire suffix_to_idx dalla dimensione del modello
            if 'suffix_vocab_size' in params:
                print(f"   Initializing feature extractor with suffix_vocab_size={params['suffix_vocab_size']}")

    return params, preprocessor, feature_extractor

def recreate_validation_split(params: dict):
    """Recreate the exact validation split used during training."""
    data_file = params["data_file"]
    seed = params.get("seed", 42)

    # Load full dataset
    df = pd.read_csv(data_file)

    # Recreate same splits with same seed
    train_val_df, test_df = train_test_split(
        df, test_size=0.1, random_state=seed, stratify=df['gender']
    )
    train_df, val_df = train_test_split(
        train_val_df, test_size=0.1, random_state=seed, stratify=train_val_df['gender']
    )

    return val_df

def create_validation_dataset(val_df, params, preprocessor, feature_extractor):
    """Create the validation dataset."""
    round_id = params.get("round", 0)

    if round_id >= 3 and feature_extractor is not None:
        # V3 dataset
        dataset = ImprovedNameGenderDataset(
            val_df,
            preprocessor,
            feature_extractor,
            mode='val',
            augmenter=None,
            augment_prob=0.0
        )
    else:
        # Standard dataset
        dataset = NameGenderDataset(
            val_df,
            preprocessor,
            mode='val'
        )

    return dataset

def infer_model_dimensions_from_checkpoint(checkpoint, model_type):
    """Infer model dimensions from checkpoint state_dict."""
    state_dict = checkpoint['model_state_dict']

    # Common dimensions
    dims = {}

    # Vocab size from char_embedding
    if 'char_embedding.weight' in state_dict:
        dims['vocab_size'] = state_dict['char_embedding.weight'].shape[0]

    # Embedding dim
    if 'char_embedding.weight' in state_dict:
        dims['embedding_dim'] = state_dict['char_embedding.weight'].shape[1]

    # Hidden size from LSTM
    if 'firstname_lstm.weight_ih_l0' in state_dict:
        dims['hidden_size'] = state_dict['firstname_lstm.weight_ih_l0'].shape[0] // 4

    # Number of layers
    n_layers = 1
    for i in range(10):
        if f'firstname_lstm.weight_ih_l{i}' in state_dict:
            n_layers = i + 1
        else:
            break
    dims['n_layers'] = n_layers

    # V3 specific
    if model_type == 'v3':
        if 'suffix_embedding.weight' in state_dict:
            dims['suffix_vocab_size'] = state_dict['suffix_embedding.weight'].shape[0]

        # Try to infer num_attention_heads
        if 'firstname_attention.query.weight' in state_dict:
            query_weight = state_dict['firstname_attention.query.weight']
            # Se hidden_size * 2 -> hidden_size, e num_heads divide hidden_size
            input_size = query_weight.shape[1]
            output_size = query_weight.shape[0]
            if output_size % 4 == 0:
                dims['num_attention_heads'] = 4
            elif output_size % 8 == 0:
                dims['num_attention_heads'] = 8
            else:
                dims['num_attention_heads'] = 4  # default

    return dims

def load_model(exp_dir: Path, params: dict, device: torch.device):
    """Load the trained model."""
    # Determine model type
    if params.get("round", 0) >= 3:
        model_type = 'v3'
    elif params.get("round", 0) >= 2:
        model_type = 'enhanced'
    else:
        model_type = 'base'

    # Load checkpoint
    ckpt_path = exp_dir / "models" / "model.pth"
    if not ckpt_path.exists():
        sys.exit(f"❌ model.pth not found in {exp_dir}/models")

    checkpoint = torch.load(ckpt_path, map_location=device)

    # Infer dimensions from checkpoint
    inferred_dims = infer_model_dimensions_from_checkpoint(checkpoint, model_type)

    # Merge with saved params (inferred takes precedence)
    model_params = {
        'vocab_size': inferred_dims.get('vocab_size', checkpoint.get('vocab_size', 100)),
        'embedding_dim': inferred_dims.get('embedding_dim', checkpoint.get('embedding_dim', 16)),
        'hidden_size': inferred_dims.get('hidden_size', checkpoint.get('hidden_size', 64)),
    }

    if model_type in ['enhanced', 'v3']:
        model_params['n_layers'] = inferred_dims.get('n_layers', checkpoint.get('n_layers', 1))
        model_params['dropout_rate'] = params.get('dropout', 0.3)

    if model_type == 'enhanced':
        model_params['dual_input'] = checkpoint.get('dual_input', params.get('dual_input', True))

    if model_type == 'v3':
        model_params['suffix_vocab_size'] = inferred_dims.get('suffix_vocab_size', 100)
        model_params['num_attention_heads'] = inferred_dims.get('num_attention_heads', 4)

    print(f"   Model type: {model_type}")
    print(f"   Inferred params: {model_params}")

    # Create model
    model = create_model(model_type, **model_params)

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return model

@torch.inference_mode()
def run_inference(model, dataloader, device, is_v3=False):
    """Run inference and collect probabilities and labels."""
    all_probs = []
    all_labels = []

    for batch in tqdm(dataloader, desc="Validation Inference"):
        if is_v3:
            # V3 model
            outputs = model(
                batch['first_name'].to(device),
                batch['last_name'].to(device),
                batch['first_suffix'].to(device),
                batch['last_suffix'].to(device),
                batch['phonetic_features'].to(device)
            )
        else:
            # Standard models
            outputs = model(
                batch['first_name'].to(device),
                batch['last_name'].to(device)
            )

        # Convert to probabilities
        probs = torch.sigmoid(outputs).cpu().numpy()
        labels = batch['gender'].cpu().numpy()

        all_probs.extend(probs.tolist())
        all_labels.extend(labels.tolist())

    return np.array(all_probs), np.array(all_labels)

def main():
    args = parse_args()
    exp_dir = Path(args.exp_dir)

    if not exp_dir.is_dir():
        sys.exit(f"❌ Experiment folder {exp_dir} not found")

    print(f"📁 Loading experiment from: {exp_dir}")

    # Load components
    params, preprocessor, feature_extractor = load_experiment_components(exp_dir)

    # Recreate validation split
    print("📊 Recreating validation split...")
    val_df = recreate_validation_split(params)
    print(f"   Validation samples: {len(val_df)}")

    # Create dataset
    val_dataset = create_validation_dataset(val_df, params, preprocessor, feature_extractor)

    # Create dataloader
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Using device: {device}")

    # Load model
    print("🤖 Loading model...")
    model = load_model(exp_dir, params, device)

    # Run inference
    print("🔮 Running inference...")
    is_v3 = params.get("round", 0) >= 3
    probs, labels = run_inference(model, val_loader, device, is_v3)

    # Save results
    out_csv = Path(args.out_csv) if args.out_csv else exp_dir / "logs" / "val_probs_labels.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame({
        'prob': probs,
        'label': labels
    })

    results_df.to_csv(out_csv, index=False)

    # Print summary
    print(f"\n✅ Saved validation probabilities to: {out_csv}")
    print(f"   Total samples: {len(labels)}")
    print(f"   Label distribution: {np.bincount(labels.astype(int))}")
    print(f"   Prob range: [{probs.min():.4f}, {probs.max():.4f}]")
    print(f"   Mean prob: {probs.mean():.4f}")

if __name__ == "__main__":
    main()
