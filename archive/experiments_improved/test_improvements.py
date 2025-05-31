#!/usr/bin/env python3
"""
Script di test per verificare che tutti i miglioramenti funzionino correttamente.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
from improvements import (
    GenderPredictorV3,
    NameFeatureExtractor,
    NameAugmenter,
    ImprovedAttentionLayer,
    CosineAnnealingWarmupScheduler,
    FocalLossImproved
)


def test_feature_extractor():
    """Test del feature extractor."""
    print("Testing NameFeatureExtractor...")
    
    extractor = NameFeatureExtractor()
    
    # Test nomi italiani
    test_cases = [
        ("Maria", "Expected female suffixes"),
        ("Giuseppe", "Expected male suffixes"),
        ("Anna", "Expected female suffixes"),
        ("Marco", "Expected male suffixes"),
        ("Ekaterina", "Expected female suffixes (Russian)"),
        ("Aleksandr", "Expected male suffixes (Russian)"),
    ]
    
    for name, expected in test_cases:
        suffixes = extractor.extract_suffix_features(name)
        phonetic = extractor.extract_phonetic_features(name)
        
        print(f"\nName: {name}")
        print(f"  Suffixes: {suffixes}")
        print(f"  Phonetic features: {phonetic}")
        print(f"  Expected: {expected}")
    
    print("✓ NameFeatureExtractor test passed!\n")


def test_augmenter():
    """Test del data augmenter."""
    print("Testing NameAugmenter...")
    
    augmenter = NameAugmenter(augment_prob=1.0)  # Always augment for testing
    
    test_names = ["Maria", "Giuseppe", "Elizabeth", "Alexander"]
    
    for name in test_names:
        print(f"\nOriginal: {name}")
        for i in range(3):
            augmented = augmenter.augment(name)
            print(f"  Augmented {i+1}: {augmented}")
    
    print("✓ NameAugmenter test passed!\n")


def test_improved_attention():
    """Test del layer di attenzione migliorato."""
    print("Testing ImprovedAttentionLayer...")
    
    batch_size = 32
    seq_len = 20
    hidden_size = 64
    
    # Create dummy LSTM output
    lstm_output = torch.randn(batch_size, seq_len, hidden_size * 2)
    
    # Test attention
    attention = ImprovedAttentionLayer(hidden_size, num_heads=4)
    output = attention(lstm_output)
    
    assert output.shape == (batch_size, hidden_size * 2), f"Wrong output shape: {output.shape}"
    
    print(f"  Input shape: {lstm_output.shape}")
    print(f"  Output shape: {output.shape}")
    print("✓ ImprovedAttentionLayer test passed!\n")


def test_model_forward():
    """Test del forward pass del modello completo."""
    print("Testing GenderPredictorV3 forward pass...")
    
    # Model parameters
    vocab_size = 60
    suffix_vocab_size = 50
    batch_size = 16
    name_length = 20
    
    # Create model
    model = GenderPredictorV3(
        vocab_size=vocab_size,
        suffix_vocab_size=suffix_vocab_size,
        embedding_dim=32,
        hidden_size=64,
        n_layers=2,
        dropout_rate=0.2,
        num_attention_heads=4
    )
    
    # Create dummy inputs
    first_names = torch.randint(0, vocab_size, (batch_size, name_length))
    last_names = torch.randint(0, vocab_size, (batch_size, name_length))
    first_suffix = torch.randint(0, suffix_vocab_size, (batch_size, 3))
    last_suffix = torch.randint(0, suffix_vocab_size, (batch_size, 3))
    phonetic_features = torch.randn(batch_size, 4)
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(first_names, last_names, first_suffix, last_suffix, phonetic_features)
    
    assert output.shape == (batch_size,), f"Wrong output shape: {output.shape}"
    
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output range: [{output.min():.3f}, {output.max():.3f}]")
    print("✓ GenderPredictorV3 forward pass test passed!\n")


def test_scheduler():
    """Test del learning rate scheduler."""
    print("Testing CosineAnnealingWarmupScheduler...")
    
    # Create dummy optimizer
    model = nn.Linear(10, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Create scheduler
    scheduler = CosineAnnealingWarmupScheduler(
        optimizer,
        warmup_epochs=3,
        max_epochs=20,
        min_lr=1e-6,
        max_lr=1e-3
    )
    
    # Test scheduling
    lrs = []
    for epoch in range(20):
        lr = scheduler.step(epoch)
        lrs.append(lr)
    
    print("  Learning rates by epoch:")
    for i, lr in enumerate(lrs[:10]):  # Show first 10
        print(f"    Epoch {i}: {lr:.6f}")
    print("  ...")
    
    # Verify warmup
    assert lrs[0] < lrs[2], "Warmup not working"
    # Verify decay
    assert lrs[-1] < lrs[5], "Decay not working"
    
    print("✓ CosineAnnealingWarmupScheduler test passed!\n")


def test_focal_loss():
    """Test della focal loss migliorata."""
    print("Testing FocalLossImproved...")
    
    batch_size = 32
    
    # Create dummy data
    logits = torch.randn(batch_size)
    targets = torch.randint(0, 2, (batch_size,)).float()
    
    # Test loss
    criterion = FocalLossImproved(alpha=0.7, gamma=2.0)
    loss = criterion(logits, targets)
    
    assert loss.dim() == 0, "Loss should be scalar"
    assert loss.item() > 0, "Loss should be positive"
    
    print(f"  Loss value: {loss.item():.4f}")
    
    # Test with auto weighting
    criterion_auto = FocalLossImproved(auto_weight=True)
    loss_auto = criterion_auto(logits, targets)
    
    print(f"  Loss with auto-weight: {loss_auto.item():.4f}")
    print("✓ FocalLossImproved test passed!\n")


def test_gradient_flow():
    """Test che il gradiente fluisca correttamente attraverso il modello."""
    print("Testing gradient flow...")
    
    # Setup
    vocab_size = 60
    suffix_vocab_size = 50
    batch_size = 8
    
    model = GenderPredictorV3(
        vocab_size=vocab_size,
        suffix_vocab_size=suffix_vocab_size,
        embedding_dim=16,
        hidden_size=32,
        n_layers=1
    )
    
    criterion = FocalLossImproved()
    
    # Dummy data
    first_names = torch.randint(0, vocab_size, (batch_size, 20))
    last_names = torch.randint(0, vocab_size, (batch_size, 20))
    first_suffix = torch.randint(0, suffix_vocab_size, (batch_size, 3))
    last_suffix = torch.randint(0, suffix_vocab_size, (batch_size, 3))
    phonetic_features = torch.randn(batch_size, 4)
    targets = torch.randint(0, 2, (batch_size,)).float()
    
    # Forward and backward
    model.train()
    output = model(first_names, last_names, first_suffix, last_suffix, phonetic_features)
    loss = criterion(output, targets)
    loss.backward()
    
    # Check gradients
    grad_norms = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_norms.append(grad_norm)
            if grad_norm == 0:
                print(f"  WARNING: Zero gradient in {name}")
    
    avg_grad_norm = np.mean(grad_norms)
    print(f"  Average gradient norm: {avg_grad_norm:.6f}")
    assert avg_grad_norm > 0, "No gradients flowing!"
    
    print("✓ Gradient flow test passed!\n")


def run_all_tests():
    """Esegue tutti i test."""
    print("=" * 60)
    print("RUNNING IMPROVEMENT TESTS")
    print("=" * 60)
    
    tests = [
        test_feature_extractor,
        test_augmenter,
        test_improved_attention,
        test_model_forward,
        test_scheduler,
        test_focal_loss,
        test_gradient_flow
    ]
    
    failed = 0
    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"✗ {test.__name__} FAILED: {e}\n")
            failed += 1
    
    print("=" * 60)
    if failed == 0:
        print("✅ ALL TESTS PASSED!")
    else:
        print(f"❌ {failed} TESTS FAILED!")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
