#!/usr/bin/env python3
"""
Test completo del sistema migrato.
Verifica che tutti i componenti funzionino correttamente.
"""

import sys
import os
import traceback
import tempfile
import pandas as pd
import torch

def test_imports():
    """Test che tutti i moduli si importino correttamente."""
    print("🔍 Testing imports...")
    
    try:
        # Test main package
        import gender_predict
        print("  ✅ Main package imported")
        
        # Test models
        from gender_predict.models import (
            GenderPredictor, GenderPredictorEnhanced, GenderPredictorV3,
            create_model, MODEL_REGISTRY
        )
        print("  ✅ Models imported")
        
        # Test data
        from gender_predict.data import (
            NamePreprocessor, NameGenderDataset, NameFeatureExtractor, NameAugmenter
        )
        print("  ✅ Data modules imported")
        
        # Test training
        from gender_predict.training import (
            FocalLoss, FocalLossImproved, BalancedBatchSampler, CosineAnnealingWarmupScheduler
        )
        print("  ✅ Training modules imported")
        
        # Test evaluation
        from gender_predict.evaluation import (
            ModelEvaluator, ErrorAnalyzer
        )
        print("  ✅ Evaluation modules imported")
        
        # Test experiments
        from gender_predict.experiments import ExperimentManager, compare_experiments
        print("  ✅ Experiment modules imported")
        
        # Test utils
        from gender_predict.utils import EarlyStopping, ensure_dir
        print("  ✅ Utility modules imported")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Import failed: {e}")
        traceback.print_exc()
        return False

def test_model_creation():
    """Test che i modelli si creino correttamente."""
    print("\n🧠 Testing model creation...")
    
    try:
        # Test factory function
        from gender_predict.models import create_model
        
        # Test base model
        model_base = create_model('base', vocab_size=100, embedding_dim=16, hidden_size=32)
        print("  ✅ Base model created")
        
        # Test enhanced model  
        model_enhanced = create_model('enhanced', 
            vocab_size=100, embedding_dim=16, hidden_size=32, n_layers=2, dual_input=True
        )
        print("  ✅ Enhanced model created")
        
        # Test V3 model
        model_v3 = create_model('v3',
            vocab_size=100, suffix_vocab_size=50, embedding_dim=32, hidden_size=64,
            n_layers=2, num_attention_heads=4
        )
        print("  ✅ V3 model created")
        
        # Test model registry
        from gender_predict.models import MODEL_REGISTRY
        print(f"  ✅ Model registry contains: {list(MODEL_REGISTRY.keys())}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Model creation failed: {e}")
        traceback.print_exc()
        return False

def test_data_processing():
    """Test preprocessing e dataset."""
    print("\n📊 Testing data processing...")
    
    try:
        from gender_predict.data import NamePreprocessor, NameGenderDataset
        
        # Test preprocessor
        preprocessor = NamePreprocessor()
        print("  ✅ NamePreprocessor created")
        
        # Test preprocessing
        result = preprocessor.preprocess_name("John Smith")
        print(f"  ✅ Name preprocessing works: {len(result['first_name'])} chars")
        
        # Test with sample data
        sample_data = pd.DataFrame({
            'primaryName': ['John Smith', 'Jane Doe', 'Maria Garcia'],
            'gender': ['M', 'W', 'W'],
            'nconst': ['nm001', 'nm002', 'nm003']
        })
        
        dataset = NameGenderDataset(sample_data, preprocessor, mode='test')
        print(f"  ✅ Dataset created with {len(dataset)} samples")
        
        # Test dataset item
        item = dataset[0]
        print(f"  ✅ Dataset item: first_name shape {item['first_name'].shape}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Data processing failed: {e}")
        traceback.print_exc()
        return False

def test_training_components():
    """Test training utilities."""
    print("\n🏋️ Testing training components...")
    
    try:
        from gender_predict.training import FocalLoss, CosineAnnealingWarmupScheduler
        import torch.optim as optim
        
        # Test focal loss
        criterion = FocalLoss(alpha=0.7, gamma=2.0)
        print("  ✅ FocalLoss created")
        
        # Test scheduler
        dummy_model = torch.nn.Linear(10, 1)
        optimizer = optim.Adam(dummy_model.parameters())
        scheduler = CosineAnnealingWarmupScheduler(
            optimizer, warmup_epochs=2, max_epochs=10
        )
        print("  ✅ Scheduler created")
        
        # Test scheduler step
        lr_before = optimizer.param_groups[0]['lr']
        scheduler.step(0)
        lr_after = optimizer.param_groups[0]['lr']
        print(f"  ✅ Scheduler works: LR {lr_before:.6f} → {lr_after:.6f}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Training components failed: {e}")
        traceback.print_exc()
        return False

def test_experiment_manager():
    """Test experiment management."""
    print("\n🔬 Testing experiment manager...")
    
    try:
        from gender_predict.experiments import ExperimentManager
        import argparse
        import tempfile
        
        # Create dummy args
        args = argparse.Namespace(
            round=1,
            loss='focal',
            alpha=0.7,
            gamma=2.0,
            note='test'
        )
        
        # Test with temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            experiment = ExperimentManager(args, base_dir=temp_dir, auto_create=True)
            print(f"  ✅ ExperimentManager created: {experiment.experiment_id}")
            
            # Test logging
            test_metrics = {'accuracy': 0.92, 'f1': 0.90}
            experiment.log_test_metrics(test_metrics)
            print("  ✅ Metrics logging works")
            
            # Test that files were created
            assert os.path.exists(experiment.experiment_dir), "Experiment directory not created"
            assert os.path.exists(experiment.test_metrics_path), "Test metrics file not created"
            print("  ✅ Files created correctly")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Experiment manager failed: {e}")
        traceback.print_exc()
        return False

def test_scripts_exist():
    """Test che gli script esistano e siano eseguibili."""
    print("\n📜 Testing scripts...")
    
    scripts = [
        'scripts/train_model.py',
        'scripts/evaluate_model.py',
        'scripts/experiment_tools.py',
        'scripts/prepare_data.py'
    ]
    
    all_exist = True
    for script in scripts:
        if os.path.exists(script):
            print(f"  ✅ {script} exists")
            
            # Test that it has help
            try:
                result = os.system(f"python {script} --help > /dev/null 2>&1")
                if result == 0:
                    print(f"     ✅ Help works")
                else:
                    print(f"     ⚠️  Help may have issues")
            except:
                print(f"     ⚠️  Could not test help")
        else:
            print(f"  ❌ {script} missing")
            all_exist = False
    
    return all_exist

def test_package_version():
    """Test package version and metadata."""
    print("\n📦 Testing package metadata...")
    
    try:
        import gender_predict
        
        # Test version
        version = getattr(gender_predict, '__version__', None)
        if version:
            print(f"  ✅ Package version: {version}")
        else:
            print("  ⚠️  No version information")
        
        # Test convenience imports
        from gender_predict import create_model, NamePreprocessor, ExperimentManager
        print("  ✅ Convenience imports work")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Package metadata failed: {e}")
        return False

def run_integration_test():
    """Test di integrazione end-to-end."""
    print("\n🔄 Running integration test...")
    
    try:
        # Import everything we need
        from gender_predict.models import create_model
        from gender_predict.data import NamePreprocessor, NameGenderDataset
        from gender_predict.training import FocalLoss
        import torch
        import pandas as pd
        
        # Create sample data
        sample_data = pd.DataFrame({
            'primaryName': ['John Smith', 'Jane Doe', 'Maria Garcia', 'David Brown', 'Sarah Wilson'],
            'gender': ['M', 'W', 'W', 'M', 'W'],
            'nconst': ['nm001', 'nm002', 'nm003', 'nm004', 'nm005']
        })
        
        # Create preprocessor and dataset
        preprocessor = NamePreprocessor()
        dataset = NameGenderDataset(sample_data, preprocessor, mode='train')
        
        # Create model
        model = create_model('base', 
            vocab_size=preprocessor.vocab_size,
            embedding_dim=16, 
            hidden_size=32
        )
        
        # Create loss and optimizer
        criterion = FocalLoss(alpha=0.7, gamma=2.0)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # Run one training step
        model.train()
        item = dataset[0]
        
        # Forward pass
        output = model(
            item['first_name'].unsqueeze(0),
            item['last_name'].unsqueeze(0)
        )
        
        loss = criterion(output, item['gender'].unsqueeze(0))
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"  ✅ Integration test passed - Loss: {loss.item():.4f}")
        return True
        
    except Exception as e:
        print(f"  ❌ Integration test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Esegui tutti i test."""
    
    print("🧪 TESTING COMPLETE MIGRATED SYSTEM")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("Model Creation", test_model_creation),
        ("Data Processing", test_data_processing),
        ("Training Components", test_training_components),
        ("Experiment Manager", test_experiment_manager),
        ("Scripts Existence", test_scripts_exist),
        ("Package Metadata", test_package_version),
        ("Integration Test", run_integration_test),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name:<20} {status}")
        if result:
            passed += 1
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Migration successful!")
        print("\n🚀 Next steps:")
        print("  1. Try: python scripts/train_model.py --help")
        print("  2. Test with real data")
        print("  3. Commit your changes")
        print("  4. Update documentation")
    else:
        print(f"\n⚠️  {total - passed} tests failed. Check the issues above.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
