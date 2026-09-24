# Testing Transliteration Support

## 📋 Quick Start

### 1. Install dependencies
```bash
cd ~/Git_Repositories/gender-predict
pip install -r requirements.txt
```

### 2. Test transliteration wrapper
```bash
cd api/
python test_transliteration.py
```

### 3. Compare original vs enhanced predictor
```bash
python test_enhanced_predictor.py
```

### 4. Quick test specific names
```bash
python3 -c "
import sys
sys.path.insert(0, '..')
sys.path.insert(0, '../scripts')

from scripts.enhanced_predictor import EnhancedGenderPredictor

config = {
    'model_path': '../models/best_v3_model/models/model.pth',
    'preprocessor_path': '../models/best_v3_model/preprocessor.pkl',
    'feature_extractor_path': '../models/best_v3_model/feature_extractor.pkl',
    'optimal_threshold': 0.52,
    'unicode_preprocessing': True,
    'enable_transliteration': True,
    'expected_performance': {
        'f1_score': 0.8996,
        'accuracy': 0.9219,
        'bias_ratio': 0.9999,
        'bias_deviation': 0.01
    }
}

predictor = EnhancedGenderPredictor(config)
predictor.load_model()

# Test Cyrillic name
result = predictor.predict_single('Екатерина')
print(f'Екатерина: {result[\"predicted_gender\"]} (was {result[\"transliterated_name\"]})')
"
```

## 📊 Expected Results

### Before Transliteration (Original Model)
- Екатерина → M (WRONG ❌)
- Владимир → M (Correct by chance)
- 王芳 → Unknown/Low confidence

### After Transliteration (Enhanced Model)
- Екатерина → ekaterina → W (CORRECT ✅)
- Владимир → vladimir → M (CORRECT ✅)
- 王芳 → wangfang → W (CORRECT ✅)

## 🚀 Deploy with Transliteration

1. The enhanced model is already integrated in `modal_deployment.py`
2. Deploy as usual:
```bash
cd api/
make deploy
```

3. Test the deployed API:
```bash
curl -X POST "YOUR_API_URL/predict" \
  -H "Content-Type: application/json" \
  -d '{"names": ["Екатерина", "王芳", "محمد"]}'
```

## 🔧 Troubleshooting

### Missing dependencies
If you get import errors:
```bash
pip install transliterate pypinyin fugashi unidic-lite romkan hangul-romanize regex
```

### Japanese transliteration not working
The Japanese tokenizer might need to download dictionary on first use:
```python
python -c "import fugashi; fugashi.Tagger()"
```

### Want to disable transliteration
Set in config:
```python
config['enable_transliteration'] = False
```