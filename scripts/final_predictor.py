#!/usr/bin/env python3
"""
FINAL PRODUCTION PREDICTOR

Clean, ready-to-use gender prediction with optimal academic-grade fairness.
This is the end result of all optimization work.

Usage:
    python final_predictor.py --input data.csv --output results.csv
"""

import pandas as pd
import torch
import numpy as np
import argparse
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# FINAL CONFIGURATION - ACADEMIC OPTIMIZED
FINAL_CONFIG = {
    'model_path': 'experiments/20250603_192912_r3_bce_h256_l3_dual_frz5/models/model.pth',
    'preprocessor_path': 'experiments/20250603_192912_r3_bce_h256_l3_dual_frz5/preprocessor.pkl',
    'optimal_threshold': 0.480,  # Ultra-fair threshold (0.01% bias deviation)
    'unicode_preprocessing': True,
    'expected_performance': {
        'f1_score': 0.8976,
        'accuracy': 0.9207,
        'bias_ratio': 0.9999,
        'bias_deviation': 0.01
    }
}

class ProductionRobustPreprocessor:
    """
    Complete production preprocessor with all optimizations.
    Includes ALL cleaning steps that improved comparison dataset performance.
    """

    def __init__(self, base_preprocessor):
        self.base_preprocessor = base_preprocessor
        self.unicode_map = self._build_unicode_mapping()
        self.stats = {
            'total_processed': 0,
            'unicode_conversions': 0,
            'encoding_fixes': 0,
            'cleaning_applied': 0
        }

    def __getattr__(self, name):
        """Delegate to base preprocessor."""
        return getattr(self.base_preprocessor, name)

    def _build_unicode_mapping(self):
        """Build comprehensive Unicode mapping."""
        return {
            # Latin with diacritics - comprehensive mapping
            'à': 'a', 'á': 'a', 'â': 'a', 'ã': 'a', 'ä': 'a', 'å': 'a', 'ā': 'a', 'ă': 'a', 'ą': 'a',
            'è': 'e', 'é': 'e', 'ê': 'e', 'ë': 'e', 'ē': 'e', 'ė': 'e', 'ę': 'e', 'ě': 'e',
            'ì': 'i', 'í': 'i', 'î': 'i', 'ï': 'i', 'ī': 'i', 'į': 'i', 'ı': 'i',
            'ò': 'o', 'ó': 'o', 'ô': 'o', 'õ': 'o', 'ö': 'o', 'ō': 'o', 'ő': 'o', 'ø': 'o',
            'ù': 'u', 'ú': 'u', 'û': 'u', 'ü': 'u', 'ū': 'u', 'ů': 'u', 'ű': 'u', 'ų': 'u',
            'ý': 'y', 'ÿ': 'y', 'ȳ': 'y',
            'ñ': 'n', 'ň': 'n', 'ń': 'n', 'ņ': 'n',
            'ç': 'c', 'č': 'c', 'ć': 'c', 'ĉ': 'c', 'ċ': 'c',
            'ş': 's', 'š': 's', 'ś': 's', 'ŝ': 's',
            'ž': 'z', 'ź': 'z', 'ż': 'z',
            'ř': 'r', 'ŕ': 'r',
            'ł': 'l', 'ľ': 'l', 'ĺ': 'l', 'ļ': 'l',
            'ď': 'd', 'đ': 'd',
            'ť': 't', 'ţ': 't',
            'ğ': 'g', 'ģ': 'g',
            'ķ': 'k',
            'ß': 'ss',

            # Uppercase variants
            'À': 'A', 'Á': 'A', 'Â': 'A', 'Ã': 'A', 'Ä': 'A', 'Å': 'A', 'Ā': 'A', 'Ă': 'A', 'Ą': 'A',
            'È': 'E', 'É': 'E', 'Ê': 'E', 'Ë': 'E', 'Ē': 'E', 'Ė': 'E', 'Ę': 'E', 'Ě': 'E',
            'Ì': 'I', 'Í': 'I', 'Î': 'I', 'Ï': 'I', 'Ī': 'I', 'Į': 'I',
            'Ò': 'O', 'Ó': 'O', 'Ô': 'O', 'Õ': 'O', 'Ö': 'O', 'Ō': 'O', 'Ő': 'O', 'Ø': 'O',
            'Ù': 'U', 'Ú': 'U', 'Û': 'U', 'Ü': 'U', 'Ū': 'U', 'Ů': 'U', 'Ű': 'U', 'Ų': 'U',
            'Ý': 'Y', 'Ÿ': 'Y',
            'Ñ': 'N', 'Ň': 'N', 'Ń': 'N', 'Ņ': 'N',
            'Ç': 'C', 'Č': 'C', 'Ć': 'C', 'Ĉ': 'C', 'Ċ': 'C',
            'Ş': 'S', 'Š': 'S', 'Ś': 'S', 'Ŝ': 'S',
            'Ž': 'Z', 'Ź': 'Z', 'Ż': 'Z',
            'Ř': 'R', 'Ŕ': 'R',
            'Ł': 'L', 'Ľ': 'L', 'Ĺ': 'L', 'Ļ': 'L',
            'Ď': 'D', 'Đ': 'D',
            'Ť': 'T', 'Ţ': 'T',
            'Ğ': 'G', 'Ģ': 'G',
            'Ķ': 'K'
        }

    def fix_encoding_issues(self, text):
        """Fix common encoding corruption issues."""
        if not isinstance(text, str):
            return text

        original_text = text

        # Common UTF-8 -> Latin-1 mistakes
        encoding_fixes = {
            'Ã¡': 'á', 'Ã©': 'é', 'Ã­': 'í', 'Ã³': 'ó', 'Ãº': 'ú',
            'Ã¢': 'â', 'Ã¤': 'ä', 'Ã¨': 'è', 'Ã¬': 'ì', 'Ã²': 'ò',
            'Ã¹': 'ù', 'Ã§': 'ç', 'Ã±': 'ñ', 'Ã¼': 'ü', 'Ã¶': 'ö',
            'â€™': "'", 'â€œ': '"', 'â€': '"', 'â€"': '-', 'â€"': '-'
        }

        for corrupted, correct in encoding_fixes.items():
            text = text.replace(corrupted, correct)

        # Remove replacement characters
        import re
        text = re.sub(r'�+', '', text)

        if text != original_text:
            self.stats['encoding_fixes'] += 1

        return text

    def conservative_clean_name(self, name):
        """Apply conservative cleaning (all steps from comparison dataset)."""
        if not isinstance(name, str) or pd.isna(name):
            return ""

        original_name = name

        # 1. Fix encoding issues
        name = self.fix_encoding_issues(name)

        # 2. Remove control characters and null bytes
        import re
        name = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]', '', name)

        # 3. Normalize Unicode
        import unicodedata
        name = unicodedata.normalize('NFD', name)
        name = unicodedata.normalize('NFC', name)

        # 4. Normalize whitespace
        name = re.sub(r'\s+', ' ', name).strip()

        # 5. Very permissive length check (only extremely long)
        if len(name) > 200:
            name = name[:200]

        if name != original_name:
            self.stats['cleaning_applied'] += 1

        return name

    def normalize_unicode(self, text):
        """Comprehensive Unicode normalization."""
        if not isinstance(text, str):
            return ""

        self.stats['total_processed'] += 1
        result = ""

        for char in text:
            if char in self.unicode_map:
                result += self.unicode_map[char]
                if char != self.unicode_map[char]:
                    self.stats['unicode_conversions'] += 1
            elif ord(char) < 128:  # ASCII character
                result += char
            else:
                # Unicode decomposition fallback
                import unicodedata
                try:
                    decomposed = unicodedata.normalize('NFD', char)
                    ascii_char = ''.join(c for c in decomposed
                                       if unicodedata.category(c) != 'Mn')

                    if ascii_char and ord(ascii_char[0]) < 128:
                        result += ascii_char
                        self.stats['unicode_conversions'] += 1
                    else:
                        result += '?'  # Conservative fallback
                except Exception:
                    result += '?'

        return result

    def preprocess_name(self, full_name):
        """
        Complete production preprocessing pipeline.
        Applies ALL optimizations found during comparison dataset analysis.
        """
        try:
            # Step 1: Conservative cleaning (encoding, control chars, etc.)
            cleaned_name = self.conservative_clean_name(full_name)

            # Step 2: Unicode normalization
            normalized_name = self.normalize_unicode(cleaned_name)

            # Step 3: Use base preprocessor
            result = self.base_preprocessor.preprocess_name(normalized_name)

            # Add metadata for monitoring
            result['_processing_metadata'] = {
                'original_name': full_name,
                'cleaned_name': cleaned_name,
                'normalized_name': normalized_name,
                'was_processed': full_name != normalized_name,
                'processing_steps': [
                    'conservative_cleaning',
                    'unicode_normalization',
                    'base_preprocessing'
                ]
            }

            return result

        except Exception as e:
            # Emergency fallback
            print(f"⚠️  Preprocessing error for '{full_name}': {e}")

            # Use base preprocessor directly as fallback
            try:
                return self.base_preprocessor.preprocess_name(str(full_name))
            except:
                # Ultimate fallback
                if hasattr(self.base_preprocessor, 'max_name_length'):
                    max_len = self.base_preprocessor.max_name_length
                    max_surname = self.base_preprocessor.max_surname_length
                else:
                    max_len = max_surname = 20

                return {
                    'first_name': [0] * max_len,
                    'last_name': [0] * max_surname,
                    '_processing_metadata': {
                        'error': str(e),
                        'fallback_used': True
                    }
                }

    def get_processing_stats(self):
        """Get comprehensive processing statistics."""
        stats = self.stats.copy()
        if stats['total_processed'] > 0:
            stats['unicode_conversion_rate'] = stats['unicode_conversions'] / stats['total_processed']
            stats['encoding_fix_rate'] = stats['encoding_fixes'] / stats['total_processed']
            stats['cleaning_rate'] = stats['cleaning_applied'] / stats['total_processed']

        return stats

class FinalGenderPredictor:
    """Final production-ready gender predictor."""

    def __init__(self, config=None):
        self.config = config or FINAL_CONFIG
        self.model = None
        self.preprocessor = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.is_v3_model = False  # Track model type

        print(f"🚀 Final Gender Predictor")
        print(f"   Device: {self.device}")
        print(f"   Optimal threshold: {self.config['optimal_threshold']}")
        print(f"   Expected F1: {self.config['expected_performance']['f1_score']:.4f}")
        print(f"   Expected bias: {self.config['expected_performance']['bias_deviation']:.2f}%")

    def load_model(self):
        """Load optimized model and preprocessor."""
        from gender_predict.evaluation.evaluator import ModelEvaluator

        print("📂 Loading model...")

        # Check if it's a V3 model by loading checkpoint first
        checkpoint = torch.load(self.config['model_path'], map_location='cpu')
        self.is_v3_model = 'suffix_vocab_size' in checkpoint

        if self.is_v3_model:
            print("🔧 Detected V3 model - will use advanced features")
        else:
            print("🔧 Detected standard model")

        # Load base evaluator
        evaluator = ModelEvaluator.from_checkpoint(
            self.config['model_path'],
            self.config['preprocessor_path'],
            self.device
        )

        # Wrap with complete production preprocessing
        if self.config['unicode_preprocessing']:
            self.preprocessor = ProductionRobustPreprocessor(evaluator.preprocessor)
            print("✅ Complete production preprocessing enabled:")
            print("   - Unicode normalization")
            print("   - Encoding issue fixes")
            print("   - Conservative cleaning")
            print("   - Control character removal")
        else:
            self.preprocessor = evaluator.preprocessor

        self.model = evaluator.model
        self.model.eval()

        print("✅ Model loaded successfully")

    def predict_single(self, name):
        """Predict gender for a single name."""
        if self.model is None:
            self.load_model()

        # Preprocess
        processed = self.preprocessor.preprocess_name(name)

        # Convert to tensors
        first_name = torch.tensor(processed['first_name'], dtype=torch.long).unsqueeze(0).to(self.device)
        last_name = torch.tensor(processed['last_name'], dtype=torch.long).unsqueeze(0).to(self.device)

        # Handle V3 model features if available
        if self.is_v3_model:
            from gender_predict.data.feature_extraction import NameFeatureExtractor

            # Crea feature extractor se non esiste già
            if not hasattr(self, 'feature_extractor'):
                self.feature_extractor = NameFeatureExtractor()

            # Estrai nome e cognome come stringhe
            first_str, last_str = self.preprocessor.split_full_name(name)

            # Genera suffix features
            first_suffix = self.feature_extractor.extract_suffix_features(first_str)
            last_suffix = self.feature_extractor.extract_suffix_features(last_str)

            # Genera phonetic features
            phonetic_first = self.feature_extractor.extract_phonetic_features(first_str)
            phonetic_last = self.feature_extractor.extract_phonetic_features(last_str)

            phonetic_features = [
                phonetic_first['ends_with_vowel'],
                phonetic_first['vowel_ratio'],
                phonetic_last['ends_with_vowel'],
                phonetic_last['vowel_ratio']
            ]

            # Pad le suffix features a lunghezza 3
            first_suffix = first_suffix + [0] * (3 - len(first_suffix))
            last_suffix = last_suffix + [0] * (3 - len(last_suffix))

            # Converti a tensori
            first_suffix_tensor = torch.tensor(first_suffix[:3], dtype=torch.long).unsqueeze(0).to(self.device)
            last_suffix_tensor = torch.tensor(last_suffix[:3], dtype=torch.long).unsqueeze(0).to(self.device)
            phonetic_tensor = torch.tensor(phonetic_features, dtype=torch.float32).unsqueeze(0).to(self.device)

            # Predict con V3 features
            with torch.no_grad():
                output = self.model(first_name, last_name, first_suffix_tensor,
                                last_suffix_tensor, phonetic_tensor)
        else:
            # Standard model prediction
            with torch.no_grad():
                output = self.model(first_name, last_name)

        # Convert to probability
        prob_female = torch.sigmoid(output).item()

        # Apply optimal threshold
        prediction = 'W' if prob_female >= self.config['optimal_threshold'] else 'M'
        confidence = max(prob_female, 1 - prob_female)

        return {
            'name': name,
            'predicted_gender': prediction,
            'probability_female': prob_female,
            'confidence': confidence,
            'threshold_used': self.config['optimal_threshold']
        }

    def predict_batch(self, names):
        """Predict gender for a list of names."""
        results = []

        print(f"🔄 Processing {len(names)} names...")

        for i, name in enumerate(names):
            if i % 1000 == 0 and i > 0:
                print(f"   Processed {i}/{len(names)}")

            try:
                result = self.predict_single(name)
                results.append(result)
            except Exception as e:
                print(f"⚠️  Error processing '{name}': {e}")
                results.append({
                    'name': name,
                    'predicted_gender': 'Unknown',
                    'probability_female': 0.5,
                    'confidence': 0.0,
                    'error': str(e)
                })

        print(f"✅ Batch prediction complete")

        # Print comprehensive preprocessing stats if available
        if hasattr(self.preprocessor, 'get_processing_stats'):
            stats = self.preprocessor.get_processing_stats()
            if stats['total_processed'] > 0:
                print(f"📊 Production Preprocessing Stats:")
                print(f"   Unicode conversions: {stats['unicode_conversions']}/{stats['total_processed']} ({stats.get('unicode_conversion_rate', 0)*100:.1f}%)")
                print(f"   Encoding fixes: {stats['encoding_fixes']} ({stats.get('encoding_fix_rate', 0)*100:.1f}%)")
                print(f"   Cleaning applied: {stats['cleaning_applied']} ({stats.get('cleaning_rate', 0)*100:.1f}%)")
        elif hasattr(self.preprocessor, 'stats'):
            # Fallback to basic stats
            stats = self.preprocessor.stats
            if stats.get('total', 0) > 0:
                print(f"📊 Unicode processing: {stats['conversions']}/{stats['total']} characters converted ({stats['conversions']/stats['total']*100:.1f}%)")

        return results

    def predict_csv(self, input_file, output_file, name_column='primaryName'):
        """Predict gender for names in CSV file."""
        print(f"📂 Loading CSV: {input_file}")

        # Load data
        df = pd.read_csv(input_file)

        if name_column not in df.columns:
            raise ValueError(f"Column '{name_column}' not found. Available: {list(df.columns)}")

        names = df[name_column].fillna('').astype(str).tolist()

        # Predict
        results = self.predict_batch(names)

        # Create results DataFrame
        results_df = pd.DataFrame(results)

        # Merge with original data
        output_df = df.copy()
        output_df['predicted_gender'] = results_df['predicted_gender']
        output_df['probability_female'] = results_df['probability_female']
        output_df['confidence'] = results_df['confidence']

        # Save results
        output_df.to_csv(output_file, index=False)

        print(f"💾 Results saved: {output_file}")

        # Summary statistics
        pred_counts = results_df['predicted_gender'].value_counts()
        print(f"\n📊 Prediction Summary:")
        for gender, count in pred_counts.items():
            print(f"   {gender}: {count:,} ({count/len(results)*100:.1f}%)")

        avg_confidence = results_df['confidence'].mean()
        print(f"   Average confidence: {avg_confidence:.3f}")

        return output_df

def main():
    parser = argparse.ArgumentParser(description="Final production gender predictor")
    parser.add_argument('--input', help='Input CSV file')
    parser.add_argument('--output', help='Output CSV file')
    parser.add_argument('--name_column', default='primaryName', help='Name column in CSV')
    parser.add_argument('--single_name', help='Predict single name (instead of CSV)')

    args = parser.parse_args()

    # Validate arguments
    if args.single_name:
        # Single prediction mode - input/output not required
        if not args.single_name.strip():
            print("❌ Error: --single_name cannot be empty")
            return
    else:
        # Batch prediction mode - input/output required
        if not args.input or not args.output:
            print("❌ Error: --input and --output are required for batch prediction")
            print("Usage examples:")
            print("  Single prediction: python final_predictor.py --single_name 'José María García'")
            print("  Batch prediction:  python final_predictor.py --input data.csv --output results.csv")
            return

    # Create predictor
    predictor = FinalGenderPredictor()

    if args.single_name:
        # Single prediction
        result = predictor.predict_single(args.single_name)
        print(f"\n🎯 Prediction for '{args.single_name}':")
        print(f"   Gender: {result['predicted_gender']}")
        print(f"   Confidence: {result['confidence']:.3f}")
        print(f"   Probability Female: {result['probability_female']:.3f}")
        print(f"   Threshold Used: {result['threshold_used']:.3f}")
    else:
        # Batch prediction
        predictor.predict_csv(args.input, args.output, args.name_column)

if __name__ == "__main__":
    main()
