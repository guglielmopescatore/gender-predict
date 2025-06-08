#!/usr/bin/env python3
"""
Intelligent Preprocessing Quality Analyzer
==========================================

This analyzer determines whether preprocessing differences are:
1. HARMFUL (production worse than training) → Align to training
2. BENEFICIAL (production better than training) → Retrain with production preprocessing  
3. NEUTRAL (equivalent quality) → Align for consistency

Key insight: If production preprocessing is actually BETTER at handling
international names, Unicode, encoding issues, etc., then the solution
is NOT to revert to training preprocessing, but to retrain the model
with the improved preprocessing!
"""

import sys
import os
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import unicodedata
import re

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from gender_predict.data.preprocessing import NamePreprocessor
from gender_predict.data.improved_preprocessing import ImprovedNamePreprocessor


class ProductionRobustPreprocessor:
    """Reproduction of ProductionRobustPreprocessor for analysis."""
    
    def __init__(self, base_preprocessor):
        self.base_preprocessor = base_preprocessor
        self.unicode_map = self._build_unicode_mapping()
        self.stats = {'total_processed': 0, 'unicode_conversions': 0, 'encoding_fixes': 0, 'cleaning_applied': 0}

    def __getattr__(self, name):
        return getattr(self.base_preprocessor, name)

    def _build_unicode_mapping(self):
        """Comprehensive Unicode mapping."""
        return {
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
            'ķ': 'k', 'ß': 'ss',
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
        encoding_fixes = {
            'Ã¡': 'á', 'Ã©': 'é', 'Ã­': 'í', 'Ã³': 'ó', 'Ãº': 'ú',
            'Ã¢': 'â', 'Ã¤': 'ä', 'Ã¨': 'è', 'Ã¬': 'ì', 'Ã²': 'ò',
            'Ã¹': 'ù', 'Ã§': 'ç', 'Ã±': 'ñ', 'Ã¼': 'ü', 'Ã¶': 'ö',
            'â€™': "'", 'â€œ': '"', 'â€': '"', 'â€"': '-', 'â€"': '-'
        }
        for corrupted, correct in encoding_fixes.items():
            text = text.replace(corrupted, correct)
        text = re.sub(r'�+', '', text)
        if text != original_text:
            self.stats['encoding_fixes'] += 1
        return text

    def conservative_clean_name(self, name):
        """Apply conservative cleaning."""
        if not isinstance(name, str) or pd.isna(name):
            return ""
        original_name = name
        name = self.fix_encoding_issues(name)
        name = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]', '', name)
        name = unicodedata.normalize('NFD', name)
        name = unicodedata.normalize('NFC', name)
        name = re.sub(r'\s+', ' ', name).strip()
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
            elif ord(char) < 128:
                result += char
            else:
                try:
                    decomposed = unicodedata.normalize('NFD', char)
                    ascii_char = ''.join(c for c in decomposed if unicodedata.category(c) != 'Mn')
                    if ascii_char and ord(ascii_char[0]) < 128:
                        result += ascii_char
                        self.stats['unicode_conversions'] += 1
                    else:
                        result += '?'
                except Exception:
                    result += '?'
        return result

    def preprocess_name(self, full_name):
        """Complete production preprocessing pipeline."""
        try:
            cleaned_name = self.conservative_clean_name(full_name)
            normalized_name = self.normalize_unicode(cleaned_name)
            result = self.base_preprocessor.preprocess_name(normalized_name)
            return result
        except Exception:
            try:
                return self.base_preprocessor.preprocess_name(str(full_name))
            except:
                return {
                    'first_name': [0] * self.base_preprocessor.max_name_length,
                    'last_name': [0] * self.base_preprocessor.max_surname_length,
                }


class IntelligentPreprocessingAnalyzer:
    """
    Determines whether preprocessing differences are beneficial or harmful.
    """
    
    def __init__(self, comparison_dataset_path="./data/raw/comparison_dataset.csv"):
        self.dataset_path = comparison_dataset_path
        self.analysis_results = {}
        
    def create_test_suite(self):
        """Create comprehensive test suite for preprocessing quality assessment."""
        
        # Load real data if available
        real_names = []
        if os.path.exists(self.dataset_path):
            try:
                df = pd.read_csv(self.dataset_path)
                real_names = df['primaryName'].dropna().head(500).tolist()
                print(f"✅ Loaded {len(real_names)} real names for testing")
            except Exception as e:
                print(f"⚠️  Error loading real data: {e}")
        
        # Synthetic test cases targeting specific preprocessing challenges
        synthetic_test_cases = {
            'unicode_diacritics': [
                'José María García',
                'François Müller',
                'Søren Østergård', 
                'Žužana Svoboda',
                'María José de la Cruz',
                'Bjørn Åkesson',
                'László Kovács',
                'Štěpán Novák'
            ],
            'encoding_corruption': [
                'JosÃ© MarÃ­a',      # Common UTF-8 → Latin-1 corruption
                'FranÃ§ois MÃ¼ller',
                'MÃ¡ria JosÃ©',
                'Ã�ngela Ã�vila'       # Corrupted accents
            ],
            'control_characters': [
                'José\x00María',         # Null bytes
                'François\x01Müller',    # Control characters  
                'María\x7f José',        # DEL character
                'Stefan\x9f Weber'       # Extended control
            ],
            'complex_international': [
                '李明华',                 # Chinese
                'محمد الأحمد',           # Arabic
                'Владимир Иванов',       # Cyrillic
                'Θεόδωρος Παπαδόπουλος', # Greek
                'יוסף כהן',             # Hebrew
                'ไพโรจน์ สุขใจ'          # Thai
            ],
            'mixed_scripts': [
                'José李明',              # Latin + Chinese
                'María محمد',            # Latin + Arabic
                'François Владимир',     # Latin + Cyrillic
            ],
            'problematic_whitespace': [
                'José  María   García',   # Multiple spaces
                ' François Müller ',     # Leading/trailing
                'María\tJosé',           # Tab character
                'Stefan\nWeber'          # Newline
            ],
            'long_names': [
                'María José de la Concepción Mercedes del Pilar Teresa de Todos los Santos',
                'Jean-Baptiste-Camille-Henri-François-Marie de Montmorency-Laval-Bouteville',
                'Wolfgang Amadeus Theophilus Gottlieb Chrysostomus Sigismundus Mozart'
            ]
        }
        
        # Combine all test cases
        all_test_names = real_names.copy()
        for category, names in synthetic_test_cases.items():
            all_test_names.extend(names)
        
        return {
            'real_names': real_names,
            'synthetic_categories': synthetic_test_cases,
            'all_test_names': all_test_names,
            'total_tests': len(all_test_names)
        }
    
    def evaluate_preprocessing_quality(self, preprocessor, test_suite, name="Unknown"):
        """Evaluate preprocessing quality across different challenges."""
        
        quality_metrics = {
            'unicode_handling': 0,
            'encoding_robustness': 0,
            'control_char_cleaning': 0,
            'international_support': 0,
            'whitespace_normalization': 0,
            'error_resilience': 0,
            'consistency': 0
        }
        
        preprocessing_stats = {
            'successful_processing': 0,
            'failed_processing': 0,
            'information_preserved': 0,
            'information_lost': 0,
            'corrupted_output': 0
        }
        
        category_scores = {}
        
        # Test each category
        for category, test_names in test_suite['synthetic_categories'].items():
            category_score = 0
            category_details = []
            
            for test_name in test_names:
                try:
                    # Process the name
                    result = preprocessor.preprocess_name(test_name)
                    
                    # Evaluate quality based on category
                    quality_score = self._evaluate_single_result(test_name, result, category)
                    category_score += quality_score
                    
                    category_details.append({
                        'input': test_name,
                        'output': result,
                        'quality_score': quality_score,
                        'issues': self._identify_processing_issues(test_name, result)
                    })
                    
                    preprocessing_stats['successful_processing'] += 1
                    
                except Exception as e:
                    preprocessing_stats['failed_processing'] += 1
                    category_details.append({
                        'input': test_name,
                        'output': None,
                        'quality_score': 0,
                        'error': str(e)
                    })
            
            # Normalize category score
            if test_names:
                category_scores[category] = {
                    'score': category_score / len(test_names),
                    'details': category_details
                }
        
        # Calculate overall quality metrics
        unicode_score = category_scores.get('unicode_diacritics', {}).get('score', 0)
        encoding_score = category_scores.get('encoding_corruption', {}).get('score', 0)
        control_score = category_scores.get('control_characters', {}).get('score', 0)
        international_score = category_scores.get('complex_international', {}).get('score', 0)
        whitespace_score = category_scores.get('problematic_whitespace', {}).get('score', 0)
        
        quality_metrics.update({
            'unicode_handling': unicode_score,
            'encoding_robustness': encoding_score,
            'control_char_cleaning': control_score,
            'international_support': international_score,
            'whitespace_normalization': whitespace_score,
            'overall_quality': np.mean([unicode_score, encoding_score, control_score, 
                                      international_score, whitespace_score])
        })
        
        return {
            'preprocessor_name': name,
            'quality_metrics': quality_metrics,
            'preprocessing_stats': preprocessing_stats,
            'category_scores': category_scores,
            'summary': self._generate_quality_summary(quality_metrics, preprocessing_stats)
        }
    
    def _evaluate_single_result(self, input_name, result, category):
        """Evaluate quality of a single preprocessing result."""
        if result is None:
            return 0.0
        
        # Category-specific evaluation
        if category == 'unicode_diacritics':
            # Check if diacritics were properly normalized
            if any(ord(c) > 127 for c in input_name):
                # Unicode input should be handled gracefully
                if 'first_name' in result and 'last_name' in result:
                    return 0.8  # Good if processed without errors
                else:
                    return 0.2
            return 1.0
            
        elif category == 'encoding_corruption':
            # Check if encoding issues were fixed
            has_corruption = 'Ã' in input_name or '€' in input_name
            if has_corruption:
                # Should clean up corruption
                return 0.9 if result else 0.1
            return 1.0
            
        elif category == 'control_characters':
            # Check if control characters were removed
            has_control = any(ord(c) < 32 and c not in [' ', '\t', '\n'] for c in input_name)
            if has_control:
                return 0.9 if result else 0.1
            return 1.0
            
        elif category == 'complex_international':
            # Check if non-Latin scripts are handled
            has_non_latin = any(ord(c) > 1000 for c in input_name)
            if has_non_latin:
                # Should gracefully handle or convert
                return 0.7 if result else 0.0
            return 1.0
            
        elif category == 'problematic_whitespace':
            # Check if whitespace is normalized
            has_bad_whitespace = '  ' in input_name or input_name != input_name.strip()
            if has_bad_whitespace:
                return 0.8 if result else 0.2
            return 1.0
            
        # Default scoring
        return 0.5 if result else 0.0
    
    def _identify_processing_issues(self, input_name, result):
        """Identify specific issues in preprocessing result."""
        issues = []
        
        if result is None:
            issues.append("Failed to process")
            return issues
        
        # Check for information loss
        if len(input_name) > 0 and all(idx == 0 for idx in result.get('first_name', [])):
            issues.append("Complete information loss in first name")
        
        if ' ' in input_name and all(idx == 0 for idx in result.get('last_name', [])):
            issues.append("Lost last name information")
        
        # Check for obvious corruption
        if any(ord(c) > 127 for c in input_name):
            issues.append("Unicode input detected")
        
        return issues
    
    def _generate_quality_summary(self, quality_metrics, preprocessing_stats):
        """Generate human-readable quality summary."""
        overall = quality_metrics['overall_quality']
        
        if overall >= 0.8:
            quality_level = "EXCELLENT"
        elif overall >= 0.6:
            quality_level = "GOOD"
        elif overall >= 0.4:
            quality_level = "MODERATE"
        else:
            quality_level = "POOR"
        
        success_rate = (preprocessing_stats['successful_processing'] / 
                       max(1, preprocessing_stats['successful_processing'] + preprocessing_stats['failed_processing']))
        
        return {
            'quality_level': quality_level,
            'overall_score': overall,
            'success_rate': success_rate,
            'strengths': [k for k, v in quality_metrics.items() if v >= 0.7],
            'weaknesses': [k for k, v in quality_metrics.items() if v < 0.5]
        }
    
    def compare_preprocessing_quality(self):
        """Compare quality of different preprocessing approaches."""
        print("🔍 INTELLIGENT PREPROCESSING QUALITY ANALYSIS")
        print("=" * 60)
        
        # Create test suite
        test_suite = self.create_test_suite()
        print(f"📊 Testing with {test_suite['total_tests']} names")
        print(f"   Real names: {len(test_suite['real_names'])}")
        print(f"   Synthetic tests: {test_suite['total_tests'] - len(test_suite['real_names'])}")
        
        # Initialize preprocessors
        base_preprocessor = NamePreprocessor()
        improved_preprocessor = ImprovedNamePreprocessor()
        production_preprocessor = ProductionRobustPreprocessor(base_preprocessor)
        
        # Evaluate each preprocessor
        print(f"\n🔄 Evaluating preprocessing quality...")
        
        base_evaluation = self.evaluate_preprocessing_quality(
            base_preprocessor, test_suite, "Base NamePreprocessor"
        )
        
        improved_evaluation = self.evaluate_preprocessing_quality(
            improved_preprocessor, test_suite, "Improved NamePreprocessor"
        )
        
        production_evaluation = self.evaluate_preprocessing_quality(
            production_preprocessor, test_suite, "Production RobustPreprocessor"
        )
        
        # Compare results
        comparison = self._compare_evaluations([
            base_evaluation,
            improved_evaluation, 
            production_evaluation
        ])
        
        self.analysis_results = {
            'test_suite': test_suite,
            'evaluations': {
                'base': base_evaluation,
                'improved': improved_evaluation,
                'production': production_evaluation
            },
            'comparison': comparison
        }
        
        return self.analysis_results
    
    def _compare_evaluations(self, evaluations):
        """Compare preprocessing evaluations and determine best approach."""
        
        # Extract quality scores
        scores = {}
        for eval_data in evaluations:
            name = eval_data['preprocessor_name']
            scores[name] = eval_data['quality_metrics']['overall_quality']
        
        # Determine ranking
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # Analysis
        best_preprocessor = ranked[0][0]
        best_score = ranked[0][1]
        
        comparison_results = {
            'rankings': ranked,
            'best_preprocessor': best_preprocessor,
            'best_score': best_score,
            'quality_differences': {},
            'recommendations': []
        }
        
        # Calculate differences
        base_score = scores.get("Base NamePreprocessor", 0)
        production_score = scores.get("Production RobustPreprocessor", 0)
        improved_score = scores.get("Improved NamePreprocessor", 0)
        
        comparison_results['quality_differences'] = {
            'production_vs_base': production_score - base_score,
            'production_vs_improved': production_score - improved_score,
            'improved_vs_base': improved_score - base_score
        }
        
        # Generate intelligent recommendations
        prod_vs_base_diff = production_score - base_score
        
        if prod_vs_base_diff > 0.1:  # Production significantly better
            comparison_results['recommendations'].extend([
                "🟢 BENEFICIAL MISMATCH DETECTED",
                "Production preprocessing is BETTER than training preprocessing",
                "Recommended action: RETRAIN model with production preprocessing",
                f"Expected gain: {prod_vs_base_diff:.1%} improvement in robustness",
                "Priority: HIGH - This could significantly improve international name handling"
            ])
            comparison_results['mismatch_type'] = 'BENEFICIAL'
            comparison_results['action'] = 'RETRAIN_WITH_PRODUCTION'
            
        elif prod_vs_base_diff < -0.1:  # Production worse
            comparison_results['recommendations'].extend([
                "🔴 HARMFUL MISMATCH DETECTED", 
                "Production preprocessing is WORSE than training preprocessing",
                "Recommended action: ALIGN production to training preprocessing",
                f"Expected gain: {abs(prod_vs_base_diff):.1%} by fixing alignment",
                "Priority: CRITICAL - Fix immediately"
            ])
            comparison_results['mismatch_type'] = 'HARMFUL'
            comparison_results['action'] = 'ALIGN_TO_TRAINING'
            
        else:  # Roughly equivalent
            comparison_results['recommendations'].extend([
                "⚪ NEUTRAL MISMATCH",
                "Preprocessing quality is roughly equivalent", 
                "Recommended action: ALIGN for consistency",
                "Priority: MEDIUM - Ensure reproducibility"
            ])
            comparison_results['mismatch_type'] = 'NEUTRAL'
            comparison_results['action'] = 'ALIGN_FOR_CONSISTENCY'
        
        # Print results
        print(f"\n📊 PREPROCESSING QUALITY COMPARISON")
        print(f"=" * 50)
        
        for i, (name, score) in enumerate(ranked, 1):
            status = "🥇" if i == 1 else "🥈" if i == 2 else "🥉"
            print(f"{status} {i}. {name}: {score:.3f}")
        
        print(f"\n💡 INTELLIGENT RECOMMENDATIONS")
        print(f"=" * 40)
        for rec in comparison_results['recommendations']:
            print(f"   {rec}")
        
        # Detailed metrics comparison
        print(f"\n📈 DETAILED QUALITY METRICS")
        print(f"=" * 40)
        metrics = ['unicode_handling', 'encoding_robustness', 'control_char_cleaning', 
                  'international_support', 'whitespace_normalization']
        
        print(f"{'Metric':<25} {'Base':<8} {'Improved':<8} {'Production':<10}")
        print(f"{'-'*55}")
        
        for metric in metrics:
            base_val = evaluations[0]['quality_metrics'].get(metric, 0)
            improved_val = evaluations[1]['quality_metrics'].get(metric, 0)
            prod_val = evaluations[2]['quality_metrics'].get(metric, 0)
            
            print(f"{metric:<25} {base_val:<8.3f} {improved_val:<8.3f} {prod_val:<10.3f}")
        
        return comparison_results
    
    def generate_v4_preprocessing_requirements(self, comparison_results):
        """Generate V4 preprocessing requirements based on analysis."""
        
        requirements = {
            'immediate_actions': [],
            'v4_requirements': [],
            'implementation_priority': 'HIGH'
        }
        
        mismatch_type = comparison_results['mismatch_type']
        action = comparison_results['action']
        
        if action == 'RETRAIN_WITH_PRODUCTION':
            requirements['immediate_actions'].extend([
                'Adopt ProductionRobustPreprocessor as the standard',
                'Retrain current V3 model with production preprocessing',
                'Measure performance improvement on international names',
                'Update all training scripts to use production preprocessing'
            ])
            
            requirements['v4_requirements'].extend([
                'Unified preprocessing module (same for training and inference)',
                'Enhanced Unicode support as demonstrated in production',
                'Robust encoding error recovery',
                'Comprehensive international name testing suite'
            ])
            
        elif action == 'ALIGN_TO_TRAINING':
            requirements['immediate_actions'].extend([
                'Fix production preprocessing to match training',
                'Investigate why production preprocessing was changed',
                'Implement preprocessing consistency checks',
                'Monitor production preprocessing behavior'
            ])
            
            requirements['v4_requirements'].extend([
                'Strict preprocessing versioning and validation',
                'Automated training/production consistency checks',
                'Improved training preprocessing robustness'
            ])
            
        else:  # ALIGN_FOR_CONSISTENCY
            requirements['immediate_actions'].extend([
                'Standardize preprocessing across all environments',
                'Choose best preprocessing approach for consistency',
                'Implement preprocessing validation pipeline'
            ])
        
        # Always add V4 requirements
        requirements['v4_requirements'].extend([
            'Preprocessing quality testing framework',
            'International name support validation',
            'Automated preprocessing regression testing',
            'Unified preprocessing API for all model versions'
        ])
        
        print(f"\n🚀 V4 PREPROCESSING REQUIREMENTS")
        print(f"=" * 45)
        print(f"📢 Immediate Actions ({requirements['implementation_priority']} priority):")
        for action in requirements['immediate_actions']:
            print(f"   • {action}")
        
        print(f"\n🔧 V4 Architecture Requirements:")
        for req in requirements['v4_requirements']:
            print(f"   • {req}")
        
        return requirements
    
    def run_complete_analysis(self):
        """Run complete intelligent preprocessing analysis."""
        
        # Run quality comparison
        results = self.compare_preprocessing_quality()
        
        # Generate V4 requirements
        v4_requirements = self.generate_v4_preprocessing_requirements(results['comparison'])
        results['v4_requirements'] = v4_requirements
        
        # Save results
        import json
        os.makedirs("./v4_analysis", exist_ok=True)
        
        with open("./v4_analysis/intelligent_preprocessing_analysis.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n💾 Analysis saved to: ./v4_analysis/intelligent_preprocessing_analysis.json")
        
        print(f"\n✅ INTELLIGENT PREPROCESSING ANALYSIS COMPLETE!")
        print(f"🎯 Key finding: {results['comparison']['mismatch_type']} mismatch detected")
        print(f"🚀 Recommended action: {results['comparison']['action']}")
        
        return results


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Intelligent Preprocessing Quality Analysis")
    parser.add_argument('--dataset', default='./data/raw/comparison_dataset.csv', 
                       help='Path to comparison dataset')
    
    args = parser.parse_args()
    
    # Run intelligent analysis
    analyzer = IntelligentPreprocessingAnalyzer(args.dataset)
    results = analyzer.run_complete_analysis()
    
    if results:
        print(f"\n🎯 NEXT STEPS BASED ON ANALYSIS:")
        action = results['comparison']['action']
        
        if action == 'RETRAIN_WITH_PRODUCTION':
            print(f"   1. This is actually GOOD NEWS - your production preprocessing is better!")
            print(f"   2. Retrain your model with the improved preprocessing")
            print(f"   3. Expected improvement in international name handling")
            print(f"   4. Use this improved preprocessing as V4 standard")
            
        elif action == 'ALIGN_TO_TRAINING':
            print(f"   1. Fix production preprocessing immediately")
            print(f"   2. This will give quick accuracy gains")
            print(f"   3. Investigate why production preprocessing was degraded")
            
        else:
            print(f"   1. Standardize preprocessing for consistency")
            print(f"   2. Focus V4 development on other improvements")


if __name__ == "__main__":
    main()
