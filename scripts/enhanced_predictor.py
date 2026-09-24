"""
Enhanced Gender Predictor with automatic transliteration support.
Extends FinalGenderPredictor to handle non-Latin scripts.
"""

import sys
from typing import Dict, List, Any

# Add parent directory to path for imports
sys.path.insert(0, '..')

from scripts.final_predictor import FinalGenderPredictor
from scripts.transliteration_wrapper import preprocess_for_model, transliterate_name


class EnhancedGenderPredictor(FinalGenderPredictor):
    """
    Gender predictor with automatic transliteration for non-Latin scripts.
    
    This class extends FinalGenderPredictor to automatically transliterate
    names from Cyrillic, Chinese, Japanese, Korean, and Arabic scripts
    to Latin/romanized form before prediction.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the enhanced predictor with transliteration support."""
        super().__init__(config)
        self.enable_transliteration = config.get('enable_transliteration', True)
        self.log_transliteration = config.get('log_transliteration', True)
        self.transliteration_stats = {
            'total': 0,
            'by_script': {},
            'transliterated': 0
        }
    
    def predict_single(self, name: str) -> Dict[str, Any]:
        """
        Predict gender for a single name with automatic transliteration.
        
        Args:
            name: Input name in any script
            
        Returns:
            Dictionary with prediction results and transliteration info
        """
        original_name = name
        script_detected = "LAT"
        
        if self.enable_transliteration:
            # Transliterate if needed
            name, script_detected = transliterate_name(name)
            
            # Update stats
            self.transliteration_stats['total'] += 1
            if script_detected != "LAT":
                self.transliteration_stats['transliterated'] += 1
            self.transliteration_stats['by_script'][script_detected] = \
                self.transliteration_stats['by_script'].get(script_detected, 0) + 1
            
            if self.log_transliteration and script_detected != "LAT":
                print(f"📝 Transliterated: {original_name} → {name} (Script: {script_detected})")
        
        # Get prediction from parent class
        result = super().predict_single(name)
        
        # Add transliteration info to result
        result['original_name'] = original_name
        result['transliterated_name'] = name
        result['detected_script'] = script_detected
        result['was_transliterated'] = (script_detected != "LAT")
        
        # Override the name field to show original
        result['name'] = original_name
        
        return result
    
    def predict_batch(self, names: List[str]) -> List[Dict[str, Any]]:
        """
        Predict gender for multiple names with automatic transliteration.
        
        Args:
            names: List of names in any script
            
        Returns:
            List of prediction dictionaries with transliteration info
        """
        if not self.enable_transliteration:
            return super().predict_batch(names)
        
        # Process each name individually to handle transliteration
        results = []
        original_names = names.copy()
        transliterated_names = []
        scripts = []
        
        # Transliterate all names
        for name in names:
            trans_name, script = transliterate_name(name)
            transliterated_names.append(trans_name)
            scripts.append(script)
            
            # Update stats
            self.transliteration_stats['total'] += 1
            if script != "LAT":
                self.transliteration_stats['transliterated'] += 1
            self.transliteration_stats['by_script'][script] = \
                self.transliteration_stats['by_script'].get(script, 0) + 1
        
        # Log transliteration summary
        if self.log_transliteration:
            non_latin_count = sum(1 for s in scripts if s != "LAT")
            if non_latin_count > 0:
                print(f"📝 Transliterated {non_latin_count}/{len(names)} names")
                script_summary = {}
                for script in scripts:
                    if script != "LAT":
                        script_summary[script] = script_summary.get(script, 0) + 1
                print(f"   Scripts: {script_summary}")
        
        # Get batch predictions from parent class
        batch_results = super().predict_batch(transliterated_names)
        
        # Add transliteration info to each result
        for i, result in enumerate(batch_results):
            result['original_name'] = original_names[i]
            result['transliterated_name'] = transliterated_names[i]
            result['detected_script'] = scripts[i]
            result['was_transliterated'] = (scripts[i] != "LAT")
            # Override the name field to show original
            result['name'] = original_names[i]
        
        return batch_results
    
    def get_transliteration_stats(self) -> Dict[str, Any]:
        """Get statistics about transliterations performed."""
        return {
            'total_names': self.transliteration_stats['total'],
            'transliterated_count': self.transliteration_stats['transliterated'],
            'transliteration_rate': (
                self.transliteration_stats['transliterated'] / 
                max(1, self.transliteration_stats['total'])
            ),
            'by_script': dict(self.transliteration_stats['by_script'])
        }