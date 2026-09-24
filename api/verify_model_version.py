#!/usr/bin/env python3
"""
Script per verificare che il modello deployato sia quello corretto
confrontando i risultati con il modello locale.
"""

import sys
import os
import json
import pickle
import torch
import urllib.request
import urllib.parse
from typing import Dict, List

# Aggiungi paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def load_local_model():
    """Carica il modello locale per confronto."""
    from final_predictor import FinalGenderPredictor

    config = {
        'model_path': '../models/best_v3_model/models/model.pth',
        'preprocessor_path': '../models/best_v3_model/preprocessor.pkl',
        'feature_extractor_path': '../models/best_v3_model/feature_extractor.pkl',
        'optimal_threshold': 0.52
    }

    predictor = FinalGenderPredictor(config)
    predictor.load_model()
    return predictor

def test_api_predictions(api_url: str, names: List[str]) -> List[Dict]:
    """Testa le predizioni dell'API usando urllib."""
    data = json.dumps({"names": names, "return_metadata": True}).encode('utf-8')

    req = urllib.request.Request(
        f"{api_url}/predict",
        data=data,
        headers={'Content-Type': 'application/json'}
    )

    try:
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode('utf-8'))
            return result['predictions']
    except urllib.error.HTTPError as e:
        error_body = e.read().decode('utf-8') if e.fp else 'No error body'
        raise Exception(f"API Error: {e.code} - {error_body}")
    except Exception as e:
        raise Exception(f"Request failed: {str(e)}")

def compare_predictions(local_results: List[Dict], api_results: List[Dict], names: List[str]):
    """Confronta i risultati locali con quelli dell'API."""
    print("\n📊 Confronto Predizioni:")
    print("=" * 80)
    print(f"{'Nome':<20} {'Locale':<15} {'API':<15} {'Match':<10} {'Diff Prob':<10}")
    print("-" * 80)

    all_match = True
    max_prob_diff = 0.0

    for i, name in enumerate(names):
        local = local_results[i]
        api = api_results[i]

        # Confronta genere predetto
        gender_match = local['predicted_gender'] == api['predicted_gender']

        # Confronta probabilità
        prob_diff = abs(local['probability_female'] - api['probability_female'])
        max_prob_diff = max(max_prob_diff, prob_diff)

        # Status
        if gender_match and prob_diff < 0.01:
            status = "✅"
        elif gender_match and prob_diff < 0.05:
            status = "⚠️"
        else:
            status = "❌"
            all_match = False

        print(f"{name:<20} {local['predicted_gender']:<15} {api['predicted_gender']:<15} "
              f"{status:<10} {prob_diff:.4f}")

    print("-" * 80)

    return all_match, max_prob_diff

def check_model_metadata(api_url: str):
    """Verifica i metadati del modello usando urllib."""
    try:
        with urllib.request.urlopen(f"{api_url}/stats") as response:
            stats = json.loads(response.read().decode('utf-8'))
            print("\n📋 Metadati Modello API:")
            print(f"   Model Path: {stats['model_info']['model_path']}")
            print(f"   Threshold: {stats['model_info']['threshold']}")
            print(f"   Expected Accuracy: {stats['model_info']['expected_accuracy']}")
    except Exception as e:
        print(f"\n⚠️  Non riesco a recuperare i metadati: {e}")

def main():
    """Main test function."""
    print("🔍 Verifica Versione Modello Deployato")
    print("=====================================")

    # Trova URL dell'API
    import subprocess
    try:
        result = subprocess.run(['modal', 'app', 'list'], capture_output=True, text=True)
        if result.returncode != 0:
            print("❌ Errore nell'esecuzione di 'modal app list'")
            print("   Assicurati che Modal CLI sia installato e configurato")
            return 1

        api_url = None

        for line in result.stdout.split('\n'):
            if 'gender-prediction-academic' in line and 'https://' in line:
                # Estrai URL
                import re
                match = re.search(r'https://[^\s]+', line)
                if match:
                    api_url = match.group(0)
                    break

        if not api_url:
            print("❌ Non riesco a trovare l'URL dell'API deployata!")
            print("   Assicurati di aver eseguito: modal deploy modal_deployment.py")
            print("\n   Output di 'modal app list':")
            print(result.stdout[:500])  # Mostra prime 500 caratteri per debug
            return 1
    except FileNotFoundError:
        print("❌ Modal CLI non trovato!")
        print("   Installa con: pip install modal")
        return 1
    except Exception as e:
        print(f"❌ Errore inaspettato: {e}")
        return 1

    print(f"✅ API URL: {api_url}")

    # Nomi di test diversificati
    test_names = [
        "Mario Rossi",      # Italiano M
        "Anna Bianchi",     # Italiano F
        "José García",      # Spagnolo M
        "María López",      # Spagnolo F
        "John Smith",       # Inglese M
        "Emma Wilson",      # Inglese F
        "François Dupont",  # Francese M
        "Sophie Martin",    # Francese F
        "Владимир",         # Russo M
        "Екатерина",        # Russo F
        "أحمد",             # Arabo M
        "فاطمة",            # Arabo F
    ]

    try:
        # 1. Carica modello locale
        print("\n1️⃣ Carico modello locale...")
        local_predictor = load_local_model()

        # 2. Predizioni locali
        print("2️⃣ Eseguo predizioni locali...")
        local_results = local_predictor.predict_batch(test_names)

        # 3. Predizioni API
        print("3️⃣ Eseguo predizioni API...")
        api_results = test_api_predictions(api_url, test_names)

        # 4. Confronta risultati
        all_match, max_prob_diff = compare_predictions(local_results, api_results, test_names)

        # 5. Check metadati
        check_model_metadata(api_url)

        # 6. Risultato finale
        print(f"\n📈 Risultato:")
        print(f"   Tutte le predizioni coincidono: {'✅ SI' if all_match else '❌ NO'}")
        print(f"   Massima differenza probabilità: {max_prob_diff:.4f}")

        if all_match and max_prob_diff < 0.01:
            print("\n✅ Il modello deployato è IDENTICO a quello locale!")
            return 0
        elif all_match and max_prob_diff < 0.05:
            print("\n⚠️ Il modello sembra essere lo stesso ma con piccole differenze numeriche.")
            print("   Questo potrebbe essere dovuto a differenze di precisione float.")
            return 0
        else:
            print("\n❌ Il modello deployato DIFFERISCE da quello locale!")
            print("   Potrebbe essere una versione cached o diversa.")
            return 1

    except Exception as e:
        print(f"\n❌ Errore durante il test: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
