#!/usr/bin/env python3
"""
Crea un dataset di esempio per testare il sistema.
"""

import pandas as pd
import numpy as np
import os

def create_sample_dataset(size=1000):
    """Crea un dataset di esempio per test."""
    
    np.random.seed(42)
    
    # Nomi comuni per test
    male_names = [
        "John Smith", "Michael Johnson", "David Brown", "James Wilson", "Robert Miller",
        "William Davis", "Christopher Garcia", "Matthew Rodriguez", "Anthony Martinez", "Mark Anderson",
        "Donald Taylor", "Steven Thomas", "Kenneth Hernandez", "Joshua Moore", "Kevin Martin",
        "Brian Jackson", "George Thompson", "Edward White", "Ronald Lopez", "Timothy Lee",
        "Daniel Gonzalez", "Joseph Harris", "Jason Clark", "Ryan Lewis", "Jacob Robinson"
    ]
    
    female_names = [
        "Mary Jones", "Patricia Williams", "Jennifer Brown", "Linda Davis", "Elizabeth Miller",
        "Barbara Wilson", "Susan Moore", "Jessica Taylor", "Sarah Anderson", "Karen Thomas",
        "Nancy Jackson", "Lisa White", "Betty Thompson", "Dorothy Lewis", "Sandra Lee",
        "Ashley Clark", "Kimberly Rodriguez", "Emily Gonzalez", "Donna Martinez", "Margaret Garcia",
        "Carol Hernandez", "Ruth Martin", "Sharon Lopez", "Michelle Robinson", "Laura Hall"
    ]
    
    # Genera campioni bilanciati
    male_samples = np.random.choice(male_names, size=size//2, replace=True)
    female_samples = np.random.choice(female_names, size=size//2, replace=True)
    
    # Combina in un dataset
    names = list(male_samples) + list(female_samples)
    genders = ['M'] * len(male_samples) + ['W'] * len(female_samples)
    ids = [f"nm{i:07d}" for i in range(len(names))]
    
    # Mescola i dati
    indices = np.random.permutation(len(names))
    
    df = pd.DataFrame({
        'primaryName': [names[i] for i in indices],
        'gender': [genders[i] for i in indices],
        'nconst': [ids[i] for i in indices]
    })
    
    return df

def main():
    """Crea dataset di esempio."""
    
    print("📊 Creating sample datasets...")
    
    # Crea directory data se non esiste
    os.makedirs('data/raw', exist_ok=True)
    
    # Crea dataset di training (più grande)
    train_df = create_sample_dataset(5000)
    train_path = 'data/raw/sample_training.csv'
    train_df.to_csv(train_path, index=False)
    print(f"✅ Training dataset created: {train_path} ({len(train_df)} samples)")
    
    # Crea dataset di test (più piccolo)
    test_df = create_sample_dataset(1000)
    test_path = 'data/raw/sample_test.csv'
    test_df.to_csv(test_path, index=False)
    print(f"✅ Test dataset created: {test_path} ({len(test_df)} samples)")
    
    # Mostra statistiche
    print(f"\nDataset statistics:")
    print(f"Training - M: {(train_df['gender'] == 'M').sum()}, W: {(train_df['gender'] == 'W').sum()}")
    print(f"Test     - M: {(test_df['gender'] == 'M').sum()}, W: {(test_df['gender'] == 'W').sum()}")
    
    print(f"\nSample data:")
    print(train_df.head())
    
    print(f"\n🎯 Now you can test the system:")
    print(f"  python scripts/train_model.py --round 1 --data_file {train_path} --epochs 5")

if __name__ == "__main__":
    main()
