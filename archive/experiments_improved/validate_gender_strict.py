#!/usr/bin/env python3
"""
Versione STRICT del validatore che identifica SOLO errori inequivocabili.
Ignora tutti i nomi ambigui o culturalmente variabili.
"""

import pandas as pd
import numpy as np
from typing import Set, Dict, List, Tuple
import argparse
from tqdm import tqdm


def get_definite_female_names() -> Set[str]:
    """Nomi SOLO ed ESCLUSIVAMENTE femminili in TUTTE le culture."""
    return {
        # Inequivocabilmente femminili
        'anna', 'maria', 'giulia', 'francesca', 'laura', 'silvia',
        'elena', 'sara', 'chiara', 'alessandra', 'monica', 'valentina',
        'martina', 'federica', 'ilaria', 'elisabetta', 'lucia', 'rosa',
        
        'mary', 'elizabeth', 'jennifer', 'patricia', 'susan', 'jessica',
        'sarah', 'nancy', 'betty', 'helen', 'sandra', 'michelle',
        'kimberly', 'deborah', 'lisa', 'dorothy', 'amanda', 'melissa',
        'stephanie', 'rebecca', 'virginia', 'kathleen', 'christina',
        'janet', 'catherine', 'christine', 'samantha', 'emma', 'sophia',
        'isabella', 'emily', 'poppy', 'lily', 'lucy', 'sophie',
        
        'carmen', 'isabel', 'dolores', 'beatriz', 'natalia',
        
        'marie', 'jeanne', 'francoise', 'monique', 'nathalie',
        'isabelle', 'jacqueline', 'sylvie', 'martine',
        
        'olga', 'tatiana', 'svetlana', 'marina', 'galina',
        'anastasia', 'ekaterina', 'katarina',
        
        'fatima', 'aisha', 'yasmin'
    }


def get_definite_male_names() -> Set[str]:
    """Nomi SOLO ed ESCLUSIVAMENTE maschili in TUTTE le culture."""
    return {
        # Inequivocabilmente maschili
        'giuseppe', 'giovanni', 'francesco', 'mario', 'luigi', 'antonio',
        'vincenzo', 'pietro', 'carlo', 'paolo', 'marco', 'roberto',
        'alberto', 'sergio', 'massimo', 'giorgio', 'alessandro', 'stefano',
        
        'james', 'john', 'robert', 'michael', 'william', 'david',
        'richard', 'joseph', 'thomas', 'charles', 'christopher',
        'matthew', 'anthony', 'donald', 'mark', 'steven', 'kenneth',
        'george', 'kevin', 'brian', 'edward', 'ronald', 'jason',
        'frank', 'scott', 'eric', 'stephen', 'raymond', 'gregory',
        'benjamin', 'patrick', 'jack', 'henry', 'justin', 'adam',
        
        'jose', 'manuel', 'juan', 'pedro', 'carlos', 'luis',
        'jorge', 'pablo', 'sergio', 'eduardo', 'javier', 'rafael',
        'enrique', 'ricardo', 'alejandro', 'santiago', 'diego',
        
        'pierre', 'michel', 'jacques', 'bernard', 'francois',
        
        'vladimir', 'aleksandr', 'sergey', 'mikhail', 'ivan',
        'nikolai', 'dmitry', 'igor', 'boris', 'konstantin'
    }


def check_only_clear_errors(df: pd.DataFrame) -> pd.DataFrame:
    """Trova SOLO errori chiari e inequivocabili."""
    
    female_names = get_definite_female_names()
    male_names = get_definite_male_names()
    
    clear_errors = []
    
    print("Checking for clear gender mislabeling...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        name = row['primaryName']
        gender = row['gender']
        
        # Estrai primo nome
        first_name = name.split()[0].lower() if name else ""
        
        # Check solo errori chiari
        if first_name in female_names and gender == 'M':
            clear_errors.append({
                'index': idx,
                'name': name,
                'first_name': first_name,
                'labeled': 'M',
                'should_be': 'W',
                'type': 'clear_female_as_male'
            })
        elif first_name in male_names and gender == 'W':
            clear_errors.append({
                'index': idx,
                'name': name,
                'first_name': first_name,
                'labeled': 'W',
                'should_be': 'M',
                'type': 'clear_male_as_female'
            })
    
    return pd.DataFrame(clear_errors)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="training_dataset.csv")
    parser.add_argument("--output", default="training_dataset_validated_strict.csv")
    parser.add_argument("--errors", default="clear_gender_errors.csv")
    
    args = parser.parse_args()
    
    print(f"Loading {args.input}...")
    df = pd.read_csv(args.input)
    
    # Trova solo errori chiari
    errors_df = check_only_clear_errors(df)
    
    if len(errors_df) > 0:
        print(f"\n=== Clear Gender Errors Found ===")
        print(f"Total: {len(errors_df)} ({len(errors_df)/len(df)*100:.2f}%)")
        
        # Analisi per tipo
        by_type = errors_df['type'].value_counts()
        print(f"\nBy type:")
        for t, count in by_type.items():
            print(f"  {t}: {count}")
        
        # Top nomi
        print(f"\nTop misclassified names:")
        top_names = errors_df['first_name'].value_counts().head(20)
        for name, count in top_names.items():
            print(f"  {name}: {count}")
        
        # Salva errori
        errors_df.to_csv(args.errors, index=False)
        print(f"\nErrors saved to {args.errors}")
        
        # Esempi
        print(f"\nExamples:")
        for _, row in errors_df.head(10).iterrows():
            print(f"  {row['name']} is labeled {row['labeled']} but should be {row['should_be']}")
        
        # Rimuovi dal dataset
        df_clean = df.drop(errors_df['index'])
        df_clean.to_csv(args.output, index=False)
        print(f"\nCleaned dataset saved to {args.output}")
        print(f"Removed: {len(errors_df)} records")
        print(f"Remaining: {len(df_clean)} records")
    else:
        print("\nNo clear errors found!")
        df.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
