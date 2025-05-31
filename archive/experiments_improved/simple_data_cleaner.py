#!/usr/bin/env python3
"""
Versione semplificata e corretta del data cleaner.
- Mantiene primaryName come stringa singola
- NON rimuove duplicati di nomi
- Rimuove solo record chiaramente problematici
"""

import pandas as pd
import numpy as np
import re
import unicodedata
from tqdm import tqdm
import argparse


def has_numbers(text):
    """Check if text contains numbers."""
    return bool(re.search(r'\d', text))


def has_special_chars(text):
    """Check if text contains special characters (excluding allowed ones)."""
    # Allowed: lettere, spazi, apostrofi, trattini
    return bool(re.search(r'[!@#$%^&*()+=\[\]{};:"\\|,<>?/]', text))


def remove_diacritics(text):
    """Remove diacritical marks from text."""
    nfd = unicodedata.normalize('NFD', text)
    return ''.join(c for c in nfd if unicodedata.category(c) != 'Mn')


def clean_dataset(input_file, output_file, 
                  remove_numbers=True,
                  remove_special=True,
                  remove_short=True,
                  remove_long=True,
                  normalize_diacritics=False,
                  min_length=3,
                  max_length=50):
    """
    Pulisce il dataset rimuovendo solo record chiaramente problematici.
    """
    
    print(f"Loading {input_file}...")
    df = pd.read_csv(input_file)
    original_size = len(df)
    
    # Track what we remove
    removed = {
        'numbers': 0,
        'special': 0,
        'too_short': 0,
        'too_long': 0,
        'empty': 0
    }
    
    # Create mask for rows to keep
    keep_mask = pd.Series(True, index=df.index)
    
    print("Analyzing dataset...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        name = row['primaryName']
        
        # Check empty
        if pd.isna(name) or name.strip() == '':
            keep_mask[idx] = False
            removed['empty'] += 1
            continue
        
        # Check numbers
        if remove_numbers and has_numbers(name):
            keep_mask[idx] = False
            removed['numbers'] += 1
            continue
            
        # Check special chars
        if remove_special and has_special_chars(name):
            keep_mask[idx] = False
            removed['special'] += 1
            continue
            
        # Check length
        if remove_short and len(name) < min_length:
            keep_mask[idx] = False
            removed['too_short'] += 1
            continue
            
        if remove_long and len(name) > max_length:
            keep_mask[idx] = False
            removed['too_long'] += 1
            continue
    
    # Apply mask
    df_clean = df[keep_mask].copy()
    
    # Normalize diacritics if requested (but keep original structure)
    if normalize_diacritics:
        print("Normalizing diacritics...")
        df_clean['primaryName'] = df_clean['primaryName'].apply(remove_diacritics)
    
    # Report
    print("\n=== Cleaning Report ===")
    print(f"Original size: {original_size:,}")
    print(f"Removed:")
    for reason, count in removed.items():
        if count > 0:
            print(f"  {reason}: {count:,}")
    print(f"Total removed: {original_size - len(df_clean):,} ({(original_size - len(df_clean))/original_size*100:.2f}%)")
    print(f"Final size: {len(df_clean):,}")
    
    # Save
    df_clean.to_csv(output_file, index=False)
    print(f"\nCleaned dataset saved to {output_file}")
    
    # Also save detailed stats
    stats_file = output_file.replace('.csv', '_stats.txt')
    with open(stats_file, 'w') as f:
        f.write(f"Cleaning Statistics\n")
        f.write(f"==================\n")
        f.write(f"Original size: {original_size:,}\n")
        f.write(f"Final size: {len(df_clean):,}\n")
        f.write(f"Removed: {original_size - len(df_clean):,} ({(original_size - len(df_clean))/original_size*100:.2f}%)\n\n")
        f.write("Removed by reason:\n")
        for reason, count in removed.items():
            f.write(f"  {reason}: {count:,}\n")


def analyze_dataset(input_file):
    """Just analyze without modifying."""
    print(f"Analyzing {input_file}...")
    df = pd.read_csv(input_file)
    
    stats = {
        'total': len(df),
        'with_numbers': 0,
        'with_special': 0,
        'too_short': 0,
        'too_long': 0,
        'with_diacritics': 0,
        'single_word': 0,
        'two_words': 0,
        'three_plus_words': 0
    }
    
    print("Scanning dataset...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        name = str(row['primaryName']) if pd.notna(row['primaryName']) else ''
        
        if has_numbers(name):
            stats['with_numbers'] += 1
        if has_special_chars(name):
            stats['with_special'] += 1
        if len(name) < 3:
            stats['too_short'] += 1
        if len(name) > 50:
            stats['too_long'] += 1
        if name != remove_diacritics(name):
            stats['with_diacritics'] += 1
            
        words = len(name.split())
        if words == 1:
            stats['single_word'] += 1
        elif words == 2:
            stats['two_words'] += 1
        elif words >= 3:
            stats['three_plus_words'] += 1
    
    print("\n=== Dataset Analysis ===")
    print(f"Total records: {stats['total']:,}")
    print(f"\nPotential issues:")
    print(f"  With numbers: {stats['with_numbers']:,}")
    print(f"  With special chars: {stats['with_special']:,}")
    print(f"  Too short (<3 chars): {stats['too_short']:,}")
    print(f"  Too long (>50 chars): {stats['too_long']:,}")
    print(f"\nCharacteristics:")
    print(f"  With diacritics: {stats['with_diacritics']:,} ({stats['with_diacritics']/stats['total']*100:.1f}%)")
    print(f"\nStructure:")
    print(f"  Single word: {stats['single_word']:,} ({stats['single_word']/stats['total']*100:.1f}%)")
    print(f"  Two words: {stats['two_words']:,} ({stats['two_words']/stats['total']*100:.1f}%)")
    print(f"  Three+ words: {stats['three_plus_words']:,} ({stats['three_plus_words']/stats['total']*100:.1f}%)")
    
    # Show examples of problematic names
    print("\nExamples of problematic names:")
    problematic = []
    for idx, row in df.iterrows():
        name = str(row['primaryName']) if pd.notna(row['primaryName']) else ''
        if has_numbers(name) or has_special_chars(name) or len(name) < 3 or len(name) > 50:
            problematic.append(name)
            if len(problematic) >= 10:
                break
    
    for name in problematic:
        print(f"  '{name}'")


def main():
    parser = argparse.ArgumentParser(description="Simple dataset cleaner")
    parser.add_argument("--input", default="training_dataset.csv")
    parser.add_argument("--output", default="training_dataset_clean.csv")
    parser.add_argument("--analyze-only", action="store_true",
                        help="Only analyze, don't clean")
    parser.add_argument("--keep-numbers", action="store_true",
                        help="Keep names with numbers")
    parser.add_argument("--keep-special", action="store_true",
                        help="Keep names with special characters")
    parser.add_argument("--normalize-diacritics", action="store_true",
                        help="Remove diacritical marks")
    parser.add_argument("--min-length", type=int, default=3,
                        help="Minimum name length")
    parser.add_argument("--max-length", type=int, default=50,
                        help="Maximum name length")
    
    args = parser.parse_args()
    
    if args.analyze_only:
        analyze_dataset(args.input)
    else:
        clean_dataset(
            args.input,
            args.output,
            remove_numbers=not args.keep_numbers,
            remove_special=not args.keep_special,
            remove_short=True,
            remove_long=True,
            normalize_diacritics=args.normalize_diacritics,
            min_length=args.min_length,
            max_length=args.max_length
        )


if __name__ == "__main__":
    main()
