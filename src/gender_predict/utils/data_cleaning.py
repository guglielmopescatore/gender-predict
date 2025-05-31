"""
Data cleaning and encoding utilities.

This module provides tools for diagnosing and fixing encoding issues,
cleaning datasets, and validating data quality.
"""

#!/usr/bin/env python3
"""
Script per diagnosticare e risolvere problemi di encoding nel dataset.
Risolve caratteri corrotti, encoding mismatch e problemi di visualizzazione.
"""

import pandas as pd
import numpy as np
import re
import chardet
import unicodedata
from tqdm import tqdm
import argparse
import os


def detect_encoding(file_path, sample_size=10000):
    """Rileva l'encoding del file."""
    print(f"🔍 Rilevamento encoding per {file_path}...")
    
    with open(file_path, 'rb') as f:
        raw_data = f.read(sample_size)
        result = chardet.detect(raw_data)
    
    print(f"   Encoding rilevato: {result['encoding']} (confidenza: {result['confidence']:.2%})")
    return result


def analyze_problematic_characters(df, column='primaryName'):
    """Analizza caratteri problematici nel dataset."""
    print(f"\n📊 Analisi caratteri problematici nella colonna '{column}'...")
    
    problematic_patterns = {
        'replacement_chars': r'�+',  # Caratteri di sostituzione
        'null_bytes': r'\x00',      # Null bytes
        'control_chars': r'[\x00-\x1F\x7F-\x9F]',  # Caratteri di controllo
        'mixed_encoding': r'[À-ÿ]{2,}',  # Possibili problemi di encoding doppio
        'suspicious_sequences': r'[^\w\s\-\'\.àáâäãåæçéèêëíìîïñóòôöõøúùûüýÿ]+',
    }
    
    stats = {}
    examples = {}
    
    for pattern_name, pattern in problematic_patterns.items():
        matches = df[column].str.contains(pattern, regex=True, na=False)
        count = matches.sum()
        stats[pattern_name] = count
        
        if count > 0:
            # Trova esempi
            examples[pattern_name] = df[matches][column].head(10).tolist()
    
    # Stampa risultati
    print(f"\nProblemi trovati in {len(df)} record:")
    for pattern_name, count in stats.items():
        if count > 0:
            print(f"  {pattern_name}: {count:,} record ({count/len(df)*100:.2f}%)")
            print(f"    Esempi: {examples[pattern_name][:3]}")
    
    return stats, examples


def clean_encoding_issues(text):
    """Pulisce problemi di encoding in una stringa."""
    if pd.isna(text) or not isinstance(text, str):
        return text
    
    # 1. Rimuovi caratteri di sostituzione
    text = re.sub(r'�+', '', text)
    
    # 2. Rimuovi null bytes e caratteri di controllo
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]', '', text)
    
    # 3. Normalizza Unicode (decompone e ricompone)
    text = unicodedata.normalize('NFD', text)
    text = unicodedata.normalize('NFC', text)
    
    # 4. Rimuovi spazi multipli
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 5. Rimuovi caratteri non stampabili rimanenti
    text = ''.join(char for char in text if char.isprintable() or char.isspace())
    
    return text


def fix_common_encoding_mistakes(text):
    """Corregge errori di encoding comuni."""
    if pd.isna(text) or not isinstance(text, str):
        return text
    
    # Mapping di caratteri corrotti comuni
    fixes = {
        # UTF-8 interpretato come Latin-1
        'Ã¡': 'á', 'Ã¢': 'â', 'Ã¤': 'ä', 'Ã¥': 'å', 'Ã¦': 'æ',
        'Ã§': 'ç', 'Ã¨': 'è', 'Ã©': 'é', 'Ãª': 'ê', 'Ã«': 'ë',
        'Ã¬': 'ì', 'Ã­': 'í', 'Ã®': 'î', 'Ã¯': 'ï', 'Ã±': 'ñ',
        'Ã²': 'ò', 'Ã³': 'ó', 'Ã´': 'ô', 'Ã¶': 'ö', 'Ã¸': 'ø',
        'Ã¹': 'ù', 'Ãº': 'ú', 'Ã»': 'û', 'Ã¼': 'ü', 'Ã½': 'ý',
        'Ã¿': 'ÿ', 'ÃŸ': 'ß', 'Ã€': 'À', 'Ã': 'Á', 'Ã‚': 'Â',
        
        # Windows-1252 common issues
        'â€™': "'", 'â€œ': '"', 'â€': '"', 'â€"': '–', 'â€"': '—',
        
        # Double encoding issues
        'Ã¡Ã¡': 'á', 'Ã©Ã©': 'é', 'Ã­Ã­': 'í', 'Ã³Ã³': 'ó', 'ÃºÃº': 'ú',
    }
    
    for corrupted, correct in fixes.items():
        text = text.replace(corrupted, correct)
    
    return text


def validate_name_after_cleaning(name):
    """Valida che un nome sia ragionevole dopo la pulizia."""
    if pd.isna(name) or not isinstance(name, str):
        return False
    
    name = name.strip()
    
    # Controlli di base
    if len(name) < 1:
        return False
    
    if len(name) > 100:  # Nomi estremamente lunghi sono sospetti
        return False
    
    # Deve contenere almeno una lettera
    if not re.search(r'[a-zA-ZÀ-ÿ]', name):
        return False
    
    # Non deve essere principalmente numeri o simboli
    letter_ratio = len(re.findall(r'[a-zA-ZÀ-ÿ]', name)) / len(name)
    if letter_ratio < 0.5:
        return False
    
    return True


def clean_dataset(input_file, output_file=None, encoding=None):
    """Pulisce l'intero dataset."""
    print(f"🧹 Pulizia del dataset: {input_file}")
    
    # Auto-rileva encoding se non specificato
    if encoding is None:
        detection = detect_encoding(input_file)
        encoding = detection['encoding']
    
    # Carica il dataset con encoding rilevato
    try:
        print(f"📂 Caricamento con encoding {encoding}...")
        df = pd.read_csv(input_file, encoding=encoding)
    except UnicodeDecodeError:
        print(f"❌ Errore con {encoding}, provo con UTF-8...")
        try:
            df = pd.read_csv(input_file, encoding='utf-8')
        except UnicodeDecodeError:
            print(f"❌ Errore con UTF-8, provo con Latin-1...")
            df = pd.read_csv(input_file, encoding='latin-1')
    
    print(f"   Caricati {len(df):,} record")
    
    # Analizza problemi prima della pulizia
    if 'primaryName' in df.columns:
        analyze_problematic_characters(df, 'primaryName')
    
    # Applica pulizia
    print(f"\n🔧 Applicazione pulizia...")
    
    cleaned_count = 0
    removed_count = 0
    
    for column in df.columns:
        if df[column].dtype == 'object':  # Colonne di testo
            print(f"   Pulizia colonna: {column}")
            
            # Applica pulizia
            original_values = df[column].copy()
            
            # Step 1: Fix encoding issues
            df[column] = df[column].apply(fix_common_encoding_mistakes)
            
            # Step 2: Clean encoding artifacts
            df[column] = df[column].apply(clean_encoding_issues)
            
            # Conta modifiche
            changed = (original_values != df[column]).sum()
            cleaned_count += changed
            print(f"      Modificati: {changed:,} valori")
    
    # Validazione speciale per primaryName
    if 'primaryName' in df.columns:
        print(f"\n✅ Validazione nomi...")
        
        original_size = len(df)
        valid_mask = df['primaryName'].apply(validate_name_after_cleaning)
        df_clean = df[valid_mask].copy()
        removed_count = original_size - len(df_clean)
        
        print(f"   Record rimossi (nomi non validi): {removed_count:,}")
        print(f"   Record rimanenti: {len(df_clean):,}")
        
        # Analizza dopo la pulizia
        print(f"\n📊 Analisi post-pulizia:")
        analyze_problematic_characters(df_clean, 'primaryName')
    else:
        df_clean = df
    
    # Salva risultato
    if output_file is None:
        name, ext = os.path.splitext(input_file)
        output_file = f"{name}_cleaned{ext}"
    
    print(f"\n💾 Salvataggio in {output_file}...")
    df_clean.to_csv(output_file, index=False, encoding='utf-8')
    
    # Report finale
    print(f"\n📋 REPORT FINALE:")
    print(f"   File originale: {input_file}")
    print(f"   File pulito: {output_file}")
    print(f"   Record originali: {len(df):,}")
    print(f"   Record finali: {len(df_clean):,}")
    print(f"   Valori modificati: {cleaned_count:,}")
    print(f"   Record rimossi: {removed_count:,}")
    print(f"   Encoding output: UTF-8")
    
    return df_clean


def sample_and_inspect(input_file, n_samples=20):
    """Campiona e ispeziona record problematici."""
    print(f"🔍 Ispezione campione da {input_file}")
    
    # Auto-rileva encoding
    detection = detect_encoding(input_file)
    encoding = detection['encoding']
    
    # Carica dataset
    try:
        df = pd.read_csv(input_file, encoding=encoding)
    except:
        df = pd.read_csv(input_file, encoding='utf-8', errors='replace')
    
    if 'primaryName' not in df.columns:
        print("❌ Colonna 'primaryName' non trovata")
        return
    
    # Trova record con caratteri problematici
    problematic = df[df['primaryName'].str.contains(r'�|[\x00-\x1F]', regex=True, na=False)]
    
    if len(problematic) == 0:
        print("✅ Nessun record problematico trovato!")
        return
    
    print(f"\n📋 Trovati {len(problematic):,} record problematici")
    print(f"Campione di {min(n_samples, len(problematic))} record:")
    
    for i, (idx, row) in enumerate(problematic.head(n_samples).iterrows()):
        name = row['primaryName']
        print(f"\n{i+1:2d}. Record {idx}")
        print(f"    Raw: {repr(name)}")
        print(f"    Displayed: {name}")
        
        # Mostra bytes
        if isinstance(name, str):
            try:
                bytes_repr = name.encode('utf-8')
                print(f"    Bytes: {bytes_repr}")
            except:
                print(f"    Bytes: [encoding error]")
        
        # Prova pulizia
        cleaned = clean_encoding_issues(fix_common_encoding_mistakes(name))
        if cleaned != name:
            print(f"    Cleaned: {cleaned}")


def main():
    parser = argparse.ArgumentParser(description="Tool per diagnosticare e riparare problemi di encoding")
    
    subparsers = parser.add_subparsers(dest='command', help='Comando da eseguire')
    
    # Comando analyze
    analyze_parser = subparsers.add_parser('analyze', help='Analizza problemi nel dataset')
    analyze_parser.add_argument('input_file', help='File CSV da analizzare')
    analyze_parser.add_argument('--samples', type=int, default=20, help='Numero di campioni da mostrare')
    
    # Comando clean
    clean_parser = subparsers.add_parser('clean', help='Pulisce il dataset')
    clean_parser.add_argument('input_file', help='File CSV da pulire')
    clean_parser.add_argument('--output', help='File di output (default: input_cleaned.csv)')
    clean_parser.add_argument('--encoding', help='Encoding da usare per leggere il file')
    
    # Comando inspect
    inspect_parser = subparsers.add_parser('inspect', help='Ispeziona record problematici')
    inspect_parser.add_argument('input_file', help='File CSV da ispezionare')
    inspect_parser.add_argument('--samples', type=int, default=10, help='Numero di campioni da ispezionare')
    
    args = parser.parse_args()
    
    if args.command == 'analyze':
        # Rileva encoding
        detect_encoding(args.input_file)
        
        # Carica e analizza
        try:
            df = pd.read_csv(args.input_file)
            if 'primaryName' in df.columns:
                analyze_problematic_characters(df, 'primaryName')
            else:
                print("❌ Colonna 'primaryName' non trovata")
        except Exception as e:
            print(f"❌ Errore nel caricamento: {e}")
    
    elif args.command == 'clean':
        clean_dataset(args.input_file, args.output, args.encoding)
    
    elif args.command == 'inspect':
        sample_and_inspect(args.input_file, args.samples)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
