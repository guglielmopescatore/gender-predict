# ImprovedNamePreprocessor - Configuration Options

## Current Implementation Status

Il V4-R1 userà `ImprovedNamePreprocessor` con parametri **default hardcoded**:

```python
# Default parameters in train_model.py line ~485
preprocessor = ImprovedNamePreprocessor()  # Se args.advanced_preprocessing=True
```

## Default Parameters Active in V4-R1

```python
ImprovedNamePreprocessor(
    max_name_length=20,           # Same as base
    max_surname_length=20,        # Same as base  
    normalize_diacritics=True,    # 🎯 NEW: Rimuove accenti (José → Jose)
    handle_hyphens='keep'         # 🎯 NEW: Mantiene trattini nei nomi
)
```

## Key Improvements Over Base Preprocessor

### ✅ **Smart Name Splitting**
- Riconosce surname prefixes: "de", "di", "van", "mc", "o'", etc.
- Gestione intelligente nomi composti: "María José de la Cruz"
- Ordine corretto first/middle/last name

### ✅ **Diacritic Normalization** 
- José → Jose
- María → Maria  
- François → Francois
- **Impact:** Risolve molti errori su nomi internazionali

### ✅ **Hyphen Handling**
- Jean-Pierre → mantiene trattino
- Mary-Ann → mantiene struttura
- **Impact:** Preserva informazione morfologica

### ✅ **Enhanced Vocabulary**
- Supporta spazi in nomi multi-parte
- Gestione apostrofi (O'Connor)
- Unknown char token per robustezza

## Expected Impact per Feature

| Feature | Error Category | Estimated Gain |
|---------|---------------|----------------|
| Smart splitting | Compound names (93.7% errors) | +0.8% |
| Diacritic norm | International names (16.5% errors) | +0.3% |
| Hyphen handling | Structured names | +0.1% |
| **TOTAL** | **Combined effect** | **+1.2%** |

## Possible Future Optimizations

### 🔧 **Advanced Parameters** (for future V4 rounds)
```python
# Potrebbero essere aggiunti al train_model.py:
--normalize_diacritics / --no-normalize_diacritics
--handle_hyphens [keep|remove|space]
--surname_detection_mode [strict|relaxed]
--compound_name_strategy [intelligent|simple]
```

### 📊 **A/B Testing Opportunities**
1. **Diacritics ON vs OFF** - Some models might benefit from keeping accents
2. **Hyphen strategies** - Test remove vs keep vs space replacement  
3. **Name order validation** - Use gender hints for name order detection

## Current Strategy: Use Defaults

Per V4-R1 usiamo i **default ottimali** per semplicità:
- Massimizza compatibility con dataset existing
- Benefici comprovati su nomi internazionali
- Zero risk di breaking changes

## Monitoring Points

Durante il training di V4-R1, monitora:

1. **Vocabulary size changes** - ImprovedPreprocessor ha vocab diverso
2. **Name processing differences** - Log alcuni esempi di preprocessing  
3. **Training stability** - Stessi parametri convergenza di B0
4. **Error pattern changes** - Focus su nomi composti e internazionali

## Implementation Notes

```python
# train_model.py setup automaticamente:
if args.advanced_preprocessing:
    preprocessor = ImprovedNamePreprocessor()  # Con defaults ottimali
    # Preprocessor verrà salvato automaticamente in experiment/preprocessor.pkl
```

✅ **Ready for V4-R1 execution** con configurazione default ottimale!