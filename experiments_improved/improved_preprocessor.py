"""
Preprocessore migliorato che gestisce meglio:
- Middle names
- Ordine nome-cognome
- Caratteri diacritici
- Nomi con trattini
"""

import unicodedata
from typing import Tuple, List, Optional
import pickle


class ImprovedNamePreprocessor:
    """Preprocessore migliorato per nomi con gestione intelligente."""
    
    def __init__(self, max_name_length=20, max_surname_length=20, 
                 normalize_diacritics=True, handle_hyphens='keep'):
        """
        Args:
            max_name_length: Lunghezza massima per i nomi
            max_surname_length: Lunghezza massima per i cognomi
            normalize_diacritics: Se True, rimuove i diacritici
            handle_hyphens: Come gestire i trattini ('keep', 'remove', 'space')
        """
        self.max_name_length = max_name_length
        self.max_surname_length = max_surname_length
        self.normalize_diacritics = normalize_diacritics
        self.handle_hyphens = handle_hyphens
        
        # Vocabolario base senza diacritici
        self.char_to_idx = {'<PAD>': 0, '<UNK>': 1}
        
        # Solo caratteri ASCII base
        for i, c in enumerate("abcdefghijklmnopqrstuvwxyz"):
            self.char_to_idx[c] = i + 2
        
        # Aggiungi trattino e apostrofo se necessario
        if handle_hyphens == 'keep':
            self.char_to_idx['-'] = len(self.char_to_idx)
        self.char_to_idx["'"] = len(self.char_to_idx)
        
        # Spazio per nomi multi-parte
        self.char_to_idx[' '] = len(self.char_to_idx)
        
        self.idx_to_char = {v: k for k, v in self.char_to_idx.items()}
        self.vocab_size = len(self.char_to_idx)
        
        # Preposizioni comuni che indicano l'inizio del cognome
        self.surname_prefixes = {
            'de', 'di', 'del', 'della', 'delle', 'degli', 'da',  # Italiano
            'van', 'von', 'der', 'den', 'ter',  # Olandese/Tedesco
            'le', 'la', 'du', 'des',  # Francese
            'mac', 'mc', "o'", 'd\'',  # Irlandese/Scozzese
            'bin', 'ibn', 'al', 'el'  # Arabo
        }
    
    def normalize_text(self, text: str) -> str:
        """Normalizza il testo secondo le impostazioni."""
        if not isinstance(text, str):
            return ""
        
        # Normalizza spazi
        text = ' '.join(text.split())
        
        # Gestisci diacritici
        if self.normalize_diacritics:
            # Decompose e rimuovi combining characters
            nfd = unicodedata.normalize('NFD', text)
            text = ''.join(c for c in nfd if unicodedata.category(c) != 'Mn')
        
        # Gestisci trattini
        if self.handle_hyphens == 'remove':
            text = text.replace('-', '')
        elif self.handle_hyphens == 'space':
            text = text.replace('-', ' ')
        
        # Converti in minuscolo
        text = text.lower()
        
        return text
    
    def intelligent_split(self, full_name: str) -> Tuple[str, str, str]:
        """
        Divide intelligentemente un nome completo in first, middle, last.
        
        Returns:
            (first_name, middle_name, last_name)
        """
        parts = full_name.split()
        
        if len(parts) == 0:
            return "", "", ""
        elif len(parts) == 1:
            return parts[0], "", ""
        elif len(parts) == 2:
            return parts[0], "", parts[1]
        
        # Più di 2 parti - serve logica intelligente
        
        # Cerca preposizioni che indicano cognome
        surname_start = -1
        for i, part in enumerate(parts[1:], 1):
            if part.lower() in self.surname_prefixes:
                surname_start = i
                break
        
        if surname_start > 0:
            # Trovata preposizione
            first = parts[0]
            middle = ' '.join(parts[1:surname_start]) if surname_start > 1 else ""
            last = ' '.join(parts[surname_start:])
        else:
            # Euristica: considera compound surnames comuni
            # Se le ultime due parti sono corte, potrebbero essere cognome composto
            if len(parts) >= 4 and len(parts[-2]) <= 3:
                first = parts[0]
                middle = ' '.join(parts[1:-2])
                last = ' '.join(parts[-2:])
            else:
                # Default: primo = nome, ultimo = cognome, resto = middle
                first = parts[0]
                middle = ' '.join(parts[1:-1]) if len(parts) > 2 else ""
                last = parts[-1]
        
        return first, middle, last
    
    def name_to_indices(self, name: str, max_length: int) -> List[int]:
        """Converte un nome in sequenza di indici."""
        name = self.normalize_text(name)
        
        indices = []
        for c in name[:max_length]:
            if c in self.char_to_idx:
                indices.append(self.char_to_idx[c])
            else:
                indices.append(self.char_to_idx['<UNK>'])  # Unknown char
        
        # Padding
        indices.extend([0] * (max_length - len(indices)))
        
        return indices
    
    def preprocess_name(self, full_name: str, 
                       use_middle_name: bool = True) -> dict:
        """
        Preprocessa un nome completo con gestione intelligente.
        
        Args:
            full_name: Nome completo
            use_middle_name: Se True, concatena middle name al cognome
            
        Returns:
            Dict con first_name e last_name come liste di indici
        """
        # Normalizza
        full_name = self.normalize_text(full_name)
        
        # Dividi intelligentemente
        first, middle, last = self.intelligent_split(full_name)
        
        # Gestisci middle name
        if use_middle_name and middle:
            # Strategia: concatena al cognome con spazio
            last = f"{middle} {last}" if last else middle
        
        # Converti in indici
        first_indices = self.name_to_indices(first, self.max_name_length)
        last_indices = self.name_to_indices(last, self.max_surname_length)
        
        return {
            'first_name': first_indices,
            'last_name': last_indices,
            'first_raw': first,
            'middle_raw': middle,
            'last_raw': last
        }
    
    def validate_name_gender_consistency(self, first_name: str, 
                                       last_name: str, 
                                       gender: str) -> float:
        """
        Valida se l'ordine nome-cognome sembra corretto dato il genere.
        
        Returns:
            Confidence score (0-1) che l'ordine sia corretto
        """
        # Pattern comuni italiani
        if gender == 'M':
            if first_name.endswith('a') and not first_name.endswith('ia'):
                # Nome maschile che finisce in 'a' è raro
                return 0.3
            if first_name.endswith('o') or first_name.endswith('e'):
                return 0.9
        elif gender == 'W':
            if first_name.endswith('o'):
                # Nome femminile che finisce in 'o' è molto raro
                return 0.2
            if first_name.endswith('a') or first_name.endswith('e'):
                return 0.9
        
        return 0.5  # Neutrale
    
    def save(self, path: str):
        """Salva il preprocessore."""
        with open(path, 'wb') as f:
            pickle.dump({
                'max_name_length': self.max_name_length,
                'max_surname_length': self.max_surname_length,
                'normalize_diacritics': self.normalize_diacritics,
                'handle_hyphens': self.handle_hyphens,
                'char_to_idx': self.char_to_idx,
                'idx_to_char': self.idx_to_char,
                'vocab_size': self.vocab_size,
                'surname_prefixes': self.surname_prefixes
            }, f)
    
    @classmethod
    def load(cls, path: str):
        """Carica un preprocessore."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        preprocessor = cls(
            max_name_length=data['max_name_length'],
            max_surname_length=data['max_surname_length'],
            normalize_diacritics=data.get('normalize_diacritics', True),
            handle_hyphens=data.get('handle_hyphens', 'keep')
        )
        
        preprocessor.char_to_idx = data['char_to_idx']
        preprocessor.idx_to_char = data['idx_to_char']
        preprocessor.vocab_size = data['vocab_size']
        preprocessor.surname_prefixes = data.get('surname_prefixes', preprocessor.surname_prefixes)
        
        return preprocessor


# Esempio di utilizzo con validazione
if __name__ == "__main__":
    preprocessor = ImprovedNamePreprocessor(
        normalize_diacritics=True,
        handle_hyphens='keep'
    )
    
    # Test cases
    test_names = [
        ("John Lee Oswald", "M"),
        ("Mirko Degli Esposti", "M"),
        ("María José de la Cruz", "W"),
        ("Jean-Pierre Dupont", "M"),
        ("Enrico Martina", "M"),  # Potenzialmente ambiguo
        ("Martina Enrico", "W"),  # Potenzialmente ambiguo
        ("李明", "M"),  # Nome cinese
        ("José María", "M"),  # Nome spagnolo
    ]
    
    print("Testing improved preprocessor:\n")
    
    for full_name, gender in test_names:
        result = preprocessor.preprocess_name(full_name)
        first, middle, last = preprocessor.intelligent_split(full_name)
        
        # Valida consistenza
        confidence = preprocessor.validate_name_gender_consistency(
            result['first_raw'], result['last_raw'], gender
        )
        
        print(f"Original: {full_name} ({gender})")
        print(f"  Split: first='{first}', middle='{middle}', last='{last}'")
        print(f"  Normalized: first='{result['first_raw']}', last='{result['last_raw']}'")
        print(f"  Confidence in order: {confidence:.2f}")
        
        if confidence < 0.5:
            print(f"  ⚠️  WARNING: Name order might be inverted!")
        
        print()
