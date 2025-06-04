"""
Test Time Augmentation (TTA) for gender prediction models.

TTA improves prediction accuracy by averaging predictions over multiple
augmented versions of the input during inference.
"""

import torch
import numpy as np
from typing import List, Tuple, Optional
from tqdm import tqdm


class TestTimeAugmentation:
    """
    Test Time Augmentation per modelli di predizione del genere.
    """

    def __init__(self, model, preprocessor, augmenter, device='cuda', feature_extractor=None):
        """
        Args:
            model: Modello trained
            preprocessor: NamePreprocessor instance
            augmenter: NameAugmenter instance
            device: Device per inferenza
            feature_extractor: NameFeatureExtractor per V3 models
        """
        self.model = model
        self.preprocessor = preprocessor
        self.augmenter = augmenter
        self.device = device
        self.feature_extractor = feature_extractor

        # Verifica che abbia split_full_name
        if not hasattr(self.preprocessor, "split_full_name"):
            raise AttributeError(
                "Il pre-processore passato a TTA non ha il metodo split_full_name()."
            )

        # Rileva tipo di modello
        self.is_v3 = self._is_v3_model()

        if self.is_v3 and feature_extractor is None:
            raise ValueError("V3 models require a feature_extractor!")

    def _is_v3_model(self):
        """Rileva se è un modello V3."""
        try:
            # V3 ha questi attributi
            return hasattr(self.model, 'suffix_embedding')
        except:
            return False

    def predict_single(self, full_name: str, n_aug: int = 5,
                    return_all: bool = False) -> Tuple[float, float]:
        """
        Predice il genere con TTA per un singolo nome.
        """
        self.model.eval()

        # Lista per i LOGIT (non probabilità!)
        logits_list = []

        # Predizione originale
        with torch.no_grad():
            orig_logit = self._predict_name(full_name).item()
            logits_list.append(orig_logit)

        # Predizioni augmentate
        for _ in range(n_aug - 1):
            aug_name = self.augmenter.augment(full_name)

            # VERIFICA: l'augmentation preserva il genere?
            # Opzionale: potresti verificare che non cambi troppo
            # es. "Paolo" -> "Paola" dovrebbe essere scartato

            with torch.no_grad():
                aug_logit = self._predict_name(aug_name).item()
                logits_list.append(aug_logit)

        if return_all:
            # Converti in probabilità per return_all
            return [torch.sigmoid(torch.tensor(l)).item() for l in logits_list]

        # Aggregazione sui LOGIT (più stabile)
        mean_logit = np.mean(logits_list)
        std_logit = np.std(logits_list)

        # Converti in probabilità DOPO l'aggregazione
        mean_prob = torch.sigmoid(torch.tensor(mean_logit)).item()

        # Confidence basata su consenso
        confidence = 1.0 / (1.0 + std_logit)  # Più robusto

        return mean_prob, confidence

    def _predict_name(self, full_name: str) -> float:
        """Predice probabilità per un singolo nome."""
        # Preprocessa
        name_data = self.preprocessor.preprocess_name(full_name)

        # Converti in tensori
        first_name = torch.tensor([name_data['first_name']], dtype=torch.long).to(self.device)
        last_name = torch.tensor([name_data['last_name']], dtype=torch.long).to(self.device)

        if self.is_v3:
            # V3 richiede features complete!
            # Assumendo che hai accesso a feature_extractor
            if hasattr(self, 'feature_extractor') and self.feature_extractor is not None:
                # Estrai features reali
                first_name_str, last_name_str = self.preprocessor.split_full_name(full_name)

                # Suffix features
                first_suffix = self.feature_extractor.extract_suffix_features(first_name_str)
                last_suffix = self.feature_extractor.extract_suffix_features(last_name_str)

                # Padding
                first_suffix = first_suffix + [0] * (3 - len(first_suffix))
                last_suffix = last_suffix + [0] * (3 - len(last_suffix))

                # Phonetic features
                phonetic_first = self.feature_extractor.extract_phonetic_features(first_name_str)
                phonetic_last = self.feature_extractor.extract_phonetic_features(last_name_str)

                phonetic_features = [
                    phonetic_first['ends_with_vowel'],
                    phonetic_first['vowel_ratio'],
                    phonetic_last['ends_with_vowel'],
                    phonetic_last['vowel_ratio']
                ]

                # Converti in tensori
                first_suffix = torch.tensor([first_suffix[:3]], dtype=torch.long).to(self.device)
                last_suffix = torch.tensor([last_suffix[:3]], dtype=torch.long).to(self.device)
                phonetic_features = torch.tensor([phonetic_features], dtype=torch.float32).to(self.device)
            else:
                # Fallback (non dovrebbe succedere)
                raise ValueError("V3 model requires feature_extractor!")

            logits = self.model(first_name, last_name, first_suffix,
                            last_suffix, phonetic_features)
        else:
            logits = self.model(first_name, last_name)

        return logits  # Ritorna LOGIT, non probabilità!

    def predict_dataset(self, dataset, batch_size=128, n_aug=5,
                       show_progress=True) -> dict:
        """
        Applica TTA a un intero dataset.

        Args:
            dataset: Dataset di test
            batch_size: Batch size per efficienza
            n_aug: Numero di augmentazioni per sample
            show_progress: Mostra barra di progresso

        Returns:
            Dict con predizioni, probabilità, confidenze
        """
        self.model.eval()

        all_predictions = []
        all_probabilities = []
        all_confidences = []
        all_true_labels = []

        # Progress bar
        iterator = range(len(dataset))
        if show_progress:
            iterator = tqdm(iterator, desc="TTA Prediction")

        for idx in iterator:
            # Get sample
            sample = dataset[idx]
            true_label = sample['gender'].item()

            # Get name from dataframe
            if hasattr(dataset, 'df'):
                full_name = dataset.df.iloc[idx]['primaryName']
            else:
                # Fallback
                full_name = "Unknown"

            # TTA prediction
            prob, conf = self.predict_single(full_name, n_aug=n_aug)

            # Collect results
            pred_label = 1 if prob >= 0.5 else 0
            all_predictions.append(pred_label)
            all_probabilities.append(prob)
            all_confidences.append(conf)
            all_true_labels.append(true_label)

        return {
            'predictions': np.array(all_predictions),
            'probabilities': np.array(all_probabilities),
            'confidences': np.array(all_confidences),
            'true_labels': np.array(all_true_labels)
        }


class SmartTTA(TestTimeAugmentation):
    """
    TTA intelligente che usa più augmentazioni per casi incerti.
    """

    def predict_single(self, full_name: str,
                       min_aug: int = 3,
                       max_aug: int = 10,
                       uncertainty_threshold: float = 0.15
                       ) -> Tuple[float, float]:
        """
        Restituisce (probabilità, confidenza) con numero di augmentazioni variabile.
        Confidenza alta ↔ bassa deviazione standard dei logit.
        """
        self.model.eval()

        # ---- raccolta LOGIT (non prob) ------------------------------------
        logits: List[float] = []

        with torch.no_grad():
            logits.append(self._predict_name(full_name).item())

        for _ in range(min_aug - 1):          # augmentazioni minime
            aug_name = self.augmenter.augment(full_name)
            with torch.no_grad():
                logits.append(self._predict_name(aug_name).item())

        # -------------------------------------------------------------------
        def agg_stats(vals: List[float]) -> Tuple[float, float]:
            return float(np.mean(vals)), float(np.std(vals))

        mean_logit, std_logit = agg_stats(logits)

        # se è “troppo incerto”, aggiungi augmentazioni finché:
        #   - raggiungi max_aug   OPPURE
        #   - la deviazione scende sotto soglia
        while std_logit > uncertainty_threshold and len(logits) < max_aug:
            aug_name = self.augmenter.augment(full_name)
            with torch.no_grad():
                logits.append(self._predict_name(aug_name).item())
            mean_logit, std_logit = agg_stats(logits)

        # ---- da logit a probabilità & confidenza -------------------------
        prob = torch.sigmoid(torch.tensor(mean_logit)).item()
        confidence = 1.0 / (1.0 + std_logit)      #  → 1 se std→0

        return prob, confidence

