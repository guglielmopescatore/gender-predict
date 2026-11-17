"""
Gender prediction from full names — inference API.

Example::

    from gender_predict import GenderPredictor

    predictor = GenderPredictor()                 # loads models/production
    predictor.predict("Maria Rossi")              # -> dict
    predictor.predict_many(["张伟", "Анна Каренина"])
    predictor.predict_dataframe(df, name_column="primaryName")

Pipeline (the one selected in June 2025, "V4-R1"):

1. optional transliteration of non-Latin scripts to a romanised form
   (also lower-cases and strips diacritics/apostrophes from Latin names);
2. robust cleaning (encoding fixes, control characters, Unicode
   normalisation) followed by the preprocessor fitted at training time;
3. V3 model forward pass with suffix and phonetic features;
4. decision at the configured threshold on the probability of "W".

Models trained with name normalisation 2 (the preprocessor saved with the model says so) replace steps 1-2
with ``gender_predict.normalization.normalize_name``, the same function used to build their training data.
For these models a name that cannot be represented (given name in an unsupported script, empty, only an
initial as given name) is not predicted: ``predicted_gender`` is "Unknown", ``probability_female`` is NaN and ``status`` says why.
"""

import logging
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

import torch

from .config import InferenceConfig
from .robust_preprocessing import ProductionRobustPreprocessor
from .transliteration import transliterate_name
from ..han import HAN_WEIGHT, han_logodds
from ..normalization import PREDICTABLE, normalize_name

logger = logging.getLogger(__name__)

_SUFFIX_LEN = 3


class GenderPredictor:
    """Production gender predictor (V3 architecture, V4-R1 weights)."""

    def __init__(
        self,
        model_dir: Union[str, Path, None] = None,
        *,
        threshold: Optional[float] = None,
        transliterate: Optional[bool] = None,
        device: str = "auto",
        lazy: bool = True,
    ):
        overrides = {}
        if threshold is not None:
            overrides["threshold"] = threshold
        if transliterate is not None:
            overrides["transliterate"] = transliterate
        self.config = InferenceConfig.load(model_dir, **overrides)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model = None
        self.preprocessor = None
        self.feature_extractor = None
        self.is_v3_model = False
        self.normalization = 1
        self.stats = {"total": 0, "transliterated": 0, "by_script": {}}
        if not lazy:
            self.load()

    # ------------------------------------------------------------------ setup
    @property
    def threshold(self) -> float:
        return self.config.threshold

    def load(self) -> "GenderPredictor":
        """Load weights, preprocessor and feature extractor."""
        from ..evaluation.evaluator import ModelEvaluator
        from ..data.feature_extraction import NameFeatureExtractor

        cfg = self.config
        logger.info("Loading model from %s (device=%s, threshold=%.3f)", cfg.model_dir, self.device, cfg.threshold)
        checkpoint = torch.load(cfg.model_path, map_location="cpu", weights_only=False)
        self.is_v3_model = "suffix_vocab_size" in checkpoint

        evaluator = ModelEvaluator.from_checkpoint(
            str(cfg.model_path), str(cfg.preprocessor_path), self.device, num_attention_heads=cfg.num_attention_heads
        )
        base = evaluator.preprocessor
        self.normalization = getattr(base, "normalization", 1)
        if self.normalization >= 2:
            self.preprocessor = base  # normalize_name does all the cleaning
        else:
            self.preprocessor = ProductionRobustPreprocessor(base) if cfg.unicode_preprocessing else base
        self.model = evaluator.model
        self.model.eval()

        self.feature_extractor = NameFeatureExtractor()
        if cfg.feature_extractor_path and cfg.feature_extractor_path.exists():
            import pickle

            with open(cfg.feature_extractor_path, "rb") as fh:
                loaded = pickle.load(fh)
            if isinstance(loaded, NameFeatureExtractor):
                self.feature_extractor = loaded
        return self

    def _ensure_loaded(self):
        if self.model is None:
            self.load()

    # -------------------------------------------------------------- inference
    def _split(self, name: str):
        if hasattr(self.preprocessor, "split_full_name"):
            return self.preprocessor.split_full_name(name)
        parts = name.strip().split()
        return (parts[0], " ".join(parts[1:])) if parts else ("", "")

    def _encode(self, name: str):
        """Model inputs for one (already transliterated) name, as plain lists of fixed length."""
        processed = self.preprocessor.preprocess_name(name)
        first, last = list(processed["first_name"]), list(processed["last_name"])
        if not self.is_v3_model:
            return first, last
        first_str, last_str = self._split(name)
        fs = self.feature_extractor.extract_suffix_features(first_str)
        ls = self.feature_extractor.extract_suffix_features(last_str)
        pf = self.feature_extractor.extract_phonetic_features(first_str)
        pl = self.feature_extractor.extract_phonetic_features(last_str)
        fs = (fs + [0] * _SUFFIX_LEN)[:_SUFFIX_LEN]
        ls = (ls + [0] * _SUFFIX_LEN)[:_SUFFIX_LEN]
        phon = [pf["ends_with_vowel"], pf["vowel_ratio"], pl["ends_with_vowel"], pl["vowel_ratio"]]
        return first, last, fs, ls, phon

    def _forward_batch(self, names: List[str]) -> List[float]:
        """Probability of "female" for a list of (already transliterated) names, in one forward pass."""
        cols = list(zip(*(self._encode(n) for n in names)))
        long_ = lambda c: torch.tensor(c, dtype=torch.long).to(self.device)
        with torch.no_grad():
            if self.is_v3_model:
                out = self.model(long_(cols[0]), long_(cols[1]), long_(cols[2]), long_(cols[3]),
                                 torch.tensor(cols[4], dtype=torch.float32).to(self.device))
            else:
                out = self.model(long_(cols[0]), long_(cols[1]))
        return torch.sigmoid(out).reshape(-1).tolist()

    def _forward(self, name: str) -> float:
        """Probability that ``name`` (already transliterated) is female."""
        return self._forward_batch([name])[0]

    def predict(self, name: str) -> Dict[str, Any]:
        """Predict the gender of one full name.

        Returns a dict with ``name``, ``predicted_gender`` ("W"/"M"),
        ``probability_female``, ``confidence``, ``threshold_used`` and, when
        transliteration is enabled, ``transliterated_name``,
        ``detected_script`` and ``was_transliterated``.
        """
        self._ensure_loaded()
        return self._predict_batch([name])[0]

    def predict_many(self, names: Iterable[str], on_error: str = "unknown", batch_size: int = 1024) -> List[Dict[str, Any]]:
        """Predict a sequence of names, ``batch_size`` names per forward pass (much faster than one by one,
        on CPU and on GPU). The results are the same as calling :meth:`predict` on each name.

        ``on_error="unknown"`` records failed names with gender "Unknown";
        ``on_error="raise"`` propagates the exception. If a batch fails, its names are processed one by one.
        """
        self._ensure_loaded()
        names = list(names)
        results: List[Dict[str, Any]] = []
        for start in range(0, len(names), max(1, batch_size)):
            chunk = names[start:start + max(1, batch_size)]
            try:
                results.extend(self._predict_batch(chunk))
            except Exception:  # noqa: BLE001 - fall back to the single-name path to isolate the bad name(s)
                for name in chunk:
                    try:
                        results.append(self.predict(name))
                    except Exception as exc:  # noqa: BLE001
                        if on_error == "raise":
                            raise
                        logger.warning("Error processing %r: %s", name, exc)
                        results.append(
                            {
                                "name": name,
                                "predicted_gender": "Unknown",
                                "probability_female": 0.5,
                                "confidence": 0.0,
                                "threshold_used": self.config.threshold,
                                "error": str(exc),
                            }
                        )
        return results

    def _prepare(self, original: str):
        """(model input, detected script, status) for one raw name."""
        if self.normalization >= 2:
            n = normalize_name(original, getattr(self.preprocessor, "initials", "undetermined"))
            return n.text, n.script, n.status, n.han
        if self.config.transliterate:
            return (*transliterate_name(original), "ok", "")
        return original, "LAT", "ok", ""

    def _predict_batch(self, names: List[Any]) -> List[Dict[str, Any]]:
        originals = [n if isinstance(n, str) else ("" if n is None else str(n)) for n in names]
        prepared = [self._prepare(o) for o in originals]
        ok = [i for i, p in enumerate(prepared) if p[2] in PREDICTABLE]
        probs = dict(zip(ok, self._forward_batch([prepared[i][0] for i in ok]))) if ok else {}
        v2, out = self.normalization >= 2, []
        for i, (original, (model_input, script, status, han)) in enumerate(zip(originals, prepared)):
            if i in probs:
                prob = probs[i]
                if han:  # name written in Han characters: the characters' evidence is added to the network's logit
                    odds = han_logodds(han)
                    q = min(max(prob, 1e-6), 1 - 1e-6)
                    prob = 1 / (1 + math.exp(-(math.log(q / (1 - q)) + HAN_WEIGHT * odds)))
                result = {
                    "name": original,
                    "predicted_gender": "W" if prob >= self.config.threshold else "M",
                    "probability_female": prob,
                    "confidence": max(prob, 1 - prob),
                    "threshold_used": self.config.threshold,
                }
            else:
                result = {
                    "name": original,
                    "predicted_gender": "Unknown",
                    "probability_female": float("nan"),
                    "confidence": 0.0,
                    "threshold_used": self.config.threshold,
                }
            if v2:
                result["status"] = status
            if v2 or self.config.transliterate:
                self.stats["total"] += 1
                self.stats["by_script"][script] = self.stats["by_script"].get(script, 0) + 1
                if script != "LAT":
                    self.stats["transliterated"] += 1
                result.update(transliterated_name=model_input, detected_script=script, was_transliterated=(script != "LAT"))
            out.append(result)
        return out

    def predict_dataframe(self, df, name_column: str = "primaryName", prefix: str = ""):
        """Return a copy of ``df`` with prediction columns appended.

        Added columns: ``predicted_gender``, ``probability_female``,
        ``confidence`` (optionally prefixed).
        """
        import pandas as pd

        if name_column not in df.columns:
            raise ValueError(f"Column '{name_column}' not found. Available: {list(df.columns)}")
        names = df[name_column].fillna("").astype(str).tolist()
        results = pd.DataFrame(self.predict_many(names), index=df.index)
        out = df.copy()
        for col in ("predicted_gender", "probability_female", "confidence", "status"):
            if col in results.columns:
                out[prefix + col] = results[col]
        return out

    def predict_csv(self, input_file, output_file, name_column: str = "primaryName"):
        """Read a CSV, predict, write a CSV. Returns the output DataFrame."""
        import pandas as pd

        out = self.predict_dataframe(pd.read_csv(input_file), name_column=name_column)
        out.to_csv(output_file, index=False)
        return out

    # ------------------------------------------------------------------ misc
    def transliteration_stats(self) -> Dict[str, Any]:
        total = self.stats["total"]
        return {
            "total_names": total,
            "transliterated_count": self.stats["transliterated"],
            "transliteration_rate": (self.stats["transliterated"] / total) if total else 0.0,
            "by_script": dict(self.stats["by_script"]),
        }

    def __repr__(self) -> str:
        return (
            f"GenderPredictor(model_dir={str(self.config.model_dir)!r}, threshold={self.config.threshold}, "
            f"transliterate={self.config.transliterate}, device={self.device!r})"
        )
