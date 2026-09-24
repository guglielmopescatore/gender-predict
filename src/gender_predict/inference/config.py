"""
Inference configuration.

The production model lives in a folder (by default ``models/production`` at
the repository root) that holds ``config.json`` plus the model weights, the
fitted preprocessor and the feature extractor. ``config.json`` is the single
source of truth for file names, decision threshold and reference metrics.

The folder is resolved, in order, from: an explicit argument, the
``GENDER_PREDICT_MODEL_DIR`` environment variable, the repository checkout
this package was imported from (editable install), the current directory.
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

ENV_VAR = "GENDER_PREDICT_MODEL_DIR"
CONFIG_FILENAME = "config.json"


def default_model_dir() -> Path:
    """Locate the production model folder."""
    env = os.environ.get(ENV_VAR)
    if env:
        return Path(env).expanduser()
    repo_root = Path(__file__).resolve().parents[3]  # src/gender_predict/inference -> repo
    candidate = repo_root / "models" / "production"
    if (candidate / CONFIG_FILENAME).exists():
        return candidate
    return Path.cwd() / "models" / "production"


@dataclass
class InferenceConfig:
    """Resolved inference settings."""

    model_dir: Path
    model_path: Path
    preprocessor_path: Path
    feature_extractor_path: Optional[Path]
    threshold: float
    transliterate: bool = True
    unicode_preprocessing: bool = True
    metrics: Dict[str, Any] = field(default_factory=dict)
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, model_dir: Union[str, Path, None] = None, **overrides) -> "InferenceConfig":
        """Read ``config.json`` from ``model_dir`` (or the default location).

        Keyword overrides (``threshold``, ``transliterate``,
        ``unicode_preprocessing``) take precedence over the file.
        """
        model_dir = Path(model_dir).expanduser() if model_dir else default_model_dir()
        cfg_path = model_dir / CONFIG_FILENAME
        if not cfg_path.exists():
            raise FileNotFoundError(
                f"No {CONFIG_FILENAME} in {model_dir}. Pass model_dir explicitly or set {ENV_VAR}."
            )
        with open(cfg_path, encoding="utf-8") as fh:
            raw = json.load(fh)
        files = raw["files"]
        fe = files.get("feature_extractor")
        return cls(
            model_dir=model_dir,
            model_path=model_dir / files["model"],
            preprocessor_path=model_dir / files["preprocessor"],
            feature_extractor_path=(model_dir / fe) if fe else None,
            threshold=float(overrides.get("threshold", raw["threshold"])),
            transliterate=bool(overrides.get("transliterate", raw.get("transliterate", True))),
            unicode_preprocessing=bool(
                overrides.get("unicode_preprocessing", raw.get("unicode_preprocessing", True))
            ),
            metrics=raw.get("metrics", {}),
            raw=raw,
        )
