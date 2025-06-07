"""
Setup script for gender-predict package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="gender-predict",
    version="3.0.0",  # Updated to reflect V3 model architecture
    author="Guglielmo Pescatore",
    author_email="guglielmo[dot]pescatore[at]unibo[dot]it",  # Anti-spam format
    description="Deep learning framework for gender prediction from names using PyTorch V3 architecture",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/guglielmopescatore/gender-predict",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",  # Updated license
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",  # Added 3.11 support
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest",
            "pytest-cov",
            "black",
            "flake8",
            "mypy",
        ],
        "api": [  # Added API deployment extras
            "fastapi>=0.104.0",
            "uvicorn>=0.24.0",
        ],
    },
    scripts=[
        "scripts/train_model.py",
        "scripts/evaluate_model.py",
        "scripts/experiment_tools.py",
        "scripts/final_predictor.py",  # Added your main predictor script
    ],
    entry_points={
        "console_scripts": [
            "gender-predict-train=scripts.train_model:main",
            "gender-predict-eval=scripts.evaluate_model:main",
            "gender-predict-tools=scripts.experiment_tools:main",
            "gender-predict=scripts.final_predictor:main",  # Added main entry point
        ],
    },
    keywords="machine-learning, deep-learning, nlp, gender-prediction, pytorch, names, classification, BiLSTM, attention",
    license="GPL-3.0",  # Explicit license
)
