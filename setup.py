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
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Deep learning models for gender prediction from names",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/gender-predict",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
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
    },
    scripts=[
        "scripts/train_model.py",
        "scripts/evaluate_model.py", 
        "scripts/experiment_tools.py",
        "scripts/prepare_data.py",
    ],
    entry_points={
        "console_scripts": [
            "gender-predict-train=scripts.train_model:main",
            "gender-predict-eval=scripts.evaluate_model:main",
            "gender-predict-tools=scripts.experiment_tools:main",
        ],
    },
)
