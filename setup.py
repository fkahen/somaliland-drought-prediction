"""
setup.py — Somaliland Drought Prediction package
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
long_description = (Path(__file__).parent / "README.md").read_text(encoding="utf-8")

# Read requirements
requirements = [
    line.strip()
    for line in (Path(__file__).parent / "requirements.txt").read_text().splitlines()
    if line.strip() and not line.startswith("#")
]

setup(
    name="somaliland-drought-prediction",
    version="2.0.0",
    description=(
        "District-level drought prediction model for Somaliland "
        "using CHIRPS, ERA5, SWALIM, FAO, and World Bank data."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Climate Data Science Team",
    author_email="climate-ds@example.org",
    url="https://github.com/your-org/somaliland-drought-prediction",
    license="MIT",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=23.0",
            "isort>=5.12",
            "flake8>=6.0",
            "mypy>=1.0",
            "pre-commit>=3.0",
        ],
        "app": [
            "streamlit>=1.28.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "drought-train=src.modeling:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords=[
        "drought", "somaliland", "africa", "climate", "machine-learning",
        "SPI", "CHIRPS", "ERA5", "SWALIM", "early-warning",
    ],
)
