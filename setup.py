#!/usr/bin/env python
"""Setup configuration for RadarTargetingSystem."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="radar-targeting-system",
    version="0.1.0",
    author="ChaLyn03",
    description="End-to-end FMCW radar demonstrator with DSP, detection, and CNN classification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ChaLyn03/RadarTargetingSystem",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "numpy",
        "scipy",
        "torch",
        "streamlit",
        "plotly",
        "pandas",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov",
            "black>=23.0",
            "isort>=5.0",
            "flake8>=6.0",
            "mypy>=1.0",
            "pylint>=3.0",
        ],
        "test": [
            "pytest>=7.0",
            "pytest-cov",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
