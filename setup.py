from setuptools import setup, find_packages

setup(
    name="tinyphobert",
    version="1.0.0",
    description="TinyPhoBERT: Multi-Level Knowledge Distillation for Vietnamese Hate Speech Detection",
    author="DangHaiNguyen",
    packages=find_packages(exclude=["tests*", "notebooks*"]),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.38.0",
        "datasets>=2.18.0",
        "scikit-learn>=1.3.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "omegaconf>=2.3.0",
        "tqdm>=4.65.0",
        "rich>=13.5.0",
    ],
)
