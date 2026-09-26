from setuptools import setup, find_packages

setup(
    name="hce-mitigation",
    version="1.0.0",
    description="Investigating and Mitigating the Alignment Paradox in Large Language Models",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.40.0",
        "datasets>=2.18.0",
        "accelerate>=0.28.0",
        "peft>=0.10.0",
        "scipy>=1.11.0",
        "scikit-learn>=1.3.0",
        "numpy>=1.24.0",
        "tqdm>=4.66.0",
    ],
)
