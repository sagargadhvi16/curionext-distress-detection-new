from setuptools import setup, find_packages

setup(
    name="curionext-distress-detection",
    version="0.1.0",
    description="Multi-modal AI system for child distress detection",
    author="Sagar",
    author_email="sagar@curionext.com",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "librosa>=0.10.0",
        "neurokit2>=0.2.4",
        "fastapi>=0.103.0",
        "shap>=0.42.0",
    ],
    python_requires=">=3.10",
)
