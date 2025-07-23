from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="toxic_detection",
    version="1.0.5",
    author="Yehor Tereshchenko",
    author_email="your.email@example.com",  # Update with your email
    description="Intelligent AI Agent for Real-time Content Moderation with 97.5% accuracy",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/Yegmina/toxic-content-detection-agent",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",

        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Text Processing :: Linguistic",
        "Topic :: Communications :: Chat",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
        "gpu": [
            "torch>=1.9.0",
            "torchvision>=0.10.0",
        ],
    },
    include_package_data=True,
    package_data={
        "toxic_validation_agent": [
            "toxicity_words.json",
            "config.json",
        ],
    },
    entry_points={
        "console_scripts": [
            "toxic-validation=toxic_validation_agent.cli:main",
        ],
    },
    keywords=[
        "ai",
        "machine-learning",
        "content-moderation",
        "toxicity-detection",
        "nlp",
        "bert",
        "chat-moderation",
        "gaming",
        "sentiment-analysis",
        "text-classification",
    ],
    project_urls={
        "Bug Reports": "https://github.com/Yegmina/toxic-content-detection-agent/issues",
        "Source": "https://github.com/Yegmina/toxic-content-detection-agent",
        "Documentation": "https://github.com/Yegmina/toxic-content-detection-agent#readme",
    },
) 