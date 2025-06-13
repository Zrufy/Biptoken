from setuptools import setup, find_packages

setup(
    name="Biptoken",
    version="0.1.0",
    author="Zrufy",
    description="Fast BPE tokenizer with guaranteed perfect text reconstruction",
    long_description="Fast BPE tokenizer with guaranteed perfect text reconstruction. 2x faster than tiktoken at encoding/decoding while preserving all whitespace and formatting.",
    long_description_content_type="text/markdown",
    url="https://github.com/Zrufy/Biptoken",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "numpy",
        "regex",
    ],
)