"""
This setup discovery in only intented to be used with "pip install -e .".
This is to force discovery of the package in the current directory,
to be able to "import" and execute anywhere.
"""
from setuptools import setup, find_packages

setup(
    name="rag_test",
    version="0.1",
    packages=find_packages(),
)