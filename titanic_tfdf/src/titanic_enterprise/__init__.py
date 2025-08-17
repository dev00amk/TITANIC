"""
Titanic Enterprise ML Infrastructure

A production-ready machine learning pipeline for Titanic survival prediction
using TensorFlow Decision Forests with enterprise-grade MLOps practices.
"""

__version__ = "1.0.0"
__author__ = "ML Engineering Team"
__email__ = "ml-team@company.com"

# Package-level imports for convenience
from titanic_enterprise.utils.config import load_config
from titanic_enterprise.utils.logging import setup_logging

__all__ = [
    "__version__",
    "__author__", 
    "__email__",
    "load_config",
    "setup_logging",
]