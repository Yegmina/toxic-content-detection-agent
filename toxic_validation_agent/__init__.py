"""
Toxic Content Detection Agent
============================

Intelligent AI Agent for Real-time Content Moderation with 97.5% accuracy.

This package provides a comprehensive, enterprise-grade hybrid pipeline for 
gaming chat toxicity detection with production-ready features including 
zero-tier word filtering, multi-stage ML pipeline, comprehensive error handling, 
and performance monitoring.

Author: Yehor Tereshchenko
License: MIT
Version: 1.0.9
"""

__version__ = "1.0.9"
__author__ = "Yehor Tereshchenko"
__email__ = "your.email@example.com"  # Update with your email
__license__ = "MIT"
__url__ = "https://github.com/Yegmina/toxic-content-detection-agent"

from .message_validator import Message_Validation, ValidationResult, PerformanceMetrics
from .exceptions import ToxicValidationError, ModelLoadError, InputValidationError

__all__ = [
    "Message_Validation",
    "ValidationResult", 
    "PerformanceMetrics",
    "ToxicValidationError",
    "ModelLoadError",
    "InputValidationError",
    "__version__",
    "__author__",
    "__email__",
    "__license__",
    "__url__",
] 