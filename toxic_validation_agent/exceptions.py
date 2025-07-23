"""
Custom exceptions for the Toxic Content Detection Agent.

This module defines all custom exceptions used throughout the package
for better error handling and debugging.
"""


class ToxicValidationError(Exception):
    """
    Base exception for all toxic validation related errors.
    
    This is the parent class for all custom exceptions in the package.
    """
    pass


class ModelLoadError(ToxicValidationError):
    """
    Exception raised when model loading fails.
    
    This exception is raised when there are issues loading
    the machine learning models or their dependencies.
    """
    pass


class InputValidationError(ToxicValidationError):
    """
    Exception raised when input validation fails.
    
    This exception is raised when the input message doesn't meet
    the validation requirements (e.g., empty, too long, invalid characters).
    """
    pass


class ConfigurationError(ToxicValidationError):
    """
    Exception raised when configuration is invalid.
    
    This exception is raised when there are issues with the
    configuration file or settings.
    """
    pass


class PipelineError(ToxicValidationError):
    """
    Exception raised when the validation pipeline fails.
    
    This exception is raised when there are issues during
    the multi-stage validation process.
    """
    pass 