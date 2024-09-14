"""
This module provides utilities for distributed training with Keras and TensorFlow.

Functions:
    get_distribution_scope(device_type: str) -> ContextManager
        Returns a context manager for executing code in a distributed environment.

Usage:
    To use the `get_distribution_scope` function, first set the `KERAS_BACKEND` environment variable to either `'jax'` or `'tensorflow'`. Then, call the function with the desired device type ('cpu', 'gpu', or 'tpu').
"""

from Corpus2GPT.training_utils.distribution_utils import get_distribution_scope

__all__ = ['get_distribution_scope']
