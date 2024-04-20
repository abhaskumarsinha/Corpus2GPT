"""
Inference Package

This package provides utilities for inference tasks.

Submodules:
- inference: Contains functions and utilities for inference.
- scale_utils: Utilities for scaling LLMs.

Sampling Strategies:
- sample_random: Implements random sampling strategy for inference.
"""

from inference.inference import *
from inference.scale_utils import *

from inference.sampling_strategies.sample_random import *
