"""
Neural Microscope - Dual-Manifold Visualization System

Provides real-time visualization of BDH neural activations in two geometric spaces:
1. Physical Space (Weight Space): Neurons positioned by learned 256D encoder weights
2. Semantic Space (Activation Space): Neurons positioned by activation patterns

Core Components:
- BDHInferenceEngine: Load checkpoints and run inference with activation capture
- NeuronPositionCalculator: Compute physical space positions (UMAP of weights)
- ActivationPositionCalculator: Compute semantic space positions (UMAP of activations)
- GPUAllocator: Manage GPU assignments for inference and visualization
"""

from .inference_engine import BDHInferenceEngine, SparseActivationCollector
from .neuron_position_calculator import NeuronPositionCalculator, get_neuron_position
from .activation_position_calculator import (
    ActivationPositionCalculator,
    get_activation_position
)
from src.utils.neural_microscope_config import GPUAllocator  # Import from top-level utils
from .validation_framework import ValidationSuite, ValidationResult, ValidatedProjection

__all__ = [
    'BDHInferenceEngine',
    'SparseActivationCollector',
    'NeuronPositionCalculator',
    'get_neuron_position',
    'ActivationPositionCalculator',
    'get_activation_position',
    'GPUAllocator',
    'ValidationSuite',
    'ValidationResult',
    'ValidatedProjection',
]

__version__ = '1.0.0'
__author__ = 'BDH Research Team'
