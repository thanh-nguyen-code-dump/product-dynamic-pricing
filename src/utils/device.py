"""
Device Utilities for Cross-Platform GPU Support

Supports:
- NVIDIA GPUs (CUDA)
- Apple Silicon (MPS)
- CPU fallback
"""

import torch
import logging

logger = logging.getLogger(__name__)


def get_best_device() -> torch.device:
    """
    Detect and return the best available device.
    
    Priority: CUDA > MPS (Apple Silicon) > CPU
    
    Returns:
        torch.device: Best available device
    """
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        logger.info(f"Using NVIDIA GPU: {torch.cuda.get_device_name(device)}")
        return device
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using Apple Silicon GPU (MPS)")
        return device
    else:
        logger.info("Using CPU")
        return torch.device("cpu")


def is_gpu_available() -> bool:
    """Check if any GPU (CUDA or MPS) is available."""
    return torch.cuda.is_available() or torch.backends.mps.is_available()


def get_device_info() -> dict:
    """Get information about the current device."""
    if torch.cuda.is_available():
        return {
            "type": "cuda",
            "name": torch.cuda.get_device_name(0),
            "memory_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
            "cuda_version": torch.version.cuda
        }
    elif torch.backends.mps.is_available():
        return {
            "type": "mps",
            "name": "Apple Silicon",
            "memory_gb": None,  # Unified memory, not directly queryable
            "cuda_version": None
        }
    else:
        return {
            "type": "cpu",
            "name": "CPU",
            "memory_gb": None,
            "cuda_version": None
        }
