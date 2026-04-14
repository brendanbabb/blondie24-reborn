"""
GPU utilities: device detection, info reporting, and memory management.

Designed for Badger-1 (RTX 5060) but works on any CUDA-capable system.
"""

import torch
import os


def get_device(requested: str = "cuda") -> torch.device:
    """
    Get the best available device.
    
    Priority: requested device → CUDA → CPU
    Prints device info on first call.
    """
    if requested == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        # Print GPU info once
        if not getattr(get_device, "_reported", False):
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            print(f"  GPU: {gpu_name} ({gpu_mem:.1f} GB)")
            print(f"  CUDA: {torch.version.cuda}")
            get_device._reported = True
        return device
    elif requested == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        if not getattr(get_device, "_reported", False):
            print("  Device: Apple MPS")
            get_device._reported = True
        return torch.device("mps")
    else:
        if requested != "cpu" and not getattr(get_device, "_reported", False):
            print(f"  Requested '{requested}' not available, falling back to CPU")
            get_device._reported = True
        return torch.device("cpu")


def gpu_memory_report() -> dict:
    """Get current GPU memory usage (returns empty dict on CPU)."""
    if not torch.cuda.is_available():
        return {}
    return {
        "allocated_mb": torch.cuda.memory_allocated() / (1024 ** 2),
        "reserved_mb": torch.cuda.memory_reserved() / (1024 ** 2),
        "max_allocated_mb": torch.cuda.max_memory_allocated() / (1024 ** 2),
    }


def optimize_for_inference():
    """
    Set PyTorch options that help inference-only workloads (no backprop).
    Call once at startup.
    """
    # Disable gradient computation globally — we never use backprop
    torch.set_grad_enabled(False)
    
    # Use TF32 on Ampere+ GPUs for faster matmuls (RTX 30xx/40xx/50xx)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    # Set number of threads for CPU operations (useful for encoding)
    cpu_count = os.cpu_count() or 4
    torch.set_num_threads(min(cpu_count, 8))


def clear_gpu_cache():
    """Free unused GPU memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
