"""
Enhanced validation utilities for DGS-GPT.

This module provides comprehensive validation for configuration parameters,
user inputs, and system requirements.
"""

import os
import torch
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

def validate_config(config) -> List[str]:
    """
    Validate configuration parameters and return list of issues.
    
    Args:
        config: Configuration object to validate
        
    Returns:
        List of validation error messages
    """
    issues = []
    
    # Validate basic types and ranges
    if not isinstance(config.batch_size, int) or config.batch_size <= 0:
        issues.append("batch_size must be a positive integer")
    
    if not isinstance(config.learning_rate, (int, float)) or config.learning_rate <= 0:
        issues.append("learning_rate must be a positive number")
    
    if not isinstance(config.n_layer, int) or config.n_layer <= 0:
        issues.append("n_layer must be a positive integer")
    
    if not isinstance(config.n_head, int) or config.n_head <= 0:
        issues.append("n_head must be a positive integer")
    
    if not isinstance(config.n_embd, int) or config.n_embd <= 0:
        issues.append("n_embd must be a positive integer")
    
    # Validate dimension compatibility
    if config.n_embd % config.n_head != 0:
        issues.append(f"n_embd ({config.n_embd}) must be divisible by n_head ({config.n_head})")
    
    if config.n_embd % 2 != 0:
        issues.append("n_embd must be even for RoPE compatibility")
    
    # Validate dropout rate
    if not isinstance(config.dropout, (int, float)) or not (0 <= config.dropout <= 1):
        issues.append("dropout must be a number between 0 and 1")
    
    # Validate MoE parameters
    if config.use_moe:
        if not isinstance(config.moe_num_experts, int) or config.moe_num_experts < 2:
            issues.append("moe_num_experts must be an integer >= 2")
        
        if not isinstance(config.moe_k, int) or not (1 <= config.moe_k <= config.moe_num_experts):
            issues.append(f"moe_k must be an integer between 1 and {config.moe_num_experts}")
    
    # Validate attention type and groups
    if config.attention_type not in ["multihead", "grouped_query"]:
        issues.append("attention_type must be 'multihead' or 'grouped_query'")
    
    if config.attention_type == "grouped_query":
        if not isinstance(config.n_query_groups, int) or config.n_query_groups <= 0:
            issues.append("n_query_groups must be a positive integer")
        
        if config.n_head % config.n_query_groups != 0:
            issues.append(f"n_head ({config.n_head}) must be divisible by n_query_groups ({config.n_query_groups})")
    
    # Validate VRAM profile
    if config.vram_profile not in ["low", "medium", "high"]:
        issues.append("vram_profile must be 'low', 'medium', or 'high'")
    
    return issues

def validate_generation_params(
    max_tokens: int,
    temperature: float,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None
) -> List[str]:
    """
    Validate text generation parameters.
    
    Args:
        max_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        top_k: Top-k sampling parameter
        top_p: Top-p (nucleus) sampling parameter
        
    Returns:
        List of validation error messages
    """
    issues = []
    
    if not isinstance(max_tokens, int) or max_tokens <= 0:
        issues.append("max_tokens must be a positive integer")
    
    if max_tokens > 10000:
        issues.append("max_tokens should not exceed 10000 for performance reasons")
    
    if not isinstance(temperature, (int, float)) or temperature <= 0:
        issues.append("temperature must be a positive number")
    
    if temperature > 5.0:
        issues.append("temperature > 5.0 may produce very random output")
    
    if top_k is not None:
        if not isinstance(top_k, int) or top_k <= 0:
            issues.append("top_k must be a positive integer or None")
    
    if top_p is not None:
        if not isinstance(top_p, (int, float)) or not (0 < top_p <= 1):
            issues.append("top_p must be a number between 0 and 1 or None")
    
    return issues

def validate_file_path(filepath: str, must_exist: bool = True, extension: str = None) -> Tuple[bool, str]:
    """
    Validate file path and accessibility.
    
    Args:
        filepath: Path to validate
        must_exist: Whether file must already exist
        extension: Required file extension (with dot, e.g., '.json')
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        path = Path(filepath)
        
        if must_exist and not path.exists():
            return False, f"File does not exist: {filepath}"
        
        if must_exist and not path.is_file():
            return False, f"Path is not a file: {filepath}"
        
        if extension and not filepath.lower().endswith(extension.lower()):
            return False, f"File must have {extension} extension"
        
        # Check if directory is writable for new files
        if not must_exist:
            parent_dir = path.parent
            if not parent_dir.exists():
                try:
                    parent_dir.mkdir(parents=True, exist_ok=True)
                except OSError as e:
                    return False, f"Cannot create directory {parent_dir}: {e}"
            
            if not os.access(parent_dir, os.W_OK):
                return False, f"Directory is not writable: {parent_dir}"
        
        return True, ""
        
    except Exception as e:
        return False, f"Invalid file path: {e}"

def validate_system_requirements() -> Dict[str, Any]:
    """
    Check system requirements and capabilities.
    
    Returns:
        Dictionary with system information and validation results
    """
    info = {
        "python_version": None,
        "torch_version": None,
        "cuda_available": False,
        "cuda_version": None,
        "gpu_memory": 0,
        "gpu_name": None,
        "warnings": [],
        "errors": []
    }
    
    try:
        import sys
        info["python_version"] = sys.version
        
        if sys.version_info < (3, 8):
            info["errors"].append("Python 3.8 or higher is required")
    except Exception as e:
        info["errors"].append(f"Cannot detect Python version: {e}")
    
    try:
        info["torch_version"] = torch.__version__
    except Exception as e:
        info["errors"].append(f"Cannot detect PyTorch version: {e}")
    
    try:
        info["cuda_available"] = torch.cuda.is_available()
        if info["cuda_available"]:
            info["cuda_version"] = torch.version.cuda
            info["gpu_memory"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            info["gpu_name"] = torch.cuda.get_device_name(0)
            
            if info["gpu_memory"] < 8:
                info["warnings"].append(f"Low GPU memory: {info['gpu_memory']:.1f}GB (8GB+ recommended)")
        else:
            info["warnings"].append("CUDA not available - training will be slow on CPU")
    except Exception as e:
        info["warnings"].append(f"Cannot detect CUDA capabilities: {e}")
    
    return info

def safe_execute(func, *args, **kwargs):
    """
    Safely execute a function with error handling.
    
    Args:
        func: Function to execute
        *args: Positional arguments
        **kwargs: Keyword arguments
        
    Returns:
        Tuple of (success, result, error_message)
    """
    try:
        result = func(*args, **kwargs)
        return True, result, None
    except Exception as e:
        logger.error(f"Error in {func.__name__}: {e}")
        return False, None, str(e)

def check_memory_usage(device) -> Dict[str, float]:
    """
    Check current memory usage.
    
    Args:
        device: PyTorch device
        
    Returns:
        Dictionary with memory usage information
    """
    memory_info = {}
    
    if device.type == "cuda":
        try:
            memory_info["allocated"] = torch.cuda.memory_allocated(device) / (1024**3)
            memory_info["cached"] = torch.cuda.memory_reserved(device) / (1024**3)
            memory_info["max_allocated"] = torch.cuda.max_memory_allocated(device) / (1024**3)
            memory_info["total"] = torch.cuda.get_device_properties(device).total_memory / (1024**3)
            memory_info["free"] = memory_info["total"] - memory_info["allocated"]
        except Exception as e:
            logger.warning(f"Cannot get CUDA memory info: {e}")
    else:
        # For CPU, we can try to get system memory info
        try:
            import psutil
            mem = psutil.virtual_memory()
            memory_info["total"] = mem.total / (1024**3)
            memory_info["available"] = mem.available / (1024**3)
            memory_info["used"] = mem.used / (1024**3)
        except ImportError:
            memory_info["note"] = "Install psutil for CPU memory monitoring"
        except Exception as e:
            logger.warning(f"Cannot get system memory info: {e}")
    
    return memory_info

def validate_json_config(filepath: str) -> Tuple[bool, Dict[str, Any], str]:
    """
    Validate and load JSON configuration file.
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        Tuple of (is_valid, config_dict, error_message)
    """
    # First validate the file path
    valid, error = validate_file_path(filepath, must_exist=True, extension='.json')
    if not valid:
        return False, {}, error
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        
        if not isinstance(config_data, dict):
            return False, {}, "Configuration file must contain a JSON object"
        
        return True, config_data, ""
        
    except json.JSONDecodeError as e:
        return False, {}, f"Invalid JSON format: {e}"
    except Exception as e:
        return False, {}, f"Error reading file: {e}"
