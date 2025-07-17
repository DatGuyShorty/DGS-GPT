"""
Validation utilities for DGS-GPT
"""
import torch
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import asdict
import logging

logger = logging.getLogger(__name__)

class ValidationError(Exception):
    """Custom exception for validation errors"""
    pass

def validate_config(config) -> List[str]:
    """
    Validate configuration parameters and return list of issues
    
    Args:
        config: Configuration object to validate
        
    Returns:
        List of validation error messages (empty if valid)
    """
    issues = []
    
    # Basic parameter validation
    if config.n_embd <= 0:
        issues.append("n_embd must be positive")
    
    if config.n_head <= 0:
        issues.append("n_head must be positive")
        
    if config.n_layer <= 0:
        issues.append("n_layer must be positive")
        
    if config.block_size <= 0:
        issues.append("block_size must be positive")
        
    if config.batch_size <= 0:
        issues.append("batch_size must be positive")
        
    # Ensure embedding dimension is divisible by number of heads
    if config.n_embd % config.n_head != 0:
        issues.append(f"n_embd ({config.n_embd}) must be divisible by n_head ({config.n_head})")
    
    # Validate dropout
    if not 0 <= config.dropout <= 1:
        issues.append("dropout must be between 0 and 1")
        
    # Validate learning rate
    if config.learning_rate <= 0:
        issues.append("learning_rate must be positive")
        
    # Validate MoE parameters
    if config.use_moe:
        if config.moe_num_experts <= 0:
            issues.append("moe_num_experts must be positive when MoE is enabled")
        if config.moe_k <= 0 or config.moe_k > config.moe_num_experts:
            issues.append(f"moe_k must be between 1 and moe_num_experts ({config.moe_num_experts})")
    
    # Validate grouped query attention
    if config.attention_type == "grouped_query":
        if config.n_query_groups <= 0:
            issues.append("n_query_groups must be positive for grouped query attention")
        if config.n_head % config.n_query_groups != 0:
            issues.append(f"n_head ({config.n_head}) must be divisible by n_query_groups ({config.n_query_groups})")
    
    # Validate VRAM profile
    valid_profiles = ["low", "medium", "high"]
    if config.vram_profile not in valid_profiles:
        issues.append(f"vram_profile must be one of {valid_profiles}")
        
    # Validate attention type
    valid_attention_types = ["multihead", "grouped_query"]
    if config.attention_type not in valid_attention_types:
        issues.append(f"attention_type must be one of {valid_attention_types}")
        
    # Validate model type
    valid_model_types = ["gpt", "compact", "llama", "mamba"]
    if config.model_type not in valid_model_types:
        issues.append(f"model_type must be one of {valid_model_types}")
    
    return issues

def validate_generation_params(max_new_tokens: int, temperature: float, 
                             top_k: Optional[int] = None, top_p: Optional[float] = None) -> List[str]:
    """
    Validate text generation parameters
    
    Args:
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        top_k: Top-k sampling parameter (optional)
        top_p: Top-p (nucleus) sampling parameter (optional)
        
    Returns:
        List of validation error messages (empty if valid)
    """
    issues = []
    
    if max_new_tokens <= 0:
        issues.append("max_new_tokens must be positive")
        
    if temperature <= 0:
        issues.append("temperature must be positive")
        
    if top_k is not None and top_k <= 0:
        issues.append("top_k must be positive if specified")
        
    if top_p is not None and not 0 < top_p <= 1:
        issues.append("top_p must be between 0 and 1 if specified")
        
    return issues

def validate_training_params(max_iters: int, plot_step_size: int) -> List[str]:
    """
    Validate training parameters
    
    Args:
        max_iters: Maximum training iterations
        plot_step_size: Steps between plot updates
        
    Returns:
        List of validation error messages (empty if valid)
    """
    issues = []
    
    if max_iters <= 0:
        issues.append("max_iters must be positive")
        
    if plot_step_size <= 0:
        issues.append("plot_step_size must be positive")
        
    if plot_step_size > max_iters:
        issues.append("plot_step_size should not be larger than max_iters")
        
    return issues

def validate_model_state(model: torch.nn.Module) -> List[str]:
    """
    Validate model state
    
    Args:
        model: PyTorch model to validate
        
    Returns:
        List of validation error messages (empty if valid)
    """
    issues = []
    
    if model is None:
        issues.append("Model is None")
        return issues
    
    # Check if model has parameters
    param_count = sum(p.numel() for p in model.parameters())
    if param_count == 0:
        issues.append("Model has no parameters")
    
    # Check for NaN or infinite parameters
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            issues.append(f"Parameter {name} contains NaN values")
        if torch.isinf(param).any():
            issues.append(f"Parameter {name} contains infinite values")
    
    return issues

def safe_execute(func, *args, **kwargs):
    """
    Safely execute a function with error handling
    
    Args:
        func: Function to execute
        *args: Positional arguments
        **kwargs: Keyword arguments
        
    Returns:
        Tuple of (success: bool, result: Any, error: Optional[str])
    """
    try:
        result = func(*args, **kwargs)
        return True, result, None
    except Exception as e:
        error_msg = f"Error in {func.__name__}: {str(e)}"
        logger.error(error_msg)
        return False, None, error_msg

def check_memory_usage(device: torch.device) -> Dict[str, Any]:
    """
    Check current memory usage
    
    Args:
        device: PyTorch device
        
    Returns:
        Dictionary with memory usage information
    """
    memory_info = {}
    
    if device.type == "cuda":
        memory_info["allocated"] = torch.cuda.memory_allocated(device) / 1024**3  # GB
        memory_info["reserved"] = torch.cuda.memory_reserved(device) / 1024**3    # GB
        memory_info["max_allocated"] = torch.cuda.max_memory_allocated(device) / 1024**3  # GB
        
        # Get total GPU memory
        try:
            total_memory = torch.cuda.get_device_properties(device).total_memory / 1024**3  # GB
            memory_info["total"] = total_memory
            memory_info["usage_percent"] = (memory_info["allocated"] / total_memory) * 100
        except Exception as e:
            logger.warning(f"Could not get total GPU memory: {e}")
            memory_info["total"] = "unknown"
            memory_info["usage_percent"] = "unknown"
    else:
        memory_info["device"] = "CPU"
        # For CPU, we could add system memory info here if needed
    
    return memory_info
