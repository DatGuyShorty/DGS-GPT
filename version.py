"""
DGS-GPT Version Information
"""

__version__ = "0.1.0-alpha"
__author__ = "DatGuyShorty"
__description__ = "A modern GPT implementation with advanced training capabilities and hyperparameter optimization"
__license__ = "MIT"

# Version history
CHANGELOG = {
    "0.1.0-alpha": [
        "Initial alpha release",
        "Core GPT model implementation with transformer architecture",
        "Multi-head attention and grouped query attention support",
        "Mixture of Experts (MoE) integration",
        "Comprehensive GUI with three-tab interface",
        "Real-time training visualization and loss plotting",
        "Optuna-powered hyperparameter optimization",
        "Cross-platform setup scripts (Windows, Linux, macOS)",
        "Character-level tokenization with custom vocabulary",
        "Model save/load functionality",
        "Memory-efficient training with mixed precision",
        "Live hyperparameter display in training tab"
    ]
}

def get_version():
    """Get the current version string"""
    return __version__

def get_version_info():
    """Get detailed version information"""
    return {
        "version": __version__,
        "author": __author__,
        "description": __description__,
        "license": __license__
    }

def print_version_info():
    """Print version information to console"""
    print(f"DGS-GPT v{__version__}")
    print(f"Author: {__author__}")
    print(f"Description: {__description__}")
    print(f"License: {__license__}")
