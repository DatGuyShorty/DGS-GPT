"""
Configuration management utilities for DGS-GPT
"""
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import asdict
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class ConfigManager:
    """Advanced configuration management with versioning and validation"""
    
    def __init__(self, config_dir: str = "configs"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        
    def save_config(self, config, name: str, description: str = "", tags: Optional[list] = None) -> bool:
        """
        Save configuration with metadata
        
        Args:
            config: Configuration object
            name: Configuration name
            description: Optional description
            tags: Optional tags for categorization
            
        Returns:
            True if successful, False otherwise
        """
        try:
            config_data = {
                'config': asdict(config),
                'metadata': {
                    'name': name,
                    'description': description,
                    'created_at': datetime.now().isoformat(),
                    'version': '1.0'
                }
            }
            
            filepath = self.config_dir / f"{name}.json"
            with open(filepath, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            logger.info(f"Configuration saved: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving configuration {name}: {e}")
            return False
    
    def load_config(self, name: str):
        """
        Load configuration by name
        
        Args:
            name: Configuration name
            
        Returns:
            Configuration object or None if not found
        """
        try:
            filepath = self.config_dir / f"{name}.json"
            
            if not filepath.exists():
                logger.warning(f"Configuration {name} not found")
                return None
            
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Import Config here to avoid circular imports
            from ShitGPT import Config
            
            config_dict = data.get('config', {})
            config = Config(**config_dict)
            
            logger.info(f"Configuration loaded: {name}")
            return config
            
        except Exception as e:
            logger.error(f"Error loading configuration {name}: {e}")
            return None
    
    def list_configs(self) -> list:
        """List all available configurations"""
        configs = []
        
        for filepath in self.config_dir.glob("*.json"):
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                metadata = data.get('metadata', {})
                configs.append({
                    'name': filepath.stem,
                    'description': metadata.get('description', ''),
                    'tags': metadata.get('tags', []),
                    'created_at': metadata.get('created_at', ''),
                })
                
            except Exception as e:
                logger.warning(f"Error reading config {filepath}: {e}")
        
        return configs
    
    def delete_config(self, name: str) -> bool:
        """Delete a configuration"""
        try:
            filepath = self.config_dir / f"{name}.json"
            
            if filepath.exists():
                filepath.unlink()
                logger.info(f"Configuration deleted: {name}")
                return True
            else:
                logger.warning(f"Configuration {name} not found")
                return False
                
        except Exception as e:
            logger.error(f"Error deleting configuration {name}: {e}")
            return False
    
    def create_preset_configs(self):
        """Create standard preset configurations"""
        from ShitGPT import Config
        
        presets = [
            {
                'name': 'tiny_gpt',
                'description': 'Tiny GPT for testing and debugging',
                'tags': ['testing', 'debug', 'small'],
                'config': Config(
                    n_layer=4,
                    n_head=4,
                    n_embd=256,
                    block_size=128,
                    batch_size=4,
                    use_moe=False,
                    attention_type='multihead'
                )
            },
            {
                'name': 'small_gpt',
                'description': 'Small GPT for experiments',
                'tags': ['experiment', 'small'],
                'config': Config(
                    n_layer=8,
                    n_head=8,
                    n_embd=512,
                    block_size=256,
                    batch_size=8,
                    use_moe=False,
                    attention_type='multihead'
                )
            },
            {
                'name': 'standard_gpt',
                'description': 'Standard GPT configuration',
                'tags': ['standard', 'default'],
                'config': Config(
                    n_layer=16,
                    n_head=16,
                    n_embd=1024,
                    block_size=512,
                    batch_size=16,
                    use_moe=False,
                    attention_type='multihead'
                )
            },
            {
                'name': 'large_gpt_moe',
                'description': 'Large GPT with Mixture of Experts',
                'tags': ['large', 'moe', 'advanced'],
                'config': Config(
                    n_layer=24,
                    n_head=24,
                    n_embd=1536,
                    block_size=1024,
                    batch_size=8,
                    use_moe=True,
                    moe_num_experts=8,
                    moe_k=2,
                    attention_type='grouped_query',
                    n_query_groups=6
                )
            }
        ]
        
        for preset in presets:
            self.save_config(
                preset['config'],
                preset['name'],
                preset['description'],
                preset['tags']
            )
        
        logger.info(f"Created {len(presets)} preset configurations")

# Configuration templates for different use cases
VRAM_PROFILES = {
    'ultra_low': {
        'description': 'For GPUs with 4-8GB VRAM',
        'batch_size': 1,
        'block_size': 128,
        'n_embd': 384,
        'n_layer': 6,
        'n_head': 6,
        'use_gradient_checkpointing': True,
        'use_moe': False
    },
    'low': {
        'description': 'For GPUs with 8-16GB VRAM',
        'batch_size': 2,
        'block_size': 512,
        'n_embd': 768,
        'n_layer': 12,
        'n_head': 12,
        'use_gradient_checkpointing': True,
        'use_moe': False
    },
    'medium': {
        'description': 'For GPUs with 16-32GB VRAM',
        'batch_size': 8,
        'block_size': 1024,
        'n_embd': 1536,
        'n_layer': 24,
        'n_head': 24,
        'use_gradient_checkpointing': False,
        'use_moe': True,
        'moe_num_experts': 4
    },
    'high': {
        'description': 'For GPUs with 32GB+ VRAM',
        'batch_size': 16,
        'block_size': 2048,
        'n_embd': 2048,
        'n_layer': 32,
        'n_head': 32,
        'use_gradient_checkpointing': False,
        'use_moe': True,
        'moe_num_experts': 8
    }
}

def get_recommended_config(vram_gb: float, use_case: str = 'general'):
    """
    Get recommended configuration based on available VRAM and use case
    
    Args:
        vram_gb: Available VRAM in GB
        use_case: 'general', 'research', 'production', 'debug'
        
    Returns:
        Dictionary with recommended configuration
    """
    # Select VRAM profile
    if vram_gb < 8:
        profile = VRAM_PROFILES['ultra_low']
    elif vram_gb < 16:
        profile = VRAM_PROFILES['low']
    elif vram_gb < 32:
        profile = VRAM_PROFILES['medium']
    else:
        profile = VRAM_PROFILES['high']
    
    # Adjust for use case
    config = profile.copy()
    
    if use_case == 'debug':
        # Smaller configs for debugging
        config['n_layer'] = min(config['n_layer'], 4)
        config['batch_size'] = min(config['batch_size'], 2)
        config['max_iters'] = 100
        
    elif use_case == 'research':
        # Balanced configs for research
        config['use_moe'] = True
        config['attention_type'] = 'grouped_query'
        
    elif use_case == 'production':
        # Optimized for inference
        config['use_gradient_checkpointing'] = False
        config['dropout'] = 0.0
    
    return config

# Global config manager instance
config_manager = ConfigManager()
