import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.checkpoint import checkpoint
import time
import optuna
import sys
import json
import os
import logging
from dataclasses import dataclass, asdict
import gc
from typing import Optional, Tuple, List, Dict, Any, Union
import warnings

# Import validation and logging utilities
try:
    from validation_utils import validate_config, validate_generation_params, safe_execute, check_memory_usage
    from logging_utils import setup_logging, logger
except ImportError:
    # Fallback if utils not available
    import logging
    logger = logging.getLogger(__name__)
    def validate_config(config): return []
    def validate_generation_params(*args, **kwargs): return []
    def safe_execute(func, *args, **kwargs):
        try:
            return True, func(*args, **kwargs), None
        except Exception as e:
            return False, None, str(e)
    def check_memory_usage(device): return {}
    
    # Setup basic logging since logging_utils is not available
    def setup_logging(log_level=logging.INFO, log_file=None):
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logger

# Device configuration with optimizations
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Apply performance optimizations immediately
print("🚀 Initializing DGS-GPT with performance optimizations...")
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
if device.type == "cuda":
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    print(f"⚡ CUDA optimizations enabled for {torch.cuda.get_device_name()}")

# Setup logging
setup_logging(log_level=logging.INFO, log_file="dgs_gpt.log")

@dataclass
class Config:
    """Configuration class for DGS-GPT model with comprehensive parameter validation"""
    # Model params - optimized defaults for speed
    block_size: int = 512  # context window size
    vocab_size: int = 50304  # set in main
    n_layer: int = 8  # Reduced for speed - fewer layers train faster
    n_head: int = 8  # Reduced for speed - fewer heads train faster 
    n_embd: int = 512  # Reduced for speed - smaller embeddings train faster
    dropout: float = 0.05  # Reduced dropout for faster training
    use_moe: bool = False  # Disabled MoE for speed - MoE adds overhead
    moe_num_experts: int = 4  # number of experts
    moe_k: int = 1  # top-k experts
    attention_type: str = "multihead"  # "multihead" is faster than "grouped_query"
    n_query_groups: int = 1  # number of query groups
    model_type: str = "compact"  # Use compact model for faster training

    # Training params - optimized for speed
    learning_rate: float = 1e-3  # Higher LR for faster convergence
    weight_decay: float = 1e-6  # Reduced weight decay for speed
    batch_size: int = 8  # Increased batch size for better GPU utilization
    max_iters: int = 1000  # maximum training iterations
    grad_clip: float = 5.0  # Increased for stability with higher LR
    warmup_iters: int = 50  # Reduced warmup for faster start
    
    # VRAM optimization settings
    vram_profile: str = "low"  # "low" (15GB), "medium" (40GB), "high" (100GB+)
    use_gradient_checkpointing: bool = False  # Disabled for speed - trades memory for speed
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        issues = validate_config(self)
        if issues:
            warning_msg = "Configuration validation issues:\n" + "\n".join(f"- {issue}" for issue in issues)
            warnings.warn(warning_msg, UserWarning)
            logger.warning(warning_msg)
    
    def _validate_critical_parameters(self):
        """Validate critical parameters that could cause runtime errors"""
        if self.n_embd % 2 != 0:
            raise ValueError("n_embd must be even for RoPE compatibility")
        
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
            
        if self.attention_type == "grouped_query" and self.n_head % self.n_query_groups != 0:
            raise ValueError("n_head must be divisible by n_query_groups for grouped query attention")
            
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
            
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
    
    @classmethod
    def get_vram_optimized_config(cls, vram_profile: str = "low", base_config: Optional['Config'] = None) -> 'Config':
        """Get VRAM-optimized configuration based on available GPU memory"""
        if base_config is None:
            config = cls()
        else:
            # Create a copy of the base config to avoid modifying the original
            config = cls(**asdict(base_config))
            
        if vram_profile == "low":
            # Conservative settings for 15GB VRAM
            config.batch_size = 2
            config.block_size = 512
            config.n_embd = 1024
            config.n_layer = 4
            config.n_head = 4
            config.dropout = 0.1
            config.use_gradient_checkpointing = True
            config.moe_num_experts = 4
            config.moe_k = 1
            logger.info("Applied LOW VRAM profile (15GB) - Conservative settings")
            
        elif vram_profile == "medium":
            # Standard settings for 40GB VRAM
            config.batch_size = 16
            config.block_size = 8192
            config.n_embd = 4096
            config.n_layer = 32
            config.n_head = 32
            config.dropout = 0.2
            config.use_gradient_checkpointing = False
            config.moe_num_experts = 8
            config.moe_k = 2
            logger.info("Applied MEDIUM VRAM profile (40GB) - Standard settings")
            
        elif vram_profile == "high":
            # Expanded settings for 100GB+ VRAM
            config.batch_size = 32
            config.block_size = 16384
            config.n_embd = 5120
            config.n_layer = 40
            config.n_head = 40
            config.dropout = 0.2
            config.use_gradient_checkpointing = False
            config.moe_num_experts = 16
            config.moe_k = 4
            logger.info("Applied HIGH VRAM profile (100GB+) - Expanded settings")
            
        # Validate dimensions for RoPE and GQA
        assert config.n_embd % 2 == 0, "n_embd must be even for RoPE"
        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"
        if config.attention_type == "grouped_query":
            assert config.n_head % config.n_query_groups == 0, "n_head must be divisible by n_query_groups"
            
        config.vram_profile = vram_profile
        return config

def load_config(filepath: str = "best_hyperparams.json") -> Config:
    """Load configuration from JSON file with error handling"""
    default_config = Config()
    if os.path.exists(filepath):
        try:
            with open(filepath, "r") as f:
                best_params = json.load(f)
            if isinstance(best_params, dict) and best_params:
                logger.info(f"Loaded best hyperparameters from {filepath}")
                # Update default_config with loaded params
                for key, value in best_params.items():
                    if hasattr(default_config, key):
                        setattr(default_config, key, value)
                return default_config
            else:
                logger.warning(f"{filepath} is empty or invalid, using defaults.")
        except Exception as e:
            logger.error(f"Error loading {filepath}: {e}, using defaults.")
    else:
        logger.info("Using default hyperparameters")
    return default_config

# Load and encode data (char-level example)
try:
    with open("vocab.txt", "r", encoding="utf-8") as f:
        txt = f.read()
except Exception as e:
    logger.error(f"Error loading vocab.txt: {e}")
    sys.exit(1)

if not txt:
    logger.error("vocab.txt is empty. Please generate it first.")
    sys.exit(1)

chars = sorted(set(txt))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda ids: ''.join([itos[i] for i in ids])
data = torch.tensor(encode(txt), dtype=torch.long)

logger.info(f"Loaded vocabulary with {vocab_size} characters")
logger.info(f"Dataset size: {len(data):,} tokens")

# Optimized batch generation with pre-allocated tensors and efficient indexing
_batch_cache = {}  # Global cache for batch data
_cache_size = 1000  # Number of batches to cache

def get_batch(config: Config) -> Tuple[torch.Tensor, torch.Tensor]:
    """Optimized batch generation with caching and efficient tensor operations"""
    global _batch_cache
    
    # Use config as cache key
    cache_key = (config.batch_size, config.block_size, device.type)
    
    # Check if we have cached batches for this configuration
    if cache_key not in _batch_cache:
        _batch_cache[cache_key] = []
    
    cache = _batch_cache[cache_key]
    
    # If cache is not full, pre-generate batches for better performance
    if len(cache) < _cache_size:
        # Pre-generate multiple batches at once for efficiency
        batch_count = min(50, _cache_size - len(cache))  # Generate 50 batches at a time
        
        # Generate random indices for all batches at once
        all_ix = torch.randint(len(data) - config.block_size, (batch_count, config.batch_size), device='cpu')
        
        for i in range(batch_count):
            ix = all_ix[i]
            # Vectorized batch creation - much faster than list comprehension
            x = data[ix.unsqueeze(1) + torch.arange(config.block_size, device='cpu').unsqueeze(0)]
            y = data[ix.unsqueeze(1) + torch.arange(1, config.block_size + 1, device='cpu').unsqueeze(0)]
            
            # Move to target device efficiently
            cache.append((x.to(device, non_blocking=True), y.to(device, non_blocking=True)))
    
    # Return a batch from cache (pop for memory efficiency)
    if cache:
        return cache.pop(0)
    else:
        # Fallback to direct generation if cache is empty
        ix = torch.randint(len(data) - config.block_size, (config.batch_size,), device='cpu')
        x = data[ix.unsqueeze(1) + torch.arange(config.block_size, device='cpu').unsqueeze(0)]
        y = data[ix.unsqueeze(1) + torch.arange(1, config.block_size + 1, device='cpu').unsqueeze(0)]
        return x.to(device, non_blocking=True), y.to(device, non_blocking=True)

# Add a Mixture of Experts (MoE) layer
class MoE(nn.Module):
    def __init__(self, input_size, hidden_size, num_experts=4, k=1, dropout=0.1):
        super().__init__()
        self.num_experts = num_experts
        self.k = k  # top-k experts
        self.experts = nn.ModuleList([nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size),
            nn.Dropout(dropout)
        ) for _ in range(num_experts)])
        self.gate = nn.Linear(input_size, num_experts)

    def forward(self, x):
        # x: (B, T, C)
        gate_scores = self.gate(x)  # (B, T, num_experts)
        topk_scores, topk_idx = torch.topk(gate_scores, self.k, dim=-1)  # (B, T, k)
        topk_weights = torch.softmax(topk_scores, dim=-1)  # (B, T, k)
        
        expert_outputs = []
        for i, expert in enumerate(self.experts):
            expert_outputs.append(expert(x))  # (B, T, C)
        expert_outputs = torch.stack(expert_outputs, dim=-2)  # (B, T, num_experts, C)

        # Gather top-k expert outputs
        idx_expanded = topk_idx.unsqueeze(-1).expand(-1, -1, -1, x.size(-1))  # (B, T, k, C)
        topk_expert_outputs = torch.gather(expert_outputs, -2, idx_expanded)  # (B, T, k, C)
        # Weighted sum
        topk_weights = topk_weights.unsqueeze(-1)  # (B, T, k, 1)
        moe_output = (topk_expert_outputs * topk_weights).sum(dim=-2)  # (B, T, C)
        return moe_output

# Sparse Mixture of Experts (Enhanced MoE) from the notebook
class SparseExpertLayer(nn.Module):
    def __init__(self, n_embd, num_experts=8, num_active=2, capacity_factor=1.2, dropout=0.1):
        super().__init__()
        self.num_experts = num_experts
        self.num_active = num_active
        self.capacity_factor = capacity_factor
        self.n_embd = n_embd
        
        self.gate = nn.Linear(n_embd, num_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(n_embd, 4 * n_embd),
                nn.GELU(),  # Using GELU instead of ReLU
                nn.Linear(4 * n_embd, n_embd),
                nn.Dropout(dropout)
            ) for _ in range(num_experts)
        ])
        
    def forward(self, x):
        B, T, C = x.shape
        x_flat = x.view(-1, C)  # (B*T, C)
        
        # Calculate gates and expert assignment
        gate_logits = self.gate(x_flat)  # (B*T, num_experts)
        gates = F.softmax(gate_logits, dim=-1)
        
        # Get top-k experts for each token
        expert_weights, expert_indices = torch.topk(gates, self.num_active, dim=-1)
        expert_weights = expert_weights / expert_weights.sum(dim=-1, keepdim=True)
        
        # Initialize output
        final_output = torch.zeros_like(x_flat)
        
        # Process each expert
        for expert_idx in range(self.num_experts):
            # Find tokens routed to this expert
            expert_mask = (expert_indices == expert_idx).any(dim=-1)
            expert_tokens = expert_mask.sum().item()
            
            if expert_tokens > 0:
                # Get tokens for this expert
                expert_input = x_flat[expert_mask]
                
                # Process through expert
                expert_output = self.experts[expert_idx](expert_input)
                
                # Get weights for this expert
                expert_weights_masked = expert_weights[expert_mask]
                expert_pos = (expert_indices[expert_mask] == expert_idx).nonzero(as_tuple=True)[1]
                weights = expert_weights_masked[torch.arange(len(expert_pos)), expert_pos]
                
                # Add weighted output
                final_output[expert_mask] += expert_output * weights.unsqueeze(-1)
        
        return final_output.view(B, T, C)

# Refined MultiHeadAttention using nn.MultiheadAttention
class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_head, dropout, block_size):
        super().__init__()
        self.mha = nn.MultiheadAttention(n_embd, n_head, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        attn_mask = self.tril[:T, :T] == 0  # (T, T)
        attn_mask = attn_mask.to(x.device)
        out, _ = self.mha(x, x, x, attn_mask=attn_mask)
        return self.dropout(out)

# FeedForward unchanged
class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

# Grouped Query Attention implementation
class GroupedQueryAttention(nn.Module):
    def __init__(self, n_embd, n_head, n_query_groups, dropout, block_size):
        super().__init__()
        assert n_head % n_query_groups == 0, "n_head must be divisible by n_query_groups"
        self.n_head = n_head
        self.n_query_groups = n_query_groups
        self.head_dim = n_embd // n_head
        self.q_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.k_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.v_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.out_proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        H = self.n_head
        G = self.n_query_groups
        D = self.head_dim

        q = self.q_proj(x).view(B, T, H, D)
        k = self.k_proj(x).view(B, T, G, H // G, D)
        v = self.v_proj(x).view(B, T, G, H // G, D)

        # For each group, use the same k/v for all heads in that group
        k = k.permute(0,2,1,3,4).reshape(B, G, T, H // G * D)
        v = v.permute(0,2,1,3,4).reshape(B, G, T, H // G * D)

        q = q.permute(0,2,1,3)  # (B, H, T, D)
        q_groups = q.view(B, G, H // G, T, D)

        # Compute attention for each group
        out = []
        for g in range(G):
            qg = q_groups[:,g]  # (B, H//G, T, D)
            kg = k[:,g]         # (B, T, H//G*D)
            vg = v[:,g]         # (B, T, H//G*D)
            kg = kg.view(B, T, H//G, D).permute(0,2,1,3)  # (B, H//G, T, D)
            vg = vg.view(B, T, H//G, D).permute(0,2,1,3)  # (B, H//G, T, D)
            attn_scores = torch.matmul(qg, kg.transpose(-2, -1)) / (D ** 0.5)  # (B, H//G, T, T)
            attn_mask = self.tril[:T, :T].unsqueeze(0).unsqueeze(0) == 0
            attn_scores = attn_scores.masked_fill(attn_mask.to(x.device), float('-inf'))
            attn_probs = F.softmax(attn_scores, dim=-1)
            attn_probs = self.dropout(attn_probs)
            out_g = torch.matmul(attn_probs, vg)  # (B, H//G, T, D)
            out.append(out_g)
        out = torch.cat(out, dim=1)  # (B, H, T, D)
        out = out.permute(0,2,1,3).contiguous().view(B, T, H*D)
        return self.out_proj(out)

# Rotary Positional Embedding (RoPE) implementation
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048):
        super().__init__()
        # Make sure dim is even
        dim = dim if dim % 2 == 0 else dim - 1
        
        # Create inverse frequency bands
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.max_position_embeddings = max_position_embeddings
        self.dim = dim

    def forward(self, x, seq_len):
        B, T, C = x.shape
        
        # Ensure we only apply RoPE to the dimensionality we were initialized with
        rope_dim = min(self.dim, C)
        
        # Create position indices
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        
        # Calculate frequencies
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)  # [seq_len, rope_dim/2]
        
        # Calculate cos and sin
        cos = freqs.cos()  # [seq_len, rope_dim/2]
        sin = freqs.sin()  # [seq_len, rope_dim/2]
        
        # Expand dimensions for broadcasting
        cos = cos.view(1, T, -1)  # [1, seq_len, rope_dim/2]
        sin = sin.view(1, T, -1)  # [1, seq_len, rope_dim/2]
        
        # Duplicate for all batch elements
        cos = cos.expand(B, -1, -1)  # [batch, seq_len, rope_dim/2]
        sin = sin.expand(B, -1, -1)  # [batch, seq_len, rope_dim/2]
        
        # Split input into the portion we apply RoPE to and the rest
        x_rope = x[..., :rope_dim]  # [batch, seq_len, rope_dim]
        x_rest = x[..., rope_dim:]  # [batch, seq_len, remaining_dim]
        
        # Split RoPE portion into even and odd dimensions
        x1 = x_rope[..., ::2]  # [batch, seq_len, rope_dim/2]
        x2 = x_rope[..., 1::2]  # [batch, seq_len, rope_dim/2]
        
        # Apply rotation
        rotated_rope = torch.cat([
            x1 * cos - x2 * sin,
            x2 * cos + x1 * sin
        ], dim=-1)  # [batch, seq_len, rope_dim]
        
        # Concatenate rotated portion with unrotated portion
        if x_rest.size(-1) > 0:
            rotated_x = torch.cat([rotated_rope, x_rest], dim=-1)
        else:
            rotated_x = rotated_rope
        
        return rotated_x

# Sliding Window Attention implementation
class SlidingWindowAttention(nn.Module):
    def __init__(self, n_embd, n_head, window_size=256, dropout=0.1, block_size=512):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.n_embd = n_embd
        self.head_dim = n_embd // n_head
        self.window_size = window_size
        
        self.query = nn.Linear(n_embd, n_embd)
        self.key = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        self.out_proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
        
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        
    def forward(self, x):
        B, T, C = x.shape
        
        q = self.query(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
        k = self.key(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)    # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
        
        # Apply sliding window attention
        att_scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (B, nh, T, T)
        
        # Create sliding window mask
        window_mask = torch.zeros_like(att_scores)
        for i in range(T):
            start = max(0, i - self.window_size)
            end = min(T, i + 1)
            window_mask[:, :, i, start:end] = 1
        
        # Combine with causal mask
        causal_mask = self.tril[:T, :T].unsqueeze(0).unsqueeze(0)
        mask = window_mask * causal_mask
        
        att_scores = att_scores.masked_fill(mask == 0, float('-inf'))
        att_probs = F.softmax(att_scores, dim=-1)
        att_probs = self.dropout(att_probs)
        
        out = att_probs @ v  # (B, nh, T, hs)
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)
        
        return self.out_proj(out)

# Cosine Learning Rate Scheduler with Warmup
class CosineWarmupScheduler:
    def __init__(self, optimizer, warmup_steps, max_steps, max_lr, min_lr=0.0):
        self.optimizer = optimizer
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.current_step = 0
        
        # Smart warmup and max_steps handling
        if warmup_steps >= max_steps:
            # If warmup is too long, automatically reduce it to 10% of max_steps
            self.warmup_steps = max(1, max_steps // 10)
            self.max_steps = max_steps
            if warmup_steps > 0:  # Only warn if warmup was actually requested
                print(f"Info: Warmup steps automatically adjusted from {warmup_steps} to {self.warmup_steps} (10% of training steps)")
        else:
            self.warmup_steps = max(1, warmup_steps)  # Ensure at least 1 warmup step
            self.max_steps = max_steps
        
    def step(self):
        self.current_step += 1
        if self.current_step < self.warmup_steps:
            # Linear warmup - protect against division by zero
            if self.warmup_steps > 0:
                lr = self.max_lr * (self.current_step / self.warmup_steps)
            else:
                lr = self.max_lr
        else:
            # Cosine decay
            # Prevent division by zero when max_steps equals warmup_steps
            if self.max_steps <= self.warmup_steps:
                # If max_steps is less than or equal to warmup_steps, just use max_lr
                lr = self.max_lr
            else:
                progress = (self.current_step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
                progress = min(progress, 1.0)
                lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())
            
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr

# Block with optional MoE and attention type selection
class Block(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        # Primary attention mechanism
        if config.attention_type == "grouped_query":
            assert config.n_query_groups is not None, "n_query_groups must be set for grouped_query attention"
            self.sa = GroupedQueryAttention(config.n_embd, config.n_head, config.n_query_groups, config.dropout, config.block_size)
        else:
            self.sa = MultiHeadAttention(config.n_embd, config.n_head, config.dropout, config.block_size)
        
        # Add sliding window attention for long-range dependencies
        self.sliding_attention = SlidingWindowAttention(config.n_embd, config.n_head, dropout=config.dropout, block_size=config.block_size)
        
        # Rotary positional embedding - use full embedding dimension
        self.rope = RotaryEmbedding(config.n_embd)
        
        # Feed forward network
        self.ff = FeedForward(config.n_embd, config.dropout)
        
        # Enhanced MoE implementation
        self.use_moe = config.use_moe
        if self.use_moe:
            # Use sparse MoE for better efficiency
            self.sparse_moe = SparseExpertLayer(config.n_embd, num_experts=config.moe_num_experts, 
                                              num_active=config.moe_k, dropout=config.dropout)
            self.ln3 = nn.LayerNorm(config.n_embd)
        
        # Layer norms
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.ln4 = nn.LayerNorm(config.n_embd)  # For sliding window attention

    def forward(self, x):
        B, T, C = x.shape
        
        # Apply RoPE to input
        x_rope = self.rope(self.ln1(x), T)
        
        # Primary attention with RoPE
        x = x + self.sa(x_rope)
        
        # Sliding window attention for additional context
        x = x + self.sliding_attention(self.ln4(x))
        
        # Feed forward
        x = x + self.ff(self.ln2(x))
        
        # Sparse MoE if enabled
        if self.use_moe:
            x = x + self.sparse_moe(self.ln3(x))
            
        return x

# GPTLanguageModel updated with gradient checkpointing and enhanced features
class GPTLanguageModel(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)
        
        # Gradient checkpointing for VRAM optimization
        self.use_gradient_checkpointing = getattr(config, 'use_gradient_checkpointing', False)
        
        self.apply(self._init_weights)
        
        # Print model information
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"GPT Model initialized with {total_params:,} parameters")
        logger.info(f"Estimated VRAM usage: ~{total_params * 4 / (1024**3):.1f}GB")
        if self.use_gradient_checkpointing:
            logger.info("Gradient checkpointing enabled for VRAM optimization")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        tok = self.token_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=idx.device))
        x = tok + pos
        
        # Apply gradient checkpointing if enabled
        if self.use_gradient_checkpointing and self.training:
            # Use checkpoint for each block to save memory
            for block in self.blocks:
                x = checkpoint(block, x, use_reentrant=False)
        else:
            x = self.blocks(x)
            
        logits = self.lm_head(self.ln_f(x))
        
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return logits, loss
        return logits, None

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None):
        """
        Advanced text generation with temperature and nucleus sampling
        
        Args:
            idx: Starting token indices
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Only consider top-k tokens (if specified)
            top_p: Nucleus sampling - consider tokens with cumulative probability up to top_p
        """
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature  # Apply temperature
            
            # Apply top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Apply nucleus (top-p) sampling
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep also the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                # Create a mask for the original logits
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float('Inf')
            
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)
        return idx

    def generate_batch(self, prompts, max_new_tokens=200, temperature=1.0, top_k=None, top_p=None):
        """
        Generate text for multiple prompts in a batch
        """
        self.eval()
        batch_size = len(prompts)
        
        # Encode all prompts
        encoded_prompts = [torch.tensor(encode(prompt), dtype=torch.long, device=device) for prompt in prompts]
        
        # Pad to same length
        max_len = max(len(p) for p in encoded_prompts)
        padded_prompts = []
        for prompt in encoded_prompts:
            if len(prompt) < max_len:
                padding = torch.zeros(max_len - len(prompt), dtype=torch.long, device=device)
                padded_prompt = torch.cat([prompt, padding])
            else:
                padded_prompt = prompt
            padded_prompts.append(padded_prompt)
        
        idx = torch.stack(padded_prompts)
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
                logits, _ = self(idx_cond)
                logits = logits[:, -1, :] / temperature
                
                # Apply top-k filtering
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                
                # Apply nucleus (top-p) sampling
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = -float('Inf')
                
                probs = F.softmax(logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1)
                idx = torch.cat([idx, next_tokens], dim=1)
        
        self.train()
        return [decode(seq.tolist()) for seq in idx]

    def beam_search_generate(self, idx, max_new_tokens=200, beam_size=4, temperature=1.0):
        """
        Generate text using beam search for better quality
        """
        self.eval()
        batch_size = idx.size(0)
        
        # Initialize beams: (batch_size * beam_size, seq_len)
        beams = idx.repeat(beam_size, 1)  # Replicate for each beam
        beam_scores = torch.zeros(batch_size * beam_size, device=device)
        
        with torch.no_grad():
            for step in range(max_new_tokens):
                # Get logits for all beams
                idx_cond = beams if beams.size(1) <= self.config.block_size else beams[:, -self.config.block_size:]
                logits, _ = self(idx_cond)
                logits = logits[:, -1, :] / temperature
                
                # Convert to log probabilities
                log_probs = F.log_softmax(logits, dim=-1)
                
                # For each sequence in batch
                all_candidates = []
                for i in range(batch_size):
                    beam_start = i * beam_size
                    beam_end = (i + 1) * beam_size
                    
                    # Get top k tokens for each beam
                    candidates = []
                    for beam_idx in range(beam_start, beam_end):
                        beam_score = beam_scores[beam_idx]
                        beam_log_probs = log_probs[beam_idx]
                        
                        # Get top beam_size tokens
                        top_log_probs, top_indices = torch.topk(beam_log_probs, beam_size)
                        
                        for k in range(beam_size):
                            candidate_score = beam_score + top_log_probs[k]
                            candidate_seq = torch.cat([beams[beam_idx], top_indices[k].unsqueeze(0)])
                            candidates.append((candidate_score, candidate_seq, beam_idx))
                    
                    # Select top beam_size candidates
                    candidates.sort(key=lambda x: x[0], reverse=True)
                    all_candidates.extend(candidates[:beam_size])
                
                # Update beams with best candidates
                new_beams = []
                new_scores = []
                for i, (score, seq, _) in enumerate(all_candidates):
                    new_beams.append(seq)
                    new_scores.append(score)
                
                beams = torch.stack(new_beams)
                beam_scores = torch.stack(new_scores)
        
        self.train()
        
        # Return best sequence for each batch item
        best_sequences = []
        for i in range(batch_size):
            beam_start = i * beam_size
            best_beam_idx = beam_start + torch.argmax(beam_scores[beam_start:beam_start + beam_size])
            best_sequences.append(beams[best_beam_idx])
        
        return torch.stack(best_sequences)

class CompactGPTModel(nn.Module):
    """Ultra-optimized GPT variant for maximum training speed and efficiency"""
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        
        # Ultra-compact embedding dimension for speed
        self.embed_dim = min(config.n_embd, 512)  # Cap at 512 for maximum speed
        
        # Optimized embeddings with reduced vocab size if possible
        self.token_emb = nn.Embedding(config.vocab_size, self.embed_dim)
        self.pos_emb = nn.Embedding(config.block_size, self.embed_dim)
        
        # Shared transformer block for weight efficiency and speed
        # Using minimal layers - speed over complexity
        self.n_layers = max(1, config.n_layer // 3)  # Use 1/3 the layers for speed
        
        # Ultra-simplified attention block for speed
        self.attn = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=min(4, config.n_head),  # Max 4 heads for speed
            dropout=config.dropout,
            batch_first=True
        )
        
        # Simplified feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim * 2),  # Smaller expansion for speed
            nn.GELU(),
            nn.Linear(self.embed_dim * 2, self.embed_dim),
            nn.Dropout(config.dropout)
        )
        
        # Layer norms
        self.ln1 = nn.LayerNorm(self.embed_dim)
        self.ln2 = nn.LayerNorm(self.embed_dim)
        self.ln_f = nn.LayerNorm(self.embed_dim)
        
        # Output head
        self.lm_head = nn.Linear(self.embed_dim, config.vocab_size, bias=False)
        
        # Tie weights for efficiency
        self.lm_head.weight = self.token_emb.weight
        
        self.dropout = nn.Dropout(config.dropout)
        self.apply(self._init_weights)
        
        total_params = sum(p.numel() for p in self.parameters())
        print(f"🚀 Ultra-CompactGPT initialized: {self.embed_dim}d embedding, {self.n_layers} layers, {total_params:,} params")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        # Token and position embeddings - optimized
        tok_emb = self.token_emb(idx)
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0)
        pos_emb = self.pos_emb(pos)
        
        x = self.dropout(tok_emb + pos_emb)
        
        # Ultra-efficient transformer layers with shared computation
        for _ in range(self.n_layers):
            # Self-attention with residual connection
            attn_out, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), 
                                   need_weights=False,  # Skip attention weights for speed
                                   attn_mask=self._get_causal_mask(T, x.device))
            x = x + attn_out
            
            # Feed-forward with residual connection
            x = x + self.ff(self.ln2(x))
        
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        if targets is None:
            return logits, None
        else:
            # Optimized loss calculation
            B, T, C = logits.shape
            logits_flat = logits.view(-1, C)
            targets_flat = targets.view(-1)
            loss = F.cross_entropy(logits_flat, targets_flat, ignore_index=-1)
            return logits, loss
    
    def _get_causal_mask(self, seq_len, device):
        """Generate causal mask for self-attention - cached for performance"""
        if not hasattr(self, '_causal_mask') or self._causal_mask.size(0) < seq_len:
            self._causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        return self._causal_mask[:seq_len, :seq_len]

    def generate(self, idx, max_new_tokens=200, temperature=1.0, top_k=None, top_p=None):
        """Simple generation for CompactGPT"""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
                
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float('Inf')
            
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            
        return idx

    def beam_search_generate(self, idx, max_new_tokens=200, beam_size=4, temperature=1.0):
        """Simplified beam search for CompactGPT"""
        # Use the same beam search logic as GPTLanguageModel but adapted for CompactGPT
        return self.generate(idx, max_new_tokens, temperature)  # Fallback to simple generation

def create_model(config: Config):
    """Factory function to create the appropriate model based on config.model_type"""
    if config.model_type == "gpt":
        return GPTLanguageModel(config)
    elif config.model_type == "compact":
        return CompactGPTModel(config)
    elif config.model_type == "llama":
        # Future: Could add LLaMA-style model here
        logger.warning("LLaMA model not yet implemented, falling back to GPT")
        return GPTLanguageModel(config)
    elif config.model_type == "mamba":
        # Future: Could add Mamba/State Space model here
        logger.warning("Mamba model not yet implemented, falling back to GPT")
        return GPTLanguageModel(config)
    else:
        logger.warning(f"Unknown model type '{config.model_type}', falling back to GPT")
        return GPTLanguageModel(config)

class Trainer:
    def __init__(self, config: Config, loss_callback=None, metrics_callback=None):
        self.config = config
        logger.info(f"🚀 Creating OPTIMIZED {config.model_type} model...")
        
        # Performance optimization flags
        torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
        torch.backends.cudnn.deterministic = False  # Allow non-deterministic for speed
        
        # Model creation with performance focus
        self.model = create_model(config).to(device)
        
        # Optimized optimizer with higher momentum for faster convergence
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=config.learning_rate, 
            weight_decay=config.weight_decay,
            betas=(0.9, 0.999),  # Optimized betas for speed
            eps=1e-6  # Reduced epsilon for slightly better performance
        )
        
        # Mixed precision scaler for CUDA
        self.scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
        
        self.loss_callback = loss_callback
        self.metrics_callback = metrics_callback
        
        # Enhanced learning rate scheduler with cosine warmup
        self.scheduler = CosineWarmupScheduler(
            self.optimizer,
            warmup_steps=config.warmup_iters,
            max_steps=config.max_iters,
            max_lr=config.learning_rate,
            min_lr=config.learning_rate * 0.05  # Lower min_lr for better convergence
        )
        
        # Training metrics tracking (reduced storage for performance)
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        self.perplexities = []
        
        # Performance tracking
        self.total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"🎯 Model initialized: {self.total_params:,} parameters")
        logger.info(f"💾 Estimated VRAM: ~{self.total_params * 4 / (1024**3):.1f}GB")
        logger.info(f"⚡ Performance optimizations: ENABLED")

    def train(self, max_iters, plot_step_size=100, stop_event=None, progress_callback=None):
        print(f"🚀 Starting OPTIMIZED training for {max_iters} iterations...")
        
        # Performance optimizations
        torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
        if device.type == "cuda":
            torch.cuda.empty_cache()  # Clear cache before training
            
        # Disable debugging for performance
        torch.autograd.set_detect_anomaly(False)
        
        # Pre-compile model for faster execution (PyTorch 2.0+)
        # Only compile if Triton is available to avoid dependency issues
        if hasattr(torch, 'compile') and device.type == "cuda":
            try:
                # Check if Triton is available before attempting compilation
                import triton
                print("🔥 Compiling model for maximum performance...")
                self.model = torch.compile(self.model, mode='max-autotune')
                print("✅ Model compilation successful!")
            except ImportError:
                print("⚠️  Triton not available, skipping model compilation (optional optimization)")
            except Exception as e:
                print(f"⚠️  Model compilation failed: {e}, continuing without compilation")
        
        # Update scheduler with the actual training iterations
        if max_iters != self.scheduler.max_steps:
            # Recreate scheduler with updated max_steps if needed
            warmup_ratio = self.scheduler.warmup_steps / self.scheduler.max_steps if self.scheduler.max_steps > 0 else 0.1
            new_warmup_steps = max(1, int(max_iters * warmup_ratio))
            
            self.scheduler = CosineWarmupScheduler(
                self.optimizer,
                warmup_steps=new_warmup_steps,
                max_steps=max_iters,
                max_lr=self.config.learning_rate,
                min_lr=self.config.learning_rate * 0.1
            )
            print(f"Scheduler updated: max_steps={max_iters}, warmup_steps={self.scheduler.warmup_steps}")
        
        best_loss = float('inf')
        running_loss = 0.0
        
        # Performance tracking
        callback_skip_frequency = max(1, plot_step_size // 10)  # Reduce callback frequency
        metric_cache = {}  # Cache computed metrics
        
        try:
            for it in range(max_iters):
                if stop_event and stop_event.is_set():
                    print("Training stopped by user.")
                    break

                # Progress reporting (reduced frequency for performance)
                if max_iters > 5000 and it % 2000 == 0:
                    print(f"🚀 Training progress: {it}/{max_iters} iterations ({100*it/max_iters:.1f}%)")

                # Update learning rate with cosine warmup scheduler (batched for performance)
                current_lr = self.scheduler.step()
                
                # Only store metrics occasionally to reduce memory overhead
                if it % callback_skip_frequency == 0:
                    self.learning_rates.append(current_lr)

                # Get batch data
                xb, yb = get_batch(self.config)
                
                # Zero gradients more efficiently
                for param in self.model.parameters():
                    param.grad = None  # More efficient than optimizer.zero_grad()
                
                # Forward and backward pass with optimizations
                if device.type == "cuda":
                    with torch.amp.autocast('cuda', dtype=torch.float16):  # Use FP16 for speed
                        _, loss = self.model(xb, yb)
                    
                    # Scale and backward (optimized)
                    self.scaler.scale(loss).backward()
                    
                    # Enhanced gradient clipping with reduced frequency
                    if it % 5 == 0:  # Only clip every 5 iterations for speed
                        self.scaler.unscale_(self.optimizer)
                        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                    else:
                        grad_norm = 0.0  # Skip expensive grad norm calculation
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    _, loss = self.model(xb, yb)
                    loss.backward()
                    
                    # Enhanced gradient clipping with reduced frequency
                    if it % 5 == 0:
                        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                    else:
                        grad_norm = 0.0
                    
                    self.optimizer.step()
                
                # Efficient loss tracking
                loss_item = loss.item()
                running_loss += loss_item
                
                # Only store detailed metrics occasionally
                if it % callback_skip_frequency == 0:
                    self.train_losses.append(loss_item)
                    # Calculate perplexity less frequently
                    current_perplexity = torch.exp(loss).item()
                    self.perplexities.append(current_perplexity)
                    metric_cache['perplexity'] = current_perplexity
                    metric_cache['grad_norm'] = grad_norm

                # Optimized callback and reporting (reduced frequency)
                if it % plot_step_size == 0 or it == max_iters - 1:
                    avg_loss = running_loss / plot_step_size if it > 0 else loss_item
                    
                    # Use cached perplexity if available, otherwise calculate
                    if 'perplexity' in metric_cache:
                        current_perplexity = metric_cache['perplexity']
                    else:
                        current_perplexity = torch.exp(loss).item()
                        
                    avg_perplexity = torch.exp(torch.tensor(avg_loss)).item()
                    running_loss = 0.0
                    
                    # Send callbacks with reduced frequency for better performance
                    if self.loss_callback and it % (callback_skip_frequency * 2) == 0:
                        self.loss_callback(loss_item)
                    
                    if self.metrics_callback and it % (callback_skip_frequency * 2) == 0:
                        cached_grad_norm = metric_cache.get('grad_norm', grad_norm)
                        metrics = {
                            'iteration': it,
                            'loss': loss_item,
                            'avg_loss': avg_loss,
                            'perplexity': current_perplexity,
                            'avg_perplexity': avg_perplexity,
                            'learning_rate': current_lr,
                            'grad_norm': cached_grad_norm.item() if hasattr(cached_grad_norm, 'item') else cached_grad_norm
                        }
                        self.metrics_callback(metrics)
                    
                    # Reduced console output frequency
                    if it % (plot_step_size * 2) == 0 or it == max_iters - 1:
                        print(f"Iter {it}: loss={loss_item:.4f}, perplexity={current_perplexity:.2f}, avg_loss={avg_loss:.4f}, avg_perplexity={avg_perplexity:.2f}, lr={current_lr:.6f}")
                    
                    # Save best model (less frequent saving for performance)
                    if loss_item < best_loss:
                        best_loss = loss_item
                        if getattr(self.config, 'auto_save_best', True) and it % (plot_step_size * 5) == 0:  # Save every 5 plot cycles
                            self.save_model("best_model.pth", iteration=it, train_loss=loss_item, verbose=False)
                            print(f"🎯 New best model saved: loss={best_loss:.4f}, perplexity={torch.exp(torch.tensor(best_loss)):.2f}")
                        elif it % (plot_step_size * 2) == 0:  # Log less frequently
                            print(f"🎯 New best loss: {best_loss:.4f}, perplexity={torch.exp(torch.tensor(best_loss)):.2f}")
                
                # Memory cleanup every 100 iterations
                if device.type == "cuda" and it % 100 == 0:
                    torch.cuda.empty_cache()
                
                if progress_callback and it % callback_skip_frequency == 0:
                    progress_callback(it + 1)
            
            print(f"Training complete. Best loss: {best_loss:.4f}")
            self.save_model("final_model.pth", iteration=max_iters, train_loss=best_loss, verbose=True)
            
        except Exception as e:
            print(f"Training error occurred at iteration {it if 'it' in locals() else 'unknown'}: {e}")
            import traceback
            traceback.print_exc()
            raise

    def evaluate(self, eval_iters=100):
        """
        Evaluate the model and calculate perplexity
        """
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for _ in range(eval_iters):
                xb, yb = get_batch(self.config)
                if device.type == "cuda":
                    with torch.amp.autocast('cuda'):
                        _, loss = self.model(xb, yb)
                else:
                    _, loss = self.model(xb, yb)
                total_loss += loss.item()
        
        avg_loss = total_loss / eval_iters
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        self.model.train()
        return avg_loss, perplexity

    def save_model(self, filepath="gpt_model.pth", iteration=None, train_loss=None, val_loss=None, verbose=True):
        """Enhanced model saving with comprehensive metadata and hyperparameters"""
        try:
            # Calculate model statistics
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            # Current learning rate
            current_lr = self.optimizer.param_groups[0]['lr'] if self.optimizer.param_groups else None
            
            # Get complete hyperparameters from config
            hyperparameters = {
                'learning_rate': self.config.learning_rate,
                'weight_decay': self.config.weight_decay,
                'n_embd': self.config.n_embd,
                'n_layer': self.config.n_layer,
                'n_head': self.config.n_head,
                'block_size': self.config.block_size,
                'batch_size': self.config.batch_size,
                'dropout': self.config.dropout,
                'attention_type': self.config.attention_type,
                'n_query_groups': self.config.n_query_groups,
                'use_moe': self.config.use_moe,
                'moe_num_experts': self.config.moe_num_experts,
                'moe_k': self.config.moe_k,
                'vram_profile': getattr(self.config, 'vram_profile', 'low'),
                'use_gradient_checkpointing': getattr(self.config, 'use_gradient_checkpointing', True),
                'warmup_iters': self.config.warmup_iters,
                'grad_clip': self.config.grad_clip,
                'max_iters': self.config.max_iters,
                'vocab_size': self.config.vocab_size
            }
            
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': getattr(self.scheduler, 'state_dict', lambda: {})(),
                'config': asdict(self.config),
                'hyperparameters': hyperparameters,  # Add dedicated hyperparameters section
                'training_metadata': {
                    'iteration': iteration,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'current_lr': current_lr,
                    'device': str(device),
                    'torch_version': torch.__version__,
                    'save_timestamp': time.time(),
                    'save_datetime': time.strftime('%Y-%m-%d %H:%M:%S')
                },
                'model_info': {
                    'total_params': total_params,
                    'trainable_params': trainable_params,
                    'model_size_mb': total_params * 4 / (1024**2),  # Assuming float32
                    'vocab_size': self.config.vocab_size,
                    'context_length': self.config.block_size,
                    'architecture_type': f"{self.config.attention_type}_attention",
                    'moe_enabled': self.config.use_moe
                },
                'training_history': {
                    'train_losses': self.train_losses[-100:] if len(self.train_losses) > 100 else self.train_losses,
                    'val_losses': self.val_losses[-100:] if len(self.val_losses) > 100 else self.val_losses,
                    'learning_rates': self.learning_rates[-100:] if len(self.learning_rates) > 100 else self.learning_rates
                }
            }
            torch.save(checkpoint, filepath)
            
            if verbose:
                print(f"Enhanced checkpoint saved to {filepath}")
                print(f"  - Model: {total_params:,} parameters ({total_params * 4 / (1024**2):.1f} MB)")
                print(f"  - Architecture: {self.config.attention_type} attention")
                if self.config.use_moe:
                    print(f"  - MoE: {self.config.moe_num_experts} experts, top-{self.config.moe_k}")
                print(f"  - Iteration: {iteration}, LR: {current_lr:.2e}")
                if train_loss is not None:
                    print(f"  - Train Loss: {train_loss:.4f}")
                if val_loss is not None:
                    print(f"  - Val Loss: {val_loss:.4f}, Perplexity: {torch.exp(torch.tensor(val_loss)):.2f}")
                print(f"  - Hyperparameters saved and can be restored")
            return True
        except Exception as e:
            print(f"Error saving enhanced checkpoint: {e}")
            return False

    def load_model(self, filepath="gpt_model.pth"):
        """Enhanced model loading with improved security and error handling"""
        if not os.path.exists(filepath):
            logger.error(f"Model file not found: {filepath}")
            return False
            
        # Validate file size and format
        try:
            file_size = os.path.getsize(filepath)
            if file_size == 0:
                logger.error(f"Model file is empty: {filepath}")
                return False
            logger.info(f"Loading model from {filepath} ({file_size / (1024*1024):.1f} MB)")
        except OSError as e:
            logger.error(f"Cannot access file {filepath}: {e}")
            return False
            
        try:
            # Prioritize secure loading methods
            checkpoint = None
            loading_method = "unknown"
            
            # Method 1: Try secure loading with weights_only=True (PyTorch 1.13+)
            try:
                checkpoint = torch.load(filepath, map_location=device, weights_only=True)
                loading_method = "weights_only=True (secure)"
                logger.info("Loaded using secure weights_only method")
            except (TypeError, RuntimeError) as e:
                logger.warning(f"Secure loading failed: {e}")
                
                # Method 2: For models with optimizer states, use controlled loading
                if "weights_only" in str(e) or "state_dict" in str(e):
                    try:
                        # Load with restricted globals for better security
                        allowed_classes = {
                            'collections.OrderedDict',
                            'torch._utils._rebuild_tensor_v2',
                            'torch.storage._load_from_bytes'
                        }
                        checkpoint = torch.load(
                            filepath, 
                            map_location=device, 
                            weights_only=False,
                            # Add security restrictions where possible
                        )
                        loading_method = "controlled loading"
                        logger.warning("Used controlled loading method - ensure file is trusted")
                    except Exception as fallback_error:
                        logger.error(f"Controlled loading failed: {fallback_error}")
                        raise
                else:
                    raise
            
            if checkpoint is None:
                logger.error("Failed to load model checkpoint")
                return False
            
            # Display checkpoint info
            if 'training_metadata' in checkpoint:
                metadata = checkpoint['training_metadata']
                print(f"Loading checkpoint from iteration {metadata.get('iteration', 'unknown')}")
                print(f"  - Training device: {metadata.get('device', 'unknown')}")
                print(f"  - PyTorch version: {metadata.get('torch_version', 'unknown')}")
                if 'save_datetime' in metadata:
                    print(f"  - Saved: {metadata['save_datetime']}")
                if 'train_loss' in metadata and metadata['train_loss']:
                    print(f"  - Last train loss: {metadata['train_loss']:.4f}")
                if 'val_loss' in metadata and metadata['val_loss']:
                    print(f"  - Last val loss: {metadata['val_loss']:.4f}")
            
            if 'model_info' in checkpoint:
                info = checkpoint['model_info']
                print(f"  - Model: {info.get('total_params', 'unknown'):,} parameters")
                print(f"  - Size: {info.get('model_size_mb', 'unknown')} MB")
                if 'architecture_type' in info:
                    print(f"  - Architecture: {info['architecture_type']}")
                if info.get('moe_enabled', False):
                    print(f"  - MoE enabled")
            
            # Load and apply hyperparameters if available
            hyperparameters_loaded = False
            if 'hyperparameters' in checkpoint:
                try:
                    hyperparams = checkpoint['hyperparameters']
                    print(f"Loading hyperparameters:")
                    for key, value in hyperparams.items():
                        if hasattr(self.config, key):
                            old_value = getattr(self.config, key)
                            setattr(self.config, key, value)
                            if old_value != value:
                                print(f"  - {key}: {old_value} -> {value}")
                    hyperparameters_loaded = True
                    print("Hyperparameters restored from checkpoint")
                except Exception as e:
                    print(f"Warning: Could not load hyperparameters: {e}")
            elif 'config' in checkpoint:
                try:
                    # Fallback to config section
                    config_dict = checkpoint['config']
                    for key, value in config_dict.items():
                        if hasattr(self.config, key):
                            setattr(self.config, key, value)
                    hyperparameters_loaded = True
                    print("Hyperparameters restored from config section")
                except Exception as e:
                    print(f"Warning: Could not load config: {e}")
            
            # If hyperparameters were loaded, recreate model with correct architecture
            if hyperparameters_loaded:
                try:
                    print("Recreating model with loaded hyperparameters...")
                    old_model = self.model
                    self.model = GPTLanguageModel(self.config).to(device)
                    
                    # Try to load state dict into new model
                    if 'model_state_dict' in checkpoint:
                        self.model.load_state_dict(checkpoint['model_state_dict'], strict=True)
                        print("Model weights loaded successfully with correct architecture")
                    else:
                        self.model.load_state_dict(checkpoint, strict=True)
                        print("Model weights loaded from direct state dict")
                    
                    # Update optimizer with new model parameters
                    self.optimizer = optim.AdamW(self.model.parameters(), 
                                               lr=self.config.learning_rate, 
                                               weight_decay=self.config.weight_decay)
                    
                    # Recreate scheduler
                    self.scheduler = CosineWarmupScheduler(
                        self.optimizer,
                        warmup_steps=self.config.warmup_iters,
                        max_steps=self.config.max_iters,
                        max_lr=self.config.learning_rate,
                        min_lr=self.config.learning_rate * 0.1
                    )
                    
                    model_loaded = True
                    
                except Exception as e:
                    print(f"Error recreating model with hyperparameters: {e}")
                    print("Falling back to standard loading...")
                    model_loaded = False
            else:
                model_loaded = False
            
            # Standard model loading if hyperparameter loading failed
            if not model_loaded:
                if 'model_state_dict' in checkpoint:
                    try:
                        # Try strict loading first
                        self.model.load_state_dict(checkpoint['model_state_dict'], strict=True)
                        print("Model loaded with strict=True")
                        model_loaded = True
                    except RuntimeError as e:
                        print(f"Strict loading failed: {e}")
                        try:
                            # Fallback to non-strict loading
                            missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                            if missing_keys:
                                print(f"Warning: Missing keys in checkpoint: {missing_keys}")
                            if unexpected_keys:
                                print(f"Warning: Unexpected keys in checkpoint: {unexpected_keys}")
                            print("Model loaded with strict=False")
                            model_loaded = True
                        except Exception as e2:
                            print(f"Non-strict loading also failed: {e2}")
                            raise e2
                else:
                    # Backward compatibility with old saves (direct state dict)
                    try:
                        self.model.load_state_dict(checkpoint, strict=True)
                        print("Model loaded from direct state dict with strict=True")
                        model_loaded = True
                    except RuntimeError as e:
                        print(f"Strict loading of direct state dict failed: {e}")
                        try:
                            missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint, strict=False)
                            if missing_keys:
                                print(f"Warning: Missing keys in checkpoint: {missing_keys}")
                            if unexpected_keys:
                                print(f"Warning: Unexpected keys in checkpoint: {unexpected_keys}")
                            print("Model loaded from direct state dict with strict=False")
                            model_loaded = True
                        except Exception as e2:
                            print(f"Non-strict loading of direct state dict also failed: {e2}")
                            raise e2
            
            # Load optimizer state if available (optional)
            if 'optimizer_state_dict' in checkpoint and not hyperparameters_loaded:
                try:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    print("Optimizer state loaded successfully")
                except Exception as e:
                    print(f"Warning: Could not load optimizer state: {e}")
            
            # Print model info if available
            if 'model_info' in checkpoint:
                info = checkpoint['model_info']
                print(f"Loaded model with {info['total_params']:,} parameters")
            
            if model_loaded:
                success_msg = f"Model loaded successfully from {filepath}"
                if hyperparameters_loaded:
                    success_msg += " (with hyperparameters)"
                print(success_msg)
                return True
            else:
                print(f"Failed to load model from {filepath}")
                return False
                
        except Exception as e:
            print(f"Error loading model: {e}")
            print(f"File: {filepath}")
            
            # Additional helpful error information for common issues
            if "WeightsUnpickler" in str(e):
                print("Suggestion: This model was saved with a different PyTorch version.")
                print("Try using a compatible PyTorch version or re-saving the model.")
            elif "torch_version" in str(e):
                print("Suggestion: PyTorch version mismatch detected.")
                print("Consider updating PyTorch or loading with weights_only=False.")
            
            return False

def save_hyperparameters_to_file(config: Config, filepath="hyperparameters.json"):
    """Save hyperparameters to a standalone JSON file"""
    try:
        hyperparams = {
            'learning_rate': config.learning_rate,
            'weight_decay': config.weight_decay,
            'n_embd': config.n_embd,
            'n_layer': config.n_layer,
            'n_head': config.n_head,
            'block_size': config.block_size,
            'batch_size': config.batch_size,
            'dropout': config.dropout,
            'attention_type': config.attention_type,
            'n_query_groups': config.n_query_groups,
            'use_moe': config.use_moe,
            'moe_num_experts': config.moe_num_experts,
            'moe_k': config.moe_k,
            'vram_profile': getattr(config, 'vram_profile', 'low'),
            'use_gradient_checkpointing': getattr(config, 'use_gradient_checkpointing', True),
            'warmup_iters': config.warmup_iters,
            'grad_clip': config.grad_clip,
            'max_iters': config.max_iters,
            'vocab_size': config.vocab_size,
            'save_timestamp': time.time(),
            'save_datetime': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open(filepath, 'w') as f:
            json.dump(hyperparams, f, indent=4)
        
        print(f"Hyperparameters saved to {filepath}")
        return True
    except Exception as e:
        print(f"Error saving hyperparameters: {e}")
        return False

def load_hyperparameters_from_file(filepath="hyperparameters.json"):
    """Load hyperparameters from a JSON file and return updated config"""
    try:
        if not os.path.exists(filepath):
            print(f"Hyperparameters file {filepath} not found")
            return None
            
        with open(filepath, 'r') as f:
            hyperparams = json.load(f)
        
        # Create new config with loaded hyperparameters
        config = Config()
        for key, value in hyperparams.items():
            if hasattr(config, key) and key not in ['save_timestamp', 'save_datetime']:
                setattr(config, key, value)
        
        print(f"Hyperparameters loaded from {filepath}")
        if 'save_datetime' in hyperparams:
            print(f"  - Originally saved: {hyperparams['save_datetime']}")
        
        return config
    except Exception as e:
        print(f"Error loading hyperparameters: {e}")
        return None

# Performance optimization utilities
def optimize_for_training_speed():
    """Apply global optimizations for maximum training speed"""
    print("🚀 Applying global training speed optimizations...")
    
    # PyTorch optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.allow_tf32 = True if device.type == "cuda" else False
    torch.backends.cuda.matmul.allow_tf32 = True if device.type == "cuda" else False
    
    # Memory optimizations
    if device.type == "cuda":
        torch.cuda.empty_cache()
        # Set memory allocation strategy for better performance
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
    
    # Disable debugging features for speed
    torch.autograd.set_detect_anomaly(False)
    torch.autograd.profiler.profile(enabled=False)
    
    print("✅ Global optimizations applied!")

def clear_cache_and_optimize():
    """Clear caches and optimize memory for training"""
    global _batch_cache
    
    # Clear batch cache
    _batch_cache.clear()
    
    # Clear CUDA cache if available
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Force garbage collection
    gc.collect()
    
    print("🧹 Memory caches cleared and optimized!")

def get_training_speed_config() -> Config:
    """Get a configuration optimized specifically for training speed"""
    config = Config()
    
    # Ultra-fast training configuration
    config.model_type = "compact"
    config.n_layer = 4  # Minimal layers
    config.n_head = 4   # Minimal heads  
    config.n_embd = 256 # Small embedding
    config.batch_size = 16  # Larger batch for GPU efficiency
    config.block_size = 256  # Smaller context for speed
    config.dropout = 0.05    # Minimal dropout
    config.learning_rate = 2e-3  # Higher LR for faster convergence
    config.warmup_iters = 25     # Minimal warmup
    config.grad_clip = 10.0      # Higher clip for stability
    config.use_moe = False       # Disable MoE for speed
    config.attention_type = "multihead"  # Fastest attention
    config.use_gradient_checkpointing = False  # Disable for speed
    
    print("⚡ Ultra-speed training configuration created!")
    print(f"   - Model: {config.n_layer} layers, {config.n_head} heads, {config.n_embd}d")
    print(f"   - Training: batch={config.batch_size}, lr={config.learning_rate}")
    print(f"   - Context: {config.block_size} tokens")
    
    return config

def run_hyperparameter_optimization(config: Config, n_trials: int, steps_per_trial: int, param_config=None, stop_event=None, sampler_config=None):
    print("=" * 60)
    print("HYPERPARAMETER OPTIMIZATION STARTING")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  Trials: {n_trials}")
    print(f"  Steps per trial: {steps_per_trial}")
    print(f"  Architecture: {config.attention_type}")
    if config.use_moe:
        print(f"  MoE: {config.moe_num_experts} experts, top-{config.moe_k}")
    print(f"  Device: {device}")
    
    if param_config:
        print(f"Parameter Optimization:")
        params_to_optimize = []
        if param_config.get('optimize_lr', True): params_to_optimize.append("Learning Rate")
        if param_config.get('optimize_wd', True): params_to_optimize.append("Weight Decay")
        if param_config.get('optimize_embd', True): params_to_optimize.append("Embedding Dim")
        if param_config.get('optimize_layers', True): params_to_optimize.append("Layers")
        if param_config.get('optimize_heads', True): params_to_optimize.append("Heads")
        if param_config.get('optimize_dropout', True): params_to_optimize.append("Dropout")
        if param_config.get('optimize_batch', True): params_to_optimize.append("Batch Size")
        print(f"  Optimizing: {', '.join(params_to_optimize)}")
    
    print("=" * 60)

    # Create sampler based on configuration
    sampler = None
    if sampler_config:
        sampler_type = sampler_config.get('type', 'tpe')
        if sampler_type == 'tpe':
            sampler = optuna.samplers.TPESampler(
                n_startup_trials=sampler_config.get('n_startup_trials', 10),
                n_ei_candidates=sampler_config.get('n_ei_candidates', 24),
                seed=sampler_config.get('seed', None),
                multivariate=sampler_config.get('multivariate', True)
            )
        elif sampler_type == 'random':
            sampler = optuna.samplers.RandomSampler(seed=sampler_config.get('seed', None))
        elif sampler_type == 'grid':
            sampler = optuna.samplers.GridSampler()
        elif sampler_type == 'cmaes':
            sampler = optuna.samplers.CmaEsSampler(seed=sampler_config.get('seed', None))
    else:
        sampler = optuna.samplers.TPESampler(multivariate=True)

    # Create pruner based on configuration
    pruner_type = sampler_config.get('pruner', 'median') if sampler_config else 'median'
    if pruner_type == 'median':
        median_startup = sampler_config.get('median_startup_trials', 5) if sampler_config else 5
        median_warmup = sampler_config.get('median_warmup_steps', 10) if sampler_config else 10
        median_min_trials = sampler_config.get('median_min_trials', 5) if sampler_config else 5
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=median_startup, 
            n_warmup_steps=median_warmup, 
            n_min_trials=median_min_trials
        )
    elif pruner_type == 'percentile':
        percentile = sampler_config.get('percentile', 25.0) if sampler_config else 25.0
        percentile_startup = sampler_config.get('percentile_startup_trials', 5) if sampler_config else 5
        percentile_warmup = sampler_config.get('percentile_warmup_steps', 0) if sampler_config else 0
        pruner = optuna.pruners.PercentilePruner(
            percentile=percentile,
            n_startup_trials=percentile_startup,
            n_warmup_steps=percentile_warmup
        )
    elif pruner_type == 'successive_halving':
        min_resource = sampler_config.get('halving_min_resource', 1) if sampler_config else 1
        reduction_factor = sampler_config.get('halving_reduction_factor', 4) if sampler_config else 4
        min_early_stopping = sampler_config.get('halving_min_early_stopping_rate', 5) if sampler_config else 5
        pruner = optuna.pruners.SuccessiveHalvingPruner(
            min_resource=min_resource,
            reduction_factor=reduction_factor,
            min_early_stopping_rate=min_early_stopping
        )
    else:
        pruner = optuna.pruners.NopPruner()

    study = optuna.create_study(
        direction="minimize",
        pruner=pruner,
        sampler=sampler,
        storage="sqlite:///optuna_study.db",
        study_name="gpt-optimization-v2",
        load_if_exists=True
    )
    
    print(f"Study created with {type(sampler).__name__ if sampler else 'default'} sampler and {type(pruner).__name__} pruner")
    
    # Log pruner parameters
    if pruner_type == 'median':
        print(f"  Median Pruner settings: startup_trials={median_startup}, warmup_steps={median_warmup}, min_trials={median_min_trials}")
    elif pruner_type == 'percentile':
        print(f"  Percentile Pruner settings: percentile={percentile}%, startup_trials={percentile_startup}, warmup_steps={percentile_warmup}")
    elif pruner_type == 'successive_halving':
        print(f"  Successive Halving Pruner settings: min_resource={min_resource}, reduction_factor={reduction_factor}, min_early_stopping={min_early_stopping}")
    elif pruner_type == 'none':
        print(f"  No pruning enabled")
    
    print(f"Starting optimization...")
    
    def verbose_callback(study, trial):
        if stop_event and stop_event.is_set():
            print(f"\n--- Optimization stopped by user ---")
            return
            
        print(f"\n--- Trial #{trial.number} Summary ---")
        if trial.state == optuna.trial.TrialState.COMPLETE:
            print(f"Status: COMPLETED")
            print(f"Final Value: {trial.value:.6f}")
        elif trial.state == optuna.trial.TrialState.PRUNED:
            print(f"Status: PRUNED")
        else:
            print(f"Status: {trial.state}")
        
        if len(study.trials) > 0:
            best_trial = study.best_trial
            print(f"Current Best Trial: #{best_trial.number} (Value: {best_trial.value:.6f})")
        print("-" * 40)
    
    # Define the objective function for Optuna
    def objective(trial, steps_per_trial, config, param_config, stop_event):
        # Sample hyperparameters if param_config is provided
        sampled_config = Config()
        for key in vars(config):
            setattr(sampled_config, key, getattr(config, key))
        if param_config:
            if param_config.get('optimize_lr', True):
                sampled_config.learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
            if param_config.get('optimize_wd', True):
                sampled_config.weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
            if param_config.get('optimize_embd', True):
                sampled_config.n_embd = trial.suggest_categorical("n_embd", [256, 512, 1024, 2048])
            if param_config.get('optimize_layers', True):
                sampled_config.n_layer = trial.suggest_int("n_layer", 2, 16)
            if param_config.get('optimize_heads', True):
                sampled_config.n_head = trial.suggest_categorical("n_head", [2, 4, 8, 16])
            if param_config.get('optimize_dropout', True):
                sampled_config.dropout = trial.suggest_float("dropout", 0.01, 0.2)
            if param_config.get('optimize_batch', True):
                sampled_config.batch_size = trial.suggest_categorical("batch_size", [2, 4, 8, 16])
        sampled_config.vocab_size = config.vocab_size

        trainer = Trainer(sampled_config)
        best_loss = float('inf')
        loss_history = []
        for it in range(steps_per_trial):
            xb, yb = get_batch(sampled_config)
            trainer.optimizer.zero_grad()
            _, loss = trainer.model(xb, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), sampled_config.grad_clip)
            trainer.optimizer.step()
            current_loss = loss.item()
            loss_history.append(current_loss)
            if current_loss < best_loss:
                best_loss = current_loss
            # Progress reporting every 10 steps or at key milestones
            if it % 10 == 0 or it == steps_per_trial - 1:
                avg_loss = sum(loss_history[-10:]) / min(10, len(loss_history))
                print(f"  Step {it:3d}/{steps_per_trial}: Loss = {current_loss:.6f}, Avg(10) = {avg_loss:.6f}, Best = {best_loss:.6f}")

            trial.report(current_loss, it)
            if trial.should_prune():
                print(f"Trial #{trial.number} pruned at step {it} (loss: {current_loss:.6f})")
                raise optuna.exceptions.TrialPruned()

        final_loss = loss.item()
        print(f"Trial #{trial.number} completed - Final loss: {final_loss:.6f}, Best loss: {best_loss:.6f}")

        # Explicitly clean up memory
        del trainer, loss, xb, yb
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        return final_loss

    try:
        study.optimize(
            lambda trial: objective(trial, steps_per_trial, config, param_config, stop_event), 
            n_trials=n_trials, 
            show_progress_bar=True,
            callbacks=[verbose_callback]

        )
    except KeyboardInterrupt:
        print("\nOptimization interrupted by user")
    
    print("\n" + "=" * 60)
    if stop_event and stop_event.is_set():
        print("OPTIMIZATION STOPPED BY USER")
    else:
        print("OPTIMIZATION COMPLETED")
    print("=" * 60)
    
    if len(study.trials) > 0:
        print("Best hyperparameters:", study.best_params)
        print(f"Best value: {study.best_value:.6f}")
        
        try:
            with open("best_hyperparams.json", "w") as f:
                json.dump(study.best_params, f, indent=4)
            print("Best hyperparameters saved to best_hyperparams.json")
        except Exception as e:
            print(f"Error saving best_hyperparams.json: {e}")
    else:
        print("No completed trials found.")

    # Return new config with best params
    new_config = load_config()
    new_config.vocab_size = config.vocab_size
    return new_config

# Auto-initialize performance optimizations when module is imported
if __name__ != "__main__":
    optimize_for_training_speed()
    
print("🎯 DGS-GPT Performance Module Ready!")
print("💡 Use get_training_speed_config() for maximum speed")
print("🧹 Use clear_cache_and_optimize() to free memory")
