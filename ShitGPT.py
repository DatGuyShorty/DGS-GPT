import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
# from live_training_plot import LiveLossPlot # Removed
import optuna
import sys
import json
import os
from dataclasses import dataclass, asdict
import gc

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

@dataclass
class Config:
    # Model params
    block_size: int = 512 # context window size
    vocab_size: int = 50304 # set in main
    n_layer: int = 16 # number of transformer blocks
    n_head: int = 16 # number of attention heads
    n_embd: int = 1024 # embedding dimension
    dropout: float = 0.01 # 0.01% dropout
    use_moe: bool = True # Mixture of Experts
    moe_num_experts: int = 4  # number of experts
    moe_k: int = 1  # top-k experts
    attention_type: str = "grouped_query"   # "multihead" or "grouped_query"
    n_query_groups: int = 1  # number of query groups

    # Training params
    learning_rate: float = 5e-4 # learning rate
    weight_decay: float = 1e-5 # weight decay
    batch_size: int = 2 # batch size
    max_iters: int = 1000 # maximum training iterations
    grad_clip: float = 1.0 # gradient clipping threshold
    warmup_iters: int = 100 # learning rate warmup iterations

def load_config(filepath="best_hyperparams.json"):
    default_config = Config()
    if os.path.exists(filepath):
        try:
            with open(filepath, "r") as f:
                best_params = json.load(f)
            if isinstance(best_params, dict) and best_params:
                print(f"Loaded best hyperparameters from {filepath}")
                # Update default_config with loaded params
                for key, value in best_params.items():
                    if hasattr(default_config, key):
                        setattr(default_config, key, value)
                return default_config
            else:
                print(f"{filepath} is empty or invalid, using defaults.")
        except Exception as e:
            print(f"Error loading {filepath}: {e}, using defaults.")
    else:
        print("Using default hyperparameters")
    return default_config

# Load and encode data (char-level example)
try:
    with open("vocab.txt", "r", encoding="utf-8") as f:
        txt = f.read()
except Exception as e:
    print(f"Error loading vocab.txt: {e}")
    sys.exit(1)

if not txt:
    print("vocab.txt is empty. Please generate it first.")
    sys.exit(1)

chars = sorted(set(txt))
vocab_size = len(chars)
stoi = {ch:i for i, ch in enumerate(chars)}
ios = {i:ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda ids: ''.join([ios[i] for i in ids])
data = torch.tensor(encode(txt), dtype=torch.long)

def get_batch(config):
    ix = torch.randint(len(data) - config.block_size, (config.batch_size,))
    x = torch.stack([data[i:i+config.block_size] for i in ix])
    y = torch.stack([data[i+1:i+config.block_size+1] for i in ix])
    return x.to(device), y.to(device)

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

# Block with optional MoE and attention type selection
class Block(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        if config.attention_type == "grouped_query":
            assert config.n_query_groups is not None, "n_query_groups must be set for grouped_query attention"
            self.sa = GroupedQueryAttention(config.n_embd, config.n_head, config.n_query_groups, config.dropout, config.block_size)
        else:
            self.sa = MultiHeadAttention(config.n_embd, config.n_head, config.dropout, config.block_size)
        self.ff = FeedForward(config.n_embd, config.dropout)
        self.use_moe = config.use_moe
        if self.use_moe:
            self.moe = MoE(config.n_embd, 4*config.n_embd, num_experts=config.moe_num_experts, k=config.moe_k, dropout=config.dropout)
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.ln3 = nn.LayerNorm(config.n_embd) if self.use_moe else None

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        if self.use_moe:
            x = x + self.moe(self.ln3(x))
        return x

# GPTLanguageModel updated to pass attention_type and n_query_groups
class GPTLanguageModel(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)
        self.apply(self._init_weights)

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
            beam_end = (i + 1) * beam_size
            # Select the best beam for this batch item
            local_beam_scores = beam_scores[beam_start:beam_end]
            local_beams = beams[beam_start:beam_end]
            best_beam_idx = torch.argmax(local_beam_scores)
            best_sequences.append(local_beams[best_beam_idx])
        
        return torch.stack(best_sequences)

class Trainer:
    def __init__(self, config: Config, loss_callback=None):
        self.config = config
        self.model = GPTLanguageModel(config).to(device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        self.scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
        self.loss_callback = loss_callback
        
        # Add learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=config.max_iters,
            eta_min=config.learning_rate * 0.1  # Min LR is 10% of initial
        )

    def train(self, max_iters, plot_step_size=100, stop_event=None, progress_callback=None):
        print(f"Starting training for {max_iters} iterations...")
        best_loss = float('inf')
        running_loss = 0.0
        
        for it in range(max_iters):
            if stop_event and stop_event.is_set():
                print("Training stopped by user.")
                break

            # Learning rate warmup
            if it < self.config.warmup_iters:
                lr = self.config.learning_rate * (it + 1) / self.config.warmup_iters
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr

            xb, yb = get_batch(self.config)
            self.optimizer.zero_grad()
            
            if device.type == "cuda":
                with torch.amp.autocast('cuda'):
                    _, loss = self.model(xb, yb)
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                _, loss = self.model(xb, yb)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                
                self.optimizer.step()

            # Update learning rate scheduler (after warmup)
            if it >= self.config.warmup_iters:
                self.scheduler.step()
            
            current_lr = self.optimizer.param_groups[0]['lr']
            running_loss += loss.item()

            if it % plot_step_size == 0 or it == max_iters - 1:
                avg_loss = running_loss / (plot_step_size if it > 0 else 1)
                running_loss = 0.0
                
                if self.loss_callback:
                    self.loss_callback(loss.item())
                print(f"Iter {it}: loss={loss.item():.4f}, avg_loss={avg_loss:.4f}, lr={current_lr:.6f}")
                
                # Save best model
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    self.save_model("best_model.pth")
                    print(f"New best model saved with loss: {best_loss:.4f}")
            
            if progress_callback:
                progress_callback(it + 1)
        
        print(f"Training complete. Best loss: {best_loss:.4f}")
        self.save_model("final_model.pth")

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

    def save_model(self, filepath="gpt_model.pth"):
        """Enhanced model saving with metadata"""
        try:
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'config': asdict(self.config),
                'model_info': {
                    'total_params': sum(p.numel() for p in self.model.parameters()),
                    'trainable_params': sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                }
            }
            torch.save(checkpoint, filepath)
            print(f"Model saved to {filepath}")
            return True
        except Exception as e:
            print(f"Error saving model: {e}")
            return False

    def load_model(self, filepath="gpt_model.pth"):
        """Enhanced model loading"""
        try:
            checkpoint = torch.load(filepath, map_location=device)
            
            # Load model state
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            else:
                # Backward compatibility with old saves
                self.model.load_state_dict(checkpoint, strict=False)
            
            # Load optimizer state if available
            if 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Print model info if available
            if 'model_info' in checkpoint:
                info = checkpoint['model_info']
                print(f"Loaded model with {info['total_params']:,} parameters")
            
            print(f"Model loaded from {filepath}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

# Hyperparameter optimization using Optuna
def objective(trial, steps_per_trial, base_config, param_config=None, stop_event=None):
    # Check if optimization should stop
    if stop_event and stop_event.is_set():
        raise optuna.exceptions.TrialPruned()
    
    print(f"\n--- Starting Trial #{trial.number} ---")
    
    # Suggest hyperparameters based on parameter configuration
    trial_config = Config()
    
    # Copy base configuration
    for attr in dir(base_config):
        if not attr.startswith('_') and hasattr(trial_config, attr):
            setattr(trial_config, attr, getattr(base_config, attr))
    
    if param_config:
        # Learning Rate
        if param_config.get('optimize_lr', True):
            lr_min = param_config.get('lr_min', 1e-5)
            lr_max = param_config.get('lr_max', 1e-2)
            trial_config.learning_rate = trial.suggest_float('learning_rate', lr_min, lr_max, log=True)
        else:
            trial_config.learning_rate = param_config.get('lr_value', base_config.learning_rate)
        
        # Weight Decay
        if param_config.get('optimize_wd', True):
            wd_min = param_config.get('wd_min', 1e-6)
            wd_max = param_config.get('wd_max', 1e-2)
            trial_config.weight_decay = trial.suggest_float('weight_decay', wd_min, wd_max, log=True)
        else:
            trial_config.weight_decay = param_config.get('wd_value', base_config.weight_decay)
        
        # Embedding Dimension
        if param_config.get('optimize_embd', True):
            embd_choices = param_config.get('embd_choices', [256, 512, 1024])
            trial_config.n_embd = trial.suggest_categorical('n_embd', embd_choices)
        else:
            trial_config.n_embd = param_config.get('embd_value', base_config.n_embd)
        
        # Number of Layers
        if param_config.get('optimize_layers', True):
            layer_min = param_config.get('layer_min', 4)
            layer_max = param_config.get('layer_max', 16)
            # Constrain based on embedding size if being optimized
            if param_config.get('optimize_embd', True):
                if trial_config.n_embd >= 1024:
                    layer_max = min(layer_max, 8)
                elif trial_config.n_embd >= 512:
                    layer_max = min(layer_max, 12)
            trial_config.n_layer = trial.suggest_int('n_layer', layer_min, layer_max)
        else:
            trial_config.n_layer = param_config.get('layer_value', base_config.n_layer)
        
        # Number of Heads
        if param_config.get('optimize_heads', True):
            head_choices = param_config.get('head_choices', [4, 8, 16])
            trial_config.n_head = trial.suggest_categorical('n_head', head_choices)
        else:
            trial_config.n_head = param_config.get('head_value', base_config.n_head)
        
        # Dropout
        if param_config.get('optimize_dropout', True):
            dropout_min = param_config.get('dropout_min', 0.0)
            dropout_max = param_config.get('dropout_max', 0.5)
            trial_config.dropout = trial.suggest_float('dropout', dropout_min, dropout_max)
        else:
            trial_config.dropout = param_config.get('dropout_value', base_config.dropout)
        
        # Batch Size
        if param_config.get('optimize_batch', True):
            batch_choices = param_config.get('batch_choices', [8, 16, 24, 32])
            trial_config.batch_size = trial.suggest_categorical('batch_size', batch_choices)
        else:
            trial_config.batch_size = param_config.get('batch_value', base_config.batch_size)
    else:
        # Fallback to original behavior
        trial_config.learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        trial_config.weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
        trial_config.n_embd = trial.suggest_categorical('n_embd', [256, 512, 1024])
        if trial_config.n_embd >= 1024:
            trial_config.n_layer = trial.suggest_int('n_layer', 4, 8)
        elif trial_config.n_embd >= 512:
            trial_config.n_layer = trial.suggest_int('n_layer', 6, 12)
        else:
            trial_config.n_layer = trial.suggest_int('n_layer', 8, 16)
        trial_config.n_head = trial.suggest_categorical('n_head', [4, 8, 16])
        trial_config.dropout = trial.suggest_float('dropout', 0.05, 0.3)
    
    # Print trial configuration
    print(f"Trial Parameters:")
    print(f"  Learning Rate: {trial_config.learning_rate:.6f}")
    print(f"  Weight Decay: {trial_config.weight_decay:.6f}")
    print(f"  Embedding Dim: {trial_config.n_embd}")
    print(f"  Layers: {trial_config.n_layer}")
    print(f"  Heads: {trial_config.n_head}")
    print(f"  Dropout: {trial_config.dropout:.3f}")
    print(f"  Batch Size: {trial_config.batch_size}")
    print(f"  Architecture: {trial_config.attention_type}")
    if trial_config.use_moe:
        print(f"  MoE: {trial_config.moe_num_experts} experts, top-{trial_config.moe_k}")
    if trial_config.attention_type == 'grouped_query':
        print(f"  Query Groups: {trial_config.n_query_groups}")
    
    # Validate n_query_groups for grouped query attention
    if trial_config.attention_type == 'grouped_query':
        if trial_config.n_head % trial_config.n_query_groups != 0:
            # Adjust n_query_groups to be compatible with n_head
            possible_groups = [g for g in [1, 2, 4, 8] if trial_config.n_head % g == 0]
            if possible_groups:
                trial_config.n_query_groups = possible_groups[0]  # Use the first valid option
            else:
                trial_config.n_query_groups = 1  # Fallback to 1

    trial_config.vocab_size = len(chars)

    # Calculate model parameters
    model = GPTLanguageModel(trial_config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model created: {total_params:,} total params, {trainable_params:,} trainable")
    
    optimizer = optim.AdamW(model.parameters(), lr=trial_config.learning_rate, weight_decay=trial_config.weight_decay)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    # Training loop (short for optimization)
    loss = torch.tensor(float('inf')) # Default loss
    best_loss = float('inf')
    loss_history = []
    
    print(f"Starting training for {steps_per_trial} steps...")
    for it in range(steps_per_trial):
        # Check for stop event
        if stop_event and stop_event.is_set():
            print(f"Trial #{trial.number} stopped by user at step {it}")
            break
            
        xb, yb = get_batch(trial_config)
        optimizer.zero_grad()
        try:
            if device.type == "cuda":
                with torch.amp.autocast('cuda'):
                    _, loss = model(xb, yb)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                _, loss = model(xb, yb)
                loss.backward()
                optimizer.step()
        except torch.cuda.OutOfMemoryError:
            print("Trial stopped: CUDA out of memory.")
            # Clean up before pruning
            del model, optimizer, scaler, xb, yb
            if device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
            return float("inf") # Prune trial

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
    del model, optimizer, scaler, loss, xb, yb
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    return final_loss

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
