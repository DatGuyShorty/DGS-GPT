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
    block_size: int = 256
    vocab_size: int = 50304 # set in main
    n_layer: int = 16
    n_head: int = 16
    n_embd: int = 1024
    dropout: float = 0.01
    use_moe: bool = False
    moe_num_experts: int = 4
    moe_k: int = 1
    attention_type: str = "multihead"
    n_query_groups: int = 1

    # Training params
    learning_rate: float = 5e-4
    weight_decay: float = 1e-5
    batch_size: int = 24
    max_iters: int = 1000

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

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)
        return idx

class Trainer:
    def __init__(self, config: Config, loss_callback=None):
        self.config = config
        self.model = GPTLanguageModel(config).to(device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        self.scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
        self.loss_callback = loss_callback

    def train(self, max_iters, plot_step_size=100, stop_event=None, progress_callback=None):
        print(f"Starting training for {max_iters} iterations...")
        for it in range(max_iters):
            if stop_event and stop_event.is_set():
                print("Training stopped by user.")
                break

            xb, yb = get_batch(self.config)
            self.optimizer.zero_grad()
            if device.type == "cuda":
                with torch.amp.autocast('cuda'):
                    _, loss = self.model(xb, yb)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                _, loss = self.model(xb, yb)
                loss.backward()
                self.optimizer.step()

            if it % plot_step_size == 0 or it == max_iters - 1:
                if self.loss_callback:
                    self.loss_callback(loss.item())
                print(f"Iter {it}: loss={loss.item():.4f}")
            
            if progress_callback:
                progress_callback(it + 1)
        print("Training complete.")

    def save_model(self, filepath="gpt_model.pth"):
        try:
            torch.save(self.model.state_dict(), filepath)
            print(f"Model saved to {filepath}")
            return True
        except Exception as e:
            print(f"Error saving model: {e}")
            return False

# Hyperparameter optimization using Optuna
def objective(trial, steps_per_trial, base_config):
    # Suggest hyperparameters
    trial_config = Config()
    trial_config.learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    trial_config.weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
    
    # Constrain layer and embedding size to prevent VRAM overflow
    trial_config.n_embd = trial.suggest_categorical('n_embd', [256, 512, 1024])
    if trial_config.n_embd >= 1024:
        trial_config.n_layer = trial.suggest_int('n_layer', 4, 8)
    elif trial_config.n_embd >= 512:
        trial_config.n_layer = trial.suggest_int('n_layer', 6, 12)
    else: # 256
        trial_config.n_layer = trial.suggest_int('n_layer', 8, 16)

    trial_config.n_head = trial.suggest_categorical('n_head', [4, 8, 16])
    trial_config.dropout = trial.suggest_float('dropout', 0.05, 0.3)
    
    # Use fixed architectural choices from base_config instead of optimizing them
    trial_config.use_moe = base_config.use_moe
    trial_config.moe_num_experts = base_config.moe_num_experts
    trial_config.moe_k = base_config.moe_k
    trial_config.attention_type = base_config.attention_type
    trial_config.n_query_groups = base_config.n_query_groups
    
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

    model = GPTLanguageModel(trial_config).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=trial_config.learning_rate, weight_decay=trial_config.weight_decay)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    # Training loop (short for optimization)
    loss = torch.tensor(float('inf')) # Default loss
    for it in range(steps_per_trial):
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

        trial.report(loss.item(), it)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    final_loss = loss.item()
    # Explicitly clean up memory
    del model, optimizer, scaler, loss, xb, yb
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    return final_loss

def run_hyperparameter_optimization(config: Config, n_trials: int, steps_per_trial: int):
    print("Running hyperparameter optimization...")

    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(),
        storage="sqlite:///optuna_study.db",
        study_name="gpt-optimization-v2",
        load_if_exists=True
    )
    study.optimize(lambda trial: objective(trial, steps_per_trial, config), n_trials=n_trials, show_progress_bar=True)
    
    print("Best hyperparameters:", study.best_params)
    try:
        with open("best_hyperparams.json", "w") as f:
            json.dump(study.best_params, f, indent=4)
        print("Best hyperparameters saved to best_hyperparams.json")
    except Exception as e:
        print(f"Error saving best_hyperparams.json: {e}")

    # Return new config with best params
    new_config = load_config()
    new_config.vocab_size = config.vocab_size
    return new_config
