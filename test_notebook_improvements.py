#!/usr/bin/env python3
"""
Test script for notebook-inspired improvements in DGS-GPT
Tests: RoPE, Sliding Window Attention, Sparse MoE, Cosine Scheduler, Enhanced Checkpointing
"""

import torch
import torch.nn as nn
import time
import os
from ShitGPT import *

def test_rope():
    """Test Rotary Positional Embedding"""
    print("Testing Rotary Positional Embedding (RoPE)...")
    try:
        rope = RotaryEmbedding(dim=64, max_position_embeddings=512)
        x = torch.randn(2, 32, 64)  # (batch, seq_len, dim)
        
        # Test forward pass
        x_rope = rope(x, seq_len=32)
        
        assert x_rope.shape == x.shape, f"RoPE output shape mismatch: {x_rope.shape} vs {x.shape}"
        assert not torch.allclose(x, x_rope), "RoPE should modify the input"
        
        print("✓ RoPE test passed!")
        return True
    except Exception as e:
        print(f"✗ RoPE test failed: {e}")
        return False

def test_sliding_window_attention():
    """Test Sliding Window Attention"""
    print("Testing Sliding Window Attention...")
    try:
        n_embd, n_head, window_size = 128, 4, 16
        sliding_attn = SlidingWindowAttention(n_embd, n_head, window_size=window_size)
        x = torch.randn(2, 32, n_embd)  # (batch, seq_len, n_embd)
        
        # Test forward pass
        output = sliding_attn(x)
        
        assert output.shape == x.shape, f"Sliding attention output shape mismatch: {output.shape} vs {x.shape}"
        assert not torch.allclose(x, output), "Sliding attention should modify the input"
        
        print("✓ Sliding Window Attention test passed!")
        return True
    except Exception as e:
        print(f"✗ Sliding Window Attention test failed: {e}")
        return False

def test_sparse_moe():
    """Test Sparse Mixture of Experts"""
    print("Testing Sparse Mixture of Experts...")
    try:
        n_embd = 128
        sparse_moe = SparseExpertLayer(n_embd, num_experts=4, num_active=2)
        x = torch.randn(2, 16, n_embd)  # (batch, seq_len, n_embd)
        
        # Test forward pass
        output = sparse_moe(x)
        
        assert output.shape == x.shape, f"Sparse MoE output shape mismatch: {output.shape} vs {x.shape}"
        
        print("✓ Sparse MoE test passed!")
        return True
    except Exception as e:
        print(f"✗ Sparse MoE test failed: {e}")
        return False

def test_cosine_warmup_scheduler():
    """Test Cosine Warmup Scheduler"""
    print("Testing Cosine Warmup Scheduler...")
    try:
        # Create dummy model and optimizer
        model = nn.Linear(10, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        
        # Create scheduler
        scheduler = CosineWarmupScheduler(
            optimizer, 
            warmup_steps=10, 
            max_steps=100, 
            max_lr=1e-3, 
            min_lr=1e-5
        )
        
        lrs = []
        for step in range(50):
            lr = scheduler.step()
            lrs.append(lr)
        
        # Check warmup phase
        assert lrs[0] < lrs[9], "Learning rate should increase during warmup"
        
        # Check cosine decay
        assert lrs[20] > lrs[40], "Learning rate should decrease after warmup"
        
        print("✓ Cosine Warmup Scheduler test passed!")
        return True
    except Exception as e:
        print(f"✗ Cosine Warmup Scheduler test failed: {e}")
        return False

def test_vram_configs():
    """Test VRAM-optimized configurations"""
    print("Testing VRAM-optimized configurations...")
    try:
        # Test all VRAM profiles
        profiles = ["low", "medium", "high"]
        configs = {}
        
        for profile in profiles:
            config = Config.get_vram_optimized_config(profile)
            configs[profile] = config
            
            # Validate constraints
            assert config.n_embd % 2 == 0, f"{profile}: n_embd must be even for RoPE"
            assert config.n_embd % config.n_head == 0, f"{profile}: n_embd must be divisible by n_head"
            
            if config.attention_type == "grouped_query":
                assert config.n_head % config.n_query_groups == 0, f"{profile}: n_head must be divisible by n_query_groups"
        
        # Check that higher profiles have larger models
        assert configs["low"].n_embd < configs["medium"].n_embd < configs["high"].n_embd
        assert configs["low"].n_layer < configs["medium"].n_layer < configs["high"].n_layer
        
        print("✓ VRAM configurations test passed!")
        return True
    except Exception as e:
        print(f"✗ VRAM configurations test failed: {e}")
        return False

def test_enhanced_model():
    """Test the enhanced GPT model with new features"""
    print("Testing Enhanced GPT Model...")
    try:
        # Create config with notebook features
        config = Config.get_vram_optimized_config("low")
        config.vocab_size = 1000  # Small vocab for testing
        config.block_size = 128   # Small context for testing
        config.n_layer = 2        # Minimal layers for testing
        
        # Create model
        model = GPTLanguageModel(config)
        
        # Test forward pass
        batch_size, seq_len = 2, 64
        idx = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        targets = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        
        # Forward pass with targets (training mode)
        logits, loss = model(idx, targets)
        
        assert logits.shape == (batch_size, seq_len, config.vocab_size)
        assert loss is not None and loss.item() > 0
        
        # Forward pass without targets (inference mode)
        model.eval()
        with torch.no_grad():
            logits_inf, loss_inf = model(idx)
            assert logits_inf.shape == (batch_size, seq_len, config.vocab_size)
            assert loss_inf is None
        
        print("✓ Enhanced GPT Model test passed!")
        return True
    except Exception as e:
        print(f"✗ Enhanced GPT Model test failed: {e}")
        return False

def test_enhanced_checkpointing():
    """Test enhanced checkpointing with metadata"""
    print("Testing Enhanced Checkpointing...")
    try:
        # Create minimal config and trainer
        config = Config.get_vram_optimized_config("low")
        config.vocab_size = 1000
        config.block_size = 64
        config.n_layer = 1
        
        trainer = Trainer(config)
        
        # Test enhanced saving
        test_path = "test_enhanced_checkpoint.pth"
        success = trainer.save_model(
            test_path, 
            iteration=100, 
            train_loss=2.5, 
            val_loss=2.8
        )
        
        assert success, "Enhanced save should succeed"
        assert os.path.exists(test_path), "Checkpoint file should exist"
        
        # Test enhanced loading
        trainer2 = Trainer(config)
        trainer2.load_model(test_path)
        
        # Clean up
        if os.path.exists(test_path):
            os.remove(test_path)
        
        print("✓ Enhanced Checkpointing test passed!")
        return True
    except Exception as e:
        print(f"✗ Enhanced Checkpointing test failed: {e}")
        return False

def test_generation_features():
    """Test advanced generation features"""
    print("Testing Advanced Generation Features...")
    try:
        # Create minimal model for testing
        config = Config.get_vram_optimized_config("low")
        config.vocab_size = 100
        config.block_size = 32
        config.n_layer = 1
        
        model = GPTLanguageModel(config)
        model.eval()
        
        # Test temperature generation
        with torch.no_grad():
            idx = torch.zeros((1, 1), dtype=torch.long)
            
            # Test different temperature values
            for temp in [0.5, 1.0, 1.5]:
                generated = model.generate(idx, max_new_tokens=10, temperature=temp)
                assert generated.shape[1] == 11  # 1 + 10 new tokens
            
            # Test top-k generation
            generated_topk = model.generate(idx, max_new_tokens=10, top_k=10)
            assert generated_topk.shape[1] == 11
            
            # Test top-p generation
            generated_topp = model.generate(idx, max_new_tokens=10, top_p=0.9)
            assert generated_topp.shape[1] == 11
            
            # Test beam search generation
            generated_beam = model.beam_search_generate(idx, max_new_tokens=10, beam_size=3)
            assert generated_beam.shape[1] == 11
        
        print("✓ Advanced Generation Features test passed!")
        return True
    except Exception as e:
        print(f"✗ Advanced Generation Features test failed: {e}")
        return False

def run_all_tests():
    """Run all notebook-inspired feature tests"""
    print("=" * 60)
    print("TESTING NOTEBOOK-INSPIRED IMPROVEMENTS")
    print("=" * 60)
    
    tests = [
        test_rope,
        test_sliding_window_attention,
        test_sparse_moe,
        test_cosine_warmup_scheduler,
        test_vram_configs,
        test_enhanced_model,
        test_enhanced_checkpointing,
        test_generation_features
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"✗ {test_func.__name__} crashed: {e}")
        print()
    
    print("=" * 60)
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("🎉 All notebook-inspired improvements working correctly!")
    else:
        print(f"⚠️  {total - passed} tests failed. Check implementation.")
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
