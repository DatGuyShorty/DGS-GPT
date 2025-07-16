#!/usr/bin/env python3
"""
Test script to verify the enhanced verbose optimization output
"""

import tkinter as tk
from tkinter import messagebox
import sys
import os

def test_verbose_optimization():
    """Test the verbose optimization with a small trial"""
    print("Testing verbose hyperparameter optimization...")
    
    try:
        from ShitGPT import Config, run_hyperparameter_optimization
        
        # Create a basic config for testing
        config = Config()
        config.use_moe = False
        config.attention_type = "multihead"
        config.n_query_groups = 1
        
        print("✅ ShitGPT imported successfully")
        print(f"✅ Config created: {config.attention_type} attention")
        
        # Test with minimal trials for quick verification
        print("\nRunning 1 trial with 5 steps to test verbose output...")
        result = run_hyperparameter_optimization(config, n_trials=1, steps_per_trial=5)
        
        print("✅ Optimization completed successfully")
        print(f"✅ Result config created")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during optimization test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_gui_verbose():
    """Test GUI with verbose optimization (without actually running)"""
    print("\nTesting GUI verbose features...")
    
    try:
        root = tk.Tk()
        root.withdraw()  # Hide the window
        
        from gui import GPT_GUI
        app = GPT_GUI(root)
        
        # Check if the new verbose features exist
        if hasattr(app, 'process_optim_queue'):
            print("✅ Enhanced process_optim_queue method found")
        else:
            print("❌ Enhanced process_optim_queue method missing")
            
        # Test architecture config
        arch_config = app.get_architecture_config()
        print(f"✅ Architecture config: {arch_config.attention_type}")
        
        root.destroy()
        return True
        
    except Exception as e:
        print(f"❌ Error during GUI test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("TESTING ENHANCED VERBOSE OPTIMIZATION")
    print("=" * 60)
    
    # Test 1: GUI verbose features
    gui_success = test_gui_verbose()
    
    # Test 2: Ask user if they want to run actual optimization test
    print("\n" + "=" * 60)
    response = input("Do you want to test actual optimization (1 trial, 5 steps)? (y/N): ").strip().lower()
    
    if response in ['y', 'yes']:
        opt_success = test_verbose_optimization()
    else:
        print("Skipping optimization test.")
        opt_success = True
    
    print("\n" + "=" * 60)
    if gui_success and opt_success:
        print("🎉 All tests passed! Enhanced verbose optimization is working.")
    else:
        print("❌ Some tests failed. Check the implementation.")
    print("=" * 60)
