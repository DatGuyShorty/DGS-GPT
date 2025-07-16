#!/usr/bin/env python3
"""
Quick test to verify the hyperparameter optimization updates work correctly
"""

import tkinter as tk
from gui import GPT_GUI

def test_hyperparameter_controls():
    """Test the hyperparameter optimization controls"""
    print("Testing hyperparameter optimization controls...")
    
    # Create a test window (but don't show it)
    root = tk.Tk()
    root.withdraw()
    
    try:
        # Create GUI instance
        app = GPT_GUI(root)
        
        # Check if new controls exist
        required_attrs = [
            'use_moe_var',
            'moe_experts_var', 
            'moe_k_var',
            'attention_type_var',
            'query_groups_var',
            'query_groups_combo'
        ]
        
        missing_attrs = []
        for attr in required_attrs:
            if not hasattr(app, attr):
                missing_attrs.append(attr)
        
        if missing_attrs:
            print(f"❌ Missing attributes: {missing_attrs}")
            return False
        
        # Test architecture config method
        arch_config = app.get_architecture_config()
        if not hasattr(arch_config, 'use_moe'):
            print("❌ Architecture config missing use_moe")
            return False
        
        print("✅ All hyperparameter optimization controls present")
        print(f"✅ Architecture config method works")
        
        # Test attention type change
        app.attention_type_var.set("grouped_query")
        app.on_attention_type_change()
        print("✅ Attention type change handler works")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        return False
    finally:
        root.destroy()

if __name__ == "__main__":
    success = test_hyperparameter_controls()
    if success:
        print("\n🎉 All tests passed! Hyperparameter optimization controls are working correctly.")
    else:
        print("\n❌ Some tests failed. Please check the implementation.")
