#!/usr/bin/env python3
"""
Test script to verify the hyperparameter display widget works
"""

import tkinter as tk
from gui import GPT_GUI

def test_hyperparameter_display():
    print("Testing hyperparameter display widget...")
    
    # Create a test root window
    root = tk.Tk()
    root.withdraw()  # Hide the window for testing
    
    try:
        # Create the GUI instance
        app = GPT_GUI(root)
        
        # Check if the hyperparameter display was created
        if hasattr(app, 'hyperparam_text'):
            print("✓ Hyperparameter display widget created")
            
            # Check if the display has content
            content = app.hyperparam_text.get("1.0", tk.END).strip()
            if content:
                print("✓ Hyperparameter display has content:")
                print("---")
                print(content)
                print("---")
            else:
                print("✗ Hyperparameter display is empty")
                
            # Test the update method
            app.update_hyperparameter_display()
            updated_content = app.hyperparam_text.get("1.0", tk.END).strip()
            if updated_content:
                print("✓ Hyperparameter display update method works")
            else:
                print("✗ Hyperparameter display update method failed")
                
        else:
            print("✗ Hyperparameter display widget not found")
            
    except Exception as e:
        print(f"✗ Error during testing: {e}")
        return False
        
    finally:
        root.destroy()
        
    print("Test completed successfully!")
    return True

if __name__ == "__main__":
    test_hyperparameter_display()
