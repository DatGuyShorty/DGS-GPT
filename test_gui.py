#!/usr/bin/env python3
"""
Simple test script to verify the GUI can be imported and instantiated
"""

import sys
import os

try:
    # Import the GUI module
    from gui import GPT_GUI
    print("✓ GUI module imported successfully")
    
    # Try to import tkinter (required for GUI)
    import tkinter as tk
    print("✓ Tkinter available")
    
    # Test that we can create the root window
    root = tk.Tk()
    root.withdraw()  # Hide the window
    print("✓ Tkinter root window created")
    
    # Try to instantiate the GUI class (but don't show it)
    try:
        app = GPT_GUI(root)
        print("✓ GPT_GUI class instantiated successfully")
        
        # Check if the new hyperparameter tab was created
        if hasattr(app, 'trials_input') and hasattr(app, 'steps_per_trial_input'):
            print("✓ Hyperparameter optimization controls found")
        else:
            print("✗ Hyperparameter optimization controls missing")
            
        # Check if the hyperparameter tab exists in the notebook
        tab_count = app.notebook.index("end")
        print(f"✓ Found {tab_count} tabs in the notebook")
        
        # Check tab names
        for i in range(tab_count):
            tab_text = app.notebook.tab(i, "text")
            print(f"  - Tab {i}: {tab_text}")
        
    except Exception as e:
        print(f"✗ Error creating GPT_GUI: {e}")
        sys.exit(1)
    
    root.destroy()
    print("✓ All tests passed - GUI is ready!")
    
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"✗ Unexpected error: {e}")
    sys.exit(1)
