import tkinter as tk
print("Starting simple GUI test...")

try:
    from gui import GPT_GUI
    print("✅ GUI imported successfully")
    
    root = tk.Tk()
    root.title("DGS-GPT - Test")
    
    # Create GUI instance
    app = GPT_GUI(root)
    print("✅ GUI instance created")
    
    # Check for new attributes
    attrs_to_check = ['use_moe_var', 'moe_experts_var', 'attention_type_var']
    for attr in attrs_to_check:
        if hasattr(app, attr):
            print(f"✅ Found attribute: {attr}")
        else:
            print(f"❌ Missing attribute: {attr}")
    
    # Test architecture config
    try:
        config = app.get_architecture_config()
        print(f"✅ Architecture config created: use_moe={config.use_moe}")
    except Exception as e:
        print(f"❌ Error getting architecture config: {e}")
    
    print("✅ All basic tests passed!")
    print("GUI window will appear - close it to complete the test.")
    
    # Show the window briefly
    root.mainloop()
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
