#!/usr/bin/env python3
"""
DGS-GPT Setup Script
Cross-platform setup for the DGS-GPT development environment
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def run_command(cmd, check=True, shell=False):
    """Run a command and return the result"""
    try:
        if isinstance(cmd, str) and not shell:
            cmd = cmd.split()
        result = subprocess.run(cmd, capture_output=True, text=True, check=check, shell=shell)
        return result.returncode == 0, result.stdout.strip(), result.stderr.strip()
    except subprocess.CalledProcessError as e:
        return False, e.stdout.strip() if e.stdout else "", e.stderr.strip() if e.stderr else ""
    except FileNotFoundError:
        return False, "", f"Command not found: {cmd[0] if isinstance(cmd, list) else cmd}"

def print_status(emoji, message):
    """Print a status message with emoji"""
    print(f"{emoji} {message}")

def check_python_version():
    """Check if Python version meets requirements"""
    version = sys.version_info
    min_version = (3, 8)
    
    if version[:2] < min_version:
        print_status("❌", f"Error: Python {version.major}.{version.minor} detected, but Python {min_version[0]}.{min_version[1]}+ is required")
        return False
    
    print_status("✅", f"Python version check passed: {version.major}.{version.minor}.{version.micro}")
    return True

def create_venv():
    """Create virtual environment"""
    venv_path = Path(".venv")
    
    if venv_path.exists():
        print_status("⚠️", "Virtual environment already exists. Removing old environment...")
        import shutil
        shutil.rmtree(venv_path)
    
    print_status("📦", "Creating virtual environment...")
    success, stdout, stderr = run_command([sys.executable, "-m", "venv", str(venv_path)])
    
    if not success:
        print_status("❌", f"Failed to create virtual environment: {stderr}")
        return False
    
    return True

def get_venv_python():
    """Get the path to the Python executable in the virtual environment"""
    system = platform.system().lower()
    venv_path = Path(".venv")
    
    if system == "windows":
        return venv_path / "Scripts" / "python.exe"
    else:
        return venv_path / "bin" / "python"

def get_pip_command():
    """Get the pip command for the virtual environment"""
    python_exe = get_venv_python()
    return [str(python_exe), "-m", "pip"]

def install_dependencies():
    """Install dependencies"""
    pip_cmd = get_pip_command()
    
    # Upgrade pip
    print_status("⬆️", "Upgrading pip...")
    success, _, stderr = run_command(pip_cmd + ["install", "--upgrade", "pip"])
    if not success:
        print_status("❌", f"Failed to upgrade pip: {stderr}")
        return False
    
    # Install PyTorch first
    print_status("🔥", "Installing PyTorch...")
    success, _, stderr = run_command(pip_cmd + ["install", "torch>=2.0.0"])
    if not success:
        print_status("❌", f"Failed to install PyTorch: {stderr}")
        return False
    
    # Install other dependencies
    if not Path("requirements.txt").exists():
        print_status("❌", "Error: requirements.txt not found")
        return False
    
    print_status("📚", "Installing dependencies from requirements.txt...")
    success, _, stderr = run_command(pip_cmd + ["install", "-r", "requirements.txt"])
    if not success:
        print_status("❌", f"Failed to install dependencies: {stderr}")
        return False
    
    return True

def test_imports():
    """Test that all required modules can be imported"""
    print_status("🧪", "Testing imports...")
    python_exe = get_venv_python()
    
    test_code = """
import torch
import optuna
import matplotlib
from gui import GPT_GUI
print('✅ All core modules import successfully')
"""
    
    success, stdout, stderr = run_command([str(python_exe), "-c", test_code])
    if success:
        print_status("✅", "Import test passed")
        return True
    else:
        print_status("❌", f"Import test failed: {stderr}")
        return False

def check_dataset():
    """Check if dataset files exist"""
    print_status("📄", "Checking for dataset files...")
    if not Path("vocab.txt").exists():
        print_status("⚠️", "vocab.txt not found. You may need to prepare your dataset.")
        print("   Run: python dataset.py (to download and prepare a sample dataset)")
        print("   Or place your own text data in vocab.txt")

def print_next_steps():
    """Print next steps for the user"""
    system = platform.system().lower()
    
    if system == "windows":
        activate_cmd = ".venv\\Scripts\\activate.bat"
    else:
        activate_cmd = "source .venv/bin/activate"
    
    print()
    print_status("🎉", "Setup completed successfully!")
    print()
    print("📋 Next steps:")
    print(f"   1. Activate the environment: {activate_cmd}")
    print("   2. Prepare dataset (if needed): python dataset.py")
    print("   3. Run the GUI: python gui.py")
    print("   4. Or run command line: python ShitGPT.py")
    print()
    print("💡 Tips:")
    print("   - The GUI provides the easiest way to train and generate text")
    print("   - Use hyperparameter optimization to find the best settings")
    print("   - Check README.md for detailed usage instructions")
    print()

def main():
    """Main setup function"""
    print_status("🚀", "Setting up DGS-GPT development environment...")
    print_status("🖥️", f"Platform: {platform.system()} {platform.release()}")
    print_status("🐍", f"Python: {sys.version}")
    
    # Check Python version
    if not check_python_version():
        return 1
    
    # Create virtual environment
    if not create_venv():
        return 1
    
    print_status("✅", "Virtual environment created")
    
    # Install dependencies
    if not install_dependencies():
        return 1
    
    # Test imports
    if not test_imports():
        return 1
    
    # Check dataset
    check_dataset()
    
    # Print next steps
    print_next_steps()
    
    return 0

if __name__ == "__main__":
    try:
        exit_code = main()
        if exit_code != 0:
            print_status("❌", "Setup failed!")
            if platform.system().lower() == "windows":
                input("Press Enter to continue...")
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print_status("⚠️", "Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print_status("❌", f"Unexpected error: {e}")
        sys.exit(1)
