#!/usr/bin/env python3
"""
Release Preparation Script for DGS-GPT
Prepares the repository for version release
"""

import os
import sys
import subprocess
import json
from pathlib import Path
from version import __version__, get_version_info

def run_command(cmd, check=True):
    """Run a command and return success status"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=check)
        return True, result.stdout.strip(), result.stderr.strip()
    except subprocess.CalledProcessError as e:
        return False, e.stdout.strip() if e.stdout else "", e.stderr.strip() if e.stderr else ""

def check_git_status():
    """Check if git repository is clean"""
    print("🔍 Checking git status...")
    success, stdout, stderr = run_command("git status --porcelain")
    if not success:
        print(f"❌ Error checking git status: {stderr}")
        return False
    
    if stdout.strip():
        print("⚠️ Working directory has uncommitted changes:")
        print(stdout)
        return False
    else:
        print("✅ Working directory is clean")
        return True

def check_required_files():
    """Check if all required files exist"""
    print("📁 Checking required files...")
    required_files = [
        "README.md",
        "LICENSE",
        "requirements.txt",
        "CHANGELOG.md",
        "CONTRIBUTING.md",
        "version.py",
        "ShitGPT.py",
        "gui.py",
        "dataset.py",
        "setup.py",
        "setup.sh",
        "setup.bat"
    ]
    
    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing required files: {', '.join(missing_files)}")
        return False
    else:
        print(f"✅ All {len(required_files)} required files present")
        return True

def validate_imports():
    """Test that all modules can be imported"""
    print("🔧 Validating imports...")
    try:
        import ShitGPT
        import gui
        import dataset
        import version
        print("✅ All core modules import successfully")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def create_version_tag():
    """Create a git tag for the version"""
    version = __version__
    tag_name = f"v{version}"
    
    print(f"🏷️ Creating version tag: {tag_name}")
    
    # Check if tag already exists
    success, stdout, stderr = run_command(f"git tag -l {tag_name}")
    if stdout.strip():
        print(f"⚠️ Tag {tag_name} already exists")
        return True
    
    # Create the tag
    version_info = get_version_info()
    tag_message = f"Release {tag_name}\n\n{version_info['description']}"
    
    success, stdout, stderr = run_command(f'git tag -a {tag_name} -m "{tag_message}"')
    if success:
        print(f"✅ Created tag {tag_name}")
        return True
    else:
        print(f"❌ Failed to create tag: {stderr}")
        return False

def generate_release_notes():
    """Generate release notes from changelog"""
    print("📝 Generating release notes...")
    
    try:
        with open("CHANGELOG.md", "r") as f:
            changelog = f.read()
        
        # Extract current version notes
        lines = changelog.split('\n')
        in_current_version = False
        release_notes = []
        
        for line in lines:
            if line.startswith('## [') and __version__ in line:
                in_current_version = True
                continue
            elif line.startswith('## [') and in_current_version:
                break
            elif in_current_version:
                release_notes.append(line)
        
        notes_content = '\n'.join(release_notes).strip()
        
        with open("RELEASE_NOTES.md", "w") as f:
            f.write(f"# DGS-GPT v{__version__} Release Notes\n\n")
            f.write(notes_content)
        
        print("✅ Release notes generated: RELEASE_NOTES.md")
        return True
        
    except Exception as e:
        print(f"❌ Error generating release notes: {e}")
        return False

def package_info():
    """Display packaging information"""
    print("📦 Package Information:")
    version_info = get_version_info()
    print(f"   Version: {version_info['version']}")
    print(f"   Author: {version_info['author']}")
    print(f"   Description: {version_info['description']}")
    print(f"   License: {version_info['license']}")

def pre_release_checklist():
    """Run pre-release checklist"""
    print("✅ Pre-Release Checklist:")
    print("   [ ] All features implemented and tested")
    print("   [ ] Documentation updated")
    print("   [ ] Cross-platform testing completed")
    print("   [ ] Security review completed")
    print("   [ ] Performance benchmarking done")
    print("   [ ] Ready for community feedback")

def main():
    """Main release preparation function"""
    print("🚀 DGS-GPT Release Preparation")
    print("=" * 40)
    
    package_info()
    print()
    
    # Run all checks
    checks = [
        check_required_files,
        validate_imports,
        check_git_status,
        generate_release_notes
    ]
    
    all_passed = True
    for check in checks:
        if not check():
            all_passed = False
        print()
    
    if all_passed:
        print("🎉 All checks passed! Repository is ready for release.")
        print()
        print("Next steps:")
        print("1. Review the generated RELEASE_NOTES.md")
        print("2. Commit any final changes")
        print("3. Run: git add . && git commit -m 'Prepare for v0.1.0-alpha release'")
        print("4. Run: git push origin main")
        print("5. Create a release on GitHub using RELEASE_NOTES.md")
        print("6. Announce the release to the community")
        print()
        pre_release_checklist()
    else:
        print("❌ Some checks failed. Please fix the issues before release.")
        return 1
    
    return 0

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n⚠️ Release preparation interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)
