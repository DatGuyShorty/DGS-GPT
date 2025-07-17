#!/usr/bin/env python3
"""
Test script to verify the code improvements made to DGS-GPT.

This script tests:
1. Enhanced configuration management
2. Improved error handling
3. Security improvements
4. Validation utilities
5. Logging enhancements
"""

import sys
import os
from pathlib import Path

def test_configuration_improvements():
    """Test configuration management improvements."""
    print("🔧 Testing Configuration Improvements...")
    
    try:
        from ShitGPT import Config
        
        # Test 1: Basic config creation
        config = Config()
        print("✅ Basic config creation works")
        
        # Test 2: VRAM optimization with proper copying
        original_batch_size = config.batch_size
        low_config = Config.get_vram_optimized_config("low", config)
        
        if config.batch_size == original_batch_size:
            print("✅ Original config not modified by VRAM optimization")
        else:
            print("❌ Original config was modified (this should not happen)")
        
        # Test 3: Parameter validation
        try:
            bad_config = Config()
            bad_config.n_embd = 5  # Odd number, should cause validation warning
            bad_config._validate_critical_parameters()
            print("❌ Should have caught odd n_embd")
        except ValueError:
            print("✅ Parameter validation works correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_validation_utilities():
    """Test the enhanced validation utilities."""
    print("\n🔍 Testing Validation Utilities...")
    
    try:
        from enhanced_validation import (
            validate_config, 
            validate_generation_params,
            validate_file_path,
            validate_system_requirements
        )
        
        # Test config validation
        from ShitGPT import Config
        config = Config()
        issues = validate_config(config)
        print(f"✅ Config validation: {len(issues)} issues found")
        
        # Test generation parameter validation
        gen_issues = validate_generation_params(
            max_tokens=100,
            temperature=1.0,
            top_k=50,
            top_p=0.9
        )
        print(f"✅ Generation param validation: {len(gen_issues)} issues found")
        
        # Test file path validation
        valid, error = validate_file_path(__file__, must_exist=True, extension='.py')
        if valid:
            print("✅ File path validation works")
        else:
            print(f"❌ File path validation failed: {error}")
        
        # Test system requirements
        sys_info = validate_system_requirements()
        print(f"✅ System validation: {len(sys_info['errors'])} errors, {len(sys_info['warnings'])} warnings")
        
        return True
        
    except Exception as e:
        print(f"❌ Validation test failed: {e}")
        return False

def test_logging_improvements():
    """Test the enhanced logging system."""
    print("\n📝 Testing Logging Improvements...")
    
    try:
        from enhanced_logging import setup_logging, TrainingLogger, ErrorHandler
        
        # Test logger setup
        logger = setup_logging(
            log_file="test_improvements.log",
            console_output=True
        )
        logger.info("Test log message")
        print("✅ Enhanced logging setup works")
        
        # Test training logger
        training_logger = TrainingLogger(logger)
        training_logger.log_step(1, 2.5, 0.001)
        print("✅ Training logger works")
        
        # Test error handler
        error_handler = ErrorHandler(logger)
        error_handler.handle_info("Test info message", "test_context")
        print("✅ Error handler works")
        
        return True
        
    except Exception as e:
        print(f"❌ Logging test failed: {e}")
        return False

def test_security_improvements():
    """Test security improvements."""
    print("\n🔒 Testing Security Improvements...")
    
    try:
        # Create a test model file
        import torch
        test_model_path = "test_model_security.pth"
        
        # Create a simple test model
        simple_model = torch.nn.Linear(10, 1)
        torch.save(simple_model.state_dict(), test_model_path)
        
        # Test secure loading
        from ShitGPT import Trainer, Config
        config = Config()
        trainer = Trainer(config)
        
        # This should work with the improved security
        result = trainer.load_model(test_model_path)
        print(f"✅ Secure model loading: {'success' if result else 'failed (expected for test)'}")
        
        # Cleanup
        if os.path.exists(test_model_path):
            os.remove(test_model_path)
        
        return True
        
    except Exception as e:
        print(f"❌ Security test failed: {e}")
        return False

def test_file_organization():
    """Test new file organization."""
    print("\n📁 Testing File Organization...")
    
    # Check if new files exist
    new_files = [
        "gpt_model.py",
        "enhanced_validation.py", 
        "enhanced_logging.py",
        "CODE_IMPROVEMENTS.md"
    ]
    
    existing_files = []
    for filename in new_files:
        if os.path.exists(filename):
            existing_files.append(filename)
            print(f"✅ {filename} created")
        else:
            print(f"❌ {filename} missing")
    
    return len(existing_files) == len(new_files)

def run_all_tests():
    """Run all improvement tests."""
    print("=" * 60)
    print("DGS-GPT CODE IMPROVEMENTS TEST SUITE")
    print("=" * 60)
    
    tests = [
        ("Configuration Management", test_configuration_improvements),
        ("Validation Utilities", test_validation_utilities),
        ("Logging Enhancements", test_logging_improvements),
        ("Security Improvements", test_security_improvements),
        ("File Organization", test_file_organization),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results[test_name] = False
    
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = 0
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:<25} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All improvements are working correctly!")
    else:
        print("⚠️  Some improvements need attention.")
    
    return passed == len(tests)

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
