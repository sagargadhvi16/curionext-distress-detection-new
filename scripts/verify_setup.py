"""Quick verification script to check if everything is set up correctly."""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def check_virtual_env():
    """Check if we're in a virtual environment."""
    print("1. Checking Virtual Environment...")
    in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    if in_venv:
        print(f"   ✓ Virtual environment active: {sys.prefix}")
        return True
    else:
        print(f"   ⚠ Not in virtual environment. Run: .\\venv\\Scripts\\Activate.ps1")
        return False


def check_dependencies():
    """Check if all required dependencies are installed."""
    print("\n2. Checking Dependencies...")
    dependencies = {
        'fastapi': 'FastAPI',
        'uvicorn': 'Uvicorn',
        'torch': 'PyTorch',
        'shap': 'SHAP',
        'pydantic': 'Pydantic',
        'yaml': 'PyYAML (yaml)'
    }
    
    all_ok = True
    for module, name in dependencies.items():
        try:
            __import__(module)
            print(f"   ✓ {name} installed")
        except ImportError:
            print(f"   ✗ {name} NOT installed")
            all_ok = False
    
    return all_ok


def check_config_files():
    """Check if config files exist."""
    print("\n3. Checking Configuration Files...")
    configs = [
        'configs/model_config.yaml',
        'configs/training_config.yaml',
        'configs/deployment_config.yaml'
    ]
    
    all_ok = True
    for config in configs:
        config_path = project_root / config
        if config_path.exists():
            print(f"   ✓ {config}")
        else:
            print(f"   ✗ {config} not found")
            all_ok = False
    
    return all_ok


def check_project_structure():
    """Check if project directories exist."""
    print("\n4. Checking Project Structure...")
    directories = [
        'src',
        'src/utils',
        'src/api',
        'src/fusion',
        'tests',
        'configs',
        'logs',
        'data',
        'models'
    ]
    
    all_ok = True
    for directory in directories:
        dir_path = project_root / directory
        if dir_path.exists():
            print(f"   ✓ {directory}/")
        else:
            print(f"   ✗ {directory}/ not found")
            all_ok = False
    
    return all_ok


def check_utils_modules():
    """Check if utility modules work."""
    print("\n5. Checking Utility Modules...")
    try:
        from src.utils.config import Config, load_config
        print("   ✓ Configuration module imports successfully")
        
        # Try loading a config
        config = load_config('configs/model_config.yaml')
        print("   ✓ Configuration file loading works")
        
        from src.utils.logger import setup_logger
        import logging
        logger = setup_logger('verify', level=logging.INFO)
        print("   ✓ Logging module works")
        logger.info("   ✓ Logger test message")
        
        return True
    except Exception as e:
        print(f"   ✗ Utility modules error: {e}")
        return False


def main():
    """Run all verification checks."""
    print("="*60)
    print("🔍 Verifying Project Setup")
    print("="*60 + "\n")
    
    checks = [
        ("Virtual Environment", check_virtual_env),
        ("Dependencies", check_dependencies),
        ("Configuration Files", check_config_files),
        ("Project Structure", check_project_structure),
        ("Utility Modules", check_utils_modules)
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"   ✗ Error: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*60)
    print("📊 Verification Summary")
    print("="*60)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{name:.<40} {status}")
        if not passed:
            all_passed = False
    
    print("="*60)
    if all_passed:
        print("\n✅ All checks passed! Your setup is ready.")
        print("\n💡 Next steps:")
        print("   - Run: python scripts/test_setup.py (for detailed tests)")
        print("   - Start developing your fusion layer and API!")
    else:
        print("\n❌ Some checks failed. Please fix the issues above.")
        print("\n💡 Common fixes:")
        print("   - Activate venv: .\\venv\\Scripts\\Activate.ps1")
        print("   - Install deps: pip install -r requirements.txt")
    print("="*60 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
