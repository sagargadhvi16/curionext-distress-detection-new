"""Test script to verify configuration and logging setup."""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import Config, load_config, merge_configs
from src.utils.logger import setup_logger
import logging


def test_config_loading():
    """Test configuration loading."""
    print("\n" + "="*50)
    print("Testing Configuration System")
    print("="*50)
    
    try:
        # Test loading single config
        model_config = load_config('configs/model_config.yaml')
        print(f"✓ Loaded model_config.yaml: {len(model_config)} top-level keys")
        
        # Test Config class
        config = Config.from_files('configs/model_config.yaml')
        print(f"✓ Created Config object")
        
        # Test attribute access
        if hasattr(config, 'audio_encoder'):
            print(f"✓ Audio encoder embedding dim: {config.audio_encoder.embedding_dim}")
        
        # Test dot notation access
        embedding_dim = config.get('audio_encoder.embedding_dim')
        print(f"✓ Retrieved via get(): {embedding_dim}")
        
        # Test merging configs
        config_merged = Config.from_files(
            'configs/model_config.yaml',
            'configs/training_config.yaml'
        )
        print(f"✓ Merged multiple config files")
        
        return True
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False


def test_logging():
    """Test logging setup."""
    print("\n" + "="*50)
    print("Testing Logging System")
    print("="*50)
    
    try:
        # Setup logger
        logger = setup_logger(
            'test_setup',
            log_file='test_setup.log',
            level=logging.DEBUG
        )
        print("✓ Logger created")
        
        # Test all log levels
        logger.debug("This is a DEBUG message")
        logger.info("This is an INFO message")
        logger.warning("This is a WARNING message")
        logger.error("This is an ERROR message")
        print("✓ All log levels tested (check console output above)")
        
        # Check if log file exists
        log_file = project_root / 'logs' / 'test_setup.log'
        if log_file.exists():
            print(f"✓ Log file created: {log_file}")
            print(f"  File size: {log_file.stat().st_size} bytes")
        else:
            print(f"⚠ Log file not found (may be expected)")
        
        return True
    except Exception as e:
        print(f"✗ Logging test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n🧪 Testing Project Setup (Configuration & Logging)")
    print("="*50)
    
    results = []
    results.append(("Configuration", test_config_loading()))
    results.append(("Logging", test_logging()))
    
    print("\n" + "="*50)
    print("Test Summary")
    print("="*50)
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{name}: {status}")
    
    all_passed = all(result[1] for result in results)
    print("\n" + "="*50)
    if all_passed:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    print("="*50 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
