# Setup Verification Guide

This guide shows you how to verify that your project setup is working correctly.

## Quick Verification

Run the comprehensive verification script:

```bash
# Activate virtual environment (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Run verification
python scripts/verify_setup.py
```

This will check:
- ✅ Virtual environment is active
- ✅ All dependencies are installed
- ✅ Configuration files exist
- ✅ Project structure is correct
- ✅ Utility modules (config & logger) work

## Detailed Testing

For more detailed tests of configuration and logging:

```bash
python scripts/test_setup.py
```

This will:
- Test YAML configuration loading
- Test configuration merging
- Test all log levels (DEBUG, INFO, WARNING, ERROR)
- Verify log file creation and rotation

## Manual Verification Steps

### 1. Check Virtual Environment

```bash
# Activate venv
.\venv\Scripts\Activate.ps1  # Windows PowerShell
# OR
venv\Scripts\activate.bat    # Windows CMD

# Verify you're in venv (should show venv path)
python -c "import sys; print(sys.prefix)"
```

### 2. Check Dependencies

```bash
# Test imports
python -c "import fastapi, uvicorn, torch, shap, pydantic, yaml; print('All dependencies OK!')"

# Check versions
pip list | findstr "fastapi uvicorn torch shap pydantic"
```

### 3. Test Configuration System

```python
# Test in Python REPL
python

>>> from src.utils.config import Config
>>> config = Config.from_files('configs/model_config.yaml')
>>> print(config.audio_encoder.embedding_dim)  # Should print: 1024
>>> print(config.get('fusion.concat_dim'))     # Should print: 1088
>>> exit()
```

### 4. Test Logging System

```python
# Test in Python REPL
python

>>> from src.utils.logger import setup_logger
>>> import logging
>>> logger = setup_logger(__name__, log_file='test.log', level=logging.DEBUG)
>>> logger.debug("Debug message")
>>> logger.info("Info message")
>>> logger.warning("Warning message")
>>> logger.error("Error message")
>>> exit()

# Check log file was created
# Windows: type logs\test.log
# Linux/Mac: cat logs/test.log
```

### 5. Verify Project Structure

```bash
# Check key directories exist
dir src\           # Should show api/, fusion/, utils/, etc.
dir configs\       # Should show *.yaml files
dir logs\          # Should exist (may be empty)
dir tests\         # Should show test files
```

### 6. Check Git Status

```bash
git status
# Should show your changes to src/utils/config.py and src/utils/logger.py
```

## Expected Results

✅ **All checks should pass** if setup is correct:
- Virtual environment is active
- All 6 dependencies (FastAPI, Uvicorn, PyTorch, SHAP, Pydantic, PyYAML) import successfully
- All 3 config files exist and can be loaded
- Project structure matches expected layout
- Config and logger modules work without errors

## Troubleshooting

### Virtual Environment Not Active
```bash
.\venv\Scripts\Activate.ps1
```

### Missing Dependencies
```bash
pip install -r requirements.txt
```

### Import Errors
- Make sure you're in the project root directory
- Ensure virtual environment is activated
- Check that `src/` directory has `__init__.py` files

### Configuration File Not Found
- Verify you're running scripts from project root
- Check that `configs/` directory exists with `.yaml` files

### Logging Not Working
- Check that `logs/` directory exists (created automatically)
- Verify write permissions in project directory

## Quick Test Commands

```bash
# One-liner to test everything
.\venv\Scripts\Activate.ps1; python scripts/verify_setup.py

# Test just imports
.\venv\Scripts\Activate.ps1; python -c "from src.utils.config import Config; from src.utils.logger import setup_logger; print('✓ All OK')"

# Check log file was created (after running tests)
dir logs\*.log
```

---

**Remember:** Always activate your virtual environment before running any Python commands or scripts!
