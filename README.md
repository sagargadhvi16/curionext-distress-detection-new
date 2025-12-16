# CurioNext Child Distress Detection System

Multi-modal AI system for real-time child distress detection using audio and biometric signals.

## 🎯 Project Status

**Timeline:** December 16, 2025 - January 15, 2026 


## 👥 Team

- **Manager : Bhaagyesh**
- **Intern 1(Nidhi):** Audio Processing Pipeline
- **Intern 2(Nikhil):** Biometric Processing Pipeline
- **Intern 3(Aryan):** Fusion Layer & Backend

## 🏗️ Architecture

<img width="934" height="1286" alt=" Architecture" src="https://github.com/user-attachments/assets/89c37c2e-6f1e-4e9b-8bf7-fdbb0603d67e" />



## 🚀 Quick Start

### Setup
```bash
git clone https://github.com/YOUR-USERNAME/curionext-distress-detection.git
cd curionext-distress-detection

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .

# Copy environment file (if exists)
# cp .env.example .env
```

### Development Workflow

```bash
# Checkout your branch
git checkout feature/audio-pipeline  # or biometric-pipeline or fusion-backend

# Work on your tasks
# ... make changes ...

# Commit and push
git add .
git commit -m "feat: implement audio normalization"
git push origin feature/audio-pipeline

# Create PR when ready
```

## 🧪 Testing

```bash
pytest                              # Run all tests
pytest tests/test_audio.py          # Run specific test
pytest --cov=src tests/             # Run with coverage
```

## ⚙️ Configuration

The project uses YAML-based configuration management. Configuration files are located in `configs/`:

- `model_config.yaml` - Model architecture and hyperparameters
- `training_config.yaml` - Training settings, loss, and data splits
- `deployment_config.yaml` - API and inference settings

### Usage Example

```python
from src.utils.config import Config

# Load single config file
config = Config.from_files('configs/model_config.yaml')

# Load and merge multiple configs (later files override earlier ones)
config = Config.from_files('configs/model_config.yaml', 'configs/training_config.yaml')

# Access configuration values
audio_dim = config.get('audio_encoder.embedding_dim')  # Dot notation
learning_rate = config.training.learning_rate  # Attribute access
```

## 📝 Logging

Structured logging is set up with console (colored) and file (rotating) handlers:

```python
from src.utils.logger import setup_logger

# Setup logger with file logging
logger = setup_logger(__name__, log_file='api.log', level=logging.DEBUG)

# Use logger
logger.debug("Debug message")
logger.info("Info message")
logger.warning("Warning message")
logger.error("Error message")
```

Logs are written to `logs/` directory with automatic rotation (10MB per file, 5 backups).

## ✅ Verify Setup

To verify everything is working correctly:

```bash
# Quick verification (recommended)
python scripts/verify_setup.py

# Detailed tests
python scripts/test_setup.py
```

See [docs/VERIFICATION.md](docs/VERIFICATION.md) for detailed verification steps.

## 📦 Tech Stack

- **ML:** PyTorch, TensorFlow Hub (YAMNet)
- **Audio:** librosa, soundfile, audiomentations
- **Biometric:** neurokit2, hrv-analysis
- **Backend:** FastAPI, uvicorn
- **XAI:** SHAP
- **Testing:** pytest
- **Configuration:** YAML-based config management
- **Logging:** Structured logging with file rotation

## 🎯 Performance Targets

- Model Size: <100MB
- Inference: <500ms
- Accuracy: >90%
- False Negative Rate: <5%

## 🔒 Security

**NEVER COMMIT:**
- Real child audio/data
- API keys or credentials
- Large model files (*.pth, *.pt)
- Personal information

## 📞 Contact

**Company:** CurioNext Labs Private Limited
**Location:** IIT Madras Research Park, Chennai

---

Built with ❤️ for child safety
