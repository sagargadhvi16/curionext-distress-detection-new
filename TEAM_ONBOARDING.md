# Team Onboarding Guide

Welcome to the CurioNext Child Distress Detection System project!

## Project Timeline

**Start Date:** December 16, 2024
**End Date:** January 15, 2025

## Team Structure

### Intern 1: Audio Processing Pipeline
**Branch:** `feature/audio-pipeline`

**Responsibilities:**
- Audio preprocessing (loading, normalization, silence trimming)
- Feature extraction (MFCC, spectral features)
- YAMNet encoder integration
- Audio-related unit tests

**Your Files:**
- `src/audio/preprocessing.py`
- `src/audio/features.py`
- `src/audio/encoder.py`
- `tests/test_audio.py`

**Key Deliverables:**
1. Audio loading and preprocessing pipeline
2. YAMNet integration for 1024-dim embeddings
3. Unit tests with >80% coverage
4. Documentation of audio processing steps

---

### Intern 2: Biometric Processing Pipeline
**Branch:** `feature/biometric-pipeline`

**Responsibilities:**
- HRV feature extraction (time and frequency domain)
- Accelerometer signal processing
- Biometric encoder (neural network)
- Biometric-related unit tests

**Your Files:**
- `src/biometric/hrv.py`
- `src/biometric/accelerometer.py`
- `src/biometric/encoder.py`
- `tests/test_biometric.py`

**Key Deliverables:**
1. HRV feature extraction (RMSSD, SDNN, LF/HF ratio)
2. Accelerometer feature extraction
3. Neural encoder for 64-dim embeddings
4. Unit tests with >80% coverage

---

### Intern 3: Fusion Layer & Backend
**Branch:** `feature/fusion-backend`

**Responsibilities:**
- Late fusion model implementation
- Binary classifier for distress detection
- SHAP-based explainability
- FastAPI backend
- Integration tests

**Your Files:**
- `src/fusion/late_fusion.py`
- `src/fusion/classifier.py`
- `src/fusion/explainer.py`
- `src/api/main.py`
- `src/api/inference.py`
- `src/api/models.py`
- `tests/test_fusion.py`
- `tests/test_api.py`

**Key Deliverables:**
1. Late fusion architecture (concat 1024+64 dims)
2. Binary classifier with weighted loss
3. SHAP integration for interpretability
4. REST API with /predict and /health endpoints
5. Integration tests

---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/curionext-distress-detection.git
cd curionext-distress-detection
```

### 2. Checkout Your Branch

**Intern 1:**
```bash
git checkout feature/audio-pipeline
```

**Intern 2:**
```bash
git checkout feature/biometric-pipeline
```

**Intern 3:**
```bash
git checkout feature/fusion-backend
```

### 3. Setup Python Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # Mac/Linux
# OR
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### 4. Verify Setup

```bash
# Test imports
python -c "import src; print('✅ Setup successful!')"

# Run tests (will show pending tests initially)
pytest tests/ -v
```

---

## Development Workflow

### Daily Workflow

1. **Pull latest changes**
   ```bash
   git pull origin feature/your-branch
   ```

2. **Make your changes**
   - Write code with clear comments
   - Follow TODO markers in template files
   - Write unit tests as you go

3. **Test your code**
   ```bash
   # Run your specific tests
   pytest tests/test_audio.py -v  # Intern 1
   pytest tests/test_biometric.py -v  # Intern 2
   pytest tests/test_fusion.py tests/test_api.py -v  # Intern 3

   # Check code style
   black src/ tests/
   flake8 src/ tests/
   ```

4. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: implement audio normalization"
   ```

5. **Push to your branch**
   ```bash
   git push origin feature/your-branch
   ```

### Commit Message Format

Use conventional commits:
- `feat:` - New feature
- `fix:` - Bug fix
- `test:` - Adding tests
- `docs:` - Documentation
- `refactor:` - Code refactoring

Examples:
```bash
git commit -m "feat: implement MFCC extraction"
git commit -m "fix: correct HRV frequency domain calculation"
git commit -m "test: add unit tests for audio preprocessing"
```


### Pull Request Process

1. **Create PR** when your feature is ready
2. **Target branch:** `develop` (NOT `main`)
3. **PR Title:** Clear description of changes
4. **Description:** What, why, and how
5. **Request review** from Sagar
6. **Address feedback** and push updates
7. **Merge** after approval

---

## Testing Requirements

### Coverage Target: >80%

Run tests with coverage:
```bash
pytest --cov=src tests/
pytest --cov=src --cov-report=html tests/
```

### Test Categories

1. **Unit Tests** - Test individual functions
2. **Integration Tests** - Test component interactions
3. **API Tests** - Test endpoints (Intern 3)

---

## Resources

### Documentation
- `docs/ARCHITECTURE.md` - System architecture
- `docs/API_DOCS.md` - API documentation
- `configs/` - Configuration files

### Datasets

**Audio Datasets:**
- FSD50K (Freesound Dataset)
- UrbanSound8K
- ESC-50

**Biometric Datasets:**
- PhysioNet
- WESAD (Wearable Stress and Affect Detection)

Download script:
```bash
python scripts/download_datasets.py
```

### Generate Synthetic Data (for testing)

```bash
# Generate synthetic audio samples
python scripts/generate_synthetic_audio.py

# Generate synthetic biometric data
python scripts/generate_synthetic_biometric.py
```

---

## Performance Targets

Your implementation should meet these targets:

- **Model Size:** < 100 MB
- **Inference Time:** < 500 ms
- **Accuracy:** > 90%
- **False Negative Rate:** < 5% (CRITICAL - missing distress cases)


### Code Reviews
- Review each other's PRs
- Provide constructive feedback
- Learn from each other's code

---

## Common Commands Cheat Sheet

```bash
# Switch branches
git checkout feature/audio-pipeline

# See your changes
git status
git diff

# Update from remote
git pull origin feature/your-branch

# Create commit
git add .
git commit -m "feat: your message"

# Push changes
git push origin feature/your-branch

# Run tests
pytest tests/test_audio.py -v
pytest --cov=src tests/

# Format code
black src/ tests/

# Check code style
flake8 src/ tests/

# Run API server (Intern 3)
uvicorn src.api.main:app --reload
```

---

## Need Help?

1. Check documentation in `docs/`
2. Ask in Discord channels
3. Create GitHub issue
4. Tag Bhaagyesh/Sagar for urgent items

---

## Security & Ethics

**NEVER commit:**
- Real child audio/video data
- Personal information
- API keys or credentials (.env files)
- Large model files (use .gitignore)

**Remember:**
- This system is for child safety
- False negatives are worse than false positives
- Privacy and data protection are paramount
- Test thoroughly before deployment


Good luck, and let's build something impactful! 🚀


