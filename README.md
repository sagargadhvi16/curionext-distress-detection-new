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

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
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

## 📦 Tech Stack

- **ML:** PyTorch, TensorFlow Hub (YAMNet)
- **Audio:** librosa, soundfile, audiomentations
- **Biometric:** neurokit2, hrv-analysis
- **Backend:** FastAPI, uvicorn
- **XAI:** SHAP
- **Testing:** pytest

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
