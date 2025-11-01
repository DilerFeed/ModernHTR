# 🖋️ ModernHTR

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Apple Silicon](https://img.shields.io/badge/Apple%20Silicon-Optimized-silver.svg)](https://developer.apple.com/metal/)

**Modern, production-ready Handwritten Text Recognition (HTR) system built with PyTorch.**

A complete, well-documented implementation of CNN+BiLSTM+CTC architecture for recognizing handwritten English text. Unlike older implementations, ModernHTR features automatic dataset downloading, comprehensive visualizations, and optimizations for Apple Silicon (M1/M2/M3).

---

## 🌟 Why ModernHTR?

### Advantages over existing solutions:

| Feature | ModernHTR | SimpleHTR | CRNN | Other |
|---------|-----------|-----------|------|-------|
| **Auto Dataset Download** | ✅ | ❌ | ❌ | ❌ |
| **Apple Silicon Optimization** | ✅ M1–M4 | ❌ | ❌ | ❌ |
| **Comprehensive Visualizations** | ✅ 15+ plots | ⚠️ Basic | ❌ | ⚠️ Basic |
| **Modern PyTorch (2.0+)** | ✅ | ❌ 1.x | ❌ Old | ⚠️ |
| **Production Ready** | ✅ | ❌ | ⚠️ | ⚠️ |
| **Well Documented** | ✅ | ⚠️ | ⚠️ | ⚠️ |
| **Active Maintenance** | ✅ 2025 | ❌ 2019 | ❌ 2017 | ⚠️ |

---

## 🏗️ Architecture

### CNN + BiLSTM + CTC

```
Input (64×800 grayscale)
    ↓
[CNN Backbone - Feature Extraction]
    Conv Block 1: 32 filters  → 32×400
    Conv Block 2: 64 filters  → 16×200
    Conv Block 3: 128 filters → 8×200
    Conv Block 4: 256 filters → 4×200
    ↓
[Reshape] → Sequence: 200 timesteps × 1024 features
    ↓
[BiLSTM - Sequence Modeling]
    2 layers, 256 hidden units
    Bidirectional (512 total)
    ↓
[Dense Layer] → 77 classes (characters + blank)
    ↓
[CTC Loss - Alignment-free Training]
    ↓
Output: Character sequence
```

**Why this architecture?**
- **CNN**: Robust feature extraction from images
- **BiLSTM**: Captures both left and right context
- **CTC**: No need for character-level annotations
- **Proven**: Used in production OCR systems

---

## 📊 Detailed Results

### Performance by Word Length

| Length | Samples | Accuracy | CER | WER |
|--------|---------|----------|-----|-----|
| 1-3 chars | ~5,000 | 75-85% | 10-15% | 15-25% |
| 4-6 chars | ~15,000 | 65-75% | 12-18% | 25-35% |
| 7-9 chars | ~12,000 | 60-70% | 15-22% | 30-40% |
| 10-12 chars | ~4,000 | 50-60% | 20-30% | 40-50% |
| 13+ chars | ~2,000 | 40-50% | 30-40% | 50-60% |

### Training Progression

| Epoch | Train Loss | Val CER | Val Acc |
|-------|------------|---------|---------|
| 1 | 3.87 | 83.93% | 12.78% |
| 10 | 1.24 | 35.42% | 48.23% |
| 20 | 0.68 | 20.15% | 58.91% |
| 30 | 0.51 | 16.34% | 62.45% |
| **44** | **0.42** | **14.60%** | **64.91%** |

---

## 🔧 Advanced Usage

### Custom Training

```python
from config import Config
from train import train_model

# Modify hyperparameters
config = Config()
config.BATCH_SIZE = 64
config.LEARNING_RATE = 0.0005
config.EPOCHS = 100

# Train
model, history = train_model(train_dataset, val_dataset, config)
```

### Inference on Custom Images

```python
import torch
from models.cnn_rnn_ctc import CNN_RNN_CTC
from utils.metrics import ctc_decode

# Load model
config = Config()
model = CNN_RNN_CTC(config).to(config.DEVICE)
checkpoint = torch.load('outputs/models/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# Predict
image = load_and_preprocess_image('path/to/image.png')
output = model(image.unsqueeze(0))
text = ctc_decode(output, config)[0]
print(f"Predicted: {text}")
```

### Generate All Visualizations

```bash
# After training, generate comprehensive visualizations
python test_and_visualize.py

# Generate CSV tables
python generate_tables.py

# Generate architecture diagrams
python visualize_architecture.py
```

---

## 📈 Monitoring Training

### Real-time Progress

```
================================================================================
EPOCH 27/50
================================================================================
Epoch 27 [Train]: 100%|█████| 958/958 [06:45<00:00, 2.36it/s]
Epoch 27 [Val]:   100%|█████| 120/120 [00:17<00:00, 7.04it/s]

📊 Epoch 27 Summary:
   Train Loss: 0.5253 | CER: 18.14% | WER: 41.20%
   Val   Loss: 0.5101 | CER: 17.07% | WER: 39.20% | Acc: 60.80%
   ✅ Best model saved! (CER: 17.07%)
```

---

## 🍎 Apple Silicon Optimization

### Performance Gains

| Device | Speed | Time/Epoch | Total (50 epochs) |
|--------|-------|------------|-------------------|
| **M2 MacBook (MPS)** | 2.5 it/s | 6-7 min | ~6 hours |
| Intel Mac (CPU) | 0.2 it/s | 50-80 min | ~50-60 hours |
| NVIDIA RTX 3080 | 8-10 it/s | 1.5-2 min | ~2 hours |

### Why MPS?

- **5-10x faster** than CPU on M1/M2/M3
- **Native support** for Apple Silicon
- **Energy efficient** - doesn't drain battery
- **No CUDA required** - works out of the box

### Setup for Mac

See detailed guide: [docs/INSTALLATION_M2.md](docs/INSTALLATION_M2.md)

---

## 📚 Dataset

### IAM Handwriting Database

- **Size**: 38,305 word images
- **Writers**: 657 different people
- **Source**: Forms, letters, and text passages
- **Format**: Grayscale PNG images
- **License**: Free for academic use

### Automatic Download

ModernHTR automatically downloads the dataset from:
1. ✅ Kaggle (primary source)
2. ✅ Google Drive (backup)
3. ⚠️ Official IAM (if available)

**No manual download needed!** Just run `python main.py`.

---

## 🧪 Testing

### Run Full Evaluation

```bash
# Test on all datasets (train/val/test)
python test_and_visualize.py

# Generate analysis tables
python generate_tables.py
```

### Outputs

**JSON Results:**
```json
{
  "test": {
    "cer": 14.60,
    "wer": 35.09,
    "acc": 64.91,
    "samples": 3831
  }
}
```

**CSV Tables** (7 files):
- Overall performance metrics
- Training progress by epoch
- Model architecture details
- Training configuration
- Comparison with baselines
- Dataset statistics
- Performance by word length

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Development Setup

```bash
# Clone your fork
git clone https://github.com/yourusername/ModernHTR.git
cd ModernHTR

# Create branch
git checkout -b feature/your-feature

# Make changes and test
python main.py

# Submit PR
git push origin feature/your-feature
```

### Areas for Contribution

- [ ] Add more datasets (RIMES, CVL, etc.)
- [ ] Implement attention mechanism
- [ ] Add transformer-based architecture
- [ ] Create Docker container
- [ ] Add ONNX export for deployment
- [ ] Improve data augmentation
- [ ] Add multi-language support

---

## 📄 Citation

If you use ModernHTR in your research, please cite:

```bibtex
@software{modernhtr2024,
  title={ModernHTR: Modern Handwritten Text Recognition with PyTorch},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/ModernHTR}
}
```

---

## 🙏 Acknowledgments

- **IAM Database**: Marti & Bunke, University of Bern
- **PyTorch Team**: For amazing deep learning framework
- **Apple**: For Metal Performance Shaders (MPS)
- **Community**: All the amazing open-source contributors

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🔗 Links

- **Documentation**: [docs/](docs/)
- **Issues**: [GitHub Issues](https://github.com/yourusername/ModernHTR/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/ModernHTR/discussions)

---

**Made with ❤️ using PyTorch**

*Modern, Fast, Production-Ready*

