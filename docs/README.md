# 🖋️ Handwritten Text Recognition (HTR) Project

Deep Learning project for recognizing handwritten English text using CNN + BiLSTM + CTC architecture.

## 🏗️ Architecture

- **CNN Backbone**: Feature extraction from images (4 convolutional blocks)
- **BiLSTM**: Sequence modeling (2 bidirectional LSTM layers)
- **CTC Loss**: Alignment without explicit segmentation

## 📊 Dataset

**IAM Handwriting Database** - English handwritten text
- ~80,000 word images
- 10 train / 10% validation / 10% test split
- Automatic download from Kaggle/Google Drive

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Project

```bash
python main.py
```

This will:
1. Automatically download IAM dataset (if needed)
2. Generate dataset visualizations
3. Create model architecture diagrams
4. Ask if you want to start training

## 📁 Project Structure

