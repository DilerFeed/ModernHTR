# 🚀 Installation Guide for Mac M2

## Prerequisites
- macOS 12.3+ (Monterey or later)
- Python 3.8 or higher
- Xcode Command Line Tools

## Step 1: Install Xcode Command Line Tools

```bash
xcode-select --install
```

## Step 2: Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

## Step 3: Install PyTorch with MPS Support

**IMPORTANT**: Install PyTorch with MPS (Metal Performance Shaders) support:

```bash
# Install PyTorch with MPS backend for Apple Silicon
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu
```

Or stable version:

```bash
pip3 install torch torchvision torchaudio
```

## Step 4: Verify MPS is Available

```bash
python3 -c "import torch; print('MPS Available:', torch.backends.mps.is_available())"
```

Expected output: `MPS Available: True`

## Step 5: Install Other Dependencies

```bash
pip install numpy opencv-python Pillow matplotlib seaborn tqdm albumentations editdistance pandas gdown scikit-learn
```

Or use requirements:

```bash
pip install -r requirements.txt
```

## Step 6: Run the Project

```bash
python main.py
```

## Performance Expectations

### Mac M2 with MPS:
- **Batch Size**: 32 (doubled from 16)
- **Workers**: 8 threads
- **Speed**: ~10-20x faster than CPU
- **Training time per epoch**: ~5-10 minutes (vs 1+ hour on CPU)

### Without MPS (CPU only):
- **Batch Size**: 16
- **Workers**: 4 threads
- **Training time per epoch**: ~1-2 hours

## Troubleshooting

### MPS Not Available

If `torch.backends.mps.is_available()` returns `False`:

1. Update macOS to 12.3 or later
2. Reinstall PyTorch:
```bash
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio
```

### Memory Issues

If you get OOM (Out of Memory) errors:

1. Reduce batch size in `config.py`:
```python
BATCH_SIZE = 16  # or even 8
```

2. Reduce number of workers:
```python
NUM_WORKERS = 4  # or 2
```

### Slow Performance

If training is still slow:

1. Check MPS is actually being used:
```python
python -c "import torch; print(torch.device('mps' if torch.backends.mps.is_available() else 'cpu'))"
```

2. Close other applications to free up GPU memory

3. Reduce image size (not recommended):
```python
IMG_WIDTH = 400  # instead of 800
```

## Monitoring GPU Usage

Install `asitop` to monitor M2 GPU usage:

```bash
pip install asitop
sudo asitop
```

This shows real-time GPU utilization during training.

## Expected Training Time

With MPS on M2:
- **1 epoch**: ~5-10 minutes
- **50 epochs**: ~4-8 hours (with early stopping likely less)

Without MPS (CPU):
- **1 epoch**: ~1-2 hours  
- **50 epochs**: ~50-100 hours (not recommended!)

## Notes

- First epoch may be slower due to MPS warmup
- Model compilation happens on first run
- Subsequent epochs should be fast and consistent
- Expected speed: ~200-400 samples/second on M2

## Verification

After installation, verify everything works:

```bash
python -c "
import torch
import cv2
import numpy as np

print('PyTorch version:', torch.__version__)
print('MPS available:', torch.backends.mps.is_available())
print('MPS built:', torch.backends.mps.is_built())
print('OpenCV version:', cv2.__version__)
print('NumPy version:', np.__version__)

if torch.backends.mps.is_available():
    x = torch.randn(1, 3, 224, 224).to('mps')
    print('✅ MPS test successful!')
else:
    print('❌ MPS not available')
"
```

Good luck with training! 🚀
