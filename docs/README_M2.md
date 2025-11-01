# 🚀 Mac M2 Optimization Guide

## Current Status

✅ **MPS is working** with CPU fallback for CTC Loss
- CNN layers: **GPU accelerated** 🚀
- LSTM layers: **GPU accelerated** 🚀  
- CTC Loss: **CPU fallback** (minor slowdown)
- Overall speedup: **~5-10x faster than full CPU**

## Why CPU Fallback?

PyTorch's CTC Loss is not yet natively supported on MPS. The fallback automatically uses CPU for CTC Loss calculations while keeping all other operations on GPU.

## Performance Expectations

### With MPS + CPU Fallback:
- **Training speed**: ~30-60 seconds/epoch
- **Total time (50 epochs)**: ~30-50 minutes
- **Speedup vs CPU**: 5-10x faster ✅

### Full CPU (no MPS):
- **Training speed**: ~5-10 minutes/epoch  
- **Total time (50 epochs)**: 4-8 hours ❌

## How to Run

### Option 1: Direct Python

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
python main.py
```

### Option 2: Using Script

```bash
chmod +x RUN_M2.sh
./RUN_M2.sh
```

## Monitoring Performance

During training, you should see:
- **Good**: ~100-200 samples/second
- **Excellent**: ~200-400 samples/second  
- **Slow** (CPU only): ~20-40 samples/second

## Troubleshooting

### If you see warnings about MPS:

