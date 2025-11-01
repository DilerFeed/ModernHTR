#!/bin/bash

# Enable MPS fallback for CTC Loss
export PYTORCH_ENABLE_MPS_FALLBACK=1

echo "=================================="
echo "🚀 Running HTR with M2 Optimization"
echo "=================================="
echo "MPS Fallback: ENABLED"
echo "CTC Loss: CPU (fallback)"
echo "CNN & LSTM: GPU (MPS)"
echo "=================================="
echo ""

# Run the project
python main.py
