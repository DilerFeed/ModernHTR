import torch
import numpy as np
import random
import warnings
import os
warnings.filterwarnings('ignore')

# Enable MPS fallback for CTC Loss
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# Optimize MPS
if torch.backends.mps.is_available():
    torch.backends.mps.flush_on_exit = False

from config import Config
from utils.data_loader import load_iam_dataset, visualize_dataset_stats
from utils.dataset import IAMDataset
from utils.visualizations import (
    visualize_samples, 
    visualize_augmentations,
    visualize_model_architecture,
    visualize_feature_maps
)
from models.cnn_rnn_ctc import CNN_RNN_CTC
from train import train_model

def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def main():
    """Main function"""
    print("\n" + "=" * 80)
    print("🖋️  HANDWRITTEN TEXT RECOGNITION (HTR) PROJECT")
    print("=" * 80)
    print("Architecture: CNN + BiLSTM + CTC")
    print("Dataset: IAM Handwriting Database")
    print("=" * 80 + "\n")
    
    # Initialize config INSTANCE
    config = Config()
    
    # Set random seed
    set_seed(config.SEED)
    
    print(f"⚙️  Configuration:")
    print(f"   Device: {config.DEVICE} ({config.DEVICE_NAME})")
    print(f"   Image Size: {config.IMG_HEIGHT}x{config.IMG_WIDTH}")
    print(f"   Batch Size: {config.BATCH_SIZE}")
    print(f"   Epochs: {config.EPOCHS}")
    print(f"   Learning Rate: {config.LEARNING_RATE}")
    print(f"   Workers: {config.NUM_WORKERS}")
    
    if config.DEVICE == 'mps':
        print(f"\n   ⚡ Apple Silicon GPU Acceleration ENABLED")
        print(f"   🚀 CNN & LSTM run on GPU")
        print(f"   ⚙️  CTC Loss uses CPU fallback")
        print(f"   ⚠️  Mixed Precision disabled (CTC compatibility)")
        print(f"   📈 Overall: ~5-8x faster than full CPU!")
    
    print()
    
    # Load dataset
    train_samples, val_samples, test_samples = load_iam_dataset(config)
    
    if train_samples is None:
        print("\n❌ Failed to load dataset. Exiting...")
        return
    
    # Visualize dataset statistics
    visualize_dataset_stats(train_samples, val_samples, test_samples, config)
    
    # Create datasets
    print("\n📦 Creating PyTorch datasets...")
    train_dataset = IAMDataset(train_samples, config, augment=True)
    val_dataset = IAMDataset(val_samples, config, augment=False)
    test_dataset = IAMDataset(test_samples, config, augment=False)
    
    print(f"✅ Train dataset: {len(train_dataset)} samples")
    print(f"✅ Val dataset: {len(val_dataset)} samples")
    print(f"✅ Test dataset: {len(test_dataset)} samples")
    
    # Visualize sample images
    visualize_samples(train_dataset, config, num_samples=16)
    
    # Visualize augmentations
    visualize_augmentations(train_dataset, config, num_samples=8)
    
    # Create model
    print("\n🏗️  Building model...")
    model = CNN_RNN_CTC(config).to(config.DEVICE)
    
    # Visualize model architecture
    visualize_model_architecture(model, config)
    
    # Visualize feature maps
    visualize_feature_maps(model, train_dataset, config, sample_idx=0)
    
    print("\n" + "=" * 80)
    print("📊 ALL VISUALIZATIONS CREATED!")
    print("=" * 80)
    print(f"Check: {config.VIS_DIR}")
    print("=" * 80)
    
    # Train model
    print("\n🎯 Ready to start training!")
    response = input("Start training? (y/n): ")
    
    if response.lower() == 'y':
        model, history = train_model(train_dataset, val_dataset, config)
        
        print("\n" + "=" * 80)
        print("🎉 PROJECT COMPLETED!")
        print("=" * 80)
        print(f"📁 Models: {config.MODEL_DIR}")
        print(f"📊 Visualizations: {config.VIS_DIR}")
        print(f"📈 Logs: {config.LOGS_DIR}")
        print("=" * 80 + "\n")
    else:
        print("\n⏸️  Training skipped. Visualizations are ready!")
        print(f"📊 Check visualizations in: {config.VIS_DIR}")

if __name__ == "__main__":
    main()
