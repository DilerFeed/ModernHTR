import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import cv2

def visualize_samples(dataset, config, num_samples=16):
    """Visualize random samples from dataset"""
    print("🖼️  Creating sample visualizations...")
    
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    rows = 4
    cols = 4
    fig, axes = plt.subplots(rows, cols, figsize=(20, 12))
    axes = axes.flatten()
    
    for idx, sample_idx in enumerate(indices):
        if idx >= len(axes):
            break
            
        img, _, text = dataset[sample_idx]
        
        # Convert tensor to numpy
        img_np = img.squeeze().numpy()
        
        axes[idx].imshow(img_np, cmap='gray')
        axes[idx].set_title(f"Text: '{text}'", fontsize=10)
        axes[idx].axis('off')
    
    # Hide unused subplots
    for idx in range(len(indices), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.VIS_DIR, 'sample_images.png'), dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {os.path.join(config.VIS_DIR, 'sample_images.png')}")
    plt.close()

def visualize_augmentations(dataset, config, num_samples=8):
    """Visualize augmentation effects"""
    print("🔄 Creating augmentation visualizations...")
    
    # Temporarily enable augmentation
    original_augment = dataset.augment
    dataset.augment = True
    
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, num_samples * 2))
    
    for row, sample_idx in enumerate(indices):
        # Get original
        dataset.augment = False
        img_orig, _, text = dataset[sample_idx]
        
        # Get two augmented versions
        dataset.augment = True
        img_aug1, _, _ = dataset[sample_idx]
        img_aug2, _, _ = dataset[sample_idx]
        
        # Plot
        axes[row, 0].imshow(img_orig.squeeze().numpy(), cmap='gray')
        axes[row, 0].set_title('Original' if row == 0 else '')
        axes[row, 0].axis('off')
        
        axes[row, 1].imshow(img_aug1.squeeze().numpy(), cmap='gray')
        axes[row, 1].set_title('Augmented 1' if row == 0 else '')
        axes[row, 1].axis('off')
        
        axes[row, 2].imshow(img_aug2.squeeze().numpy(), cmap='gray')
        axes[row, 2].set_title('Augmented 2' if row == 0 else '')
        axes[row, 2].axis('off')
        
        axes[row, 0].set_ylabel(f"'{text}'", rotation=0, labelpad=40, va='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.VIS_DIR, 'augmentations.png'), dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {os.path.join(config.VIS_DIR, 'augmentations.png')}")
    plt.close()
    
    # Restore original augmentation setting
    dataset.augment = original_augment

def visualize_model_architecture(model, config):
    """Visualize model architecture summary"""
    print("🏗️  Creating model architecture visualization...")
    
    from models.cnn_rnn_ctc import count_parameters
    total_params, trainable_params = count_parameters(model)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.axis('off')
    
    architecture_text = f"""
    🏗️  MODEL ARCHITECTURE: CNN + BiLSTM + CTC
    
    {'=' * 70}
    INPUT LAYER
    {'=' * 70}
    Shape: (batch, 1, {config.IMG_HEIGHT}, {config.IMG_WIDTH})
    Description: Grayscale handwritten text images
    
    {'=' * 70}
    CNN BACKBONE - Feature Extraction
    {'=' * 70}
    Conv Block 1:  Conv2d(1 → 32)   + BatchNorm + ReLU + MaxPool
                   Output: ({config.IMG_HEIGHT//2}, {config.IMG_WIDTH//2})
    
    Conv Block 2:  Conv2d(32 → 64)  + BatchNorm + ReLU + MaxPool
                   Output: ({config.IMG_HEIGHT//4}, {config.IMG_WIDTH//4})
    
    Conv Block 3:  Conv2d(64 → 128) + BatchNorm + ReLU + MaxPool
                   Output: ({config.IMG_HEIGHT//8}, {config.IMG_WIDTH//4})
    
    Conv Block 4:  Conv2d(128 → 256) + BatchNorm + ReLU + MaxPool
                   Output: ({config.IMG_HEIGHT//16}, {config.IMG_WIDTH//4})
    
    Feature Size: {model.cnn_output_size} per timestep
    Sequence Length: {model.feature_width} timesteps
    
    {'=' * 70}
    RNN LAYERS - Sequence Modeling
    {'=' * 70}
    BiLSTM:        Input: {model.cnn_output_size}
                   Hidden: {config.LSTM_HIDDEN} (x2 for bidirectional)
                   Layers: {config.LSTM_LAYERS}
                   Dropout: {config.DROPOUT}
                   Output: {config.LSTM_HIDDEN * 2}
    
    {'=' * 70}
    OUTPUT LAYER - Character Prediction
    {'=' * 70}
    Linear Layer:  Input: {config.LSTM_HIDDEN * 2}
                   Output: {config.NUM_CLASSES} (characters + blank)
    
    CTC Loss:      Alignment without segmentation
    
    {'=' * 70}
    MODEL STATISTICS
    {'=' * 70}
    Total Parameters: {total_params:,}
    Trainable Parameters: {trainable_params:,}
    Model Size: ~{total_params * 4 / (1024**2):.2f} MB (float32)
    
    {'=' * 70}
    """
    
    ax.text(0.05, 0.95, architecture_text, fontsize=9, 
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.VIS_DIR, 'model_architecture.png'), dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {os.path.join(config.VIS_DIR, 'model_architecture.png')}")
    plt.close()
    
    print(f"\n📊 Model Summary:")
    print(f"   Total Parameters: {total_params:,}")
    print(f"   Trainable Parameters: {trainable_params:,}")

def plot_training_history(history, config):
    """Plot training and validation metrics"""
    print("📈 Creating training history plots...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss plot
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # CER plot
    axes[1].plot(epochs, history['train_cer'], 'b-', label='Train CER', linewidth=2)
    axes[1].plot(epochs, history['val_cer'], 'r-', label='Val CER', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Character Error Rate (%)')
    axes[1].set_title('Character Error Rate')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # WER plot  
    axes[2].plot(epochs, history['train_wer'], 'b-', label='Train WER', linewidth=2)
    axes[2].plot(epochs, history['val_wer'], 'r-', label='Val WER', linewidth=2)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Word Error Rate (%)')
    axes[2].set_title('Word Error Rate')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.VIS_DIR, 'training_history.png'), dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {os.path.join(config.VIS_DIR, 'training_history.png')}")
    plt.close()

def visualize_predictions(model, dataset, config, num_samples=10):
    """Visualize model predictions"""
    print("🔮 Creating prediction visualizations...")
    
    model.eval()
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    fig, axes = plt.subplots(num_samples, 1, figsize=(15, num_samples * 2))
    if num_samples == 1:
        axes = [axes]
    
    with torch.no_grad():
        for idx, sample_idx in enumerate(indices):
            img, _, true_text = dataset[sample_idx]
            
            # Get prediction
            img_batch = img.unsqueeze(0).to(config.DEVICE)
            output = model(img_batch)
            
            # Decode
            from utils.metrics import ctc_decode
            pred_text = ctc_decode(output, config)[0]
            
            # Plot
            img_np = img.squeeze().numpy()
            axes[idx].imshow(img_np, cmap='gray', aspect='auto')
            axes[idx].set_title(f"True: '{true_text}' | Predicted: '{pred_text}'", 
                              fontsize=10, pad=10)
            axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.VIS_DIR, 'predictions.png'), dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {os.path.join(config.VIS_DIR, 'predictions.png')}")
    plt.close()

def visualize_feature_maps(model, dataset, config, sample_idx=0):
    """Visualize CNN feature maps"""
    print("🗺️  Creating feature map visualizations...")
    
    model.eval()
    img, _, text = dataset[sample_idx]
    
    with torch.no_grad():
        img_batch = img.unsqueeze(0).to(config.DEVICE)
        features = model.get_feature_maps(img_batch)
    
    fig = plt.figure(figsize=(20, 12))
    
    # Original image
    ax = plt.subplot(5, 1, 1)
    ax.imshow(img.squeeze().numpy(), cmap='gray')
    ax.set_title(f"Input Image: '{text}'", fontsize=12, fontweight='bold')
    ax.axis('off')
    
    # Feature maps from each conv block
    conv_blocks = ['conv1', 'conv2', 'conv3', 'conv4']
    for i, block_name in enumerate(conv_blocks):
        feature_map = features[block_name][0].cpu()  # (channels, H, W)
        
        # Show first 8 channels
        num_channels = min(8, feature_map.shape[0])
        
        for j in range(num_channels):
            ax = plt.subplot(5, 8, (i+1)*8 + j + 1)
            ax.imshow(feature_map[j].numpy(), cmap='viridis')
            if j == 0:
                ax.set_ylabel(f'{block_name}', fontsize=10, fontweight='bold')
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.VIS_DIR, 'feature_maps.png'), dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {os.path.join(config.VIS_DIR, 'feature_maps.png')}")
    plt.close()

def visualize_confusion_examples(model, dataset, config, num_samples=20):
    """Visualize best and worst predictions"""
    print("🎯 Creating confusion examples visualization...")
    
    model.eval()
    
    results = []
    indices = np.random.choice(len(dataset), min(100, len(dataset)), replace=False)
    
    with torch.no_grad():
        for idx in indices:
            img, _, true_text = dataset[idx]
            img_batch = img.unsqueeze(0).to(config.DEVICE)
            output = model(img_batch)
            
            from utils.metrics import ctc_decode
            pred_text = ctc_decode(output, config)[0]
            
            # Calculate character error
            import editdistance
            distance = editdistance.eval(pred_text, true_text)
            error_rate = distance / max(len(true_text), 1)
            
            results.append((idx, img, true_text, pred_text, error_rate))
    
    # Sort by error rate
    results.sort(key=lambda x: x[4])
    
    # Get best and worst
    best_results = results[:num_samples//2]
    worst_results = results[-num_samples//2:]
    
    fig, axes = plt.subplots(num_samples, 1, figsize=(15, num_samples * 1.5))
    
    for i in range(num_samples//2):
        # Best predictions
        _, img, true_text, pred_text, error = best_results[i]
        axes[i].imshow(img.squeeze().numpy(), cmap='gray', aspect='auto')
        axes[i].set_title(f"✅ BEST #{i+1} | True: '{true_text}' | Pred: '{pred_text}' | Error: {error:.2%}", 
                         fontsize=9, color='green')
        axes[i].axis('off')
        
        # Worst predictions
        _, img, true_text, pred_text, error = worst_results[i]
        axes[num_samples//2 + i].imshow(img.squeeze().numpy(), cmap='gray', aspect='auto')
        axes[num_samples//2 + i].set_title(f"❌ WORST #{i+1} | True: '{true_text}' | Pred: '{pred_text}' | Error: {error:.2%}", 
                                          fontsize=9, color='red')
        axes[num_samples//2 + i].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.VIS_DIR, 'confusion_examples.png'), dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {os.path.join(config.VIS_DIR, 'confusion_examples.png')}")
    plt.close()

def create_final_report(history, model, val_dataset, config):
    """Create comprehensive final report"""
    print("📄 Creating final report...")
    
    from models.cnn_rnn_ctc import count_parameters
    total_params, _ = count_parameters(model)
    
    fig = plt.figure(figsize=(16, 10))
    
    # Main title
    fig.suptitle('🖋️ HTR Project - Final Report', fontsize=20, fontweight='bold', y=0.98)
    
    # 1. Training curves (top row)
    ax1 = plt.subplot(2, 3, 1)
    epochs = range(1, len(history['train_loss']) + 1)
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Validation', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(epochs, history['train_cer'], 'b-', label='Train', linewidth=2)
    ax2.plot(epochs, history['val_cer'], 'r-', label='Validation', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('CER (%)')
    ax2.set_title('Character Error Rate')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(epochs, history['val_acc'], 'g-', linewidth=2)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Accuracy (%)')
    ax3.set_title('Validation Accuracy')
    ax3.grid(True, alpha=0.3)
    
    # 2. Final metrics (middle left)
    ax4 = plt.subplot(2, 3, 4)
    ax4.axis('off')
    
    best_epoch = np.argmin(history['val_cer']) + 1
    best_cer = min(history['val_cer'])
    final_cer = history['val_cer'][-1]
    final_wer = history['val_wer'][-1]
    final_acc = history['val_acc'][-1]
    
    metrics_text = f"""
    📊 FINAL METRICS
    
    Best Epoch: {best_epoch}
    Best CER: {best_cer:.2f}%
    
    Final Results (Epoch {len(epochs)}):
    ├─ CER: {final_cer:.2f}%
    ├─ WER: {final_wer:.2f}%
    └─ Accuracy: {final_acc:.2f}%
    
    Total Epochs: {len(epochs)}
    Training Time: ~{len(epochs) * 7:.0f} minutes
    """
    
    ax4.text(0.1, 0.5, metrics_text, fontsize=11, 
            verticalalignment='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    # 3. Model info (middle center)
    ax5 = plt.subplot(2, 3, 5)
    ax5.axis('off')
    
    model_text = f"""
    🏗️  MODEL INFO
    
    Architecture:
    ├─ CNN: 4 blocks
    ├─ BiLSTM: {config.LSTM_LAYERS} layers
    └─ Output: {config.NUM_CLASSES} classes
    
    Parameters:
    └─ Total: {total_params:,}
    
    Training:
    ├─ Batch Size: {config.BATCH_SIZE}
    ├─ Learning Rate: {config.LEARNING_RATE}
    └─ Device: {config.DEVICE_NAME}
    """
    
    ax5.text(0.1, 0.5, model_text, fontsize=11,
            verticalalignment='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    # 4. Dataset info (middle right)
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    dataset_text = f"""
    📚 DATASET INFO
    
    Total Samples:
    ├─ Train: {len(val_dataset) * 8}
    ├─ Val: {len(val_dataset)}
    └─ Test: {len(val_dataset)}
    
    Characters: {config.NUM_CLASSES - 1}
    
    Augmentation:
    ├─ Rotation
    ├─ Noise
    ├─ Blur
    └─ Elastic Transform
    """
    
    ax6.text(0.1, 0.5, dataset_text, fontsize=11,
            verticalalignment='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.VIS_DIR, 'final_report.png'), dpi=200, bbox_inches='tight')
    print(f"✅ Saved: {os.path.join(config.VIS_DIR, 'final_report.png')}")
    plt.close()
