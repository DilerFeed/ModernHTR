import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import json

# Enable MPS fallback for unsupported operations
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

from config import Config
from models.cnn_rnn_ctc import CNN_RNN_CTC
from utils.dataset import IAMDataset, collate_fn
from utils.metrics import calculate_cer, calculate_wer, calculate_accuracy
from utils.visualizations import plot_training_history, visualize_predictions

def train_epoch(model, dataloader, criterion, optimizer, config, epoch, scaler=None):
    """Train for one epoch"""
    model.train()
    
    total_loss = 0
    total_cer = 0
    total_wer = 0
    num_batches = 0
    
    use_amp = config.USE_AMP and scaler is not None
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    
    for batch_idx, (images, targets, target_lengths, texts) in enumerate(progress_bar):
        images = images.to(config.DEVICE)
        targets = targets.to(config.DEVICE)
        target_lengths = target_lengths.to(config.DEVICE)
        
        optimizer.zero_grad()
        
        # Regular forward pass (no AMP issues)
        outputs = model(images)
        
        output_lengths = torch.full(
            size=(images.size(0),),
            fill_value=outputs.size(0),
            dtype=torch.long
        ).to(config.DEVICE)
        
        # Debug first batch
        if batch_idx == 0 and epoch == 1:
            print(f"\n🐛 DEBUG INFO:")
            print(f"   Output shape: {outputs.shape}")  # Should be (T, batch, num_classes)
            print(f"   Output lengths: {output_lengths}")
            print(f"   Target shape: {targets.shape}")
            print(f"   Target lengths: {target_lengths}")
            print(f"   Num classes: {config.NUM_CLASSES}")
            print(f"   Sample target: {targets[0][:target_lengths[0]]}")
            print(f"   Sample text: '{texts[0]}'")
            print(f"   Output min/max: {outputs.min():.4f} / {outputs.max():.4f}")
        
        # CTC Loss
        try:
            loss = criterion(outputs, targets, output_lengths, target_lengths)
            
            if batch_idx == 0 and epoch == 1:
                print(f"   Loss value: {loss.item():.4f}")
                print(f"   Loss is finite: {torch.isfinite(loss).item()}")
            
        except Exception as e:
            print(f"\n❌ CTC Loss Error at batch {batch_idx}:")
            print(f"   Error: {e}")
            print(f"   Output shape: {outputs.shape}")
            print(f"   Target shape: {targets.shape}")
            print(f"   Output lengths: {output_lengths}")
            print(f"   Target lengths: {target_lengths}")
            continue
        
        # Check for NaN/Inf
        if not torch.isfinite(loss):
            print(f"\n⚠️  Non-finite loss at batch {batch_idx}: {loss.item()}")
            continue
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        
        # Metrics
        with torch.no_grad():
            cer = calculate_cer(outputs, texts, config)
            wer = calculate_wer(outputs, texts, config)
        
        total_loss += loss.item()
        total_cer += cer
        total_wer += wer
        num_batches += 1
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'cer': f'{cer:.2f}%',
            'wer': f'{wer:.2f}%'
        })
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    avg_cer = total_cer / num_batches if num_batches > 0 else 0
    avg_wer = total_wer / num_batches if num_batches > 0 else 0
    
    return avg_loss, avg_cer, avg_wer

def validate(model, dataloader, criterion, config, epoch):
    """Validate model"""
    model.eval()
    
    total_loss = 0
    total_cer = 0
    total_wer = 0
    total_acc = 0
    num_batches = 0
    
    use_amp = config.USE_AMP
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]  ")
    
    with torch.no_grad():
        for images, targets, target_lengths, texts in progress_bar:
            images = images.to(config.DEVICE)
            targets = targets.to(config.DEVICE)
            target_lengths = target_lengths.to(config.DEVICE)
            
            # Mixed precision forward pass
            if use_amp:
                with torch.autocast(device_type=config.DEVICE, dtype=torch.float16):
                    outputs = model(images)
                    
                    output_lengths = torch.full(
                        size=(images.size(0),),
                        fill_value=outputs.size(0),
                        dtype=torch.long
                    ).to(config.DEVICE)
                    
                    loss = criterion(outputs, targets, output_lengths, target_lengths)
            else:
                outputs = model(images)
                
                output_lengths = torch.full(
                    size=(images.size(0),),
                    fill_value=outputs.size(0),
                    dtype=torch.long
                ).to(config.DEVICE)
                
                loss = criterion(outputs, targets, output_lengths, target_lengths)
            
            # Metrics
            cer = calculate_cer(outputs, texts, config)
            wer = calculate_wer(outputs, texts, config)
            acc = calculate_accuracy(outputs, texts, config)
            
            total_loss += loss.item()
            total_cer += cer
            total_wer += wer
            total_acc += acc
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'cer': f'{cer:.2f}%',
                'wer': f'{wer:.2f}%',
                'acc': f'{acc:.2f}%'
            })
    
    avg_loss = total_loss / num_batches
    avg_cer = total_cer / num_batches
    avg_wer = total_wer / num_batches
    avg_acc = total_acc / num_batches
    
    return avg_loss, avg_cer, avg_wer, avg_acc

def train_model(train_dataset, val_dataset, config):
    """Main training loop"""
    print("\n" + "=" * 80)
    print("🚀 STARTING TRAINING")
    print("=" * 80)
    
    # Optimize for MPS
    use_mps = config.DEVICE == 'mps'
    if use_mps:
        print("⚡ Apple Silicon GPU (MPS) detected - optimizing for M2...")
        print("   - Increased batch size to 32")
        print("   - Using 8 worker threads")
        print("   - MPS fallback enabled for CTC Loss (uses CPU)")
        print("   - CNN & LSTM still run on GPU (major speedup!)")
        print("   - Mixed Precision DISABLED (incompatible with CTC on MPS)")
        print("   - MPS lazy mode ENABLED")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=False if use_mps else (config.DEVICE == 'cuda'),
        persistent_workers=True if config.NUM_WORKERS > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=False if use_mps else (config.DEVICE == 'cuda'),
        persistent_workers=True if config.NUM_WORKERS > 0 else False
    )
    
    # Create model
    model = CNN_RNN_CTC(config).to(config.DEVICE)
    
    # Loss and optimizer
    criterion = nn.CTCLoss(blank=config.BLANK_IDX, zero_infinity=True)
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    
    # Gradient scaler for mixed precision
    scaler = torch.cuda.amp.GradScaler() if config.USE_AMP else None
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5
    )
    
    # Training history
    history = {
        'train_loss': [],
        'train_cer': [],
        'train_wer': [],
        'val_loss': [],
        'val_cer': [],
        'val_wer': [],
        'val_acc': []
    }
    
    best_val_cer = float('inf')
    patience_counter = 0
    max_patience = 10
    
    print(f"\n📊 Training Configuration:")
    print(f"   Device: {config.DEVICE} ({config.DEVICE_NAME})")
    print(f"   Batch Size: {config.BATCH_SIZE}")
    print(f"   Epochs: {config.EPOCHS}")
    print(f"   Learning Rate: {config.LEARNING_RATE}")
    print(f"   Workers: {config.NUM_WORKERS}")
    print(f"   Mixed Precision: {'ENABLED' if config.USE_AMP else 'DISABLED'}")
    print(f"   Train Samples: {len(train_dataset)}")
    print(f"   Val Samples: {len(val_dataset)}")
    print(f"   Batches per Epoch: {len(train_loader)}")
    print("=" * 80 + "\n")
    
    # Training loop
    for epoch in range(1, config.EPOCHS + 1):
        print(f"\n{'='*80}")
        print(f"EPOCH {epoch}/{config.EPOCHS}")
        print(f"{'='*80}")
        
        # Train
        train_loss, train_cer, train_wer = train_epoch(
            model, train_loader, criterion, optimizer, config, epoch, scaler
        )
        
        # Validate
        val_loss, val_cer, val_wer, val_acc = validate(
            model, val_loader, criterion, config, epoch
        )
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_cer'].append(train_cer)
        history['train_wer'].append(train_wer)
        history['val_loss'].append(val_loss)
        history['val_cer'].append(val_cer)
        history['val_wer'].append(val_wer)
        history['val_acc'].append(val_acc)
        
        # Print epoch summary
        print(f"\n📊 Epoch {epoch} Summary:")
        print(f"   Train Loss: {train_loss:.4f} | CER: {train_cer:.2f}% | WER: {train_wer:.2f}%")
        print(f"   Val   Loss: {val_loss:.4f} | CER: {val_cer:.2f}% | WER: {val_wer:.2f}% | Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_cer < best_val_cer:
            best_val_cer = val_cer
            patience_counter = 0
            
            checkpoint_path = os.path.join(config.MODEL_DIR, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_cer': val_cer,
                'val_loss': val_loss,
                'config': config
            }, checkpoint_path)
            print(f"   ✅ Best model saved! (CER: {val_cer:.2f}%)")
        else:
            patience_counter += 1
            print(f"   ⏳ No improvement. Patience: {patience_counter}/{max_patience}")
        
        # Early stopping
        if patience_counter >= max_patience:
            print(f"\n⚠️  Early stopping triggered after {epoch} epochs")
            break
        
        # Save checkpoint every 5 epochs
        if epoch % 5 == 0:
            checkpoint_path = os.path.join(config.MODEL_DIR, f'checkpoint_epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_cer': val_cer,
                'val_loss': val_loss,
                'config': config
            }, checkpoint_path)
            print(f"   💾 Checkpoint saved")
    
    # Save final model
    final_path = os.path.join(config.MODEL_DIR, 'final_model.pth')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_cer': val_cer,
        'val_loss': val_loss,
        'config': config
    }, final_path)
    
    # Save training history
    history_path = os.path.join(config.OUTPUT_DIR, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4)
    
    print("\n" + "=" * 80)
    print("✅ TRAINING COMPLETED")
    print("=" * 80)
    print(f"Best Validation CER: {best_val_cer:.2f}%")
    print(f"Models saved in: {config.MODEL_DIR}")
    print("=" * 80 + "\n")
    
    print("📊 Creating visualizations...")
    
    # Plot training history
    plot_training_history(history, config)
    
    # Visualize predictions on validation set
    model.eval()
    visualize_predictions(model, val_dataset, config, num_samples=10)
    
    # NEW: Confusion examples
    from utils.visualizations import visualize_confusion_examples, create_final_report
    visualize_confusion_examples(model, val_dataset, config, num_samples=20)
    
    # NEW: Final comprehensive report
    create_final_report(history, model, val_dataset, config)
    
    print("\n✅ All visualizations created!")
    print(f"📁 Check: {config.VIS_DIR}")
    
    return model, history
