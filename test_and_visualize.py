import torch
import numpy as np
import os
import json
from tqdm import tqdm

from config import Config
from models.cnn_rnn_ctc import CNN_RNN_CTC
from utils.dataset import IAMDataset, collate_fn
from utils.data_loader import load_iam_dataset
from utils.metrics import calculate_cer, calculate_wer, calculate_accuracy, ctc_decode
from utils.visualizations import (
    visualize_predictions, 
    visualize_confusion_examples,
    create_final_report
)
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import editdistance

def load_best_model(config):
    """Load best trained model"""
    print("🔄 Loading best model...")
    
    checkpoint_path = os.path.join(config.MODEL_DIR, 'best_model.pth')
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Model not found: {checkpoint_path}")
        return None, None
    
    # Load checkpoint with weights_only=False to allow loading Config object
    checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE, weights_only=False)
    
    # Create model
    model = CNN_RNN_CTC(config).to(config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ Model loaded from epoch {checkpoint['epoch']}")
    print(f"   Best Val CER: {checkpoint['val_cer']:.2f}%")
    
    return model, checkpoint

def test_on_dataset(model, dataset, config, dataset_name="Test"):
    """Test model on entire dataset"""
    print(f"\n🧪 Testing on {dataset_name} dataset...")
    
    dataloader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        collate_fn=collate_fn
    )
    
    all_predictions = []
    all_targets = []
    all_images = []
    
    total_cer = 0
    total_wer = 0
    total_acc = 0
    num_batches = 0
    
    model.eval()
    
    with torch.no_grad():
        for images, targets, target_lengths, texts in tqdm(dataloader, desc=f"Testing {dataset_name}"):
            images = images.to(config.DEVICE)
            
            # Forward pass
            outputs = model(images)
            
            # Decode predictions
            predictions = ctc_decode(outputs, config)
            
            # Calculate metrics
            cer = calculate_cer(outputs, texts, config)
            wer = calculate_wer(outputs, texts, config)
            acc = calculate_accuracy(outputs, texts, config)
            
            total_cer += cer
            total_wer += wer
            total_acc += acc
            num_batches += 1
            
            # Store for detailed analysis
            for i, (pred, target, img) in enumerate(zip(predictions, texts, images)):
                all_predictions.append(pred)
                all_targets.append(target)
                all_images.append(img.cpu())
    
    # Calculate averages
    avg_cer = total_cer / num_batches
    avg_wer = total_wer / num_batches
    avg_acc = total_acc / num_batches
    
    print(f"\n📊 {dataset_name} Results:")
    print(f"   CER: {avg_cer:.2f}%")
    print(f"   WER: {avg_wer:.2f}%")
    print(f"   Accuracy: {avg_acc:.2f}%")
    print(f"   Total samples: {len(all_predictions)}")
    
    return {
        'predictions': all_predictions,
        'targets': all_targets,
        'images': all_images,
        'cer': avg_cer,
        'wer': avg_wer,
        'acc': avg_acc
    }

def create_detailed_error_analysis(results, config, dataset_name="Test"):
    """Create detailed error analysis visualization"""
    print(f"\n📈 Creating detailed error analysis for {dataset_name}...")
    
    predictions = results['predictions']
    targets = results['targets']
    
    # Calculate per-sample errors
    errors = []
    for pred, target in zip(predictions, targets):
        distance = editdistance.eval(pred, target)
        error_rate = distance / max(len(target), 1)
        errors.append({
            'pred': pred,
            'target': target,
            'distance': distance,
            'error_rate': error_rate,
            'correct': pred == target
        })
    
    # Statistics
    total_correct = sum(1 for e in errors if e['correct'])
    accuracy = (total_correct / len(errors)) * 100
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # 1. Error rate distribution
    error_rates = [e['error_rate'] * 100 for e in errors]
    axes[0, 0].hist(error_rates, bins=50, color='steelblue', edgecolor='black')
    axes[0, 0].axvline(np.mean(error_rates), color='red', linestyle='--', 
                       linewidth=2, label=f'Mean: {np.mean(error_rates):.2f}%')
    axes[0, 0].set_xlabel('Error Rate (%)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title(f'{dataset_name} Set - Error Rate Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Word length vs error
    word_lengths = [len(e['target']) for e in errors]
    axes[0, 1].scatter(word_lengths, error_rates, alpha=0.3, s=10)
    axes[0, 1].set_xlabel('Word Length (characters)')
    axes[0, 1].set_ylabel('Error Rate (%)')
    axes[0, 1].set_title('Word Length vs Error Rate')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Accuracy breakdown
    length_bins = [0, 3, 6, 9, 12, 100]
    length_labels = ['1-3', '4-6', '7-9', '10-12', '13+']
    binned_accuracy = []
    
    for i in range(len(length_bins) - 1):
        bin_errors = [e for e in errors 
                      if length_bins[i] < len(e['target']) <= length_bins[i+1]]
        if bin_errors:
            bin_acc = (sum(1 for e in bin_errors if e['correct']) / len(bin_errors)) * 100
            binned_accuracy.append(bin_acc)
        else:
            binned_accuracy.append(0)
    
    axes[1, 0].bar(length_labels, binned_accuracy, color='lightgreen', edgecolor='black')
    axes[1, 0].set_xlabel('Word Length Range')
    axes[1, 0].set_ylabel('Accuracy (%)')
    axes[1, 0].set_title('Accuracy by Word Length')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    for i, v in enumerate(binned_accuracy):
        axes[1, 0].text(i, v + 2, f'{v:.1f}%', ha='center', fontweight='bold')
    
    # 4. Summary statistics
    axes[1, 1].axis('off')
    
    summary_text = f"""
    📊 {dataset_name.upper()} SET STATISTICS
    
    {'='*40}
    Overall Performance:
    ├─ Total Samples: {len(errors):,}
    ├─ Correct: {total_correct:,} ({accuracy:.2f}%)
    ├─ Incorrect: {len(errors) - total_correct:,}
    ├─ CER: {results['cer']:.2f}%
    └─ WER: {results['wer']:.2f}%
    
    {'='*40}
    Error Analysis:
    ├─ Mean Error Rate: {np.mean(error_rates):.2f}%
    ├─ Median Error Rate: {np.median(error_rates):.2f}%
    ├─ Std Dev: {np.std(error_rates):.2f}%
    ├─ Min Error: {np.min(error_rates):.2f}%
    └─ Max Error: {np.max(error_rates):.2f}%
    
    {'='*40}
    Word Length:
    ├─ Mean Length: {np.mean(word_lengths):.1f}
    ├─ Median Length: {np.median(word_lengths):.0f}
    └─ Max Length: {max(word_lengths)}
    """
    
    axes[1, 1].text(0.1, 0.5, summary_text, fontsize=10,
                   verticalalignment='center', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.VIS_DIR, f'{dataset_name.lower()}_error_analysis.png'), 
                dpi=200, bbox_inches='tight')
    print(f"✅ Saved: {dataset_name.lower()}_error_analysis.png")
    plt.close()
    
    return errors

def create_confusion_matrix_characters(results, config):
    """Create character-level confusion analysis"""
    print("\n🔤 Creating character confusion analysis...")
    
    from collections import defaultdict
    
    predictions = results['predictions']
    targets = results['targets']
    
    # Collect character confusions
    confusions = defaultdict(int)
    correct_chars = defaultdict(int)
    total_chars = defaultdict(int)
    
    for pred, target in zip(predictions, targets):
        # Align strings for character comparison
        max_len = max(len(pred), len(target))
        pred_padded = pred + ' ' * (max_len - len(pred))
        target_padded = target + ' ' * (max_len - len(target))
        
        for p, t in zip(pred_padded, target_padded):
            if t != ' ':
                total_chars[t] += 1
                if p == t:
                    correct_chars[t] += 1
                elif p != ' ':
                    confusions[(t, p)] += 1
    
    # Get top confusions
    top_confusions = sorted(confusions.items(), key=lambda x: x[1], reverse=True)[:20]
    
    # Character accuracy
    char_accuracy = {}
    for char in total_chars:
        char_accuracy[char] = (correct_chars.get(char, 0) / total_chars[char]) * 100
    
    # Sort by frequency
    top_chars = sorted(total_chars.items(), key=lambda x: x[1], reverse=True)[:20]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1. Most common character confusions
    if top_confusions:
        conf_labels = [f"'{t}'→'{p}'" for (t, p), count in top_confusions]
        conf_counts = [count for (t, p), count in top_confusions]
        
        axes[0].barh(range(len(conf_labels)), conf_counts, color='salmon')
        axes[0].set_yticks(range(len(conf_labels)))
        axes[0].set_yticklabels(conf_labels)
        axes[0].set_xlabel('Frequency')
        axes[0].set_title('Top 20 Character Confusions')
        axes[0].invert_yaxis()
        axes[0].grid(True, alpha=0.3, axis='x')
    
    # 2. Character-wise accuracy
    char_labels = [f"'{char}'" for char, count in top_chars]
    char_accs = [char_accuracy[char] for char, count in top_chars]
    
    colors = ['green' if acc > 90 else 'orange' if acc > 70 else 'red' for acc in char_accs]
    axes[1].barh(range(len(char_labels)), char_accs, color=colors)
    axes[1].set_yticks(range(len(char_labels)))
    axes[1].set_yticklabels(char_labels)
    axes[1].set_xlabel('Accuracy (%)')
    axes[1].set_title('Top 20 Most Frequent Characters - Accuracy')
    axes[1].invert_yaxis()
    axes[1].grid(True, alpha=0.3, axis='x')
    axes[1].axvline(90, color='green', linestyle='--', alpha=0.5, label='90%')
    axes[1].axvline(70, color='orange', linestyle='--', alpha=0.5, label='70%')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.VIS_DIR, 'character_confusion_analysis.png'), 
                dpi=200, bbox_inches='tight')
    print(f"✅ Saved: character_confusion_analysis.png")
    plt.close()

def main():
    """Main testing and visualization function"""
    print("\n" + "=" * 80)
    print("🧪 HTR MODEL TESTING & VISUALIZATION")
    print("=" * 80 + "\n")
    
    # Initialize config
    config = Config()
    
    # Load dataset
    print("📂 Loading dataset...")
    train_samples, val_samples, test_samples = load_iam_dataset(config)
    
    if test_samples is None:
        print("❌ Failed to load dataset")
        return
    
    # Create datasets
    train_dataset = IAMDataset(train_samples, config, augment=False)
    val_dataset = IAMDataset(val_samples, config, augment=False)
    test_dataset = IAMDataset(test_samples, config, augment=False)
    
    print(f"✅ Train: {len(train_dataset)} samples")
    print(f"✅ Val: {len(val_dataset)} samples")
    print(f"✅ Test: {len(test_dataset)} samples")
    
    # Load best model
    model, checkpoint = load_best_model(config)
    
    if model is None:
        return
    
    # Load training history
    history_path = os.path.join(config.OUTPUT_DIR, 'training_history.json')
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    print("\n" + "=" * 80)
    print("🧪 TESTING ON ALL DATASETS")
    print("=" * 80)
    
    # Test on all datasets
    train_results = test_on_dataset(model, train_dataset, config, "Train")
    val_results = test_on_dataset(model, val_dataset, config, "Validation")
    test_results = test_on_dataset(model, test_dataset, config, "Test")
    
    print("\n" + "=" * 80)
    print("📊 CREATING VISUALIZATIONS")
    print("=" * 80)
    
    # Create all visualizations
    visualize_predictions(model, test_dataset, config, num_samples=15)
    visualize_confusion_examples(model, test_dataset, config, num_samples=20)
    
    # Detailed error analysis for each set
    train_errors = create_detailed_error_analysis(train_results, config, "Train")
    val_errors = create_detailed_error_analysis(val_results, config, "Validation")
    test_errors = create_detailed_error_analysis(test_results, config, "Test")
    
    # Character confusion analysis
    create_confusion_matrix_characters(test_results, config)
    
    # Final comprehensive report
    create_final_report(history, model, val_dataset, config)
    
    # Save detailed results
    detailed_results = {
        'train': {
            'cer': train_results['cer'],
            'wer': train_results['wer'],
            'acc': train_results['acc'],
            'samples': len(train_results['predictions'])
        },
        'validation': {
            'cer': val_results['cer'],
            'wer': val_results['wer'],
            'acc': val_results['acc'],
            'samples': len(val_results['predictions'])
        },
        'test': {
            'cer': test_results['cer'],
            'wer': test_results['wer'],
            'acc': test_results['acc'],
            'samples': len(test_results['predictions'])
        },
        'model_info': {
            'best_epoch': checkpoint['epoch'],
            'total_epochs': len(history['train_loss']),
            'best_val_cer': checkpoint['val_cer']
        }
    }
    
    results_path = os.path.join(config.OUTPUT_DIR, 'final_test_results.json')
    with open(results_path, 'w') as f:
        json.dump(detailed_results, f, indent=4)
    
    print("\n" + "=" * 80)
    print("✅ TESTING & VISUALIZATION COMPLETED!")
    print("=" * 80)
    print(f"📁 Visualizations: {config.VIS_DIR}")
    print(f"📊 Results saved: {results_path}")
    print("=" * 80 + "\n")
    
    print("📊 FINAL SUMMARY:")
    print(f"   Train:      CER {train_results['cer']:.2f}% | WER {train_results['wer']:.2f}% | Acc {train_results['acc']:.2f}%")
    print(f"   Validation: CER {val_results['cer']:.2f}% | WER {val_results['wer']:.2f}% | Acc {val_results['acc']:.2f}%")
    print(f"   Test:       CER {test_results['cer']:.2f}% | WER {test_results['wer']:.2f}% | Acc {test_results['acc']:.2f}%")
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
