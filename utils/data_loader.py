import os
import random
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

def find_dataset_structure(config):
    """Auto-detect IAM dataset structure"""
    print("🔍 Auto-detecting dataset structure...")
    
    # Possible structures
    structures = [
        # Structure 1: Kaggle dataset
        {
            'words_txt': os.path.join(config.DATA_ROOT, 'iam_words', 'words.txt'),
            'words_dir': os.path.join(config.DATA_ROOT, 'iam_words', 'words')
        },
        # Structure 2: Alternative Kaggle
        {
            'words_txt': os.path.join(config.DATA_ROOT, 'words_new.txt'),
            'words_dir': os.path.join(config.DATA_ROOT, 'iam_words', 'words')
        },
        # Structure 3: Original IAM
        {
            'words_txt': os.path.join(config.DATA_ROOT, 'words.txt'),
            'words_dir': os.path.join(config.DATA_ROOT, 'words')
        },
    ]
    
    for struct in structures:
        if os.path.exists(struct['words_txt']) and os.path.exists(struct['words_dir']):
            print(f"✅ Found dataset structure:")
            print(f"   words.txt: {struct['words_txt']}")
            print(f"   words dir: {struct['words_dir']}")
            config.WORDS_TXT = struct['words_txt']
            config.WORDS_DIR = struct['words_dir']
            return True
    
    # Manual search
    print("🔍 Searching for dataset files...")
    for root, dirs, files in os.walk(config.DATA_ROOT):
        for file in files:
            if 'words' in file.lower() and file.endswith('.txt'):
                words_txt_path = os.path.join(root, file)
                print(f"   Found: {words_txt_path}")
                
                # Try to find words directory
                parent_dir = os.path.dirname(words_txt_path)
                possible_words_dirs = [
                    os.path.join(parent_dir, 'words'),
                    os.path.join(config.DATA_ROOT, 'iam_words', 'words'),
                    os.path.join(config.DATA_ROOT, 'words')
                ]
                
                for words_dir in possible_words_dirs:
                    if os.path.exists(words_dir):
                        print(f"   Found: {words_dir}")
                        config.WORDS_TXT = words_txt_path
                        config.WORDS_DIR = words_dir
                        return True
    
    return False

def load_iam_dataset(config):
    """
    Loads IAM dataset and creates character mappings
    Returns: train_samples, val_samples, test_samples
    """
    print("📂 Loading IAM dataset...")
    
    # Auto-detect dataset structure
    if not os.path.exists(config.WORDS_TXT):
        if not find_dataset_structure(config):
            print(f"❌ Dataset not found in {config.DATA_ROOT}")
            print("\n📌 Please ensure you have:")
            print("   1. Downloaded from: https://www.kaggle.com/datasets/nibinv23/iam-handwriting-word-database")
            print("   2. Extracted to: data/iam/")
            print("   3. Structure should contain 'words.txt' and 'words/' folder")
            print("\n📂 Current directory structure:")
            for root, dirs, files in os.walk(config.DATA_ROOT):
                level = root.replace(config.DATA_ROOT, '').count(os.sep)
                indent = ' ' * 2 * level
                print(f"{indent}{os.path.basename(root)}/")
                subindent = ' ' * 2 * (level + 1)
                for file in files[:5]:  # Show first 5 files
                    print(f"{subindent}{file}")
                if len(files) > 5:
                    print(f"{subindent}... and {len(files) - 5} more files")
                if level > 2:  # Limit depth
                    break
            return None, None, None
    
    samples = []
    all_chars = set()
    
    # Read words.txt
    print(f"📄 Reading: {config.WORDS_TXT}")
    with open(config.WORDS_TXT, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print("📝 Parsing annotations...")
    for line in tqdm(lines):
        if line.startswith('#'):
            continue
        
        parts = line.strip().split()
        if len(parts) < 9:
            continue
        
        # Format: word_id segmentation_result graylevel x y w h grammar_tag transcription
        word_id = parts[0]
        segmentation = parts[1]
        text = parts[-1]
        
        # Skip poorly segmented
        if segmentation != 'ok':
            continue
        
        # Path to image: words/a01/a01-000u/a01-000u-00-00.png
        path_parts = word_id.split('-')
        img_path = os.path.join(
            config.WORDS_DIR,
            path_parts[0],
            f"{path_parts[0]}-{path_parts[1]}",
            f"{word_id}.png"
        )
        
        if os.path.exists(img_path):
            samples.append((img_path, text))
            all_chars.update(text)
    
    if len(samples) == 0:
        print(f"❌ No valid samples found!")
        print(f"   Checked directory: {config.WORDS_DIR}")
        print(f"   Example expected path: {config.WORDS_DIR}/a01/a01-000u/a01-000u-00-00.png")
        return None, None, None
    
    print(f"✅ Loaded {len(samples)} samples")
    print(f"🔤 Unique characters: {len(all_chars)}")
    
    # Create character mappings - use instance method
    config.build_char_mappings(''.join(all_chars))
    
    # Shuffle
    random.seed(config.SEED)
    random.shuffle(samples)
    
    # Split into train/val/test
    n = len(samples)
    train_end = int(n * config.TRAIN_SPLIT)
    val_end = train_end + int(n * config.VAL_SPLIT)
    
    train_samples = samples[:train_end]
    val_samples = samples[train_end:val_end]
    test_samples = samples[val_end:]
    
    print(f"📊 Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}")
    
    return train_samples, val_samples, test_samples

def visualize_dataset_stats(train_samples, val_samples, test_samples, config):
    """Visualize dataset statistics"""
    print("📊 Creating dataset visualizations...")
    
    # Extract texts
    train_texts = [text for _, text in train_samples]
    val_texts = [text for _, text in val_samples]
    test_texts = [text for _, text in test_samples]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Word length distribution
    train_lengths = [len(text) for text in train_texts]
    val_lengths = [len(text) for text in val_texts]
    test_lengths = [len(text) for text in test_texts]
    
    axes[0, 0].hist([train_lengths, val_lengths, test_lengths], 
                    label=['Train', 'Val', 'Test'], bins=30, alpha=0.7)
    axes[0, 0].set_xlabel('Word Length (characters)')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Word Length Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Dataset split sizes
    sizes = [len(train_samples), len(val_samples), len(test_samples)]
    axes[0, 1].bar(['Train', 'Val', 'Test'], sizes, color=['#3498db', '#e74c3c', '#2ecc71'])
    axes[0, 1].set_ylabel('Number of Samples')
    axes[0, 1].set_title('Dataset Split Sizes')
    for i, v in enumerate(sizes):
        axes[0, 1].text(i, v + 100, str(v), ha='center', va='bottom', fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # 3. Top-20 most frequent characters
    all_chars = ''.join(train_texts + val_texts + test_texts)
    char_counter = Counter(all_chars)
    top_chars = char_counter.most_common(20)
    chars, counts = zip(*top_chars)
    
    axes[1, 0].barh(range(len(chars)), counts, color='#9b59b6')
    axes[1, 0].set_yticks(range(len(chars)))
    axes[1, 0].set_yticklabels([f"'{c}'" for c in chars])
    axes[1, 0].set_xlabel('Count')
    axes[1, 0].set_title('Top-20 Most Frequent Characters')
    axes[1, 0].invert_yaxis()
    axes[1, 0].grid(True, alpha=0.3, axis='x')
    
    # 4. Statistics
    stats_text = f"""
    📊 DATASET STATISTICS
    
    ═══════════════════════════
    Total samples: {len(train_samples) + len(val_samples) + len(test_samples)}
    Train: {len(train_samples)}
    Val: {len(val_samples)}
    Test: {len(test_samples)}
    
    ═══════════════════════════
    Unique characters: {len(config.CHARS)}
    Avg word length: {sum(train_lengths) / len(train_lengths):.1f}
    Min length: {min(train_lengths)}
    Max length: {max(train_lengths)}
    
    ═══════════════════════════
    Characters: {config.CHARS[:50]}...
    """
    
    axes[1, 1].text(0.1, 0.5, stats_text, fontsize=10, 
                    verticalalignment='center', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.VIS_DIR, 'dataset_stats.png'), dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {os.path.join(config.VIS_DIR, 'dataset_stats.png')}")
    plt.close()
