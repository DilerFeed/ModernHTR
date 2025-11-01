import os
import torch

# Enable MPS fallback for unsupported operations (like CTC Loss)
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# Optimize MPS performance
if torch.backends.mps.is_available():
    torch.backends.mps.flush_on_exit = False  # Faster exit

class Config:
    # Paths
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    DATA_ROOT = os.path.join(PROJECT_ROOT, 'data', 'iam')
    
    # Kaggle dataset has structure: data/iam/iam_words/words/ and data/iam/iam_words/words.txt
    WORDS_DIR = os.path.join(DATA_ROOT, 'iam_words', 'words')
    WORDS_TXT = os.path.join(DATA_ROOT, 'iam_words', 'words.txt')
    
    # Fallback paths if structure is different
    if not os.path.exists(WORDS_TXT):
        # Try alternative: data/iam/words.txt
        WORDS_TXT_ALT = os.path.join(DATA_ROOT, 'words_new.txt')
        if os.path.exists(WORDS_TXT_ALT):
            WORDS_TXT = WORDS_TXT_ALT
    
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'outputs')
    MODEL_DIR = os.path.join(OUTPUT_DIR, 'models')
    VIS_DIR = os.path.join(OUTPUT_DIR, 'visualizations')
    LOGS_DIR = os.path.join(OUTPUT_DIR, 'logs')
    
    # Create directories
    os.makedirs(DATA_ROOT, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(VIS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)
    
    # Image parameters
    IMG_HEIGHT = 64
    IMG_WIDTH = 800
    IMG_CHANNELS = 1  # Grayscale
    
    # Model parameters
    CNN_FILTERS = [32, 64, 128, 256]  # Number of filters in CNN layers
    LSTM_HIDDEN = 256
    LSTM_LAYERS = 2
    DROPOUT = 0.3
    
    # Training parameters
    BATCH_SIZE = 32  # Increased for M2 - was 16
    EPOCHS = 50
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-5
    
    # Data split
    TRAIN_SPLIT = 0.8
    VAL_SPLIT = 0.1
    TEST_SPLIT = 0.1
    
    # Device selection with MPS support for Apple Silicon
    if torch.cuda.is_available():
        DEVICE = 'cuda'
        DEVICE_NAME = torch.cuda.get_device_name(0)
        USE_AMP = True  # Automatic Mixed Precision
    elif torch.backends.mps.is_available():
        DEVICE = 'mps'
        DEVICE_NAME = 'Apple Silicon GPU (MPS)'
        USE_AMP = False  # Disable AMP - causes issues with CTC Loss on MPS
    else:
        DEVICE = 'cpu'
        DEVICE_NAME = 'CPU'
        USE_AMP = False
    
    # Optimize workers for M2
    NUM_WORKERS = 8 if DEVICE == 'mps' else 4  # M2 has 8 performance cores
    
    # Other
    SEED = 42
    
    def __init__(self):
        """Initialize instance-specific character mappings"""
        # Characters (will be filled from dataset)
        self.CHARS = ''
        self.CHAR_TO_IDX = {}
        self.IDX_TO_CHAR = {}
        self.NUM_CLASSES = 0
        self.BLANK_IDX = 0  # For CTC
    
    def build_char_mappings(self, chars):
        """Creates character mappings"""
        self.CHARS = ''.join(sorted(set(chars)))
        self.BLANK_IDX = 0
        self.CHAR_TO_IDX = {char: idx + 1 for idx, char in enumerate(self.CHARS)}
        self.IDX_TO_CHAR = {idx: char for char, idx in self.CHAR_TO_IDX.items()}
        self.IDX_TO_CHAR[self.BLANK_IDX] = ''  # Blank for CTC
        self.NUM_CLASSES = len(self.CHARS) + 1  # +1 for blank
        
        print(f"✅ Character mappings created:")
        print(f"   Total characters: {len(self.CHARS)}")
        print(f"   Num classes (with blank): {self.NUM_CLASSES}")
        print(f"   Sample chars: {self.CHARS[:20]}...")
        
    def __str__(self):
        device_name = 'MPS' if self.DEVICE == 'mps' else ('CUDA' if self.DEVICE == 'cuda' else 'CPU')
        return f"Config(device={device_name}, batch_size={self.BATCH_SIZE}, epochs={self.EPOCHS})"
