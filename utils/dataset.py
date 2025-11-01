import os
import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class IAMDataset(Dataset):
    def __init__(self, samples, config, transform=None, augment=False):
        """
        Args:
            samples: list of tuples (img_path, text)
            config: Config object
            transform: transformations
            augment: use augmentation
        """
        self.samples = samples
        self.config = config
        self.transform = transform
        self.augment = augment
        
        # Augmentation for training set
        if self.augment:
            self.augmentation = A.Compose([
                A.Rotate(limit=3, p=0.3),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                A.GaussianBlur(blur_limit=(3, 5), p=0.2),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.2),
            ])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, text = self.samples[idx]
        
        # Load image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            # If image failed to load, create empty one
            img = np.ones((self.config.IMG_HEIGHT, self.config.IMG_WIDTH), dtype=np.uint8) * 255
        
        # Augmentation
        if self.augment and self.augmentation:
            augmented = self.augmentation(image=img)
            img = augmented['image']
        
        # Resize
        img = cv2.resize(img, (self.config.IMG_WIDTH, self.config.IMG_HEIGHT))
        
        # Normalization
        img = img.astype(np.float32) / 255.0
        
        # Inversion (if background is dark)
        # img = 1.0 - img
        
        # Add channel
        img = np.expand_dims(img, axis=0)  # (1, H, W)
        
        # Convert to tensor
        img = torch.FloatTensor(img)
        
        # Encode text
        encoded_text = self.encode_text(text)
        
        return img, encoded_text, text
    
    def encode_text(self, text):
        """Encodes text to indices"""
        encoded = []
        for char in text:
            if char in self.config.CHAR_TO_IDX:
                encoded.append(self.config.CHAR_TO_IDX[char])
            else:
                # Debug: print unknown characters
                if len(encoded) == 0:  # Only print once per text
                    print(f"⚠️  Unknown char '{char}' (ord={ord(char)}) in text: '{text}'")
        
        # Return at least empty tensor (not None)
        if len(encoded) == 0:
            # Return empty tensor - will be filtered out
            return torch.LongTensor([])
        
        return torch.LongTensor(encoded)

def collate_fn(batch):
    """Function to collate batch with different text lengths"""
    images, encoded_texts, texts = zip(*batch)
    
    # Filter out samples with empty targets
    valid_indices = [i for i, text in enumerate(encoded_texts) if len(text) > 0]
    
    if len(valid_indices) == 0:
        # All samples have empty targets - this shouldn't happen
        print("❌ WARNING: All samples in batch have empty targets!")
        # Return dummy batch
        return (
            torch.stack(images[:1], dim=0),
            torch.zeros(1, 1, dtype=torch.long),
            torch.LongTensor([1]),
            [texts[0]]
        )
    
    # Filter batch
    images = [images[i] for i in valid_indices]
    encoded_texts = [encoded_texts[i] for i in valid_indices]
    texts = [texts[i] for i in valid_indices]
    
    # Stack images
    images = torch.stack(images, dim=0)
    
    # Calculate text lengths
    text_lengths = torch.LongTensor([len(text) for text in encoded_texts])
    
    # Pad texts
    max_len = max(text_lengths)
    padded_texts = torch.zeros(len(encoded_texts), max_len, dtype=torch.long)
    for i, text in enumerate(encoded_texts):
        padded_texts[i, :len(text)] = text
    
    return images, padded_texts, text_lengths, texts
