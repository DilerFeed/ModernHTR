import torch
import torch.nn as nn

class CNN_RNN_CTC(nn.Module):
    """
    HTR Architecture: CNN + BiLSTM + CTC
    
    CNN: Extracts features from image
    BiLSTM: Models sequences
    CTC: Alignment without segmentation
    """
    def __init__(self, config):
        super(CNN_RNN_CTC, self).__init__()
        self.config = config
        
        # ============ CNN BACKBONE ============
        # Extracts visual features from image
        
        # Block 1: 1 -> 32
        self.conv1 = nn.Sequential(
            nn.Conv2d(config.IMG_CHANNELS, config.CNN_FILTERS[0], 
                     kernel_size=3, padding=1),
            nn.BatchNorm2d(config.CNN_FILTERS[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # H/2, W/2
        )
        
        # Block 2: 32 -> 64
        self.conv2 = nn.Sequential(
            nn.Conv2d(config.CNN_FILTERS[0], config.CNN_FILTERS[1], 
                     kernel_size=3, padding=1),
            nn.BatchNorm2d(config.CNN_FILTERS[1]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # H/4, W/4
        )
        
        # Block 3: 64 -> 128
        self.conv3 = nn.Sequential(
            nn.Conv2d(config.CNN_FILTERS[1], config.CNN_FILTERS[2], 
                     kernel_size=3, padding=1),
            nn.BatchNorm2d(config.CNN_FILTERS[2]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))  # H/8, W/4
        )
        
        # Block 4: 128 -> 256
        self.conv4 = nn.Sequential(
            nn.Conv2d(config.CNN_FILTERS[2], config.CNN_FILTERS[3], 
                     kernel_size=3, padding=1),
            nn.BatchNorm2d(config.CNN_FILTERS[3]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))  # H/16, W/4
        )
        
        # Calculate feature size after CNN
        # After all pooling: H = 64/16 = 4, W = 800/4 = 200
        self.feature_height = config.IMG_HEIGHT // 16
        self.feature_width = config.IMG_WIDTH // 4
        self.cnn_output_size = config.CNN_FILTERS[3] * self.feature_height
        
        # ============ RNN LAYERS ============
        # Models feature sequence
        
        self.rnn = nn.LSTM(
            input_size=self.cnn_output_size,
            hidden_size=config.LSTM_HIDDEN,
            num_layers=config.LSTM_LAYERS,
            bidirectional=True,
            dropout=config.DROPOUT if config.LSTM_LAYERS > 1 else 0,
            batch_first=True
        )
        
        # ============ OUTPUT LAYER ============
        # Character prediction
        
        self.dropout = nn.Dropout(config.DROPOUT)
        self.fc = nn.Linear(config.LSTM_HIDDEN * 2, config.NUM_CLASSES)  # *2 for bidirectional
    
    def forward(self, x):
        """
        Args:
            x: (batch, 1, H, W) - images
        Returns:
            output: (W', batch, num_classes) - logits for CTC
        """
        batch_size = x.size(0)
        
        # ============ CNN FEATURE EXTRACTION ============
        x = self.conv1(x)  # (batch, 32, H/2, W/2)
        x = self.conv2(x)  # (batch, 64, H/4, W/4)
        x = self.conv3(x)  # (batch, 128, H/8, W/4)
        x = self.conv4(x)  # (batch, 256, H/16, W/4)
        
        # Reshape for RNN
        # (batch, channels, height, width) -> (batch, width, channels*height)
        x = x.permute(0, 3, 1, 2)  # (batch, W', 256, H')
        batch, width, channels, height = x.size()
        x = x.reshape(batch, width, channels * height)  # (batch, W', 256*H')
        
        # ============ RNN SEQUENCE MODELING ============
        x, _ = self.rnn(x)  # (batch, W', lstm_hidden*2)
        
        # ============ OUTPUT PROJECTION ============
        x = self.dropout(x)
        x = self.fc(x)  # (batch, W', num_classes)
        
        # Permute for CTC: (W', batch, num_classes)
        x = x.permute(1, 0, 2)
        
        # Log softmax for CTC loss
        x = torch.nn.functional.log_softmax(x, dim=2)
        
        return x
    
    def get_feature_maps(self, x):
        """Get intermediate feature maps for visualization"""
        features = {}
        
        features['input'] = x
        x = self.conv1(x)
        features['conv1'] = x
        x = self.conv2(x)
        features['conv2'] = x
        x = self.conv3(x)
        features['conv3'] = x
        x = self.conv4(x)
        features['conv4'] = x
        
        return features

def count_parameters(model):
    """Count model parameters"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
