import torch
import numpy as np
import editdistance

def ctc_decode(predictions, config):
    """
    Decodes CTC predictions to text
    
    Args:
        predictions: (T, batch, num_classes) - model output
        config: Config object
    Returns:
        List of decoded strings
    """
    batch_size = predictions.size(1)
    predictions = predictions.permute(1, 0, 2)  # (batch, T, num_classes)
    
    decoded_texts = []
    
    for b in range(batch_size):
        # Get most probable characters
        pred = predictions[b]  # (T, num_classes)
        _, max_indices = torch.max(pred, dim=1)  # (T,)
        max_indices = max_indices.cpu().numpy()
        
        # CTC decode: remove blanks and repeated characters
        decoded = []
        prev_idx = None
        
        for idx in max_indices:
            if idx != config.BLANK_IDX and idx != prev_idx:
                if idx in config.IDX_TO_CHAR:
                    decoded.append(config.IDX_TO_CHAR[idx])
            prev_idx = idx
        
        decoded_texts.append(''.join(decoded))
    
    return decoded_texts

def calculate_cer(predictions, targets, config):
    """
    Calculate Character Error Rate
    
    Args:
        predictions: (T, batch, num_classes) - model output
        targets: list of target strings
        config: Config object
    Returns:
        CER as percentage
    """
    decoded = ctc_decode(predictions, config)
    
    total_distance = 0
    total_length = 0
    
    for pred, target in zip(decoded, targets):
        distance = editdistance.eval(pred, target)
        total_distance += distance
        total_length += len(target)
    
    cer = (total_distance / total_length) * 100 if total_length > 0 else 0
    return cer

def calculate_wer(predictions, targets, config):
    """
    Calculate Word Error Rate
    
    Args:
        predictions: (T, batch, num_classes) - model output
        targets: list of target strings
        config: Config object
    Returns:
        WER as percentage
    """
    decoded = ctc_decode(predictions, config)
    
    total_distance = 0
    total_words = len(targets)
    
    for pred, target in zip(decoded, targets):
        # Exact match comparison for word-level
        if pred != target:
            total_distance += 1
    
    wer = (total_distance / total_words) * 100 if total_words > 0 else 0
    return wer

def calculate_accuracy(predictions, targets, config):
    """
    Calculate exact match accuracy
    
    Args:
        predictions: (T, batch, num_classes) - model output
        targets: list of target strings
        config: Config object
    Returns:
        Accuracy as percentage
    """
    decoded = ctc_decode(predictions, config)
    
    correct = sum(pred == target for pred, target in zip(decoded, targets))
    accuracy = (correct / len(targets)) * 100 if len(targets) > 0 else 0
    
    return accuracy
