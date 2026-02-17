import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from tqdm.notebook import tqdm
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    roc_auc_score, confusion_matrix
)
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============ GLOBAL CONFIGURATION ============
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================
# TEST MODE - CHANGE THESE FOR QUICK TESTING
# ============================================
TEST_MODE = False  # SET TO False FOR FULL EXPERIMENTS

if TEST_MODE:
    print("⚠️  TEST MODE ENABLED - Running quick validation")
    NUM_EPOCHS = 3              # Reduced from 50
    PATIENCE = 2                # Reduced from 10
    NUM_RUNS = 1                # Test with 1 run only
    SAMPLE_SIZES_TO_TEST = [5]  # Test with just 5-shot
    BATCH_SIZE = 16             # Smaller batch for speed
else:
    print("🚀 FULL EXPERIMENT MODE")
    NUM_EPOCHS = 50
    PATIENCE = 10
    NUM_RUNS = 2
    SAMPLE_SIZES_TO_TEST = [5, 10, 20, 50, 100, 200]  # Full sweep
    BATCH_SIZE = 32

# Dataset paths (auto-detected)
DATASET_CONFIGS = {
    # 'covid19': {
    #     'splits_path': '/kaggle/working/data_splits_covid19',
    #     'results_path': '/kaggle/working/results_covid19',
    #     'name': 'COVID-19 Radiography'
    # },
    'pneumonia': {
        'splits_path': '/kaggle/working/data_splits_pneumonia',
        'results_path': '/kaggle/working/results_pneumonia',
        'name': 'Chest X-Ray Pneumonia'
    }
   
   
}

# Training hyperparameters
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 1e-4
MIN_DELTA = 0.001

# Experimental design
MODELS_TO_TEST = [
    'resnet50',           # Classic baseline
    'efficientnet_b4',    # EfficientNet family
    'vit_b_16',           # Vision Transformer

]

FINETUNE_STRATEGIES = [
    'head_only',          # Freeze backbone, train head only
    'layer_wise',         # Freeze early layers, train late + head
]

AUGMENTATION_STRATEGIES = [
    'standard',           # Basic augmentations

]

SEEDS = [42, 456]  # For reproducibility

print(f"{'='*70}")
print(f"IEEE ACCESS EXPERIMENTS - COMPLETE PIPELINE")
print(f"{'='*70}")
print(f"Device: {DEVICE}")
print(f"Test Mode: {TEST_MODE}")
print(f"Datasets: {len(DATASET_CONFIGS)}")
print(f"Models: {len(MODELS_TO_TEST)}")
print(f"Fine-tuning strategies: {len(FINETUNE_STRATEGIES)}")
print(f"Augmentation strategies: {len(AUGMENTATION_STRATEGIES)}")
print(f"Sample sizes: {SAMPLE_SIZES_TO_TEST}")
print(f"Independent runs: {NUM_RUNS}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Max epochs: {NUM_EPOCHS}")
print(f"{'='*70}")

# ============ DATASET CLASS ============

class MedicalImageDataset(Dataset):
    """Unified dataset loader for all medical imaging modalities"""
    
    def __init__(self, json_file, transform=None, mixup_alpha=0.0, cutmix_alpha=0.0):
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        self.image_paths = data['image_paths']
        self.labels = data['labels']
        self.transform = transform
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        
        # Verify and filter valid images
        valid_indices = []
        for idx, path in enumerate(self.image_paths):
            if os.path.exists(path):
                valid_indices.append(idx)
        
        self.image_paths = [self.image_paths[i] for i in valid_indices]
        self.labels = [self.labels[i] for i in valid_indices]
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            image = Image.new('RGB', (224, 224), color='black')
        
        if self.transform:
            image = self.transform(image)
        
        label = self.labels[idx]
        
        return image, label


# ============ MIXUP & CUTMIX FUNCTIONS ============

def mixup_data(x, y, alpha=1.0):
    """Apply MixUp augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def cutmix_data(x, y, alpha=1.0):
    """Apply CutMix augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    # Generate random box
    W = x.size()[2]
    H = x.size()[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # Uniform sampling
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    
    # Adjust lambda to match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    y_a, y_b = y, y[index]
    
    return x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """MixUp/CutMix loss"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ============ DATA TRANSFORMS ============

def get_transforms(train=True, img_size=224, augmentation='standard'):
    """
    Data transforms with augmentation strategy
    augmentation: 'standard', 'mixup', 'cutmix'
    Note: MixUp/CutMix are applied during training, not in transform
    """
    
    if train:
        if augmentation in ['standard', 'mixup', 'cutmix']:
            # Same base transforms for all
            return transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        
def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """MixUp/CutMix loss"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ============ DATA TRANSFORMS ============

def get_transforms(train=True, img_size=224, augmentation='standard'):
    """
    Data transforms with augmentation strategy
    augmentation: 'standard', 'mixup', 'cutmix'
    Note: MixUp/CutMix are applied during training, not in transform
    """
    
    if train:
        if augmentation in ['standard', 'mixup', 'cutmix']:
            # Same base transforms for all
            return transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
