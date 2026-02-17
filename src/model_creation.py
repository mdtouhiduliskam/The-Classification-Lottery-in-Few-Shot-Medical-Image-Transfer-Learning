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
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============ MODEL CREATION ============

def create_model(model_name, num_classes, freeze_strategy='none'):
    """
    Create model with specified architecture and freezing strategy
    
    model_name: 'resnet50', 'efficientnet_b4', 'vit_b_16', 'convnext_tiny', 'resnet50_radimagenet'
    freeze_strategy: 'none', 'head_only', 'early_layers'
    """
    
    if model_name == 'resnet50':
        model = models.resnet50(weights='IMAGENET1K_V1')
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
        
        if freeze_strategy == 'head_only':
            for param in model.parameters():
                param.requires_grad = False
            for param in model.fc.parameters():
                param.requires_grad = True
                
        elif freeze_strategy == 'early_layers':
            # Freeze conv1, bn1, layer1, layer2
            for name, param in model.named_parameters():
                if any(layer in name for layer in ['conv1', 'bn1', 'layer1', 'layer2']):
                    param.requires_grad = False
                else:
                    param.requires_grad = True
    
    elif model_name == 'resnet50_radimagenet':
        # Load ResNet50 architecture
        model = models.resnet50(weights=None)  # No ImageNet weights
        
        # TODO: Load RadImageNet weights here
        # For now, we'll use ImageNet as placeholder
        # In production, you'd load: model.load_state_dict(torch.load('radimagenet_resnet50.pth'))
        print("⚠️  WARNING: RadImageNet weights not loaded - using random init")
        print("    Download weights from: https://github.com/BMEII-AI/RadImageNet")
        
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
        
        if freeze_strategy == 'head_only':
            for param in model.parameters():
                param.requires_grad = False
            for param in model.fc.parameters():
                param.requires_grad = True
                
        elif freeze_strategy == 'early_layers':
            for name, param in model.named_parameters():
                if any(layer in name for layer in ['conv1', 'bn1', 'layer1', 'layer2']):
                    param.requires_grad = False
                else:
                    param.requires_grad = True
    
    elif model_name == 'efficientnet_b4':
        model = models.efficientnet_b4(weights='IMAGENET1K_V1')
        num_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_features, num_classes)
        
        if freeze_strategy == 'head_only':
            for param in model.parameters():
                param.requires_grad = False
            for param in model.classifier.parameters():
                param.requires_grad = True
                
        elif freeze_strategy == 'early_layers':
            # Freeze first 3 feature blocks
            for name, param in model.named_parameters():
                if any(f'features.{i}' in name for i in range(3)):
                    param.requires_grad = False
                else:
                    param.requires_grad = True
    
    elif model_name == 'vit_b_16':
        model = models.vit_b_16(weights='IMAGENET1K_V1')
        num_features = model.heads.head.in_features
        model.heads.head = nn.Linear(num_features, num_classes)
        
        if freeze_strategy == 'head_only':
            for param in model.parameters():
                param.requires_grad = False
            for param in model.heads.parameters():
                param.requires_grad = True
                
        elif freeze_strategy == 'early_layers':
            # Freeze first 6 transformer blocks (out of 12)
            for name, param in model.named_parameters():
                if 'encoder.layers.encoder_layer_' in name:
                    layer_num = int(name.split('encoder_layer_')[1].split('.')[0])
                    if layer_num < 6:
                        param.requires_grad = False
                    else:
                        param.requires_grad = True
                elif 'conv_proj' in name or 'class_token' in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
    
    elif model_name == 'convnext_tiny':
        model = models.convnext_tiny(weights='IMAGENET1K_V1')
        num_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(num_features, num_classes)
        
        if freeze_strategy == 'head_only':
            for param in model.parameters():
                param.requires_grad = False
            for param in model.classifier.parameters():
                param.requires_grad = True
                
        elif freeze_strategy == 'early_layers':
            # Freeze first 2 stages (out of 4)
            for name, param in model.named_parameters():
                if 'features.0' in name or 'features.1' in name or 'features.2' in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
    
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model.to(DEVICE)
