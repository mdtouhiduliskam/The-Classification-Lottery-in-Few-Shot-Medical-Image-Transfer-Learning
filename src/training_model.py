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

from data_loading_and_preprocess import DEVICE, cutmix_data, mixup_criterion, mixup_data
warnings.filterwarnings('ignore')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============ TRAINING LOOP ============

def train_epoch(model, dataloader, criterion, optimizer, augmentation='standard'):
    """Single training epoch with optional MixUp/CutMix"""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    for images, labels in dataloader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        # Apply MixUp or CutMix
        if augmentation == 'mixup':
            images, labels_a, labels_b, lam = mixup_data(images, labels, alpha=1.0)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            
        elif augmentation == 'cutmix':
            images, labels_a, labels_b, lam = cutmix_data(images, labels, alpha=1.0)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            
        else:  # standard
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion):
    """Validation/test evaluation"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    val_loss = running_loss / len(dataloader.dataset)
    val_acc = accuracy_score(all_labels, all_preds)
    
    return val_loss, val_acc, all_preds, all_labels, np.array(all_probs)


# ============ METRICS CALCULATION ============

def calculate_metrics(y_true, y_pred, y_probs, num_classes):
    """Comprehensive evaluation metrics"""
    
    metrics = {}
    
    # Primary metrics
    metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
    
    # Weighted metrics
    precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    metrics['precision_weighted'] = float(precision_w)
    metrics['recall_weighted'] = float(recall_w)
    metrics['f1_weighted'] = float(f1_w)
    
    # Macro metrics
    precision_m, recall_m, f1_m, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    metrics['precision_macro'] = float(precision_m)
    metrics['recall_macro'] = float(recall_m)
    metrics['f1_macro'] = float(f1_m)
    
    # Per-class metrics
    precision_pc, recall_pc, f1_pc, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    metrics['per_class'] = {
        'precision': precision_pc.tolist(),
        'recall': recall_pc.tolist(),
        'f1_score': f1_pc.tolist(),
        'support': support.tolist()
    }
    
    # AUC-ROC
    try:
        if num_classes == 2:
            metrics['auc_roc'] = float(roc_auc_score(y_true, y_probs[:, 1]))
        else:
            metrics['auc_roc'] = float(roc_auc_score(
                y_true, y_probs, multi_class='ovr', average='weighted'
            ))
    except Exception:
        metrics['auc_roc'] = None
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm.tolist()
    
    # Sensitivity & Specificity
    if num_classes == 2:
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        metrics['sensitivity'] = float(sensitivity)
        metrics['specificity'] = float(specificity)
    else:
        sensitivities = []
        specificities = []
        
        for i in range(num_classes):
            tp = cm[i, i]
            fn = cm[i, :].sum() - tp
            fp = cm[:, i].sum() - tp
            tn = cm.sum() - (tp + fn + fp)
            
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            sensitivities.append(sens)
            specificities.append(spec)
        
        metrics['sensitivity_per_class'] = sensitivities
        metrics['specificity_per_class'] = specificities
        metrics['sensitivity_avg'] = float(np.mean(sensitivities))
        metrics['specificity_avg'] = float(np.mean(specificities))
    
    return metrics