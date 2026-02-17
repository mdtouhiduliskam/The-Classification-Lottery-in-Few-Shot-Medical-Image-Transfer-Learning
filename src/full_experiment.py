import os
import json
import numpy as np
import pandas as pd
from pydantic import create_model
import torch
import torch.nn as nn
from duckdb import torch


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

from data_loading_and_preprocess import AUGMENTATION_STRATEGIES, BATCH_SIZE, DATASET_CONFIGS, DEVICE, FINETUNE_STRATEGIES, LEARNING_RATE, MIN_DELTA, MODELS_TO_TEST, NUM_EPOCHS, NUM_RUNS, PATIENCE, SAMPLE_SIZES_TO_TEST, SEEDS, TEST_MODE, WEIGHT_DECAY, MedicalImageDataset, get_transforms
from training_model import calculate_metrics, train_epoch, validate
warnings.filterwarnings('ignore')



# ============ SINGLE EXPERIMENT ============

def run_experiment(dataset_key, model_name, sample_size, finetune_strategy, 
                   augmentation, num_classes, run_seed, dataset_name):
    """Run single experiment configuration"""
    
    # Set seed
    torch.manual_seed(run_seed)
    np.random.seed(run_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(run_seed)
    
    config = DATASET_CONFIGS[dataset_key]
    splits_path = config['splits_path']
    
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name} | {sample_size}-shot")
    print(f"Model: {model_name} | Strategy: {finetune_strategy}")
    print(f"Augmentation: {augmentation} | Seed: {run_seed}")
    print(f"{'='*60}")
    
    # Load data
    split_dir = os.path.join(splits_path, f"split_{sample_size}_shot")
    
    train_dataset = MedicalImageDataset(
        os.path.join(split_dir, 'train.json'),
        transform=get_transforms(train=True, augmentation=augmentation)
    )
    val_dataset = MedicalImageDataset(
        os.path.join(split_dir, 'val.json'),
        transform=get_transforms(train=False)
    )
    test_dataset = MedicalImageDataset(
        os.path.join(split_dir, 'test.json'),
        transform=get_transforms(train=False)
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
        num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
        num_workers=2, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, 
        num_workers=2, pin_memory=True
    )
    
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")
    
    # Create model
    freeze_map = {
        'head_only': 'head_only',
        'layer_wise': 'early_layers',    # ← ADD THIS LINE!
        'full_finetuning': 'none'
    }
    freeze_strategy = freeze_map[finetune_strategy]
    
    model = create_model(model_name, num_classes, freeze_strategy=freeze_strategy)
    
    # Count trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,} | Trainable: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")
    
    # Optimizer & scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=LEARNING_RATE, 
        weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    train_history = []
    best_model_state = None
    
    start_time = time.time()
    
    for epoch in range(1, NUM_EPOCHS + 1):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, augmentation=augmentation
        )
        
        # Validate
        val_loss, val_acc, _, _, _ = validate(model, val_loader, criterion)
        
        # Update scheduler
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Save history
        train_history.append({
            'epoch': epoch,
            'train_loss': float(train_loss),
            'train_acc': float(train_acc),
            'val_loss': float(val_loss),
            'val_acc': float(val_acc),
            'lr': float(current_lr)
        })
        
        # Early stopping
        if val_loss < best_val_loss - MIN_DELTA:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break
        
        if epoch % 5 == 0 or TEST_MODE:
            print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")
    
    training_time = time.time() - start_time
    
    # Load best model
    model.load_state_dict({k: v.to(DEVICE) for k, v in best_model_state.items()})
    
    # Test evaluation
    test_loss, test_acc, test_preds, test_labels, test_probs = validate(
        model, test_loader, criterion
    )
    test_metrics = calculate_metrics(test_labels, test_preds, test_probs, num_classes)
    
    print(f"\n{'='*60}")
    print(f"TEST RESULTS")
    print(f"{'='*60}")
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"F1 (weighted): {test_metrics['f1_weighted']:.4f}")
    print(f"F1 (macro): {test_metrics['f1_macro']:.4f}")
    if test_metrics.get('auc_roc'):
        print(f"AUC-ROC: {test_metrics['auc_roc']:.4f}")
    
    # Compile results
    results = {
        'experiment_config': {
            'dataset': dataset_key,
            'dataset_name': dataset_name,
            'model_name': model_name,
            'sample_size': sample_size,
            'finetune_strategy': finetune_strategy,
            'augmentation': augmentation,
            'pretrained_source': 'RadImageNet' if 'radimagenet' in model_name else 'ImageNet',
            'num_classes': num_classes,
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
            'weight_decay': WEIGHT_DECAY,
            'num_epochs_trained': len(train_history),
            'training_time_seconds': float(training_time),
            'total_parameters': int(total_params),
            'trainable_parameters': int(trainable_params),
            'seed': run_seed,
            'test_mode': TEST_MODE
        },
        'training_history': train_history,
        'best_val_loss': float(best_val_loss),
        'test_metrics': test_metrics,
        'timestamp': datetime.now().isoformat()
    }
    
    # Cleanup
    del model
    torch.cuda.empty_cache()
    
    return results


# ============ MAIN EXECUTION ============

def run_all_experiments():
    """Execute all experiments across all datasets"""
    
    print(f"\n{'#'*70}")
    print(f"# STARTING COMPLETE EXPERIMENTS")
    print(f"{'#'*70}\n")
    
    # Calculate total experiments
    total_experiments = 0
    for dataset_key in DATASET_CONFIGS:
        total_experiments += (
            len(MODELS_TO_TEST) * 
            len(SAMPLE_SIZES_TO_TEST) * 
            len(FINETUNE_STRATEGIES) * 
            len(AUGMENTATION_STRATEGIES) * 
            NUM_RUNS
        )
    
    print(f"Total experiments to run: {total_experiments}")
    if TEST_MODE:
        print(f"⚠️  TEST MODE: Quick validation only")
    else:
        print(f"Estimated time: ~{total_experiments * 10} minutes")
    print()
    
    experiment_counter = 0
    
    # Run experiments per dataset
    for dataset_key in DATASET_CONFIGS:
        config = DATASET_CONFIGS[dataset_key]
        dataset_name = config['name']
        splits_path = config['splits_path']
        results_path = config['results_path']
        
        # Load dataset metadata
        with open(os.path.join(splits_path, 'splits_info.json'), 'r') as f:
            splits_info = json.load(f)
        
        num_classes = splits_info['num_classes']
        
        # Create results directory
        os.makedirs(results_path, exist_ok=True)
        
        print(f"\n{'#'*70}")
        print(f"# DATASET: {dataset_name.upper()}")
        print(f"# Classes: {num_classes} | Total samples: {splits_info['total_samples']['all']}")
        print(f"{'#'*70}\n")
        
        all_results = []
        
        for model_name in MODELS_TO_TEST:
            for sample_size in SAMPLE_SIZES_TO_TEST:
                for finetune_strategy in FINETUNE_STRATEGIES:
                    for augmentation in AUGMENTATION_STRATEGIES:
                        
                        run_results = []
                        
                        for run_idx, seed in enumerate(SEEDS[:NUM_RUNS]):
                            experiment_counter += 1
                            
                            print(f"\n[{experiment_counter}/{total_experiments}] Running experiment...")
                            
                            try:
                                results = run_experiment(
                                    dataset_key, model_name, sample_size, 
                                    finetune_strategy, augmentation,
                                    num_classes, seed, dataset_name
                                )
                                
                                run_results.append(results)
                                all_results.append(results)
                                
                                # Save individual result
                                filename = (f"{dataset_key}_{model_name}_{sample_size}shot_"
                                          f"{finetune_strategy}_{augmentation}_seed{seed}.json")
                                with open(os.path.join(results_path, filename), 'w') as f:
                                    json.dump(results, f, indent=2)
                                
                            except Exception as e:
                                print(f"✗ Experiment failed: {e}")
                                import traceback
                                traceback.print_exc()
                        
                        # Aggregate results across runs
                        if run_results:
                            aggregate_metrics = {
                                'accuracy_mean': float(np.mean([r['test_metrics']['accuracy'] for r in run_results])),
                                'accuracy_std': float(np.std([r['test_metrics']['accuracy'] for r in run_results])),
                                'f1_weighted_mean': float(np.mean([r['test_metrics']['f1_weighted'] for r in run_results])),
                                'f1_weighted_std': float(np.std([r['test_metrics']['f1_weighted'] for r in run_results])),
                                'f1_macro_mean': float(np.mean([r['test_metrics']['f1_macro'] for r in run_results])),
                                'f1_macro_std': float(np.std([r['test_metrics']['f1_macro'] for r in run_results])),
                            }
                            
                            aggregate_result = {
                                'experiment_config': run_results[0]['experiment_config'],
                                'num_runs': len(run_results),
                                'aggregate_metrics': aggregate_metrics,
                                'individual_results': run_results
                            }
                            
                            agg_filename = (f"{dataset_key}_{model_name}_{sample_size}shot_"
                                          f"{finetune_strategy}_{augmentation}_aggregate.json")
                            with open(os.path.join(results_path, agg_filename), 'w') as f:
                                json.dump(aggregate_result, f, indent=2)
        
        # Save all results for this dataset
        all_results_file = os.path.join(results_path, f'all_results_{dataset_key}.json')
        with open(all_results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\n {dataset_name} complete! Results saved to: {results_path}")
    
    print(f"\n{'#'*70}")
    print(f"# ALL EXPERIMENTS COMPLETE!")
    print(f"{'#'*70}")
