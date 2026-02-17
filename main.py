
import json
import os
from data_loading_and_preprocess import DATASET_CONFIGS, SAMPLE_SIZES_TO_TEST
from full_experiment import run_all_experiments, run_experiment


def verify_pipeline():
    """Quick sanity check - test ONE experiment"""
    print("\n" + "="*70)
    print("PIPELINE VERIFICATION - Testing 1 experiment")
    print("="*70)
    
    # Use first available dataset
    dataset_key = list(DATASET_CONFIGS.keys())[0]
    config = DATASET_CONFIGS[dataset_key]
    
    # Load metadata
    with open(os.path.join(config['splits_path'], 'splits_info.json'), 'r') as f:
        splits_info = json.load(f)
    
    num_classes = splits_info['num_classes']
    
    # Test single configuration
    try:
        results = run_experiment(
            dataset_key=dataset_key,
            model_name='vit_b_16',  # Fastest to test
            sample_size=SAMPLE_SIZES_TO_TEST[0],
            finetune_strategy='head_only',
            augmentation='standard',
            num_classes=num_classes,
            run_seed=42,
            dataset_name=config['name']
        )
        
        print("\n" + "="*70)
        print("✅ PIPELINE VERIFICATION PASSED!")
        print("="*70)
        print(f"Test accuracy: {results['test_metrics']['accuracy']:.4f}")
        print(f"Training time: {results['experiment_config']['training_time_seconds']:.1f}s")
        print("\nYou can now run full experiments with TEST_MODE=False")
        
        return True
        
    except Exception as e:
        print("\n" + "="*70)
        print("✗ PIPELINE VERIFICATION FAILED!")
        print("="*70)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False



    
    # Uncomment ONE of these options:
    
# Option 1: Quick verification (recommended first)
# verify_pipeline()
    
# Option 2: Run all experiments
run_all_experiments()