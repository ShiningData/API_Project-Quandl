#!/usr/bin/env python3
"""
SIMPLE EXECUTABLE SCRIPT - AutoGluon Imbalanced Classification
===============================================================

This script will guide you through running the model step by step.
Just follow the prompts!

Before running:
1. Install dependencies: pip install -r requirements.txt
2. Prepare your data CSV file with 24 numerical features + 1 target column
3. Run this script: python run_model.py
"""

import os
import sys

def print_header(text):
    """Print a formatted header"""
    print("\n" + "="*80)
    print(f"  {text}")
    print("="*80 + "\n")

def check_dependencies():
    """Check if required packages are installed"""
    print_header("STEP 1: Checking Dependencies")
    
    required_packages = {
        'pandas': 'pandas',
        'numpy': 'numpy',
        'sklearn': 'scikit-learn',
        'autogluon.tabular': 'autogluon.tabular',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn'
    }
    
    missing_packages = []
    
    for package, pip_name in required_packages.items():
        try:
            __import__(package)
            print(f"✓ {pip_name} installed")
        except ImportError:
            print(f"✗ {pip_name} NOT installed")
            missing_packages.append(pip_name)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("\nPlease install them by running:")
        print("  pip install -r requirements.txt")
        print("\nOr install manually:")
        print(f"  pip install {' '.join(missing_packages)}")
        return False
    
    print("\n✓ All dependencies are installed!")
    return True

def get_user_input():
    """Get configuration from user"""
    print_header("STEP 2: Configuration")
    
    print("Please provide the following information:\n")
    
    # Data file
    while True:
        data_file = input("Enter path to your CSV file (or 'demo' for demo data): ").strip()
        if data_file.lower() == 'demo':
            print("  → Using demo data")
            break
        elif os.path.exists(data_file):
            print(f"  → Found file: {data_file}")
            break
        else:
            print(f"  ✗ File not found: {data_file}")
            print("  Please try again or type 'demo' for demo data\n")
    
    # Target column
    if data_file.lower() != 'demo':
        target_column = input("\nEnter name of target column (default: 'target'): ").strip()
        target_column = target_column if target_column else 'target'
    else:
        target_column = 'target'
    print(f"  → Target column: {target_column}")
    
    # Training time
    print("\nHow long should training take?")
    print("  1. Quick test (10 minutes)")
    print("  2. Medium quality (30 minutes)")
    print("  3. High quality (1 hour)")
    print("  4. Best quality (2+ hours)")
    print("  5. Custom time")
    
    time_choice = input("\nSelect option [1-5] (default: 2): ").strip()
    time_choice = time_choice if time_choice else '2'
    
    time_limits = {
        '1': (600, 'medium_quality'),
        '2': (1800, 'good_quality'),
        '3': (3600, 'high_quality'),
        '4': (7200, 'best_quality'),
    }
    
    if time_choice == '5':
        custom_time = input("Enter time in seconds (e.g., 3600 for 1 hour): ").strip()
        time_limit = int(custom_time)
        preset = 'good_quality'
    else:
        time_limit, preset = time_limits.get(time_choice, (1800, 'good_quality'))
    
    print(f"  → Training time: {time_limit} seconds ({time_limit/60:.0f} minutes)")
    print(f"  → Quality preset: {preset}")
    
    # Evaluation metric
    print("\nWhich evaluation metric to optimize?")
    print("  1. ROC-AUC (overall performance)")
    print("  2. Average Precision (recommended for imbalanced)")
    print("  3. F1 Score (balance precision/recall)")
    print("  4. Recall (minimize false negatives)")
    
    metric_choice = input("\nSelect option [1-4] (default: 2): ").strip()
    metric_choice = metric_choice if metric_choice else '2'
    
    metrics = {
        '1': 'roc_auc',
        '2': 'average_precision',
        '3': 'f1',
        '4': 'recall'
    }
    eval_metric = metrics.get(metric_choice, 'average_precision')
    print(f"  → Evaluation metric: {eval_metric}")
    
    # Advanced options
    print("\nUse advanced features?")
    print("  1. Basic (sample weights only - recommended for large datasets)")
    print("  2. Advanced (includes SMOTE resampling - better results)")
    
    advanced_choice = input("\nSelect option [1-2] (default: 1): ").strip()
    use_advanced = advanced_choice == '2'
    
    if use_advanced:
        print("\nSelect resampling method:")
        print("  1. SMOTE (standard)")
        print("  2. ADASYN (adaptive)")
        print("  3. BorderlineSMOTE (focus on boundary)")
        print("  4. No resampling (sample weights only)")
        
        resample_choice = input("\nSelect option [1-4] (default: 1): ").strip()
        resample_choice = resample_choice if resample_choice else '1'
        
        resampling_methods = {
            '1': 'smote',
            '2': 'adasyn',
            '3': 'borderline_smote',
            '4': 'none'
        }
        resampling_method = resampling_methods.get(resample_choice, 'smote')
        print(f"  → Resampling method: {resampling_method}")
    else:
        resampling_method = 'none'
    
    config = {
        'data_file': data_file,
        'target_column': target_column,
        'time_limit': time_limit,
        'preset': preset,
        'eval_metric': eval_metric,
        'use_advanced': use_advanced,
        'resampling_method': resampling_method
    }
    
    return config

def run_model(config):
    """Run the model with given configuration"""
    print_header("STEP 3: Running Model")
    
    print("Importing libraries... (this may take a moment)")
    
    if config['use_advanced']:
        from advanced_autogluon_pipeline import ImbalancedClassificationPipeline
        import pandas as pd
        from sklearn.model_selection import train_test_split
        
        print("✓ Libraries imported\n")
        
        # Load data
        print("Loading data...")
        if config['data_file'].lower() == 'demo':
            print("  Creating demo dataset...")
            from autogluon_imbalanced_model import create_sample_data
            df = create_sample_data()
        else:
            df = pd.read_csv(config['data_file'])
        
        print(f"✓ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        print(f"  Target distribution:\n{df[config['target_column']].value_counts()}\n")
        
        # Split data
        print("Splitting data into train/test (80%/20%)...")
        train_data, test_data = train_test_split(
            df,
            test_size=0.2,
            random_state=42,
            stratify=df[config['target_column']]
        )
        print(f"✓ Train set: {len(train_data)} samples")
        print(f"✓ Test set: {len(test_data)} samples\n")
        
        # Initialize pipeline
        print("Initializing advanced pipeline...")
        pipeline = ImbalancedClassificationPipeline(
            target_column=config['target_column'],
            random_state=42
        )
        print("✓ Pipeline initialized\n")
        
        # Train model
        print("Training model... (this will take a while)")
        print(f"  Time limit: {config['time_limit']} seconds")
        print(f"  Preset: {config['preset']}")
        print(f"  Resampling: {config['resampling_method']}\n")
        
        predictor = pipeline.train_model(
            train_data=train_data,
            time_limit=config['time_limit'],
            preset=config['preset'],
            eval_metric=config['eval_metric'],
            use_sample_weights=True,
            resampling_method=config['resampling_method']
        )
        
        print("\n✓ Training completed!\n")
        
        # Evaluate
        print("Evaluating model...")
        results = pipeline.comprehensive_evaluation(
            test_data=test_data,
            thresholds=[0.3, 0.5, 0.7]
        )
        
        print("\n✓ Evaluation completed!\n")
        
        # Generate plots
        print("Generating visualizations...")
        pipeline.plot_comprehensive_results(save_path='./comprehensive_evaluation.png')
        pipeline.save_results(filepath='./results.json')
        
        print("✓ Visualizations saved to: ./comprehensive_evaluation.png")
        print("✓ Results saved to: ./results.json")
        
    else:
        # Basic pipeline
        from autogluon_imbalanced_model import (
            create_sample_data,
            train_autogluon_model,
            evaluate_model,
            find_optimal_threshold,
            plot_evaluation_metrics
        )
        import pandas as pd
        from sklearn.model_selection import train_test_split
        
        print("✓ Libraries imported\n")
        
        # Load data
        print("Loading data...")
        if config['data_file'].lower() == 'demo':
            print("  Creating demo dataset...")
            df = create_sample_data()
        else:
            df = pd.read_csv(config['data_file'])
        
        print(f"✓ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        print(f"  Target distribution:\n{df[config['target_column']].value_counts()}\n")
        
        # Split data
        print("Splitting data into train/test (80%/20%)...")
        train_data, test_data = train_test_split(
            df,
            test_size=0.2,
            random_state=42,
            stratify=df[config['target_column']]
        )
        print(f"✓ Train set: {len(train_data)} samples")
        print(f"✓ Test set: {len(test_data)} samples\n")
        
        # Train model
        print("Training model... (this will take a while)")
        print(f"  Time limit: {config['time_limit']} seconds")
        print(f"  Preset: {config['preset']}\n")
        
        predictor = train_autogluon_model(
            train_data=train_data,
            target_column=config['target_column'],
            time_limit=config['time_limit'],
            preset=config['preset'],
            eval_metric=config['eval_metric']
        )
        
        print("\n✓ Training completed!\n")
        
        # Evaluate
        print("Evaluating model at default threshold (0.5)...")
        results = evaluate_model(
            predictor=predictor,
            test_data=test_data,
            target_column=config['target_column'],
            threshold=0.5
        )
        
        print("\n✓ Initial evaluation completed!\n")
        
        # Optimize threshold
        print("Finding optimal threshold...")
        optimal_threshold = find_optimal_threshold(results, metric='f1')
        
        print("\nRe-evaluating with optimal threshold...")
        results_optimal = evaluate_model(
            predictor=predictor,
            test_data=test_data,
            target_column=config['target_column'],
            threshold=optimal_threshold
        )
        
        print("\n✓ Evaluation completed!\n")
        
        # Generate plots
        print("Generating visualizations...")
        plot_evaluation_metrics(results_optimal, save_path='./evaluation_plots.png')
        
        print("✓ Visualizations saved to: ./evaluation_plots.png")
    
    print_header("STEP 4: Results Summary")
    
    print("Model training and evaluation completed successfully!")
    print("\nGenerated files:")
    if config['use_advanced']:
        print("  📊 comprehensive_evaluation.png - Detailed visualizations")
        print("  📄 results.json - Evaluation metrics")
    else:
        print("  📊 evaluation_plots.png - Performance visualizations")
    
    print(f"  🤖 Model saved in: {predictor.path}")
    
    print("\nTo load and use the model later:")
    print("  from autogluon.tabular import TabularPredictor")
    print(f"  predictor = TabularPredictor.load('{predictor.path}')")
    print("  predictions = predictor.predict(new_data)")
    print("  probabilities = predictor.predict_proba(new_data)")
    
    print("\n" + "="*80)
    print("  ✓ ALL DONE!")
    print("="*80)

def main():
    """Main execution"""
    print("""
    ╔══════════════════════════════════════════════════════════════════════════╗
    ║                                                                          ║
    ║         AutoGluon Ensemble Model for Imbalanced Classification          ║
    ║                                                                          ║
    ║  This script will guide you through training a model for your           ║
    ║  highly imbalanced dataset (2.3M passes vs 14.8K declines)              ║
    ║                                                                          ║
    ╚══════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Check dependencies
    if not check_dependencies():
        print("\nPlease install dependencies and run again.")
        sys.exit(1)
    
    # Get configuration
    config = get_user_input()
    
    # Confirm
    print_header("Configuration Summary")
    print("Please confirm your settings:")
    print(f"  Data file: {config['data_file']}")
    print(f"  Target column: {config['target_column']}")
    print(f"  Training time: {config['time_limit']} seconds ({config['time_limit']/60:.0f} minutes)")
    print(f"  Quality preset: {config['preset']}")
    print(f"  Evaluation metric: {config['eval_metric']}")
    print(f"  Advanced features: {'Yes' if config['use_advanced'] else 'No'}")
    if config['use_advanced']:
        print(f"  Resampling method: {config['resampling_method']}")
    
    confirm = input("\nProceed with training? [Y/n]: ").strip().lower()
    if confirm and confirm != 'y':
        print("Training cancelled.")
        sys.exit(0)
    
    # Run model
    try:
        run_model(config)
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error occurred: {str(e)}")
        print("\nFull error details:")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
