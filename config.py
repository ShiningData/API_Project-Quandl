"""
Configuration file for AutoGluon imbalanced classification
Modify these parameters based on your specific requirements
"""

# Data Configuration
DATA_CONFIG = {
    'data_path': 'your_data.csv',  # Path to your dataset
    'target_column': 'target',      # Name of target column
    'test_size': 0.2,               # Proportion of data for testing
    'random_state': 42,             # Random seed for reproducibility
}

# Class Distribution
CLASS_DISTRIBUTION = {
    'n_passes': 2_302_090,
    'n_declines': 14_800,
    'imbalance_ratio': 155.55,  # passes/declines
}

# Training Configuration - OPTIMIZED FOR RECALL
TRAINING_CONFIG = {
    'time_limit': 3600,  # Training time in seconds (3600 = 1 hour)
    
    # Preset quality levels:
    # - 'best_quality': Highest quality, longest training time (RECOMMENDED for production)
    # - 'high_quality': High quality, balanced time
    # - 'good_quality': Good balance of quality and speed
    # - 'medium_quality': Faster training, reasonable quality
    'preset': 'best_quality',
    
    # Evaluation metric for model selection
    # CRITICAL: Using 'recall' to minimize False Negatives
    'eval_metric': 'recall',
    
    # Cross-validation settings
    'num_bag_folds': 8,      # More folds for better minority class coverage
    'num_bag_sets': 1,       # Number of bagging sets
    'num_stack_levels': 2,   # Deeper stacking for better ensemble
    
    # Other settings
    'auto_stack': True,      # Enable automatic stacking
    'verbosity': 2,          # Logging verbosity (0-4)
}

# Hyperparameter Configurations - OPTIMIZED FOR RECALL (Minimize False Negatives)
HYPERPARAMETERS_CONFIG = {
    # Gradient Boosting Machine (LightGBM) - Aggressive minority class focus
    'GBM': {
        'num_boost_round': 2000,
        'learning_rate': 0.005,  # Lower LR for better generalization
        'max_depth': 8,
        'num_leaves': 255,
        'min_child_weight': 1,  # Lower = more sensitive to minority class
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        # scale_pos_weight calculated dynamically based on class imbalance
    },
    
    # XGBoost - Recall optimization
    'XGB': {
        'n_estimators': 2000,
        'learning_rate': 0.005,
        'max_depth': 8,
        'min_child_weight': 1,  # More sensitive
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 0,  # No pruning = catch more patterns
        # scale_pos_weight calculated dynamically
    },
    
    # CatBoost - Balanced weights
    'CAT': {
        'auto_class_weights': 'Balanced',
        'iterations': 2000,
        'learning_rate': 0.005,
        'depth': 8,
        'l2_leaf_reg': 1,  # Lower regularization for more sensitivity
    },
    
    # Random Forest - Aggressive class balancing
    'RF': {
        'class_weight': 'balanced_subsample',  # More aggressive than 'balanced'
        'n_estimators': 500,
        'max_depth': 20,  # Deeper trees for complex patterns
        'min_samples_leaf': 1,  # More sensitive
        'min_samples_split': 2,
        'max_features': 'sqrt',
    },
    
    # Extra Trees - Aggressive balancing
    'XT': {
        'class_weight': 'balanced_subsample',
        'n_estimators': 500,
        'max_depth': 20,
        'min_samples_leaf': 1,
        'min_samples_split': 2,
    },
}

# Evaluation Configuration - RECALL FOCUSED
EVALUATION_CONFIG = {
    'default_threshold': 0.3,  # Lower threshold for better recall (was 0.5)
    'optimize_threshold_metric': 'recall',  # CRITICAL: Optimize for recall
    'metrics_to_calculate': [
        'recall',  # PRIMARY METRIC
        'precision',
        'f1',
        'roc_auc',
        'average_precision',
    ],
    # Cost parameters for False Negatives vs False Positives
    'fn_cost': 10,  # False Negative costs 10x False Positive
    'fp_cost': 1,
}

# Output Configuration
OUTPUT_CONFIG = {
    'model_save_path': './autogluon_models',
    'plots_save_path': './evaluation_plots.png',
    'results_save_path': './results.json',
}

# Imbalanced Data Handling Strategy
IMBALANCE_STRATEGY = {
    # Strategy for handling class imbalance
    # Options: 'sample_weights', 'class_weights', 'both'
    'strategy': 'both',
    
    # Whether to use SMOTE or other sampling techniques (not directly in AutoGluon)
    'use_resampling': False,
    
    # Custom weight calculation method
    # Options: 'balanced', 'effective_samples'
    'weight_method': 'balanced',
}
