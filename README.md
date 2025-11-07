# AutoGluon Ensemble Model for Imbalanced Binary Classification

This repository contains Python scripts for building an ensemble model using AutoGluon Tabular to handle highly imbalanced binary classification tasks.

## Dataset Specifications

- **Target Variable**: Binary (Pass/Decline)
- **Class Distribution**: 
  - Passes: 2,302,090
  - Declines: 14,800
  - Imbalance Ratio: ~155:1
- **Features**: 24 numerical input variables

## Features

- ✅ **AutoGluon Tabular** ensemble modeling with automatic hyperparameter tuning
- ✅ **Imbalanced data handling**: Sample weights, class weights, and resampling techniques
- ✅ **Multiple resampling methods**: SMOTE, ADASYN, BorderlineSMOTE, etc.
- ✅ **Threshold optimization** for different metrics (F1, Precision, Recall)
- ✅ **Comprehensive evaluation**: ROC-AUC, Average Precision, F1, MCC, Cohen's Kappa
- ✅ **Model interpretability**: Feature importance and model leaderboard
- ✅ **Visualization**: ROC curves, PR curves, confusion matrices, and more

## Installation

### Option 1: Using pip

```bash
pip install -r requirements.txt
```

### Option 2: Install AutoGluon separately

```bash
# For CPU only
pip install autogluon.tabular[all]

# For GPU support (if available)
pip install autogluon.tabular[all,ray]

# Install other dependencies
pip install pandas numpy scikit-learn matplotlib seaborn imbalanced-learn
```

## Files Overview

1. **`autogluon_imbalanced_model.py`** - Main script with core functionality
   - Data loading and preparation
   - AutoGluon model training with hyperparameter tuning
   - Model evaluation and visualization
   - Threshold optimization

2. **`advanced_autogluon_pipeline.py`** - Advanced features
   - Resampling techniques (SMOTE, ADASYN, etc.)
   - Cost-sensitive threshold optimization
   - Comprehensive multi-threshold evaluation
   - Enhanced visualization and reporting

3. **`config.py`** - Configuration file
   - Easy parameter adjustment
   - Hyperparameter configurations for different models
   - Training and evaluation settings

4. **`requirements.txt`** - Python package dependencies

## Quick Start

### Basic Usage

```python
from autogluon_imbalanced_model import main
import pandas as pd

# Load your data
df = pd.read_csv('your_data.csv')

# The main function will:
# 1. Split data into train/test
# 2. Train AutoGluon ensemble with imbalanced handling
# 3. Evaluate model performance
# 4. Generate visualizations
predictor, results = main()
```

### Custom Configuration

```python
from autogluon_imbalanced_model import (
    load_and_prepare_data,
    train_autogluon_model,
    evaluate_model,
    find_optimal_threshold,
    plot_evaluation_metrics
)

# Load your data
train_data, test_data = load_and_prepare_data(
    data_path='your_data.csv',
    target_column='target'
)

# Train model with custom settings
predictor = train_autogluon_model(
    train_data=train_data,
    target_column='target',
    time_limit=3600,  # 1 hour
    preset='best_quality',  # or 'high_quality', 'good_quality', 'medium_quality'
    eval_metric='roc_auc'  # or 'average_precision', 'f1', 'precision', 'recall'
)

# Evaluate model
results = evaluate_model(
    predictor=predictor,
    test_data=test_data,
    target_column='target',
    threshold=0.5
)

# Find optimal threshold
optimal_threshold = find_optimal_threshold(results, metric='f1')

# Re-evaluate with optimal threshold
results_optimal = evaluate_model(
    predictor=predictor,
    test_data=test_data,
    target_column='target',
    threshold=optimal_threshold
)

# Generate plots
plot_evaluation_metrics(results_optimal)
```

### Advanced Pipeline with Resampling

```python
from advanced_autogluon_pipeline import ImbalancedClassificationPipeline
import pandas as pd
from sklearn.model_selection import train_test_split

# Initialize pipeline
pipeline = ImbalancedClassificationPipeline(
    target_column='target',
    random_state=42
)

# Load data
df = pd.read_csv('your_data.csv')
train_data, test_data = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df['target']
)

# Train with SMOTE resampling
predictor = pipeline.train_model(
    train_data=train_data,
    time_limit=3600,
    preset='best_quality',
    eval_metric='roc_auc',
    use_sample_weights=True,
    resampling_method='smote',  # Options: 'smote', 'adasyn', 'borderline_smote', etc.
)

# Comprehensive evaluation at multiple thresholds
results = pipeline.comprehensive_evaluation(
    test_data=test_data,
    thresholds=[0.3, 0.5, 0.7]
)

# Generate comprehensive visualizations
pipeline.plot_comprehensive_results()

# Save results
pipeline.save_results(filepath='./results.json')
```

## Hyperparameter Tuning

AutoGluon automatically tunes hyperparameters, but you can specify custom configurations:

```python
hyperparameters = {
    'GBM': [
        {
            'num_boost_round': 1000,
            'learning_rate': 0.01,
            'max_depth': 6,
        },
        {
            'num_boost_round': 500,
            'learning_rate': 0.05,
            'max_depth': 8,
        }
    ],
    'XGB': [
        {
            'n_estimators': 1000,
            'learning_rate': 0.01,
            'max_depth': 6,
        }
    ],
    'CAT': [
        {
            'iterations': 1000,
            'learning_rate': 0.01,
            'depth': 6,
        }
    ],
    'RF': [
        {
            'n_estimators': 300,
            'max_depth': 15,
        }
    ],
}

predictor.fit(
    train_data=train_data,
    hyperparameters=hyperparameters,
    # ... other parameters
)
```

## Imbalanced Data Handling Strategies

### 1. Sample Weights (Recommended for Large Datasets)
```python
# Automatically applied in the script
# Weights are inversely proportional to class frequency
```

### 2. Class Weights
```python
# Applied in model-specific hyperparameters
# Example: 'scale_pos_weight' for XGBoost/LightGBM
# Example: 'class_weight': 'balanced' for Random Forest
```

### 3. Resampling Techniques

#### SMOTE (Synthetic Minority Over-sampling)
```python
pipeline.train_model(
    train_data=train_data,
    resampling_method='smote',
    # ...
)
```

#### ADASYN (Adaptive Synthetic Sampling)
```python
pipeline.train_model(
    train_data=train_data,
    resampling_method='adasyn',
    # ...
)
```

#### Combined Techniques
```python
# SMOTE + Tomek Links
resampling_method='smote_tomek'

# SMOTE + Edited Nearest Neighbors
resampling_method='smote_enn'
```

## Evaluation Metrics

For imbalanced classification, the script calculates:

- **ROC-AUC**: Overall discriminative ability
- **Average Precision (AP)**: Area under PR curve
- **F1 Score**: Harmonic mean of precision and recall
- **Precision**: Minimize false positives
- **Recall**: Minimize false negatives
- **Matthews Correlation Coefficient (MCC)**: Balanced measure
- **Cohen's Kappa**: Agreement beyond chance
- **Balanced Accuracy**: Average of sensitivity and specificity

## Threshold Optimization

The script optimizes classification thresholds for different objectives:

```python
# Optimize for F1 score
threshold_f1 = find_optimal_threshold(results, metric='f1')

# Optimize for precision (minimize false positives)
threshold_precision = find_optimal_threshold(results, metric='precision')

# Optimize for recall (minimize false negatives)
threshold_recall = find_optimal_threshold(results, metric='recall')

# Optimize using Youden's J statistic
threshold_j = find_optimal_threshold(results, metric='j_statistic')
```

## Model Interpretability

### Feature Importance
```python
feature_importance = predictor.feature_importance(test_data)
print(feature_importance)
```

### Model Leaderboard
```python
leaderboard = predictor.leaderboard(test_data)
print(leaderboard)
```

### Best Model Details
```python
best_model = predictor.get_model_best()
print(f"Best model: {best_model}")
```

## Output Files

After running the scripts, you'll get:

1. **Trained Model** - Saved in `./autogluon_models_YYYYMMDD_HHMMSS/`
2. **Evaluation Plots** - `evaluation_plots.png` or `comprehensive_evaluation.png`
3. **Threshold Optimization Plots** - `threshold_optimization_*.png`
4. **Results JSON** - `results.json` with all metrics
5. **Model Artifacts** - Feature importance, leaderboard, etc.

## Loading Saved Models

```python
from autogluon.tabular import TabularPredictor

# Load the trained model
predictor = TabularPredictor.load('./autogluon_models_YYYYMMDD_HHMMSS/')

# Make predictions on new data
predictions = predictor.predict(new_data)
prediction_probabilities = predictor.predict_proba(new_data)
```

## Performance Tips

### For Large Datasets
- Use `preset='medium_quality'` or `preset='good_quality'` for faster training
- Reduce `num_bag_folds` to 3 instead of 5
- Set appropriate `time_limit` based on your compute resources

### For Best Performance
- Use `preset='best_quality'`
- Increase `time_limit` (e.g., 3600 seconds or more)
- Use `num_bag_folds=5` or higher
- Enable GPU support if available

### Memory Optimization
```python
# For large datasets, consider sampling during development
train_sample = train_data.sample(frac=0.1, random_state=42)
```

## Troubleshooting

### Issue: Out of Memory
**Solution**: Reduce `num_bag_folds`, use data sampling, or increase system memory

### Issue: Training Takes Too Long
**Solution**: Reduce `time_limit`, use lower quality preset, or reduce `num_bag_folds`

### Issue: Poor Performance on Minority Class
**Solutions**:
- Try different resampling methods (SMOTE, ADASYN)
- Adjust `scale_pos_weight` in hyperparameters
- Use `eval_metric='average_precision'` or `eval_metric='f1'`
- Optimize threshold for recall

### Issue: Model Overfitting
**Solutions**:
- Increase regularization in hyperparameters
- Reduce model complexity (max_depth, num_leaves)
- Use more cross-validation folds

## References

- [AutoGluon Documentation](https://auto.gluon.ai/)
- [Imbalanced-learn Documentation](https://imbalanced-learn.org/)
- [Handling Imbalanced Data](https://machinelearningmastery.com/tactics-to-combat-imbalanced-classes-in-your-machine-learning-dataset/)

## License

MIT License - feel free to use and modify for your projects.

## Contact

For questions or issues, please open an issue in the repository.
