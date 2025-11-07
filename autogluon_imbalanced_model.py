"""
AutoGluon Ensemble Model for Highly Imbalanced Binary Classification
Target: Binary (Pass/Decline)
Class Distribution: 2,302,090 passes to 14,800 declines (ratio ~155:1)
Features: 24 numerical variables
"""

import pandas as pd
import numpy as np
from autogluon.tabular import TabularPredictor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def load_and_prepare_data(data_path, target_column='target'):
    """
    Load and prepare the dataset
    
    Parameters:
    -----------
    data_path : str
        Path to your dataset (CSV, parquet, etc.)
    target_column : str
        Name of the target column
    
    Returns:
    --------
    train_data, test_data : pd.DataFrame
    """
    print("Loading data...")
    # Load your data - adjust based on your file format
    df = pd.read_csv(data_path)  # or pd.read_parquet(data_path)
    
    print(f"Dataset shape: {df.shape}")
    print(f"\nTarget distribution:\n{df[target_column].value_counts()}")
    print(f"Imbalance ratio: {df[target_column].value_counts().max() / df[target_column].value_counts().min():.2f}:1")
    
    # Split data - stratified to maintain class distribution
    train_data, test_data = train_test_split(
        df, 
        test_size=0.2, 
        random_state=RANDOM_STATE,
        stratify=df[target_column]
    )
    
    print(f"\nTraining set size: {len(train_data)}")
    print(f"Test set size: {len(test_data)}")
    
    return train_data, test_data


def create_sample_data():
    """
    Create sample imbalanced dataset for demonstration
    Replace this with your actual data loading
    """
    print("Creating sample imbalanced dataset...")
    
    n_passes = 2302090
    n_declines = 14800
    n_features = 24
    
    # For demonstration, using smaller sample
    sample_ratio = 0.01  # Use 1% for faster training in this example
    n_passes_sample = int(n_passes * sample_ratio)
    n_declines_sample = int(n_declines * sample_ratio)
    
    # Generate synthetic data
    np.random.seed(RANDOM_STATE)
    
    # Passes (class 0)
    X_passes = np.random.randn(n_passes_sample, n_features) + np.random.uniform(-0.5, 0.5, n_features)
    y_passes = np.zeros(n_passes_sample)
    
    # Declines (class 1) - slightly different distribution
    X_declines = np.random.randn(n_declines_sample, n_features) + np.random.uniform(0.5, 1.5, n_features)
    y_declines = np.ones(n_declines_sample)
    
    # Combine
    X = np.vstack([X_passes, X_declines])
    y = np.hstack([y_passes, y_declines])
    
    # Create DataFrame
    feature_names = [f'feature_{i+1}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y.astype(int)
    
    # Shuffle
    df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    
    print(f"Sample dataset created: {df.shape}")
    print(f"Target distribution:\n{df['target'].value_counts()}")
    
    return df


def train_autogluon_model(
    train_data, 
    target_column='target',
    time_limit=3600,
    preset='best_quality',
    eval_metric='recall'
):
    """
    Train AutoGluon ensemble model optimized for RECALL (minimizing False Negatives)
    
    Parameters:
    -----------
    train_data : pd.DataFrame
        Training dataset
    target_column : str
        Name of target column
    time_limit : int
        Training time limit in seconds
    preset : str
        AutoGluon preset quality level
        Options: 'best_quality', 'high_quality', 'good_quality', 'medium_quality'
    eval_metric : str
        Evaluation metric for model selection (default: 'recall' for FN minimization)
    
    Returns:
    --------
    predictor : TabularPredictor
    """
    print("\n" + "="*80)
    print("TRAINING AUTOGLUON MODEL - OPTIMIZED FOR RECALL")
    print("Goal: MINIMIZE FALSE NEGATIVES (Catch all declines)")
    print("="*80)
    
    # Calculate sample weights with AGGRESSIVE weighting for minority class
    class_counts = train_data[target_column].value_counts()
    minority_class = class_counts.idxmin()
    majority_class = class_counts.idxmax()
    
    # AGGRESSIVE weighting: 3x stronger emphasis on minority class
    imbalance_ratio = class_counts[majority_class] / class_counts[minority_class]
    weight_minority = imbalance_ratio * 1.5  # Extra boost for recall
    weight_majority = 1.0
    
    print(f"\nImbalance Ratio: {imbalance_ratio:.2f}:1")
    print(f"Aggressive Class Weights (Recall-Optimized):")
    print(f"  Majority class ({majority_class}): {weight_majority:.4f}")
    print(f"  Minority class ({minority_class}): {weight_minority:.4f}")
    print(f"  → Minority class weighted {weight_minority:.1f}x higher")
    
    # Add sample weights column
    train_data_weighted = train_data.copy()
    train_data_weighted['sample_weight'] = train_data_weighted[target_column].map({
        majority_class: weight_majority,
        minority_class: weight_minority
    })
    
    # Hyperparameters optimized for RECALL (catching minority class)
    # Scale_pos_weight is CRITICAL for gradient boosting models
    scale_pos_weight = weight_minority / weight_majority
    
    hyperparameters = {
        'GBM': {
            # LightGBM with aggressive minority class focus
            'scale_pos_weight': scale_pos_weight,
            'num_boost_round': 2000,
            'learning_rate': 0.005,  # Lower LR for better generalization
            'max_depth': 8,
            'num_leaves': 255,
            'min_child_weight': 1,  # Lower = more sensitive to minority class
            'subsample': 0.8,
            'colsample_bytree': 0.8,
        },
        'XGB': {
            # XGBoost with recall optimization
            'scale_pos_weight': scale_pos_weight,
            'n_estimators': 2000,
            'learning_rate': 0.005,
            'max_depth': 8,
            'min_child_weight': 1,  # More sensitive
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0,  # No pruning = catch more patterns
        },
        'CAT': {
            # CatBoost with balanced weights
            'auto_class_weights': 'Balanced',
            'iterations': 2000,
            'learning_rate': 0.005,
            'depth': 8,
            'l2_leaf_reg': 1,  # Lower regularization
        },
        'RF': {
            # Random Forest with aggressive class balancing
            'class_weight': 'balanced_subsample',  # More aggressive than 'balanced'
            'n_estimators': 500,
            'max_depth': 20,  # Deeper trees
            'min_samples_leaf': 1,  # More sensitive
            'min_samples_split': 2,
            'max_features': 'sqrt',
        },
        'XT': {
            # Extra Trees with aggressive balancing
            'class_weight': 'balanced_subsample',
            'n_estimators': 500,
            'max_depth': 20,
            'min_samples_leaf': 1,
            'min_samples_split': 2,
        },
    }
    
    # Initialize predictor with RECALL as primary metric
    predictor = TabularPredictor(
        label=target_column,
        eval_metric=eval_metric,
        problem_type='binary',
        sample_weight='sample_weight',
        path=f'./autogluon_recall_optimized_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    )
    
    # Train with settings optimized for recall
    print(f"\nTraining Configuration (Recall-Optimized):")
    print(f"  Time limit: {time_limit} seconds ({time_limit/60:.1f} minutes)")
    print(f"  Preset: {preset}")
    print(f"  Eval metric: {eval_metric} ← MINIMIZE FALSE NEGATIVES")
    print(f"  Sample weights: Enabled (Aggressive)")
    print(f"  Cross-validation folds: 8 (more folds = better minority class coverage)")
    
    predictor.fit(
        train_data=train_data_weighted,
        time_limit=time_limit,
        presets=preset,
        hyperparameters=hyperparameters,
        num_bag_folds=8,  # More folds for better minority class representation
        num_bag_sets=1,
        num_stack_levels=2,  # Deeper stacking for better ensemble
        verbosity=2,
        auto_stack=True,
    )
    
    print("\n" + "="*80)
    print("TRAINING COMPLETED - RECALL-OPTIMIZED MODEL READY")
    print("="*80)
    
    return predictor


def evaluate_model(predictor, test_data, target_column='target', threshold=0.5):
    """
    Comprehensive evaluation focused on RECALL and FALSE NEGATIVES
    
    Parameters:
    -----------
    predictor : TabularPredictor
        Trained AutoGluon predictor
    test_data : pd.DataFrame
        Test dataset
    target_column : str
        Name of target column
    threshold : float
        Classification threshold (default 0.5, but lower threshold recommended for recall)
    
    Returns:
    --------
    results : dict
        Dictionary containing evaluation metrics
    """
    print("\n" + "="*80)
    print("MODEL EVALUATION - RECALL OPTIMIZATION FOCUS")
    print("="*80)
    
    # Get predictions
    y_true = test_data[target_column]
    y_pred_proba = predictor.predict_proba(test_data.drop(columns=[target_column]))
    
    # Get probability for positive class
    if isinstance(y_pred_proba, pd.DataFrame):
        y_pred_proba_pos = y_pred_proba.iloc[:, 1].values
    else:
        y_pred_proba_pos = y_pred_proba
    
    y_pred = (y_pred_proba_pos >= threshold).astype(int)
    
    # Calculate metrics with focus on minority class
    roc_auc = roc_auc_score(y_true, y_pred_proba_pos)
    avg_precision = average_precision_score(y_true, y_pred_proba_pos)
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Calculate additional metrics
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = recall  # Same as recall
    
    print(f"\n{'='*80}")
    print(f"CRITICAL METRICS - MINIMIZE FALSE NEGATIVES")
    print(f"{'='*80}")
    print(f"\n🎯 PRIMARY GOAL: RECALL (Catch all declines)")
    print(f"   Recall: {recall:.4f} ({recall*100:.2f}%)")
    print(f"   → Out of {tp+fn} actual declines, we caught {tp}")
    print(f"   ❌ FALSE NEGATIVES: {fn} ← MINIMIZE THIS!")
    
    print(f"\n📊 CONFUSION MATRIX:")
    print(f"   True Negatives (TN):  {tn:,} (Correctly predicted passes)")
    print(f"   False Positives (FP): {fp:,} (Incorrectly flagged as declines)")
    print(f"   False Negatives (FN): {fn:,} ← MISSED DECLINES (BAD!)")
    print(f"   True Positives (TP):  {tp:,} (Correctly caught declines)")
    
    print(f"\n📈 SUPPORTING METRICS:")
    print(f"   Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"   F1 Score: {f1:.4f}")
    print(f"   ROC-AUC: {roc_auc:.4f}")
    print(f"   Average Precision: {avg_precision:.4f}")
    print(f"   Specificity: {specificity:.4f}")
    
    # Calculate cost-based metric (customize based on business needs)
    # Example: FN costs 10x more than FP
    fn_cost = 10
    fp_cost = 1
    total_cost = (fn * fn_cost) + (fp * fp_cost)
    print(f"\n💰 COST ANALYSIS (FN cost = {fn_cost}x FP cost):")
    print(f"   Total Cost: {total_cost:,} units")
    print(f"   FN Cost: {fn * fn_cost:,} ({fn} × {fn_cost})")
    print(f"   FP Cost: {fp * fp_cost:,} ({fp} × {fp_cost})")
    
    print("\n" + "="*80)
    print("Classification Report:")
    print("="*80)
    print(classification_report(y_true, y_pred, target_names=['Pass', 'Decline'], digits=4))
    
    # Feature importance
    print("\n" + "="*80)
    print("Top 15 Feature Importances:")
    print("="*80)
    feature_importance = predictor.feature_importance(test_data)
    print(feature_importance.head(15).to_string())
    
    # Model leaderboard
    print("\n" + "="*80)
    print("Model Leaderboard (Top 10):")
    print("="*80)
    leaderboard = predictor.leaderboard(test_data, silent=True)
    print(leaderboard.head(10)[['model', 'score_val', 'score_test', 'fit_time']].to_string())
    
    results = {
        'roc_auc': roc_auc,
        'average_precision': avg_precision,
        'recall': recall,
        'precision': precision,
        'f1_score': f1,
        'specificity': specificity,
        'confusion_matrix': cm,
        'true_negatives': tn,
        'false_positives': fp,
        'false_negatives': fn,
        'true_positives': tp,
        'total_cost': total_cost,
        'feature_importance': feature_importance,
        'leaderboard': leaderboard,
        'y_true': y_true,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba_pos
    }
    
    return results


def plot_evaluation_metrics(results, save_path='./evaluation_plots.png'):
    """
    Create visualization of model performance
    
    Parameters:
    -----------
    results : dict
        Results from evaluate_model function
    save_path : str
        Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Confusion Matrix
    cm = results['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
    axes[0, 0].set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('True Label')
    axes[0, 0].set_xlabel('Predicted Label')
    axes[0, 0].set_xticklabels(['Pass', 'Decline'])
    axes[0, 0].set_yticklabels(['Pass', 'Decline'])
    
    # 2. Precision-Recall Curve
    precision, recall, thresholds = precision_recall_curve(
        results['y_true'], 
        results['y_pred_proba']
    )
    axes[0, 1].plot(recall, precision, linewidth=2)
    axes[0, 1].set_xlabel('Recall')
    axes[0, 1].set_ylabel('Precision')
    axes[0, 1].set_title(
        f'Precision-Recall Curve (AP={results["average_precision"]:.3f})', 
        fontsize=14, 
        fontweight='bold'
    )
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Feature Importance (Top 10)
    feature_imp = results['feature_importance'].head(10)
    axes[1, 0].barh(range(len(feature_imp)), feature_imp['importance'])
    axes[1, 0].set_yticks(range(len(feature_imp)))
    axes[1, 0].set_yticklabels(feature_imp['feature'])
    axes[1, 0].set_xlabel('Importance')
    axes[1, 0].set_title('Top 10 Feature Importances', fontsize=14, fontweight='bold')
    axes[1, 0].invert_yaxis()
    
    # 4. Prediction Distribution
    axes[1, 1].hist(
        results['y_pred_proba'][results['y_true'] == 0], 
        bins=50, 
        alpha=0.6, 
        label='Pass (Actual)', 
        color='blue'
    )
    axes[1, 1].hist(
        results['y_pred_proba'][results['y_true'] == 1], 
        bins=50, 
        alpha=0.6, 
        label='Decline (Actual)', 
        color='red'
    )
    axes[1, 1].set_xlabel('Predicted Probability')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Prediction Distribution by True Class', fontsize=14, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nEvaluation plots saved to: {save_path}")
    plt.close()


def find_optimal_threshold(results, metric='recall', fn_cost=10, fp_cost=1):
    """
    Find optimal classification threshold optimized for RECALL and minimizing False Negatives
    
    Parameters:
    -----------
    results : dict
        Results from evaluate_model function
    metric : str
        Metric to optimize ('recall', 'f1', 'cost_sensitive')
    fn_cost : float
        Cost of False Negative (default: 10)
    fp_cost : float
        Cost of False Positive (default: 1)
    
    Returns:
    --------
    optimal_threshold : float
    """
    from sklearn.metrics import precision_score, recall_score
    
    print("\n" + "="*80)
    print(f"THRESHOLD OPTIMIZATION - {metric.upper()}")
    print("="*80)
    
    thresholds = np.linspace(0.01, 0.99, 99)
    scores = []
    fn_counts = []
    
    for threshold in thresholds:
        y_pred = (results['y_pred_proba'] >= threshold).astype(int)
        
        if metric == 'recall':
            score = recall_score(results['y_true'], y_pred, zero_division=0)
        elif metric == 'f1':
            score = f1_score(results['y_true'], y_pred, zero_division=0)
        elif metric == 'precision':
            score = precision_score(results['y_true'], y_pred, zero_division=0)
        elif metric == 'cost_sensitive':
            # Minimize cost (FN weighted heavily)
            cm = confusion_matrix(results['y_true'], y_pred)
            tn, fp, fn, tp = cm.ravel()
            cost = (fn * fn_cost) + (fp * fp_cost)
            score = -cost  # Negative because we want to maximize score (minimize cost)
        else:
            score = recall_score(results['y_true'], y_pred, zero_division=0)
        
        scores.append(score)
        
        # Track False Negatives
        cm = confusion_matrix(results['y_true'], y_pred)
        fn_counts.append(cm[1, 0])
    
    optimal_idx = np.argmax(scores)
    optimal_threshold = thresholds[optimal_idx]
    optimal_score = scores[optimal_idx]
    optimal_fn = fn_counts[optimal_idx]
    
    # Also find threshold for maximum recall (usually lower threshold)
    recall_scores = []
    for threshold in thresholds:
        y_pred = (results['y_pred_proba'] >= threshold).astype(int)
        recall_scores.append(recall_score(results['y_true'], y_pred, zero_division=0))
    
    max_recall_idx = np.argmax(recall_scores)
    max_recall_threshold = thresholds[max_recall_idx]
    max_recall = recall_scores[max_recall_idx]
    max_recall_fn = fn_counts[max_recall_idx]
    
    print(f"\n🎯 OPTIMAL THRESHOLD (optimizing {metric}):")
    print(f"   Threshold: {optimal_threshold:.3f}")
    print(f"   {metric.capitalize()}: {optimal_score if metric != 'cost_sensitive' else -optimal_score:.4f}")
    print(f"   False Negatives: {optimal_fn}")
    
    print(f"\n🎯 MAXIMUM RECALL THRESHOLD:")
    print(f"   Threshold: {max_recall_threshold:.3f}")
    print(f"   Recall: {max_recall:.4f} ({max_recall*100:.2f}%)")
    print(f"   False Negatives: {max_recall_fn}")
    
    print(f"\n💡 RECOMMENDATION:")
    if metric == 'recall':
        print(f"   Use threshold: {optimal_threshold:.3f}")
        print(f"   This minimizes False Negatives while maintaining reasonable precision")
    else:
        print(f"   For maximum recall (catch all declines): {max_recall_threshold:.3f}")
        print(f"   For balanced approach: {optimal_threshold:.3f}")
    
    # Show trade-off at different thresholds
    print(f"\n📊 THRESHOLD COMPARISON:")
    print(f"{'Threshold':<12} {'Recall':<10} {'Precision':<12} {'F1':<10} {'FN':<8}")
    print("-" * 52)
    
    for t in [0.1, 0.2, 0.3, optimal_threshold, 0.5, 0.7]:
        y_pred = (results['y_pred_proba'] >= t).astype(int)
        r = recall_score(results['y_true'], y_pred, zero_division=0)
        p = precision_score(results['y_true'], y_pred, zero_division=0)
        f = f1_score(results['y_true'], y_pred, zero_division=0)
        cm = confusion_matrix(results['y_true'], y_pred)
        fn = cm[1, 0]
        marker = " ← Optimal" if abs(t - optimal_threshold) < 0.01 else ""
        print(f"{t:<12.3f} {r:<10.4f} {p:<12.4f} {f:<10.4f} {fn:<8}{marker}")
    
    return optimal_threshold


def main():
    """
    Main execution flow - OPTIMIZED FOR RECALL (Minimize False Negatives)
    """
    print("="*80)
    print("AUTOGLUON RECALL-OPTIMIZED CLASSIFICATION PIPELINE")
    print("Goal: MINIMIZE FALSE NEGATIVES - Catch ALL Declines")
    print("="*80)
    
    # Option 1: Load your actual data
    # train_data, test_data = load_and_prepare_data('your_data.csv', target_column='target')
    
    # Option 2: Use sample data for demonstration
    df = create_sample_data()
    train_data, test_data = train_test_split(
        df, 
        test_size=0.2, 
        random_state=RANDOM_STATE,
        stratify=df['target']
    )
    
    # Train model with RECALL optimization
    predictor = train_autogluon_model(
        train_data=train_data,
        target_column='target',
        time_limit=600,  # Adjust based on your needs (600=10min, 1800=30min, 3600=1hr)
        preset='good_quality',  # Use 'best_quality' for production
        eval_metric='recall'  # CRITICAL: Optimizing for recall
    )
    
    # Evaluate model with lower threshold (better for recall)
    print("\n" + "="*80)
    print("INITIAL EVALUATION - Default Threshold (0.5)")
    print("="*80)
    results = evaluate_model(
        predictor=predictor,
        test_data=test_data,
        target_column='target',
        threshold=0.5
    )
    
    # Find optimal threshold for recall
    print("\n" + "="*80)
    print("FINDING OPTIMAL THRESHOLD FOR RECALL")
    print("="*80)
    optimal_threshold = find_optimal_threshold(results, metric='recall')
    
    # Re-evaluate with optimal threshold
    print("\n" + "="*80)
    print(f"FINAL EVALUATION - Optimal Threshold ({optimal_threshold:.3f})")
    print("="*80)
    results_optimal = evaluate_model(
        predictor=predictor,
        test_data=test_data,
        target_column='target',
        threshold=optimal_threshold
    )
    
    # Also evaluate with aggressive threshold for maximum recall
    aggressive_threshold = 0.2  # Lower threshold = higher recall
    print("\n" + "="*80)
    print(f"AGGRESSIVE EVALUATION - Very Low Threshold ({aggressive_threshold})")
    print("This maximizes recall but may increase false positives")
    print("="*80)
    results_aggressive = evaluate_model(
        predictor=predictor,
        test_data=test_data,
        target_column='target',
        threshold=aggressive_threshold
    )
    
    # Create visualizations
    plot_evaluation_metrics(results_optimal, save_path='./evaluation_plots_optimal.png')
    plot_evaluation_metrics(results_aggressive, save_path='./evaluation_plots_aggressive.png')
    
    # Save model
    print(f"\n{'='*80}")
    print("MODEL SAVED AND READY FOR USE")
    print(f"{'='*80}")
    print(f"\nModel location: {predictor.path}")
    print(f"\nGenerated files:")
    print(f"  📊 evaluation_plots_optimal.png (threshold={optimal_threshold:.3f})")
    print(f"  📊 evaluation_plots_aggressive.png (threshold={aggressive_threshold})")
    
    print(f"\n💡 RECOMMENDATIONS:")
    print(f"   1. Use threshold {optimal_threshold:.3f} for balanced recall optimization")
    print(f"   2. Use threshold {aggressive_threshold} for maximum recall (catches more declines)")
    print(f"   3. Adjust threshold based on business cost of False Negatives vs False Positives")
    
    print("\nTo load the model later:")
    print(f"  from autogluon.tabular import TabularPredictor")
    print(f"  predictor = TabularPredictor.load('{predictor.path}')")
    print(f"  probabilities = predictor.predict_proba(new_data)")
    print(f"  predictions = (probabilities.iloc[:, 1] >= {optimal_threshold:.3f}).astype(int)")
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETED - RECALL-OPTIMIZED MODEL READY")
    print("="*80)
    
    return predictor, results_optimal


if __name__ == "__main__":
    predictor, results = main()
