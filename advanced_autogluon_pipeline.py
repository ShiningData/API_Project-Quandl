"""
Advanced AutoGluon Script with Enhanced Imbalanced Data Handling
Features:
- SMOTE/ADASYN resampling options
- Advanced threshold optimization
- Cost-sensitive learning
- Detailed model comparison
"""

import pandas as pd
import numpy as np
from autogluon.tabular import TabularPredictor
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek, SMOTEENN
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    cohen_kappa_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')


class ImbalancedClassificationPipeline:
    """
    Comprehensive pipeline for imbalanced binary classification using AutoGluon
    """
    
    def __init__(self, target_column='target', random_state=42):
        self.target_column = target_column
        self.random_state = random_state
        self.predictor = None
        self.results = {}
        
    def apply_resampling(self, X_train, y_train, method='smote', sampling_strategy='auto'):
        """
        Apply resampling techniques for imbalanced data
        
        Parameters:
        -----------
        X_train : pd.DataFrame
            Training features
        y_train : pd.Series
            Training labels
        method : str
            Resampling method: 'smote', 'adasyn', 'borderline_smote', 
            'smote_tomek', 'smote_enn', 'random_undersample', 'none'
        sampling_strategy : str or float
            Sampling strategy
        
        Returns:
        --------
        X_resampled, y_resampled : Resampled data
        """
        print(f"\nApplying resampling method: {method}")
        print(f"Original class distribution: {y_train.value_counts().to_dict()}")
        
        if method == 'none':
            return X_train, y_train
        
        # Select resampling method
        resamplers = {
            'smote': SMOTE(random_state=self.random_state, sampling_strategy=sampling_strategy),
            'adasyn': ADASYN(random_state=self.random_state, sampling_strategy=sampling_strategy),
            'borderline_smote': BorderlineSMOTE(random_state=self.random_state, sampling_strategy=sampling_strategy),
            'smote_tomek': SMOTETomek(random_state=self.random_state, sampling_strategy=sampling_strategy),
            'smote_enn': SMOTEENN(random_state=self.random_state, sampling_strategy=sampling_strategy),
            'random_undersample': RandomUnderSampler(random_state=self.random_state, sampling_strategy=sampling_strategy),
        }
        
        if method not in resamplers:
            raise ValueError(f"Unknown resampling method: {method}")
        
        resampler = resamplers[method]
        X_resampled, y_resampled = resampler.fit_resample(X_train, y_train)
        
        print(f"Resampled class distribution: {pd.Series(y_resampled).value_counts().to_dict()}")
        
        # Convert back to DataFrame
        X_resampled = pd.DataFrame(X_resampled, columns=X_train.columns)
        y_resampled = pd.Series(y_resampled, name=y_train.name)
        
        return X_resampled, y_resampled
    
    def calculate_sample_weights(self, y_train, method='balanced'):
        """
        Calculate sample weights for imbalanced data
        
        Parameters:
        -----------
        y_train : pd.Series
            Training labels
        method : str
            Weight calculation method: 'balanced', 'effective_samples'
        
        Returns:
        --------
        weights : np.array
        """
        class_counts = y_train.value_counts()
        
        if method == 'balanced':
            # Standard balanced weights: n_samples / (n_classes * n_samples_per_class)
            weights = len(y_train) / (len(class_counts) * y_train.map(class_counts))
        
        elif method == 'effective_samples':
            # Effective number of samples (for extreme imbalance)
            beta = 0.9999
            effective_num = 1.0 - np.power(beta, class_counts)
            weights_per_class = (1.0 - beta) / effective_num
            weights_per_class = weights_per_class / weights_per_class.sum() * len(class_counts)
            weights = y_train.map(weights_per_class)
        
        else:
            raise ValueError(f"Unknown weight method: {method}")
        
        return weights.values
    
    def train_model(
        self,
        train_data,
        time_limit=3600,
        preset='best_quality',
        eval_metric='recall',
        use_sample_weights=True,
        resampling_method='none',
        hyperparameters=None
    ):
        """
        Train AutoGluon model optimized for RECALL (minimizing False Negatives)
        """
        print("\n" + "="*80)
        print("TRAINING AUTOGLUON MODEL - RECALL OPTIMIZED")
        print("Goal: MINIMIZE FALSE NEGATIVES")
        print("="*80)
        
        X_train = train_data.drop(columns=[self.target_column])
        y_train = train_data[self.target_column]
        
        # Apply resampling if specified
        if resampling_method != 'none':
            X_train, y_train = self.apply_resampling(
                X_train, 
                y_train, 
                method=resampling_method,
                sampling_strategy='auto'
            )
            train_data = pd.concat([X_train, y_train], axis=1)
        
        # Calculate AGGRESSIVE class weights for recall optimization
        class_counts = y_train.value_counts()
        minority_class = class_counts.idxmin()
        majority_class = class_counts.idxmax()
        
        imbalance_ratio = class_counts[majority_class] / class_counts[minority_class]
        # AGGRESSIVE: Weight minority class 2x the imbalance ratio
        scale_pos_weight = imbalance_ratio * 2.0
        
        print(f"\nImbalance ratio: {imbalance_ratio:.2f}:1")
        print(f"Aggressive scale_pos_weight: {scale_pos_weight:.2f} (2x imbalance ratio)")
        
        # Add sample weights if specified
        train_data_final = train_data.copy()
        sample_weight_col = None
        
        if use_sample_weights:
            # Aggressive weighting for minority class
            weights = y_train.map({
                majority_class: 1.0,
                minority_class: scale_pos_weight / 2  # Balance with scale_pos_weight
            }).values
            train_data_final['sample_weight'] = weights
            sample_weight_col = 'sample_weight'
            print(f"Aggressive sample weights applied")
        
        # Hyperparameters optimized for RECALL
        if hyperparameters is None:
            hyperparameters = {
                'GBM': {
                    'scale_pos_weight': scale_pos_weight,
                    'num_boost_round': 2000,
                    'learning_rate': 0.005,
                    'max_depth': 8,
                    'num_leaves': 255,
                    'min_child_weight': 1,  # More sensitive
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                },
                'XGB': {
                    'scale_pos_weight': scale_pos_weight,
                    'n_estimators': 2000,
                    'learning_rate': 0.005,
                    'max_depth': 8,
                    'min_child_weight': 1,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'gamma': 0,
                },
                'CAT': {
                    'auto_class_weights': 'Balanced',
                    'iterations': 2000,
                    'learning_rate': 0.005,
                    'depth': 8,
                    'l2_leaf_reg': 1,
                },
                'RF': {
                    'class_weight': 'balanced_subsample',
                    'n_estimators': 500,
                    'max_depth': 20,
                    'min_samples_leaf': 1,
                    'min_samples_split': 2,
                },
                'XT': {
                    'class_weight': 'balanced_subsample',
                    'n_estimators': 500,
                    'max_depth': 20,
                    'min_samples_leaf': 1,
                },
            }
        
        # Initialize and train predictor
        self.predictor = TabularPredictor(
            label=self.target_column,
            eval_metric=eval_metric,
            problem_type='binary',
            sample_weight=sample_weight_col,
            path=f'./autogluon_recall_optimized_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        )
        
        print(f"\nTraining with:")
        print(f"  Time limit: {time_limit}s ({time_limit/60:.1f} min)")
        print(f"  Preset: {preset}")
        print(f"  Eval metric: {eval_metric} ← MINIMIZE FALSE NEGATIVES")
        print(f"  Resampling: {resampling_method}")
        print(f"  CV folds: 8 (better minority class coverage)")
        
        self.predictor.fit(
            train_data=train_data_final,
            time_limit=time_limit,
            presets=preset,
            hyperparameters=hyperparameters,
            num_bag_folds=8,  # More folds for better minority class coverage
            num_bag_sets=1,
            num_stack_levels=2,  # Deeper stacking
            verbosity=2,
            auto_stack=True,
        )
        
        print("\nTraining completed - Recall-optimized model ready!")
        return self.predictor
    
    def optimize_threshold(self, y_true, y_pred_proba, metric='recall', plot=True):
        """
        Find optimal classification threshold - OPTIMIZED FOR RECALL
        
        Parameters:
        -----------
        y_true : array
            True labels
        y_pred_proba : array
            Predicted probabilities
        metric : str
            Metric to optimize: 'recall' (default), 'f1', 'precision', 'j_statistic', 'cost_sensitive'
        plot : bool
            Whether to plot threshold vs metric
        
        Returns:
        --------
        optimal_threshold : float
        """
        print(f"\n🎯 Optimizing threshold for: {metric.upper()}")
        print(f"   Goal: MINIMIZE FALSE NEGATIVES")
        
        thresholds = np.linspace(0.01, 0.99, 99)
        scores = []
        fn_counts = []
        recall_scores = []
        
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            
            # Calculate recall regardless of metric
            recall = recall_score(y_true, y_pred, zero_division=0)
            recall_scores.append(recall)
            
            # Count False Negatives
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            fn_counts.append(fn)
            
            if metric == 'recall':
                score = recall
            elif metric == 'f1':
                score = f1_score(y_true, y_pred, zero_division=0)
            elif metric == 'precision':
                score = precision_score(y_true, y_pred, zero_division=0)
            elif metric == 'j_statistic':
                # Youden's J statistic = sensitivity + specificity - 1
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                score = sensitivity + specificity - 1
            elif metric == 'cost_sensitive':
                # False negative cost is 10x false positive cost
                cost = fp * 1 + fn * 10
                score = -cost
            else:
                score = recall
            
            scores.append(score)
        
        optimal_idx = np.argmax(scores)
        optimal_threshold = thresholds[optimal_idx]
        optimal_score = scores[optimal_idx]
        optimal_fn = fn_counts[optimal_idx]
        optimal_recall = recall_scores[optimal_idx]
        
        # Also find max recall threshold
        max_recall_idx = np.argmax(recall_scores)
        max_recall_threshold = thresholds[max_recall_idx]
        max_recall = recall_scores[max_recall_idx]
        max_recall_fn = fn_counts[max_recall_idx]
        
        print(f"\n   Optimal threshold: {optimal_threshold:.3f}")
        print(f"   {metric.capitalize()}: {optimal_score if metric != 'cost_sensitive' else -optimal_score:.4f}")
        print(f"   Recall at optimal: {optimal_recall:.4f}")
        print(f"   False Negatives: {optimal_fn}")
        
        print(f"\n   Maximum recall threshold: {max_recall_threshold:.3f}")
        print(f"   Max recall: {max_recall:.4f}")
        print(f"   False Negatives: {max_recall_fn}")
        
        if plot:
            plt.figure(figsize=(12, 6))
            
            # Plot 1: Metric vs Threshold
            plt.subplot(1, 2, 1)
            plt.plot(thresholds, scores, linewidth=2, label=metric.capitalize())
            plt.axvline(optimal_threshold, color='r', linestyle='--', 
                       label=f'Optimal: {optimal_threshold:.3f}')
            plt.xlabel('Threshold', fontsize=12)
            plt.ylabel(f'{metric.capitalize()} Score', fontsize=12)
            plt.title(f'Threshold Optimization for {metric.upper()}', fontsize=14, fontweight='bold')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Plot 2: False Negatives vs Threshold
            plt.subplot(1, 2, 2)
            plt.plot(thresholds, fn_counts, linewidth=2, color='red', label='False Negatives')
            plt.axvline(optimal_threshold, color='g', linestyle='--', 
                       label=f'Optimal: {optimal_threshold:.3f}')
            plt.xlabel('Threshold', fontsize=12)
            plt.ylabel('False Negative Count', fontsize=12)
            plt.title('False Negatives vs Threshold', fontsize=14, fontweight='bold')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f'./threshold_optimization_{metric}_recall_focus.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        return optimal_threshold
    
    def comprehensive_evaluation(self, test_data, thresholds=[0.3, 0.5, 0.7]):
        """
        Comprehensive model evaluation at multiple thresholds
        """
        print("\n" + "="*80)
        print("COMPREHENSIVE MODEL EVALUATION")
        print("="*80)
        
        X_test = test_data.drop(columns=[self.target_column])
        y_test = test_data[self.target_column]
        
        # Get predictions
        y_pred_proba = self.predictor.predict_proba(X_test)
        
        if isinstance(y_pred_proba, pd.DataFrame):
            y_pred_proba_pos = y_pred_proba.iloc[:, 1].values
        else:
            y_pred_proba_pos = y_pred_proba
        
        # Calculate threshold-independent metrics
        roc_auc = roc_auc_score(y_test, y_pred_proba_pos)
        avg_precision = average_precision_score(y_test, y_pred_proba_pos)
        
        print(f"\nThreshold-Independent Metrics:")
        print(f"  ROC-AUC: {roc_auc:.4f}")
        print(f"  Average Precision: {avg_precision:.4f}")
        
        # Find optimal thresholds for different metrics
        optimal_thresholds = {}
        for metric in ['f1', 'j_statistic', 'precision', 'recall']:
            optimal_thresholds[metric] = self.optimize_threshold(
                y_test, 
                y_pred_proba_pos, 
                metric=metric, 
                plot=True
            )
        
        # Evaluate at multiple thresholds
        print(f"\nPerformance at Different Thresholds:")
        print("-" * 80)
        
        threshold_results = []
        for threshold in thresholds:
            y_pred = (y_pred_proba_pos >= threshold).astype(int)
            
            cm = confusion_matrix(y_test, y_pred)
            tn, fp, fn, tp = cm.ravel()
            
            metrics = {
                'threshold': threshold,
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1': f1_score(y_test, y_pred, zero_division=0),
                'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
                'mcc': matthews_corrcoef(y_test, y_pred),
                'cohen_kappa': cohen_kappa_score(y_test, y_pred),
                'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp
            }
            
            threshold_results.append(metrics)
            
            print(f"\nThreshold: {threshold}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  F1: {metrics['f1']:.4f}")
            print(f"  MCC: {metrics['mcc']:.4f}")
            print(f"  Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
        
        # Store results
        self.results = {
            'roc_auc': roc_auc,
            'average_precision': avg_precision,
            'optimal_thresholds': optimal_thresholds,
            'threshold_results': threshold_results,
            'y_test': y_test,
            'y_pred_proba': y_pred_proba_pos,
            'feature_importance': self.predictor.feature_importance(test_data),
            'leaderboard': self.predictor.leaderboard(test_data, silent=True)
        }
        
        return self.results
    
    def plot_comprehensive_results(self, save_path='./comprehensive_evaluation.png'):
        """
        Create comprehensive visualization of results
        """
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. ROC Curve
        ax1 = fig.add_subplot(gs[0, 0])
        fpr, tpr, _ = roc_curve(self.results['y_test'], self.results['y_pred_proba'])
        ax1.plot(fpr, tpr, linewidth=2, label=f'ROC-AUC = {self.results["roc_auc"]:.3f}')
        ax1.plot([0, 1], [0, 1], 'k--', linewidth=1)
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('ROC Curve', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Precision-Recall Curve
        ax2 = fig.add_subplot(gs[0, 1])
        precision, recall, _ = precision_recall_curve(
            self.results['y_test'], 
            self.results['y_pred_proba']
        )
        ax2.plot(recall, precision, linewidth=2)
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title(
            f'PR Curve (AP={self.results["average_precision"]:.3f})',
            fontweight='bold'
        )
        ax2.grid(True, alpha=0.3)
        
        # 3. Feature Importance
        ax3 = fig.add_subplot(gs[0, 2])
        feat_imp = self.results['feature_importance'].head(15)
        ax3.barh(range(len(feat_imp)), feat_imp['importance'])
        ax3.set_yticks(range(len(feat_imp)))
        ax3.set_yticklabels(feat_imp['feature'], fontsize=8)
        ax3.set_xlabel('Importance')
        ax3.set_title('Top 15 Features', fontweight='bold')
        ax3.invert_yaxis()
        
        # 4-6. Confusion Matrices at different thresholds
        for idx, threshold in enumerate([0.3, 0.5, 0.7]):
            ax = fig.add_subplot(gs[1, idx])
            y_pred = (self.results['y_pred_proba'] >= threshold).astype(int)
            cm = confusion_matrix(self.results['y_test'], y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
            ax.set_title(f'CM at threshold={threshold}', fontweight='bold')
            ax.set_ylabel('True')
            ax.set_xlabel('Predicted')
        
        # 7. Prediction Distribution
        ax7 = fig.add_subplot(gs[2, 0])
        ax7.hist(
            self.results['y_pred_proba'][self.results['y_test'] == 0],
            bins=50, alpha=0.6, label='Negative Class', color='blue', density=True
        )
        ax7.hist(
            self.results['y_pred_proba'][self.results['y_test'] == 1],
            bins=50, alpha=0.6, label='Positive Class', color='red', density=True
        )
        ax7.set_xlabel('Predicted Probability')
        ax7.set_ylabel('Density')
        ax7.set_title('Prediction Distribution', fontweight='bold')
        ax7.legend()
        ax7.set_yscale('log')
        
        # 8. Model Leaderboard
        ax8 = fig.add_subplot(gs[2, 1:])
        leaderboard = self.results['leaderboard'].head(10)
        ax8.barh(range(len(leaderboard)), leaderboard['score_val'])
        ax8.set_yticks(range(len(leaderboard)))
        ax8.set_yticklabels(leaderboard['model'], fontsize=9)
        ax8.set_xlabel('Validation Score')
        ax8.set_title('Model Leaderboard (Top 10)', fontweight='bold')
        ax8.invert_yaxis()
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nComprehensive evaluation plots saved to: {save_path}")
        plt.close()
    
    def save_results(self, filepath='./results.json'):
        """Save results to JSON file"""
        results_to_save = {
            'roc_auc': float(self.results['roc_auc']),
            'average_precision': float(self.results['average_precision']),
            'optimal_thresholds': {k: float(v) for k, v in self.results['optimal_thresholds'].items()},
            'threshold_results': [
                {k: (float(v) if isinstance(v, (int, float, np.number)) else v) 
                 for k, v in result.items()}
                for result in self.results['threshold_results']
            ],
        }
        
        with open(filepath, 'w') as f:
            json.dump(results_to_save, f, indent=2)
        
        print(f"\nResults saved to: {filepath}")


def main():
    """Main execution"""
    # Initialize pipeline
    pipeline = ImbalancedClassificationPipeline(target_column='target', random_state=42)
    
    # Load your data
    # df = pd.read_csv('your_data.csv')
    
    # For demonstration, create sample data
    from autogluon_imbalanced_model import create_sample_data
    df = create_sample_data()
    
    # Split data
    train_data, test_data = train_test_split(
        df, 
        test_size=0.2, 
        random_state=42,
        stratify=df['target']
    )
    
    # Train model with various options
    predictor = pipeline.train_model(
        train_data=train_data,
        time_limit=600,
        preset='medium_quality',
        eval_metric='roc_auc',
        use_sample_weights=True,
        resampling_method='none',  # Options: 'none', 'smote', 'adasyn', etc.
    )
    
    # Comprehensive evaluation
    results = pipeline.comprehensive_evaluation(
        test_data=test_data,
        thresholds=[0.3, 0.5, 0.7]
    )
    
    # Plot results
    pipeline.plot_comprehensive_results(save_path='./comprehensive_evaluation.png')
    
    # Save results
    pipeline.save_results(filepath='./results.json')
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("="*80)
    
    return pipeline


if __name__ == "__main__":
    pipeline = main()
