# HOW TO RUN - Step by Step Guide

## Complete Installation and Execution Guide

Follow these steps exactly to run the AutoGluon model on your data.

---

## STEP 1: Download the Files

Download all the files from this project:
- `run_model.py` (main script to run)
- `autogluon_imbalanced_model.py`
- `advanced_autogluon_pipeline.py`
- `config.py`
- `requirements.txt`

Save them all in the same folder (e.g., `autogluon_project/`)

---

## STEP 2: Install Python

Make sure you have Python 3.8 or higher installed.

**Check your Python version:**
```bash
python --version
```
or
```bash
python3 --version
```

If you don't have Python, download it from: https://www.python.org/downloads/

---

## STEP 3: Install Required Packages

Open a terminal/command prompt in your project folder and run:

### Option A: Install from requirements.txt (RECOMMENDED)
```bash
pip install -r requirements.txt
```

### Option B: Install packages individually
```bash
pip install autogluon.tabular[all]
pip install pandas numpy scikit-learn matplotlib seaborn imbalanced-learn
```

**Note:** This may take 5-10 minutes as AutoGluon is a large package.

**For GPU support (optional, if you have NVIDIA GPU):**
```bash
pip install autogluon.tabular[all,ray]
```

---

## STEP 4: Prepare Your Data

Your CSV file should have:
- **24 numerical feature columns** (named anything you like)
- **1 target column** (binary: 0 for Pass, 1 for Decline)

Example CSV format:
```csv
feature_1,feature_2,feature_3,...,feature_24,target
0.523,1.234,0.876,...,0.456,0
1.145,0.367,1.234,...,1.567,1
0.789,0.912,0.345,...,0.678,0
```

**Important:** Make sure your target column contains only 0s and 1s (or you can rename the column to 'target' for simplicity).

---

## STEP 5: Run the Model

### Method 1: Interactive Script (EASIEST - RECOMMENDED)

Simply run:
```bash
python run_model.py
```

The script will ask you questions and guide you through the process:
1. It checks if all packages are installed
2. Asks for your data file path (or type 'demo' to use demo data)
3. Asks for configuration options (training time, quality, etc.)
4. Trains the model
5. Evaluates and generates visualizations

**Example interaction:**
```
Enter path to your CSV file (or 'demo' for demo data): my_data.csv
Enter name of target column (default: 'target'): target
Select training time [1-5] (default: 2): 2
Select evaluation metric [1-4] (default: 2): 2
Use advanced features [1-2] (default: 1): 1
Proceed with training? [Y/n]: Y
```

---

### Method 2: Python Script (Custom Control)

Create a new Python file (e.g., `my_run.py`) and add:

```python
# Import the necessary functions
from autogluon_imbalanced_model import (
    load_and_prepare_data,
    train_autogluon_model,
    evaluate_model,
    find_optimal_threshold,
    plot_evaluation_metrics
)

# Load and prepare your data
train_data, test_data = load_and_prepare_data(
    data_path='your_data.csv',  # Change to your file path
    target_column='target'       # Change to your target column name
)

# Train the model
predictor = train_autogluon_model(
    train_data=train_data,
    target_column='target',
    time_limit=1800,              # 30 minutes (adjust as needed)
    preset='good_quality',        # Options: 'medium_quality', 'good_quality', 'high_quality', 'best_quality'
    eval_metric='average_precision'  # Recommended for imbalanced data
)

# Evaluate the model
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

# Generate visualizations
plot_evaluation_metrics(results_optimal)

print(f"\nModel saved to: {predictor.path}")
print("Evaluation plots saved to: ./evaluation_plots.png")
```

Then run:
```bash
python my_run.py
```

---

### Method 3: Advanced Pipeline with SMOTE

Create a Python file (e.g., `my_advanced_run.py`):

```python
from advanced_autogluon_pipeline import ImbalancedClassificationPipeline
import pandas as pd
from sklearn.model_selection import train_test_split

# Initialize pipeline
pipeline = ImbalancedClassificationPipeline(
    target_column='target',
    random_state=42
)

# Load your data
df = pd.read_csv('your_data.csv')

# Split data
train_data, test_data = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df['target']
)

# Train model with SMOTE
predictor = pipeline.train_model(
    train_data=train_data,
    time_limit=1800,
    preset='good_quality',
    eval_metric='average_precision',
    use_sample_weights=True,
    resampling_method='smote'  # Options: 'none', 'smote', 'adasyn', 'borderline_smote'
)

# Comprehensive evaluation
results = pipeline.comprehensive_evaluation(
    test_data=test_data,
    thresholds=[0.3, 0.5, 0.7]
)

# Generate plots and save results
pipeline.plot_comprehensive_results()
pipeline.save_results()

print(f"\nModel saved to: {predictor.path}")
print("Results saved to: ./results.json")
print("Plots saved to: ./comprehensive_evaluation.png")
```

Then run:
```bash
python my_advanced_run.py
```

---

## STEP 6: Understanding the Output

After training completes, you'll see:

### 1. **Console Output**
- Training progress and model performance
- Feature importance
- Model leaderboard (which models performed best)
- Evaluation metrics (ROC-AUC, Average Precision, F1, etc.)
- Confusion matrix

### 2. **Generated Files**

#### Model Directory (e.g., `autogluon_models_20250104_143022/`)
- Contains the trained model
- Can be loaded later for predictions

#### Evaluation Plots (`evaluation_plots.png` or `comprehensive_evaluation.png`)
- Confusion matrix
- Precision-Recall curve
- Feature importance chart
- Prediction distribution

#### Results File (`results.json` - if using advanced pipeline)
- All metrics in JSON format
- Can be loaded for analysis

---

## STEP 7: Making Predictions on New Data

After training, use your model to make predictions:

```python
from autogluon.tabular import TabularPredictor
import pandas as pd

# Load the saved model
predictor = TabularPredictor.load('./autogluon_models_20250104_143022/')  # Use your actual path

# Load new data (must have same features as training data)
new_data = pd.read_csv('new_data.csv')

# Get predictions
predictions = predictor.predict(new_data)  # Returns 0 or 1
probabilities = predictor.predict_proba(new_data)  # Returns probabilities

# Save predictions
new_data['prediction'] = predictions
new_data['probability_decline'] = probabilities.iloc[:, 1]
new_data.to_csv('predictions.csv', index=False)

print("Predictions saved to predictions.csv")
```

### Using Custom Threshold

```python
# If you found optimal threshold (e.g., 0.3) during training
optimal_threshold = 0.3

probabilities = predictor.predict_proba(new_data)
custom_predictions = (probabilities.iloc[:, 1] >= optimal_threshold).astype(int)

new_data['custom_prediction'] = custom_predictions
```

---

## Troubleshooting

### Problem: "ModuleNotFoundError: No module named 'autogluon'"
**Solution:** Install AutoGluon
```bash
pip install autogluon.tabular[all]
```

### Problem: "Memory Error" or system runs out of RAM
**Solutions:**
1. Use a smaller sample of your data for testing
2. Reduce `num_bag_folds` in the code (e.g., set to 3)
3. Use `preset='medium_quality'` instead of 'best_quality'
4. Close other applications

### Problem: Training is too slow
**Solutions:**
1. Reduce `time_limit` (e.g., 600 seconds = 10 minutes)
2. Use `preset='medium_quality'` or 'good_quality'
3. Use fewer cross-validation folds

### Problem: "FileNotFoundError" when loading data
**Solutions:**
1. Check the file path is correct
2. Use absolute path (e.g., `C:/Users/YourName/data.csv` on Windows or `/home/user/data.csv` on Linux)
3. Make sure you're in the correct directory

### Problem: Poor performance on minority class (declines)
**Solutions:**
1. Use `eval_metric='average_precision'` or 'f1'
2. Try advanced pipeline with `resampling_method='smote'`
3. Optimize threshold for recall: `find_optimal_threshold(results, metric='recall')`
4. Increase training time for better quality

### Problem: "TypeError" or unexpected errors
**Solutions:**
1. Make sure your target column contains only 0 and 1 (not strings)
2. Ensure all feature columns are numerical
3. Check for missing values: `df.isnull().sum()`
4. Update AutoGluon: `pip install --upgrade autogluon.tabular`

---

## Quick Reference - Common Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run interactive script
python run_model.py

# Check Python version
python --version

# Install AutoGluon only
pip install autogluon.tabular[all]

# Upgrade AutoGluon
pip install --upgrade autogluon.tabular

# List installed packages
pip list
```

---

## System Requirements

**Minimum:**
- Python 3.8+
- 8 GB RAM
- 10 GB free disk space

**Recommended:**
- Python 3.9 or 3.10
- 16+ GB RAM
- 20+ GB free disk space
- Multi-core CPU (4+ cores)

**For Large Datasets (2M+ rows):**
- 32+ GB RAM
- SSD storage
- GPU (optional, but speeds up neural networks)

---

## Getting Help

If you encounter issues:

1. Check the error message carefully
2. Review the troubleshooting section above
3. Make sure all packages are installed correctly
4. Try running with demo data first: type 'demo' when asked for data file
5. Check AutoGluon documentation: https://auto.gluon.ai/

---

## Quick Start Example (Copy-Paste)

```bash
# 1. Install packages
pip install -r requirements.txt

# 2. Run the model (will ask for your data file)
python run_model.py

# 3. When prompted, enter your CSV file path or type 'demo'
# 4. Follow the prompts to configure training
# 5. Wait for training to complete
# 6. Check output files: evaluation_plots.png and model directory
```

That's it! You're ready to run the model.
