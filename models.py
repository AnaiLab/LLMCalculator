import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.utils import resample
import joblib  # Import joblib for model saving

# Load the data
data = pd.read_csv('./nsqip_clean.csv')

# Define the targets and the common features
targets = ['returnor', 'nothdvt', 'noprenafl', 'noupneumo', 'ncdarrest', 
           'nothsysep', 'nurninfec', 'readmission1']
common_features = ['age', 'sex', 'fnstatus2', 'electsurg', 'asaclas', 'ascites', 
                   'prsepis', 'diabetes', 'hxchf', 'steroid', 'dialysis', 'renafail', 
                   'hxcopd', 'ventilat', 'smoke', 'dyspnea', 'hypermed', 'discancr', 'BMI']

# Dictionary to hold model results
model_results = {}

# Bootstrapping parameters
n_bootstraps = 1000
alpha = 0.05

# Loop over each target to create and evaluate a model
for target in targets:
    # Drop rows where target is not 0 or 1
    filtered_data = data[data[target].isin([0, 1])]
    
    # Create the feature list specific to the current target
    features = common_features + [f'cpt_risk_{target}']
    
    # Drop rows with any NaNs in the used features or target
    final_data = filtered_data.dropna(subset=features + [target])
    
    # Create the feature matrix X and target vector y
    X = final_data[features]
    y = final_data[target]
    
    # Split the data into training and testing sets with stratification
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Initialize and fit the logistic regression model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    # Save the model to disk
    joblib.dump(model, f'./models/{target}_model.joblib')

    # Bootstrapping
    auc_scores, brier_scores = [], []
    for i in range(n_bootstraps):
        # Resample the data
        X_test_resampled, y_test_resampled = resample(X_test, y_test)
        # Predict probabilities on the resampled test set
        y_probs_resampled = model.predict_proba(X_test_resampled)[:, 1]
        # Calculate and store the metrics
        auc_scores.append(roc_auc_score(y_test_resampled, y_probs_resampled))
        brier_scores.append(brier_score_loss(y_test_resampled, y_probs_resampled))
    
    # Calculate the 95% confidence intervals
    auc_lower = np.percentile(auc_scores, 100 * alpha/2)
    auc_upper = np.percentile(auc_scores, 100 * (1 - alpha/2))
    brier_lower = np.percentile(brier_scores, 100 * alpha/2)
    brier_upper = np.percentile(brier_scores, 100 * (1 - alpha/2))
    
    # Store the results
    model_results[target] = {
        'C-statistic': {'Mean': np.mean(auc_scores), '95% CI': (auc_lower, auc_upper)},
        'Brier Score': {'Mean': np.mean(brier_scores), '95% CI': (brier_lower, brier_upper)}
    }

# Print all results
for target, metrics in model_results.items():
    print(f"Model for {target}:")
    print(f"C-statistic: {metrics['C-statistic']['Mean']:.4f}, 95% CI: {metrics['C-statistic']['95% CI']}")
    print(f"Brier Score: {metrics['Brier Score']['Mean']:.4f}, 95% CI: {metrics['Brier Score']['95% CI']}\n")
