import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.utils import resample
import joblib
import os

# Load the dataset
df = pd.read_csv('nsqip_synthetic_processed_with_risks.csv')

# Define the targets
targets = ['MORTALITY', 'CARDIAC', 'RENALFAILURE', 'NOUPNEUMO', 
           'MORBIDITY', 'NOTHDVT', 'NSUPINFEC', 'NURNINFEC']

# Drop rows that don't have 0 or 1 in target columns
df = df[df[targets].isin([0, 1]).all(axis=1)]

# Define the common features
common_features = ['Age', 'SEX', 'FNSTATUS2','EMERGNCY','ASACLAS','ASCITES','PRSEPIS',
                   'DIABETES', 'HXCHF', 'STEROID', 'DIALYSIS','RENAFAIL', 'HXCOPD', 
                   'VENTILAT', 'SMOKE', 'DYSPNEA', 'HYPERMED', 'DISCANCR','BMI']

# Function to calculate metrics and confidence intervals
def calculate_metrics_with_ci(model, X_test, y_test, n_bootstraps=1000, random_state=42):
    np.random.seed(random_state)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    c_statistic = roc_auc_score(y_test, y_pred_proba)
    brier_score = brier_score_loss(y_test, y_pred_proba)
    
    c_statistic_bootstrap = []
    brier_score_bootstrap = []
    
    for i in range(n_bootstraps):
        X_resampled, y_resampled = resample(X_test, y_test)
        y_pred_resampled = model.predict_proba(X_resampled)[:, 1]
        
        c_statistic_bootstrap.append(roc_auc_score(y_resampled, y_pred_resampled))
        brier_score_bootstrap.append(brier_score_loss(y_resampled, y_pred_resampled))
    
    c_statistic_ci = np.percentile(c_statistic_bootstrap, [2.5, 97.5])
    brier_score_ci = np.percentile(brier_score_bootstrap, [2.5, 97.5])
    
    return {
        'C-statistic': c_statistic,
        'C-statistic CI': c_statistic_ci,
        'Brier Score': brier_score,
        'Brier Score CI': brier_score_ci
    }

# Directory to save models
os.makedirs('./models', exist_ok=True)

# Dictionary to store results
results = {}

for target in targets:
    # Define the specific risk feature for this target
    risk_feature = f'CPT_{target}_RISK'
    
    # Create the list of features for this target model
    features = common_features + [risk_feature]
    
    # Define X and y
    X = df[features]
    y = df[target]
    
    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize the Logistic Regression model
    model = LogisticRegression(max_iter=1000)
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Save the model
    model_filename = f'./models/{target}_model.joblib'
    joblib.dump(model, model_filename)
    
    # Calculate metrics and confidence intervals
    metrics = calculate_metrics_with_ci(model, X_test, y_test)
    
    # Store the results
    results[target] = metrics

# Convert results to DataFrame for easy viewing
results_df = pd.DataFrame(results).T

# Display the results
print(results_df)
