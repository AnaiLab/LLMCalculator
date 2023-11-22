import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, brier_score_loss

from joblib import dump

# Load the data
df = pd.read_csv('./nsqip_clean.csv')

# List of common features for all models
common_features = ['age', 'sex', 'fnstatus2', 'electsurg', 'asaclas', 'ascites', 'prsepis', 'diabetes', 
                   'hxchf', 'steroid', 'dialysis', 'renafail', 'hxcopd', 'ventilat', 'smoke', 'dyspnea',
                   'hypermed', 'discancr', 'BMI']

# List of target variables
targets = ['returnor', 'nothdvt', 'noprenafl', 'noupneumo', 'ncdarrest', 'nothsysep', 'nurninfec', 'readmission1']

# Dictionary to store models
models = {}

# Loop over each target to create a logistic regression model
for target in targets:
    # Filter rows where target is 0 or 1
    filtered_df = df[df[target].isin([0, 1])]
    
    # Create feature list for this model
    feature_list = common_features + [f'cpt_risk_{target}']
    
    # Split data into training and test sets
    X = filtered_df[feature_list]
    y = filtered_df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Initialize and fit logistic regression model with increased max_iter
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    # Calculate and print metrics (Brier score and ROC AUC)
    y_prob = model.predict_proba(X_test)[:, 1]
    print(f"Metrics for {target}:")
    print(f"  Brier Score: {brier_score_loss(y_test, y_prob)}")
    print(f"  ROC AUC: {roc_auc_score(y_test, y_prob)}")
    
    # Store the model
    models[target] = model

# Export each model to a separate file
for target, model in models.items():
    dump(model, f'./models/{target}_model.joblib')

