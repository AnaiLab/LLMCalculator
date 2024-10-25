import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
df = pd.read_csv('./aggregated_nsqip.csv')

# Function to categorize age
def categorize_age(age):
    if age == '90+':
        return 4
    age = int(age)
    if age < 65:
        return 1
    elif 65 <= age <= 74:
        return 2
    elif 75 <= age <= 84:
        return 3
    else:
        return 4

# Apply age categorization
df['Age'] = df['Age'].apply(categorize_age)

# Encode SEX
df['SEX'] = df['SEX'].map({'male': 0, 'female': 1})

# Encode FNSTATUS2
df['FNSTATUS2'] = df['FNSTATUS2'].map({'Independent': 1, 'Partially Dependent': 2, 'Totally Dependent': 3})

# Clean ASACLAS and convert to numeric
df['ASACLAS'] = pd.to_numeric(df['ASACLAS'].str[0], errors='coerce')

# Drop rows where ASACLAS couldn't be cast to a number
df = df.dropna(subset=['ASACLAS'])

# Encode PRSEPIS
df['PRSEPIS'] = df['PRSEPIS'].map({'None': 1, 'SIRS': 2, 'Sepsis': 3, 'Septic Shock': 4}).fillna(1)

# Encode DIABETES
df['DIABETES'] = df['DIABETES'].map({'NO': 1, 'NON-INSULIN': 2, 'ORAL': 2, 'INSULIN': 3})

# Encode Yes/No columns
yes_no_columns = [
    'EMERGNCY', 'ASCITES', 'HXCHF', 'STEROID', 'DIALYSIS', 'RENAFAIL', 'HXCOPD',
    'VENTILAT', 'SMOKE', 'DYSPNEA', 'HYPERMED', 'DISCANCR'
]
df[yes_no_columns] = df[yes_no_columns].applymap(lambda x: 1 if x == 'Yes' else 0)

# Convert numeric fields (1 if non-zero, 0 if zero)
numeric_columns = [
    'NSUPINFEC', 'NDEHIS', 'NOUPNEUMO', 'NREINTUB', 'NPULEMBOL', 'NFAILWEAN', 
    'NRENAINSF', 'NOPRENAFL', 'NURNINFEC', 'NCNSCVA', 'NCDARREST', 'NCDMI', 
    'NOTHDVT', 'NOTHSYSEP'
]
df[numeric_columns] = df[numeric_columns].applymap(lambda x: 1 if x != 0 else 0)

# Transform DOpertoD to MORTALITY
df['MORTALITY'] = df['DOpertoD'].apply(lambda x: 0 if x == -99 else 1)
df = df.drop(columns=['DOpertoD'])

# Create BMI column
df['BMI'] = (df['WEIGHT'] / (df['HEIGHT'] ** 2)) * 703

# Create MORBIDITY column
df['MORBIDITY'] = df[numeric_columns].any(axis=1).astype(int)

# Create CARDIAC column
df['CARDIAC'] = df[['NCDARREST', 'NCDMI']].any(axis=1).astype(int)

# Create RENALFAILURE column
df['RENALFAILURE'] = df[['NOPRENAFL', 'NRENAINSF']].any(axis=1).astype(int)

# Ensure all columns are numeric
df = df.apply(pd.to_numeric, errors='coerce')

# Drop any rows with NaNs (non-numeric values after conversion)
df = df.dropna()

# Calculate CPT-specific risk scores
outcomes = ['NOTHDVT', 'NOUPNEUMO', 'NSUPINFEC', 'MORBIDITY', 'MORTALITY', 'NURNINFEC', 'RENALFAILURE', 'CARDIAC']
cpt_risks = df.groupby('CPT')[outcomes].mean()

# Scale the risk scores
scaler = MinMaxScaler()
cpt_risks_scaled = pd.DataFrame(scaler.fit_transform(cpt_risks), columns=cpt_risks.columns, index=cpt_risks.index)

# Create a dictionary lookup table for each outcome
cpt_risk_dict = cpt_risks_scaled.to_dict(orient='index')

# Add the scaled risk scores to the dataset
for outcome in outcomes:
    df[f'CPT_{outcome}_RISK'] = df['CPT'].map(cpt_risk_dict).apply(lambda x: x[outcome] if pd.notnull(x) else np.nan)

# Ensure no missing values for the new columns
df = df.dropna()

# Save the processed DataFrame
df.to_csv('./nsqip_synthetic_processed_with_risks.csv', index=False)

# Save the lookup table as a dictionary
import json
with open('./cpt_risk_lookup.json', 'w') as f:
    json.dump(cpt_risk_dict, f)
