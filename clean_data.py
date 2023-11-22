import pandas as pd

# Load the data
df = pd.read_csv('./NSQIP.csv')

# Mappings for initial encoding
age_mapping = {
    '90+': 4
}
df['age'] = df['age'].replace(age_mapping).astype(float)
df['age'] = pd.cut(df['age'], bins=[0, 64, 74, 84, float('inf')], labels=[1, 2, 3, 4])

sex_mapping = {
    'male': 0,
    'female': 1
}
fnstatus2_mapping = {
    'Independent': 1,
    'Partially Dependent': 2,
    'Totally Dependent': 3
}
prsepis_mapping = {
    'None': 1,
    'SIRS': 2,
    'Sepsis': 3,
    'Septic Shock': 4
}
diabetes_mapping = {
    'NO': 1,
    'NON-INSULIN': 2,
    'INSULIN': 3
}
yes_no_mapping = {
    'Yes': 1,
    'No': 0
}

# Apply mappings
df['sex'] = df['sex'].map(sex_mapping)
df['fnstatus2'] = df['fnstatus2'].map(fnstatus2_mapping)
df['prsepis'] = df['prsepis'].map(prsepis_mapping)
df['diabetes'] = df['diabetes'].map(diabetes_mapping)
df['asaclas'] = df['asaclas'].str[0]
df = df[pd.to_numeric(df['asaclas'], errors='coerce').notnull()]
df['asaclas'] = df['asaclas'].astype(int)
yes_no_columns = [
    'electsurg', 'ascites', 'hxchf', 'steroid', 'dialysis',
    'renafail', 'hxcopd', 'ventilat', 'smoke', 'returnor',
    'readmission1', 'dyspnea', 'hypermed', 'discancr'
]
for col in yes_no_columns:
    df[col] = df[col].map(yes_no_mapping)

# Drop rows with missing values
required_columns = ['age', 'sex', 'fnstatus2', 'asaclas', 'prsepis', 'diabetes'] + yes_no_columns
df = df.dropna(subset=required_columns)

# Encode additional target variables
additional_targets = ['returnor', 'tothlos', 'nothdvt', 'noprenafl', 'noupneumo', 'ncdarrest', 'nothsysep', 'nurninfec', 'readmission1']
df = df.dropna(subset=additional_targets)

# Calculate CPT-specific risk scores
target_outcomes = ['returnor', 'tothlos', 'nothdvt', 'noprenafl', 'noupneumo', 'ncdarrest', 'nothsysep', 'nurninfec', 'readmission1']
cpt_risk_scores = {}
for outcome in target_outcomes:
    outcome_by_cpt = df.groupby('cpt')[outcome].mean()
    outcome_by_cpt = (outcome_by_cpt - outcome_by_cpt.min()) / (outcome_by_cpt.max() - outcome_by_cpt.min())
    df[f'cpt_risk_{outcome}'] = df['cpt'].map(outcome_by_cpt)
    cpt_risk_scores[outcome] = outcome_by_cpt.to_dict()

# Calculate BMI
df['BMI'] = (df['weight'] / (df['height'] ** 2)) * 703
df.to_csv('nsqip_clean.csv', index=False)


# Save the cpt_risk_scores dictionary to a Python file
with open('cptRisks.py', 'w') as f:
    f.write('riskCodes = ')
    f.write(str(cpt_risk_scores))
