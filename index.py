from flask import Flask, render_template, request, redirect, url_for
from cptRisks import riskCodes
import numpy as np
from joblib import load

outcome_display_names = {
    'returnor': 'Return to OR',
    'nothdvt': 'DVT',
    'noprenafl': 'Renal Failure',
    'noupneumo': 'Pneumonia',
    'ncdarrest': 'Cardiac Arrest',
    'nothsysep': 'Sepsis',
    'nurninfec': 'UTI',
    'readmission1': 'Readmission'
}

# Dictionary to store the imported models
imported_models = {}

# List of target variables (same as before)
targets = ['returnor', 'nothdvt', 'noprenafl', 'noupneumo', 'ncdarrest', 'nothsysep', 'nurninfec', 'readmission1']

# Load each model from its file
for target in targets:
    imported_models[target] = load(f'./models/{target}_model.joblib')

app = Flask(__name__)

def encode_variables(form_data):
    encoded_data = {}
    
    # Age encoding
    age = int(form_data['age'])
    if age < 65:
        encoded_data['age'] = 1
    elif age < 75:
        encoded_data['age'] = 2
    elif age < 85:
        encoded_data['age'] = 3
    else:
        encoded_data['age'] = 4

    # Sex encoding
    encoded_data['sex'] = 0 if form_data['sex'] == 'Male' else 1

    # Functional Status encoding
    fnstatus2_map = {'Independent': 1, 'Partially Dependent': 2, 'Totally Dependent': 3}
    encoded_data['fnstatus2'] = fnstatus2_map[form_data['fnstatus2']]
    
    # ASA Class encoding
    asaclas_map = {'Healthy patient': 1, 'Mild systemic disease': 2, 'Severe systemic disease': 3,
                   'Severe systemic disease/constant threat to life': 4, 'Moribund/not expected to survive surgery': 5}
    encoded_data['asaclas'] = asaclas_map[form_data['asaclas']]
    
    # Systemic Sepsis encoding
    prsepis_map = {'None': 1, 'SIRS': 2, 'Sepsis': 3, 'Septic Shock': 4}
    encoded_data['prsepis'] = prsepis_map[form_data['prsepis']]
    
    # Diabetes encoding
    diabetes_map = {'No': 1, 'Oral': 2, 'Insulin': 3}
    encoded_data['diabetes'] = diabetes_map[form_data['diabetes']]
    
    # Binary Yes/No encoding, inverting electsurg
    binary_fields = ['electsurg', 'ascites', 'hxchf', 'steroid', 'dialysis', 'renafail', 'hxcopd', 
                     'ventilat', 'smoke', 'dyspnea', 'hypermed', 'discancr']
    for field in binary_fields:
        encoded_data[field] = 1 if form_data[field] == 'Yes' else 0
    
    encoded_data['electsurg'] = 1 - encoded_data['electsurg']  # Inverting electsurg
    
    # Calculate BMI
    height_in = float(form_data['in'])
    weight_lb = float(form_data['lbs'])
    bmi = (weight_lb * 703) / (height_in ** 2)
    encoded_data['bmi'] = bmi
    
    
    # Adding CPT risk codes
    cpt_code = form_data['cpt']
    
    for outcome in ['returnor', 'tothlos', 'nothdvt', 'noprenafl', 'noupneumo', 'ncdarrest', 'nothsysep', 'nurninfec', 'readmission1']:
        outcome_dict = riskCodes.get(outcome, {})
        risk_score = outcome_dict.get(cpt_code, np.mean(list(outcome_dict.values())))
        encoded_data[f'cpt_risk_{outcome}'] = risk_score
    
    return encoded_data


@app.route('/', methods=['GET', 'POST'])
def input_page():
    if request.method == 'POST':
        form_data = request.form
        encoded_data = encode_variables(form_data)
        
        # Rename 'bmi' to 'BMI'
        encoded_data['BMI'] = encoded_data.pop('bmi')
        
        # Risk prediction
        risk_scores = {}
        for outcome, model in imported_models.items():
            features = ['age', 'sex', 'fnstatus2', 'electsurg', 'asaclas', 'ascites', 'prsepis', 'diabetes', 
                        'hxchf', 'steroid', 'dialysis', 'renafail', 'hxcopd', 'ventilat', 'smoke', 'dyspnea',
                        'hypermed', 'discancr', 'BMI', f'cpt_risk_{outcome}']
            X = np.array([encoded_data[feature] for feature in features]).reshape(1, -1)
            prob = model.predict_proba(X)[0][1]
            risk_scores[outcome] = prob
        
        return render_template('results_page.html', risk_scores=risk_scores, outcome_display_names=outcome_display_names)
        
    return render_template('input_page.html')

if __name__ == '__main__':
    app.run(debug=True)