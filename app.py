from flask import Flask, render_template, request
import json
import joblib
import os

app = Flask(__name__)

# Load the risk scores from the JSON file
with open('cpt_risk_lookup.json', 'r') as f:
    riskCodes = json.load(f)

# Load all the models into a dictionary
imported_models = {}
outcomes = ["MORTALITY", "CARDIAC", "RENALFAILURE", "NOUPNEUMO", 
            "MORBIDITY", "NOTHDVT", "NSUPINFEC", "NURNINFEC"]

for outcome in outcomes:
    model_path = os.path.join('models', f'{outcome}_model.joblib')
    imported_models[outcome] = joblib.load(model_path)

# Mapping for more readable output
outcome_mapping = {
    "MORTALITY": "Mortality",
    "MORBIDITY": "Morbidity",
    "NOUPNEUMO": "Pneumonia",
    "CARDIAC": "Cardiac arrest or MI",
    "NSUPINFEC": "SSI",
    "NURNINFEC": "UTI",
    "NOTHDVT": "Venous Thromboembolism",
    "RENALFAILURE": "Renal Failure"
}

@app.route('/')
def input_page():
    return render_template('input_page.html')

@app.route('/results', methods=['POST'])
def results():
    # Retrieve form data
    data = request.form

    # Encode variables
    age = int(data['Age'])
    if age < 65:
        Age = 1
    elif 65 <= age < 75:
        Age = 2
    elif 75 <= age < 85:
        Age = 3
    else:
        Age = 4

    SEX = 0 if data['SEX'] == 'Male' else 1

    FNSTATUS2 = {'Independent': 1, 'Partially Dependent': 2, 'Totally Dependent': 3}[data['FNSTATUS2']]

    ASACLAS = {
        'Healthy patient': 1,
        'Mild systemic disease': 2,
        'Severe systemic disease': 3,
        'Severe systemic disease/constant threat to life': 4,
        'Moribund/not expected to survive surgery': 5
    }[data['ASACLAS']]

    PRSEPIS = {'None': 1, 'SIRS': 2, 'Sepsis': 3, 'Septic Shock': 4}[data['PRSEPIS']]

    DIABETES = {'No': 1, 'Oral': 2, 'Insulin': 3}[data['DIABETES']]

    # Binary encoding for Yes/No questions
    binary_fields = [
        'EMERGNCY', 'ASCITES', 'HXCHF', 'STEROID', 'DIALYSIS', 
        'RENAFAIL', 'HXCOPD', 'VENTILAT', 'SMOKE', 'DYSPNEA', 
        'HYPERMED', 'DISCANCR'
    ]

    encoded_data = {field: 1 if data[field] == 'Yes' else 0 for field in binary_fields}

    # Convert height and weight to BMI
    height_in_inches = int(data['IN'])
    weight_in_pounds = int(data['LBS'])
    BMI = (weight_in_pounds / (height_in_inches ** 2)) * 703

    # Collect all encoded data
    encoded_data.update({
        'Age': Age,
        'SEX': SEX,
        'FNSTATUS2': FNSTATUS2,
        'ASACLAS': ASACLAS,
        'PRSEPIS': PRSEPIS,
        'DIABETES': DIABETES,
        'BMI': round(BMI, 2)
    })

    # Convert CPT code to integer
    cpt_code = int(data.get('CPT_CODE'))
    
    if str(cpt_code) in riskCodes:
        cpt_risk = riskCodes[str(cpt_code)]
    else:
        # Calculate mean risk scores if CPT code is not found
        mean_risk = {}
        for target in riskCodes[next(iter(riskCodes))].keys():
            mean_risk[target] = sum(risk[target] for risk in riskCodes.values()) / len(riskCodes)
        cpt_risk = mean_risk

    # Add CPT risk scores to encoded_data
    for target, risk in cpt_risk.items():
        encoded_data[f'CPT_{target}_RISK'] = risk

    # Prepare input data for models
    input_features = [
        'Age', 'SEX', 'FNSTATUS2','EMERGNCY','ASACLAS','ASCITES','PRSEPIS',
        'DIABETES', 'HXCHF', 'STEROID', 'DIALYSIS','RENAFAIL', 'HXCOPD', 
        'VENTILAT', 'SMOKE', 'DYSPNEA', 'HYPERMED', 'DISCANCR','BMI'
    ]

    risk_scores = {}
    for outcome in outcomes:
        # Add specific CPT risk score to input features
        input_data = [encoded_data[feature] for feature in input_features]
        input_data.append(encoded_data[f'CPT_{outcome}_RISK'])

        # Predict risk score using the model
        model = imported_models[outcome]
        risk_probability = model.predict_proba([input_data])[0][1]  # Probability of the positive class
        risk_scores[outcome_mapping[outcome]] = round(risk_probability * 100, 2)  # Convert to percentage and round

    # Display the risk scores
    return render_template('results_page.html', risk_scores=risk_scores)

if __name__ == '__main__':
    app.run(debug=True)
