from flask import Flask, render_template, request, redirect, url_for
import joblib
import numpy as np

app = Flask(__name__)

# Load the model and scaler
model = joblib.load('heart_failure_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    features = [
        float(request.form['age']),
        int(request.form['anaemia']),
        float(request.form['creatinine_phosphokinase']),
        int(request.form['diabetes']),
        float(request.form['ejection_fraction']),
        int(request.form['high_blood_pressure']),
        float(request.form['platelets']),
        float(request.form['serum_creatinine']),
        float(request.form['serum_sodium']),
        int(request.form['sex']),
        int(request.form['smoking']),
        float(request.form['time'])
    ]
    
    # Calculate interaction terms
    age_ejection_fraction = features[0] * features[4]
    creatinine_phosphokinase_serum_creatinine = features[2] * features[7]
    
    # Combine all features
    all_features = features + [age_ejection_fraction, creatinine_phosphokinase_serum_creatinine]
    
    # Convert to numpy array and reshape
    features_array = np.array(all_features).reshape(1, -1)
    
    # Scale the numerical features
    num_cols_indices = [0, 2, 4, 6, 7, 8, 11, 12, 13]  # Indices of numerical features
    features_array[:, num_cols_indices] = scaler.transform(features_array[:, num_cols_indices])
    
    # Make prediction
    prediction = model.predict(features_array)
    probability = model.predict_proba(features_array)[0][1]
    
    # Store results in session and redirect to results page
    return redirect(url_for('results', 
                          prediction=int(prediction[0]), 
                          probability=f"{probability*100:.2f}"))

@app.route('/results')
def results():
    prediction = request.args.get('prediction', '0')
    probability = request.args.get('probability', '0')
    
    result = {
        'prediction': 'High Risk of Heart Failure' if prediction == '1' else 'Low Risk of Heart Failure',
        'probability': f"{probability}%"
    }
    
    return render_template('results.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)