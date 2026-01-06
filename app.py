from flask import Flask, request, jsonify, render_template
import joblib
import sqlite3
import numpy as np
import os

app = Flask(__name__)

# Load Models
model = joblib.load('sleep_model.pkl')
scaler = joblib.load('scaler.pkl')
le_gender = joblib.load('le_gender.pkl')
le_occup = joblib.load('le_occup.pkl')

def init_db():
    conn = sqlite3.connect('sleep_data.db')
    c = conn.cursor()
    c.execute("CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, prediction TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)")
    conn.commit()
    conn.close()

init_db()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # Safely handle categorical data
        try: gender_enc = le_gender.transform([data['gender']])[0]
        except: gender_enc = 0
        
        try: occup_enc = le_occup.transform([data['occupation']])[0]
        except: occup_enc = 0

        features = [
            float(data['age']), gender_enc, occup_enc,
            float(data['daily_screen']), float(data['night_screen']), int(data['blue_light']),
            float(data['bmi']), int(data['heart_rate']), int(data['stress']),
            int(data['phys_act']), int(data['snoring']), int(data['night_walking']), int(data['coffee'])
        ]
        
        final_features = scaler.transform([features])
        prediction = model.predict(final_features)[0]
        
        insight = "Great habits! Keep it up."
        if prediction == 'Insomnia': insight = "Limit night screen time. Try blue light filters."
        elif prediction == 'Sleep Apnea': insight = "High risk detected. Please consult a doctor."

        return jsonify({'prediction': prediction, 'insight': insight})
    except Exception as e:
        print("ERROR:", e)
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
