import os
import sys
import subprocess

# --- 1. SETUP FOLDERS ---
print("📂 Creating folder structure...")
os.makedirs('templates', exist_ok=True)

# --- 2. CREATE DATA GENERATOR ---
print("📄 Writing Data Generator...")
with open('generate_dataset.py', 'w', encoding='utf-8') as f:
    f.write(r'''import pandas as pd
import numpy as np
import random

NUM_SAMPLES = 5000
np.random.seed(42)

def generate_data():
    data = []
    for _ in range(NUM_SAMPLES):
        age = np.random.randint(18, 60)
        gender = np.random.choice(['Male', 'Female'])
        occupation = np.random.choice(['Student', 'Engineer', 'Doctor', 'Artist', 'Manager'])
        daily_screen = np.round(np.random.uniform(2.0, 12.0), 1)
        night_screen = np.round(np.random.uniform(0.0, 4.0), 1)
        if night_screen > daily_screen: night_screen = daily_screen
        blue_light = np.random.choice([0, 1])
        bmi = np.round(np.random.uniform(18.5, 35.0), 1)
        heart_rate = np.random.randint(60, 95)
        stress = np.random.randint(1, 10)
        phys_act = np.random.randint(0, 120)
        snoring = np.random.choice([0, 1])
        night_walking = np.random.choice([0, 1], p=[0.95, 0.05])
        coffee = np.random.randint(0, 5)

        # Logic for Disorders
        insomnia_score = 0
        if night_screen > 2.0: insomnia_score += 3
        if blue_light == 0: insomnia_score += 1
        if stress > 6: insomnia_score += 3
        if coffee > 2: insomnia_score += 2
        
        apnea_score = 0
        if bmi > 28: apnea_score += 4
        if snoring == 1: apnea_score += 3
        if heart_rate > 80: apnea_score += 1

        if apnea_score >= 5: disorder = "Sleep Apnea"
        elif insomnia_score >= 5: disorder = "Insomnia"
        else: disorder = "Healthy"

        data.append([age, gender, occupation, daily_screen, night_screen, blue_light, bmi, heart_rate, stress, phys_act, snoring, night_walking, coffee, disorder])
    
    columns = ['Age', 'Gender', 'Occupation', 'DailyScreenTime', 'NightScreenTime', 'BlueLightFilter', 'BMI', 'HeartRate', 'StressLevel', 'PhysicalActivity', 'Snoring', 'NightWalking', 'CoffeeIntake', 'Disorder']
    pd.DataFrame(data, columns=columns).to_csv('sleep_disorder_dataset.csv', index=False)
    print("✅ Dataset Created!")

if __name__ == "__main__":
    generate_data()
''')

# --- 3. CREATE MODEL TRAINER ---
print("📄 Writing Model Trainer...")
with open('train_model.py', 'w', encoding='utf-8') as f:
    f.write(r'''import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
import joblib

df = pd.read_csv('sleep_disorder_dataset.csv')

le_gender = LabelEncoder()
df['Gender'] = le_gender.fit_transform(df['Gender'])
le_occup = LabelEncoder()
df['Occupation'] = le_occup.fit_transform(df['Occupation'])

X = df.drop('Disorder', axis=1)
y = df['Disorder']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# MLP Model
mlp = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=1000, random_state=42)
mlp.fit(X_train_scaled, y_train)

# Save files
joblib.dump(mlp, 'sleep_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(le_gender, 'le_gender.pkl')
joblib.dump(le_occup, 'le_occup.pkl')
print("✅ Model Trained!")
''')

# --- 4. CREATE FLASK BACKEND ---
print("📄 Writing App Backend...")
with open('app.py', 'w', encoding='utf-8') as f:
    f.write(r'''from flask import Flask, request, jsonify, render_template
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
''')

# --- 5. CREATE FRONTEND (PASTEL THEME) ---
print("📄 Writing Frontend Template...")
with open('templates/index.html', 'w', encoding='utf-8') as f:
    f.write(r'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sleep Disorder Prediction</title>
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&family=Montserrat:wght@700&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root { --bg-color: #FDF2F8; --card-bg: rgba(255, 255, 255, 0.85); }
        body { font-family: 'Poppins', sans-serif; background: linear-gradient(135deg, #FDF2F8 0%, #E0F2FE 100%); display: flex; justify-content: center; align-items: center; min-height: 100vh; margin: 0; }
        .dashboard-container { background: var(--card-bg); backdrop-filter: blur(12px); border-radius: 24px; padding: 30px; width: 90%; max-width: 850px; box-shadow: 0 20px 50px rgba(0,0,0,0.1); text-align: center; }
        h1 { font-family: 'Montserrat', sans-serif; color: #8B5CF6; margin-bottom: 5px; }
        .subtitle { color: #6B7280; font-size: 0.9rem; margin-bottom: 30px; }
        
        .form-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 15px; text-align: left; }
        label { font-size: 0.85rem; font-weight: 600; color: #374151; }
        input, select { width: 100%; padding: 12px; border: 2px solid #E5E7EB; border-radius: 10px; box-sizing: border-box; }
        
        .analyze-btn { grid-column: span 2; width: 100%; padding: 15px; background: linear-gradient(to right, #C4B5FD, #86EFAC); border: none; border-radius: 12px; color: white; font-weight: bold; font-size: 1rem; cursor: pointer; margin-top: 20px; transition: transform 0.2s; }
        .analyze-btn:hover { transform: scale(1.02); }

        #result-box { display: none; margin-top: 30px; padding-top: 20px; border-top: 1px solid #eee; animation: fadeIn 0.5s; }
        .prediction-badge { padding: 10px 25px; border-radius: 50px; color: white; font-weight: bold; font-size: 1.5rem; display: inline-block; margin-bottom: 15px; }
        .bg-healthy { background-color: #6EE7B7; } 
        .bg-insomnia { background-color: #FCA5A5; } 
        .bg-apnea { background-color: #FDBA74; }
        
        @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
    </style>
</head>
<body>

<div class="dashboard-container">
    <h1>Sleep Disorder Prediction</h1>
    <p class="subtitle">VIA SMARTPHONE SCREEN TIME PATTERNS</p>

    <form id="predictionForm">
        <div class="form-grid">
            <div><label>Age</label><input type="number" id="age" value="24" required></div>
            <div><label>Gender</label><select id="gender"><option value="Male">Male</option><option value="Female">Female</option></select></div>
            <div><label>Daily Screen Time (Hrs)</label><input type="number" step="0.1" id="daily_screen" value="6.0" required></div>
            <div><label>Nighttime Usage (Hrs)</label><input type="number" step="0.1" id="night_screen" value="1.0" required></div>
            <div><label>BMI</label><input type="number" step="0.1" id="bmi" value="22.5" required></div>
            <div><label>Heart Rate</label><input type="number" id="heart_rate" value="72" required></div>
        </div>

        <input type="hidden" id="occupation" value="Student">
        <input type="hidden" id="blue_light" value="0">
        <input type="hidden" id="phys_act" value="30">
        <input type="hidden" id="stress" value="5">
        <input type="hidden" id="snoring" value="0">
        <input type="hidden" id="night_walking" value="0">
        <input type="hidden" id="coffee" value="1">

        <button type="submit" class="analyze-btn">Analyze Patterns ✨</button>
    </form>

    <div id="result-box">
        <h3>Prediction Result</h3>
        <div id="predBadge" class="prediction-badge"></div>
        <p id="insightText" style="color: #555;"></p>
        <div style="height: 200px; width: 100%;"><canvas id="sleepChart"></canvas></div>
    </div>
</div>

<script>
    document.getElementById('predictionForm').addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const data = {
            age: document.getElementById('age').value,
            gender: document.getElementById('gender').value,
            occupation: document.getElementById('occupation').value,
            daily_screen: document.getElementById('daily_screen').value,
            night_screen: document.getElementById('night_screen').value,
            blue_light: document.getElementById('blue_light').value,
            bmi: document.getElementById('bmi').value,
            heart_rate: document.getElementById('heart_rate').value,
            stress: document.getElementById('stress').value,
            phys_act: document.getElementById('phys_act').value,
            snoring: document.getElementById('snoring').value,
            night_walking: document.getElementById('night_walking').value,
            coffee: document.getElementById('coffee').value
        };

        try {
            const response = await fetch('/predict', {
                method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(data)
            });
            const result = await response.json();
            
            if(result.error) { alert("Server Error: " + result.error); return; }

            const resBox = document.getElementById('result-box');
            resBox.style.display = 'block';
            
            const badge = document.getElementById('predBadge');
            badge.innerText = result.prediction;
            badge.className = 'prediction-badge ' + (result.prediction === 'Healthy' ? 'bg-healthy' : result.prediction === 'Insomnia' ? 'bg-insomnia' : 'bg-apnea');
            
            document.getElementById('insightText').innerText = result.insight;

            // Chart
            const ctx = document.getElementById('sleepChart').getContext('2d');
            if(window.myChart) window.myChart.destroy();
            window.myChart = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: ['Your Night Screen Time', 'Recommended Limit'],
                    datasets: [{
                        label: 'Hours',
                        data: [data.night_screen, 1.0],
                        backgroundColor: ['#FCA5A5', '#6EE7B7']
                    }]
                },
                options: { responsive: true, maintainAspectRatio: false }
            });

        } catch (err) { alert("Connection Failed. Is python app.py running?"); }
    });
</script>

</body>
</html>
''')

# --- 6. AUTO-RUN SETUP COMMANDS ---
print("\n🚀 STARTING AUTOMATED SETUP...")
print("--------------------------------")

try:
    print("1️⃣  Generating Dataset...")
    subprocess.run([sys.executable, 'generate_dataset.py'], check=True)
    
    print("2️⃣  Training ML Model (This takes 5-10 seconds)...")
    subprocess.run([sys.executable, 'train_model.py'], check=True)
    
    print("\n✅ SETUP COMPLETE! EVERYTHING IS READY.")
    print("👉 TYPE THIS COMMAND TO START:  python app.py")

except Exception as e:
    print(f"❌ Error during auto-setup: {e}")