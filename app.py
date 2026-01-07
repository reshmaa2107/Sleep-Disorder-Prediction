from flask import Flask, request, jsonify, render_template, session, make_response
import joblib
import sqlite3
import datetime
from fpdf import FPDF

app = Flask(__name__)
app.secret_key = 'super_secret_key_for_session'

# Load Models
model = joblib.load('sleep_model.pkl')
scaler = joblib.load('scaler.pkl')
le_gender = joblib.load('le_gender.pkl')
le_occup = joblib.load('le_occup.pkl')

def init_db():
    conn = sqlite3.connect('sleep_data.db')
    c = conn.cursor()
    c.execute("CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, prediction TEXT, timestamp TEXT)")
    c.execute("CREATE TABLE IF NOT EXISTS feedback (id INTEGER PRIMARY KEY, message TEXT, rating INTEGER, timestamp TEXT)")
    conn.commit()
    conn.close()

init_db()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form
        
        # Categorical Encoding
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
        
        # --- NEW LOGIC FOR MESSAGES ---
        details = {}
        
        if prediction == 'Healthy':
            # POSITIVE MESSAGE (No medical jargon)
            details = {
                'is_bad': False, # Flag to tell HTML this is good news
                'title': 'Healthy 🎉',
                'desc': 'Your sleep patterns and screen time habits are well-balanced. You are maintaining a healthy lifestyle.',
                'action': 'Keep up the consistent schedule and continue prioritizing your rest!'
            }
        else:
            # NEGATIVE MESSAGE (Medical Details)
            details = {
                'is_bad': True, # Flag to tell HTML to show medical info
                'title': prediction,
                'desc': 'A sleep disorder where you have trouble falling and/or staying asleep.' if prediction == 'Insomnia' else 'A serious disorder where breathing repeatedly stops and starts.',
                'cause': 'High screen time (Blue light), Stress, irregular sleep schedule.' if prediction == 'Insomnia' else 'High BMI (Obesity), Snoring, airway anatomy, genetics.',
                'remedy': 'Follow the 20-20-20 rule, use Blue Light filters, avoid caffeine.' if prediction == 'Insomnia' else 'Weight management, side-sleeping, or consulting a doctor.'
            }

        # DB Save
        conn = sqlite3.connect('sleep_data.db')
        c = conn.cursor()
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        c.execute("INSERT INTO users (prediction, timestamp) VALUES (?, ?)", (prediction, timestamp))
        conn.commit()
        conn.close()

        # Session Save
        session['report_data'] = {
            'prediction': prediction,
            'timestamp': timestamp,
            'details': details,
            'stats': {'stress': data['stress'], 'night_screen': data['night_screen']}
        }

        return render_template('result.html', prediction=prediction, details=details, form_data=data)

    except Exception as e:
        return f"Error: {e}"

@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    try:
        data = request.json
        conn = sqlite3.connect('sleep_data.db')
        c = conn.cursor()
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        c.execute("INSERT INTO feedback (message, rating, timestamp) VALUES (?, ?, ?)", (data['message'], int(data['rating']), ts))
        conn.commit()
        conn.close()
        return jsonify({'status': 'success', 'message': 'Feedback saved!'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download_report')
def download_report():
    if 'report_data' not in session: return "No report found."
    data = session['report_data']
    
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="SLEEP HEALTH ANALYSIS REPORT", ln=True, align='C')
    pdf.ln(10)

    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Date: {data['timestamp']}", ln=True)
    
    pdf.set_font("Arial", 'B', 14)
    if data['prediction'] == 'Healthy':
        pdf.set_text_color(0, 128, 0)
    else:
        pdf.set_text_color(255, 0, 0)
        
    pdf.cell(200, 10, txt=f"Prediction Result: {data['prediction']}", ln=True)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(5)

    # DYNAMIC CONTENT BASED ON HEALTH
    pdf.set_font("Arial", size=12)
    
    if data['details']['is_bad']:
        # Show Medical Info
        pdf.multi_cell(0, 10, txt=f"Definition: {data['details']['desc']}")
        pdf.ln(2)
        pdf.multi_cell(0, 10, txt=f"Potential Causes: {data['details']['cause']}")
        pdf.ln(2)
        pdf.multi_cell(0, 10, txt=f"Remedies: {data['details']['remedy']}")
    else:
        # Show Happy Message
        pdf.multi_cell(0, 10, txt=f"{data['details']['desc']}")
        pdf.ln(5)
        pdf.set_font("Arial", 'I', 12)
        pdf.multi_cell(0, 10, txt=f"Advice: {data['details']['action']}")

    pdf.ln(10)
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(0, 10, txt="DISCLAIMER: This is an AI prediction and not a medical diagnosis. Please consult a doctor.", ln=True, align='C')

    response = make_response(pdf.output(dest='S').encode('latin-1'))
    response.headers['Content-Type'] = 'application/pdf'
    response.headers['Content-Disposition'] = 'attachment; filename=Sleep_Report.pdf'
    return response

if __name__ == "__main__":
    app.run(debug=True)