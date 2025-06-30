import os
import logging
import socket
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, session, jsonify, redirect
from werkzeug.utils import secure_filename
import google.generativeai as genai
from dotenv import load_dotenv
import PyPDF2
import re
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn import svm

load_dotenv()

app = Flask(__name__)
app.config.update({
    'UPLOAD_FOLDER': os.path.abspath('uploads'),
    'ALLOWED_EXTENSIONS': {'pdf'},
    'MAX_CONTENT_LENGTH': 16 * 1024 * 1024,
    'SECRET_KEY': os.urandom(24)
})

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Medical report analyzer keywords
MEDICAL_KEYWORDS = {
    'blood', 'cholesterol', 'urine', 'report', 'test', 'doctor', 'symptom', 'diagnosis', 'treatment',
    'creatinine', 'wbc', 'neutrophils', 'lymphocytes', 'vldl', 'hba1c', 'rbc', 'platelet', 'cbc',
    'hdl', 'ldl', 'triglycerides', 'glucose', 'bilirubin', 'protein', 'bun', 'sodium', 'potassium',
    'chlorides', 'uric acid', 'phosphorous', 'calcium', 'mcv', 'mch', 'mchc', 'rdw', 'esr', 'thyroid',
    'vitamin', 'infection', 'urinalysis', 'urine culture', 'liver', 'kidney', 'hematology', 'cbc', 'lipid'
}

# Configure Gemini API
genai.configure(api_key=os.getenv('GEMINI_API_KEY', ''))
MODEL = genai.GenerativeModel('gemini-1.5-pro')

# ===== BREAST CANCER MODEL SETUP =====
try:
    # Load and preprocess breast cancer data
    bc_data = pd.read_csv('data/data.csv')
    bc_data['diagnosis'] = bc_data['diagnosis'].map({'M':1, 'B':0})
    
    # Remove unnecessary columns
    cols_to_drop = ['id', 'Unnamed: 32']
    bc_data = bc_data.drop(columns=[col for col in cols_to_drop if col in bc_data.columns], errors='ignore')
    
    bc_X = bc_data.drop(columns=['diagnosis'])
    bc_Y = bc_data['diagnosis']
    bc_feature_names = bc_X.columns.tolist()
    
    # Preprocessing
    bc_imputer = SimpleImputer(strategy='mean')
    bc_X_imputed = bc_imputer.fit_transform(bc_X)
    bc_scaler = StandardScaler()
    bc_X_scaled = bc_scaler.fit_transform(bc_X_imputed)
    
    bc_X_train, bc_X_test, bc_Y_train, bc_Y_test = train_test_split(bc_X_scaled, bc_Y, test_size=0.2, stratify=bc_Y, random_state=42)
    bc_model = LogisticRegression(max_iter=2000)
    bc_model.fit(bc_X_train, bc_Y_train)
    
    # Feature descriptions
    bc_feature_details = {
        "radius_mean": "Mean of distances from center to points on the perimeter",
        "texture_mean": "Standard deviation of gray-scale values",
        "perimeter_mean": "Mean size of the core tumor perimeter",
        "area_mean": "Mean size of the core tumor area",
        "smoothness_mean": "Mean of local variation in radius lengths",
        "compactness_mean": "Mean of perimeter² / area - 1.0",
        "concavity_mean": "Mean of severity of concave portions of the contour",
        "concave points_mean": "Mean of number of concave portions of the contour",
        "symmetry_mean": "Mean of symmetry",
        "fractal_dimension_mean": "Mean of 'coastline approximation' - 1",
        # Add the rest of your feature descriptions here
    }
    
    logging.info("Breast cancer model loaded successfully")
except Exception as e:
    logging.error(f"Error loading breast cancer model: {str(e)}")
    bc_model = None

# ===== HEART DISEASE MODEL SETUP =====
try:
    # Load and prepare heart disease data
    heart_data = pd.read_csv('data/heart.csv')
    hd_X = heart_data.drop(columns='target', axis=1)
    hd_y = heart_data['target']
    
    # Split data
    hd_X_train, hd_X_test, hd_y_train, hd_y_test = train_test_split(hd_X, hd_y, test_size=0.2, stratify=hd_y, random_state=2)
    
    # Train SVM model
    hd_model = svm.SVC(kernel='rbf', probability=True)
    hd_model.fit(hd_X_train, hd_y_train)
    
    # Heart disease feature descriptions
    hd_feature_details = {
        'age': 'Age in years',
        'sex': 'Sex (1 = male, 0 = female)',
        'cp': 'Chest pain type (0-3)',
        'trestbps': 'Resting blood pressure (mm Hg)',
        'chol': 'Serum cholesterol (mg/dl)',
        'fbs': 'Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)',
        'restecg': 'Resting ECG results (0-2)',
        'thalach': 'Maximum heart rate achieved',
        'exang': 'Exercise induced angina (1 = yes, 0 = no)',
        'oldpeak': 'ST depression induced by exercise relative to rest',
        'slope': 'Slope of the peak exercise ST segment (0-2)',
        'ca': 'Number of major vessels colored by fluoroscopy (0-3)',
        'thal': 'Thalassemia (0 = normal, 1 = fixed defect, 2 = reversible defect)'
    }
    
    logging.info("Heart disease model loaded successfully")
except Exception as e:
    logging.error(f"Error loading heart disease model: {str(e)}")
    hd_model = None

# ===== UTILITY FUNCTIONS =====
def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def is_medical_query(query):
    return any(keyword in query.lower() for keyword in MEDICAL_KEYWORDS)

def process_pdf(filepath):
    try:
        with open(filepath, 'rb') as f:
            return '\n'.join(
                page.extract_text()
                for page in PyPDF2.PdfReader(f).pages
                if page.extract_text()
            )
    except Exception as e:
        logging.error(f"PDF processing failed: {str(e)}")
        return None

def parse_results(analysis):
    results = []
    lines = re.findall(r"[-*•] (.+)", analysis)

    for line in lines:
        text = line.strip()
        lowered = text.lower()

        if any(w in lowered for w in ['high', 'elevated', 'increased', 'above']):
            status = 'high'
        elif any(w in lowered for w in ['low', 'decreased', 'reduced', 'below']):
            status = 'low'
        elif any(w in lowered for w in ['normal', 'within range', 'okay']):
            status = 'normal'
        else:
            status = 'normal'  # Default fallback

        results.append({
            'label': text,
            'status': status
        })

    return results

# ===== ROUTES =====
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/report-analyzer', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']
        if not (file and allowed_file(file.filename)):
            return render_template('report_analyzer.html', error="Invalid file. Please upload a PDF.")

        try:
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            report_text = process_pdf(filepath)
            if not report_text:
                raise ValueError("Could not extract text from PDF")

            # Ask Gemini to analyze
            analysis = MODEL.generate_content(
                f"""You are a medical expert. Analyze the following report and provide:
- 🚨 Abnormal Findings (bullet points)
- ✅ Normal Results (bullet points)
- 💡 Simple Explanations
- 📋 Recommendations (1-2 lines each)

Format in clear markdown. Be concise and helpful.

Medical Report:
{report_text}
"""
            ).text

            session['analysis'] = analysis
            structured_results = parse_results(analysis)
            return render_template('results.html', results=structured_results)

        except Exception as e:
            logging.error(str(e))
            return render_template('report_analyzer.html', error="Error processing file. Please try again.")
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)

    return render_template('report_analyzer.html')

@app.route('/ask', methods=['POST'])
def chatbot():
    try:
        data = request.get_json()
        user_question = data.get('question', '').strip()

        if not is_medical_query(user_question):
            return jsonify({
                'response': "I can only answer questions about your medical report or related health results. Please ask about your test results, findings, or medical terms."
            })

        prompt = (
            f"You are a medical report assistant. Only answer questions about the user's uploaded medical report or related health results. "
            f"If the question is unrelated, politely reply: 'I can only answer questions about your medical report or related health results.'\n\n"
            f"User's question: {user_question}\n\n"
            f"Medical Report Analysis:\n{session.get('analysis', '')}\n"
            f"Keep your answer under 3 sentences, clear and professional."
        )

        response = MODEL.generate_content(prompt)
        return jsonify({'response': response.text})

    except Exception as e:
        logging.error(f"Chatbot error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/speak-item/<int:item_id>', methods=['GET'])
def speak_item(item_id):
    try:
        analysis = session.get('analysis', '')
        if not analysis:
            return jsonify({'error': 'No analysis available'}), 404
        
        # Parse the report to get individual items
        results = parse_results(analysis)
        if item_id < 0 or item_id >= len(results):
            return jsonify({'error': 'Item not found'}), 404
            
        item = results[item_id]
        return jsonify({'text': f"{item['label']} - Status: {item['status']}"})
        
    except Exception as e:
        logging.error(f"Speech item error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/predict-breast-cancer', methods=['GET', 'POST'])
def predict_bc():
    if bc_model is None:
        return render_template('error.html', message="Breast cancer prediction model is not available")
    
    prediction = None
    input_values = {feat: request.form.get(feat, '') for feat in bc_feature_names}

    if request.method == 'POST' and 'reset' not in request.form:
        try:
            features = [float(request.form[feat]) for feat in bc_feature_names]
            input_data = np.array(features).reshape(1, -1)
            input_data = bc_scaler.transform(bc_imputer.transform(input_data))
            pred = bc_model.predict(input_data)
            prediction = 'Malignant (Cancer Detected)' if pred[0] == 1 else 'Benign (No Cancer Detected)'
        except Exception as e:
            prediction = f"Error: {e}"

    # On reset, clear all input values
    if request.method == 'POST' and 'reset' in request.form:
        input_values = {feat: '' for feat in bc_feature_names}
        prediction = None

    return render_template(
        'predict_bc.html',
        prediction=prediction,
        feature_names=bc_feature_names,
        feature_details=bc_feature_details,
        input_values=input_values
    )

@app.route('/predict-heart-disease', methods=['GET', 'POST'])
def predict_hd():
      # Load and prepare data
    heart_data = pd.read_csv('data/heart.csv')
    X = heart_data.drop(columns='target', axis=1)
    y = heart_data['target']

# Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

# Train SVM model
    model = svm.SVC(kernel='rbf', probability=True)
    model.fit(X_train, y_train)
    prediction = None
    if request.method == 'POST':
        try:
            features = [
                int(request.form['age']),
                int(request.form['sex']),
                int(request.form['cp']),
                int(request.form['trestbps']),
                int(request.form['chol']),
                int(request.form['fbs']),
                int(request.form['restecg']),
                int(request.form['thalach']),
                int(request.form['exang']),
                float(request.form['oldpeak']),
                int(request.form['slope']),
                int(request.form['ca']),
                int(request.form['thal'])
            ]
            input_data = np.array(features).reshape(1, -1)
            pred = model.predict(input_data)
            if pred[0] == 0:
                prediction = 'The Person does not have Heart Disease'
            else:
                prediction = 'The Person has Heart Disease'
        except Exception as e:
            prediction = f"Error: {e}"
    return render_template('predict_hd.html', prediction=prediction)



if __name__ == '__main__':
    port = find_free_port()
    logging.info(f"Starting server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=True)
