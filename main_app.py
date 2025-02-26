import joblib
import pandas as pd
from flask import Flask, request, jsonify, render_template
from fpdf import FPDF
import os
import json
from textwrap import wrap

# Load the trained model
model = joblib.load("models/random_forest_model.pkl")

# Define feature names
FEATURES = ['receiving_blood_transfusion', 'red_sore_around_nose', 'abnormal_menstruation', 'continuous_sneezing',
            'breathlessness', 'blackheads', 'shivering', 'dizziness', 'back_pain', 'unsteadiness',
            'yellow_crust_ooze', 'muscle_weakness', 'loss_of_balance', 'chills', 'ulcers_on_tongue',
            'stomach_bleeding', 'lack_of_concentration', 'coma', 'neck_pain', 'weakness_of_one_body_side',
            'diarrhoea', 'receiving_unsterile_injections', 'headache', 'family_history', 'fast_heart_rate',
            'pain_behind_the_eyes', 'sweating', 'mucoid_sputum', 'spotting_ urination', 'sunken_eyes',
            'dischromic _patches', 'nausea', 'dehydration', 'loss_of_appetite', 'abdominal_pain', 'stomach_pain',
            'yellowish_skin', 'altered_sensorium', 'chest_pain', 'muscle_wasting', 'vomiting', 'mild_fever',
            'high_fever', 'red_spots_over_body', 'dark_urine', 'itching', 'yellowing_of_eyes', 'fatigue',
            'joint_pain', 'muscle_pain']

# Load Disease Data from JSON - Use relative path and error handling
try:
    with open("json_data/main_json.json", "r", encoding='utf-8') as file:
        DISEASE_DATA = json.load(file)
except FileNotFoundError:
    print("Error: JSON file not found. Please check the file path.")
    DISEASE_DATA = []
except json.JSONDecodeError:
    print("Error: Invalid JSON format.")
    DISEASE_DATA = []

# Flask App Configuration
app = Flask(__name__)
REPORT_FOLDER = "reports"
app.config['REPORT_FOLDER'] = REPORT_FOLDER
if not os.path.exists(REPORT_FOLDER):
    os.makedirs(REPORT_FOLDER)


class PDF(FPDF):
    def multi_cell_with_wrap(self, w, h, txt, border=0, align='L'):
        # Split text into lines that fit within the specified width
        lines = wrap(txt, width=85)  # Adjust width parameter as needed
        for line in lines:
            self.cell(w, h, txt=line, border=border, ln=True, align=align)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/braincancer.html')
def brain_cancer():
    return render_template('braincancer.html')


@app.route('/breastcancer.html')
def breast_cancer():
    return render_template('breastcancer.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        input_data = pd.DataFrame([data], columns=FEATURES)
        predicted_disease = model.predict(input_data)[0]

        # Debug print
        print(f"Predicted disease: {predicted_disease}")
        print(f"Number of diseases in DISEASE_DATA: {len(DISEASE_DATA)}")

        # Retrieve detailed disease information from JSON with case-insensitive matching
        disease_info = next(
            (d for d in DISEASE_DATA if d["name"].lower() == predicted_disease.lower()),
            None
        )

        # Debug print
        print(f"Found disease info: {disease_info is not None}")

        # Generate PDF Report
        pdf = PDF()
        pdf.add_page()
        pdf.set_font("Arial", style="B", size=16)
        pdf.cell(200, 10, txt="Patient Diagnosis Report", ln=True, align='C')
        pdf.ln(10)

        # Predicted Disease
        pdf.set_font("Arial", style="B", size=14)
        pdf.cell(200, 10, txt=f"Predicted Disease: {predicted_disease}", ln=True, align='L')
        pdf.ln(5)

        # Display Selected Symptoms
        pdf.set_font("Arial", style="B", size=12)
        pdf.cell(200, 10, txt="Selected Symptoms:", ln=True, align='L')
        pdf.ln(5)

        pdf.set_font("Arial", size=12)
        selected_symptoms = [symptom for symptom, value in data.items() if value == 1]
        for symptom in selected_symptoms:
            pdf.cell(200, 10, txt=f"- {symptom.replace('_', ' ').title()}", ln=True, align='L')

        pdf.ln(10)

        # If disease details are found, add them to the report
        if disease_info:
            # Symptoms Explanation
            if disease_info.get("symptoms"):
                pdf.set_font("Arial", style="B", size=12)
                pdf.cell(200, 10, txt="Symptoms Explanation:", ln=True, align='L')
                pdf.ln(5)

                pdf.set_font("Arial", size=12)
                for symptom, desc in disease_info["symptoms"].items():
                    pdf.set_font("Arial", style="B", size=12)
                    pdf.cell(200, 10, txt=f"{symptom.replace('_', ' ').title()}:", ln=True, align='L')
                    pdf.set_font("Arial", size=12)
                    pdf.multi_cell_with_wrap(200, 10, desc)
                    pdf.ln(5)

            # Possible Causes
            if disease_info.get("possible_causes"):
                pdf.ln(5)
                pdf.set_font("Arial", style="B", size=12)
                pdf.cell(200, 10, txt="Possible Causes:", ln=True, align='L')
                pdf.ln(5)

                pdf.set_font("Arial", size=12)
                for cause in disease_info["possible_causes"]:
                    pdf.multi_cell_with_wrap(200, 10, f"- {cause}")

            # Medical Relief
            if disease_info.get("medical_relief"):
                pdf.ln(5)
                pdf.set_font("Arial", style="B", size=12)
                pdf.cell(200, 10, txt="Medical Relief:", ln=True, align='L')
                pdf.ln(5)

                pdf.set_font("Arial", size=12)
                for relief in disease_info["medical_relief"]:
                    pdf.multi_cell_with_wrap(200, 10, f"- {relief}")
        else:
            pdf.set_font("Arial", style="I", size=12)
            pdf.cell(200, 10, txt="Detailed disease information not found in database.", ln=True, align='L')

        pdf.ln(10)
        pdf.set_font("Arial", 'B', size=14)
        pdf.cell(200, 10, "Technical Information", ln=True, align='L')
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(190, 5,
                       "Analysis performed using deep learning-based image classification model trained on medical imaging datasets. The model analyzes visual patterns and features to categorize the image into predefined classes.",
                       align='L')

        # Recommendations
        pdf.ln(25)
        pdf.set_font("Arial", 'B', size=14)
        pdf.cell(200, 10, "Recommendations", ln=True, align='L')
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(190, 5,
                       "1. Review this report with your healthcare provider\n2. Discuss any concerns or questions about the results\n3. Follow your healthcare provider's recommendations for any additional testing or monitoring\n4. Keep this report for your medical records",
                       align='L')

        # Footer with Additional Disclaimers
        pdf.ln(35)
        pdf.set_font("Arial", 'I', size=8)
        pdf.multi_cell(190, 4,
                       "This report is generated automatically and should be interpreted by qualified medical professionals. The analysis is based on machine learning algorithms and should be used as a supporting tool, not as a replacement for professional medical judgment. Regular medical check-ups and consultations are essential for proper healthcare management.",
                       align='L')
        # Save PDF
        pdf_filename = os.path.join(REPORT_FOLDER, "patient_report.pdf")
        pdf.output(pdf_filename)

        return jsonify({
            "prediction": predicted_disease,
            "report": pdf_filename,
            "success": True
        })

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return jsonify({
            "error": str(e),
            "success": False
        }), 500


if __name__ == '__main__':
    app.run(debug=True)