import os
import tensorflow as tf
import numpy as np
from flask import Flask, request, jsonify, send_file
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from flask_cors import CORS
from fpdf import FPDF

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the trained model
model_path = r"models/brain_tumor_model.h5"
model = tf.keras.models.load_model(model_path)

UPLOAD_FOLDER = 'uploads'
REPORT_FOLDER = 'reports'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['REPORT_FOLDER'] = REPORT_FOLDER

# Ensure necessary folders exist
for folder in [UPLOAD_FOLDER, REPORT_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

def preprocess_image(image_path):
    img_height, img_width = 150, 150
    img = load_img(image_path, target_size=(img_height, img_width))
    img_array = img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def generate_pdf_report(image_path, result_text, confidence, filename):
    """Generates a PDF report with prediction details."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", 'B', size=16)

    # Title
    pdf.cell(200, 10, "Brain Tumor Detection Report", ln=True, align='C')
    pdf.ln(10)  # Space after title

    # Image Section
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, "Uploaded Image:", ln=True, align='L')
    pdf.image(image_path, x=10, y=50, w=80)  # Adding image

    pdf.ln(120)  # Increased space after image

    # Prediction Result
    pdf.set_font("Arial", 'B', size=12)
    pdf.cell(200, 10, f"Prediction: {result_text}", ln=True, align='L')
    pdf.cell(200, 10, f"Tumor detection %: {confidence}", ln=True, align='L')
    # Technical Details
    pdf.ln(25)
    pdf.set_font("Arial", 'B', size=14)
    pdf.cell(200, 10, "3. Technical Information", ln=True, align='L')
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(190, 5,
                   "Analysis performed using deep learning-based image classification model trained on medical imaging datasets. The model analyzes visual patterns and features to categorize the image into predefined classes.",
                   align='L')

    # Recommendations
    pdf.ln(25)
    pdf.set_font("Arial", 'B', size=14)
    pdf.cell(200, 10, "4. Recommendations", ln=True, align='L')
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
    report_path = os.path.join(REPORT_FOLDER, f"{filename}.pdf")
    pdf.output(report_path)

    return report_path

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Save uploaded image
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # Process the image
        processed_img = preprocess_image(file_path)

        # Predict
        prediction = model.predict(processed_img)
        probability = 1 - prediction[0][0]  # Convert to float for JSON serialization

        result_text = "No Tumor Detection" if probability < 0.5 else "Tumor Detected"
        confidence = f"{probability:.2%}"

        # Generate PDF Report
        pdf_filename = os.path.splitext(file.filename)[0]  # Remove file extension
        pdf_path = generate_pdf_report(file_path, result_text, confidence, pdf_filename)

        # Clean up - remove uploaded file
        os.remove(file_path)

        return jsonify({
            'prediction': result_text,
            'confidence': confidence
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download_report/<filename>', methods=['GET'])
def download_report(filename):
    """Allows the user to download the generated PDF report."""
    report_path = os.path.join(REPORT_FOLDER, filename)
    if os.path.exists(report_path):
        return send_file(report_path, as_attachment=True)
    else:
        return jsonify({'error': 'Report not found'}), 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9500, debug=True)