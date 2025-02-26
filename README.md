# 🏥 Medical Diagnosis System

A comprehensive **Medical Diagnosis System** that includes multiple **disease detection modules** using **Machine Learning** and **Deep Learning** models. The system currently supports:
✅ **Brain Tumor Detection**  
✅ **Breast Cancer Classification**  
✅ **General Disease Prediction** based on symptoms  

---

## 🚀 Features

🔬 **Brain Tumor Detection**: Analyzes **MRI images** to detect the presence of tumors 🧠  
🎗 **Breast Cancer Classification**: Classifies **breast cancer images** into **Benign, Malignant, or Normal** 🏥  
🤒 **Symptom-Based Disease Prediction**: Predicts **possible diseases** based on input symptoms 💉  
📄 **Automated Report Generation**: Generates **detailed PDF reports** for all diagnoses 📝  
🌐 **Cross-Origin Support**: Enabled **CORS** for web application integration 🔗  
🔄 **RESTful API**: Simple and consistent **API endpoints** for all services ⚙️  

---

## 📌 Prerequisites

Before running the system, make sure you have the following installed:

✅ **Python 3.8+** 🐍  
✅ **Flask** 🌐  
✅ **TensorFlow** 🤖  
✅ **NumPy** 📊  
✅ **pandas** 🏗  
✅ **joblib** 🏎  
✅ **FPDF** 📄  
✅ **flask-cors** 🔄  

---

## 📥 Installation

1️⃣ Clone the repository:
```bash
$ git clone https://github.com/Rathoreatri03/Murphy-Systum.git
$ cd medical-diagnosis-system
```

2️⃣ Install required packages:
```bash
$ pip install -r requirements.txt
```

3️⃣ Ensure you have the following **model files** in the root directory and you can download from Release Tab:
📌 `brain_tumor_model.h5` (Brain Tumor Model)  
📌 `boba.h5` (Breast Cancer Model)  
📌 `random_forest_model.pkl` (Disease Prediction Model)  

---

## 📂 Project Structure

```
medical-diagnosis-system/
├── app.py
├── models/
│   ├── brain_tumor_model.h5
│   ├── boba.h5
│   └── random_forest_model.pkl
├── uploads/
├── reports/
├── templates/
│   ├── index.html
│   ├── braincancer.html
│   └── breastcancer.html
└── README.md
```

---

## 🔗 API Endpoints

### 🧠 Brain Tumor Detection
- **Endpoint**: `/predict`
- **Method**: `POST`
- **Input**: Image file (MRI scan)
- **Output**: JSON with **prediction result** and **confidence score**

### 🎗 Breast Cancer Classification
- **Endpoint**: `/predict`
- **Method**: `POST`
- **Input**: Image file
- **Output**: JSON with classification result (**Benign, Malignant, Normal**)

### 🤒 Disease Prediction
- **Endpoint**: `/predict`
- **Method**: `POST`
- **Input**: JSON with **symptom indicators**
- **Output**: JSON with **predicted disease** and **report file**

### 📄 Report Download
- **Endpoint**: `/download_report/<filename>`
- **Method**: `GET`
- **Output**: PDF **report file**

---

## 💻 Usage Examples

### 🧠 Brain Tumor Detection
```python
import requests

url = "http://localhost:9500/predict"
files = {'file': open('brain_scan.jpg', 'rb')}
response = requests.post(url, files=files)
print(response.json())
```

### 🎗 Breast Cancer Classification
```python
import requests

url = "http://localhost:8000/predict"
files = {'file': open('breast_scan.jpg', 'rb')}
response = requests.post(url, files=files)
print(response.json())
```

### 🤒 Disease Prediction
```python
import requests
import json

url = "http://localhost:5000/predict"
symptoms = {
    "headache": 1,
    "fever": 1,
    "fatigue": 1,
}
response = requests.post(url, json=symptoms)
print(response.json())
```

---

## ⚙️ Running the Applications

Each service can be run independently:
```bash
# 🧠 Brain Tumor Detection Service
$ python app.py --service brain_tumor --port 9500

# 🎗 Breast Cancer Classification Service
$ python app.py --service breast_cancer --port 8000

# 🤒 Disease Prediction Service
$ python app.py --service disease_prediction --port 5000
```

---

## 🔒 Security Notes

⚠️ The application currently runs in **debug mode**, which should be **disabled in production** ❌  
🛡 Implement **proper input validation** and **file type checking** before deployment 🔍  
🔑 Add **authentication and authorization** mechanisms for production use 🔒  
🌍 Secure the API endpoints using **HTTPS** in production 🌐  

---

## 📜 License

This project is licensed under the **MIT License** - see the `LICENSE` file for details 📝  
