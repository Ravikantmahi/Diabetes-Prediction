# 🩺 Diabetes Prediction Web Application

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/flask-2.0+-green.svg)

---

## 🧠 Overview

This is a **Flask-based web application** that predicts whether a patient is **diabetic** using a **Random Forest Classifier** trained on a dataset of 10,000 patient records.

Users can input patient details like gender, age, hypertension, heart disease, smoking history, BMI, HbA1c level, and blood glucose level. The app then returns:

- ✅ Diabetes prediction (Yes/No)
- 📈 Probability score
- 📊 Model performance metrics (accuracy, precision, recall, F1 score)
- 📉 Visualizations (feature importance & confusion matrix)

---

## 🌟 Features

- 🔘 **User-Friendly Web Form** for input
- 📊 **Model Metrics**: Accuracy, precision, recall, F1 score
- 🧮 **Prediction Probability**
- 📌 **Feature Importance & Confusion Matrix**
- ⚖️ **Balanced Class Weights** to handle class imbalance
- 🚦 **Error Handling** and form validation

---
## 🗂 Project Structure
```
diabetes_prediction/
│
├── data/
│ └── diabetes_data.csv # Patient dataset
├── src/
│ ├── data_preprocessing.py # Cleans and preprocesses data
│ ├── model_training.py # Trains the ML model
│ ├── evaluation.py # Calculates model metrics
│ └── visualization.py # Generates plots (base64)
├── static/
│ ├── css/
│ │ └── styles.css # Web UI styling
│ └── js/
│ └── script.js # Form logic
├── templates/
│ └── index.html # Web form UI
├── app.py # Flask app runner
├── requirements.txt # Python dependencies
└── README.md # Project documentation
```

---

## 🔧 Prerequisites

- Python 3.8 or higher
- Git
- Dataset: `diabetes_data.csv` with columns:  
  `gender`, `age`, `hypertension`, `heart_disease`, `smoking_history`, `bmi`, `HbA1c_level`, `blood_glucose_level`, `diabetes`

---

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/diabetes_prediction_web.git
cd diabetes_prediction_web
pip install -r requirements.txt

Or install manually:

pip install flask pandas numpy scikit-learn matplotlib seaborn
```

## Add Dataset
Place diabetes_data.csv inside the data/ folder.

✅ Make sure the dataset has valid entries and column values like:
smoking_history: never, No Info, current, former, ever, not current

▶️ Running the App
 ```
  python app.py
```
Open your browser and go to:
👉 http://127.0.0.1:5000

🧪 Usage
Step 1: Fill Patient Details
Gender: Male / Female

Age: e.g., 50

Hypertension: 1 (Yes) or 0 (No)

Heart Disease: 1 (Yes) or 0 (No)

Smoking History: Never, Current, Former, etc.

BMI: e.g., 27.5

HbA1c Level: e.g., 6.6

Blood Glucose Level: e.g., 140

Step 2: Click “Predict”
🩸 Diabetes Prediction: Yes/No

🔢 Probability of diabetes (%)

📊 Model Performance Metrics

📈 Visuals: Feature importance & confusion matrix

📉 Model Details
Algorithm: Random Forest Classifier (100 trees)

Class Handling: class_weight='balanced' for imbalance

Encoding: One-hot encoding for categorical features

Scaling: StandardScaler for numeric features

🧾 Example Performance
| Metric    | Value  |
| --------- | ------ |
| Accuracy  | 96.99% |
| Precision | 94.67% |
| Recall    | 68.68% |
| F1 Score  | 79.61% |


columns

🤝 Contributing
Contributions are welcome!
```
 # Fork & Clone
git clone https://github.com/your-username/diabetes_prediction_web.git

# Create Feature Branch
git checkout -b feature/your-feature

# Commit Changes
git commit -m "Add your feature"

# Push & PR
git push origin feature/your-feature
```
📬 Contact
Ravi Kant

Built with ❤️ for medical decision support and machine learning enthusiasts.
