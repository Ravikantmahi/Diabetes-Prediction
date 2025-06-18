from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import base64
from io import BytesIO
from src.data_preprocessing import preprocess_data, get_preprocessor
from src.model_training import train_model
from src.evaluation import evaluate_model
from src.visualization import plot_feature_importance, plot_confusion_matrix

app = Flask(__name__)

# Load and preprocess data, train model
data = pd.read_csv("data/diabetes_data.csv")
X_train, X_test, y_train, y_test, feature_names = preprocess_data(data)
model = train_model(X_train, y_train)
metrics, _ = evaluate_model(model, X_test, y_test)
preprocessor = get_preprocessor()  # Get the fitted preprocessor

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    data = request.form
    input_data = pd.DataFrame({
        'gender': [data['gender']],
        'age': [float(data['age'])],
        'hypertension': [int(data['hypertension'])],
        'heart_disease': [int(data['heart_disease'])],
        'smoking_history': [data['smoking_history']],
        'bmi': [float(data['bmi'])],
        'HbA1c_level': [float(data['hba1c'])],
        'blood_glucose_level': [float(data['glucose'])]
    })
    
    # Preprocess input using the fitted preprocessor
    input_processed = preprocessor.transform(input_data)
    
    # Predict
    prediction = model.predict(input_processed)[0]
    prob = model.predict_proba(input_processed)[0][1]
    
    # Generate visualizations
    feature_importance_img = plot_feature_importance(model, feature_names)
    confusion_matrix_img = plot_confusion_matrix(model, X_test, y_test)
    
    return jsonify({
        'prediction': 'Diabetic' if prediction == 1 else 'Not Diabetic',
        'probability': round(prob * 100, 2),
        'metrics': metrics,
        'feature_importance': feature_importance_img,
        'confusion_matrix': confusion_matrix_img
    })

if __name__ == '__main__':
    app.run(debug=True)