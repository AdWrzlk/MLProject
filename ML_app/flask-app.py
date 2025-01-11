from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Konfiguracja zbiorów danych
DATASETS_CONFIG = {
    'heart_disease': {
        'features': [
            ('sex', 'Płeć pacjenta (0 = kobieta, 1 = mężczyzna)'),
            ('age', 'Wiek pacjenta'),
            ('cp', 'Typ bólu w klatce piersiowej'),
            ('trestbps', 'Ciśnienie tętnicze krwi w spoczynku'),
            ('chol', 'Poziom cholesterolu'),
            ('fbs', 'Cukier we krwi na czczo'),
            ('restecg', 'Wyniki EKG spoczynkowego'),
            ('thalachh', 'Maksymalne tętno'),
            ('exng', 'Dławica wysiłkowa'),
            ('oldpeak', 'Obniżenie odcinka ST'),
            ('slp', 'Nachylenie odcinka ST'),
            ('caa', 'Liczba głównych naczyń wieńcowych'),
            ('thall', 'Wynik testu Thallium')
        ]
    },
    'diabetes': {
        'features': [
            ('Pregnancies', 'Liczba ciąż'),
            ('Glucose', 'Poziom glukozy'),
            ('BloodPressure', 'Ciśnienie krwi'),
            ('SkinThickness', 'Grubość fałdu skórnego'),
            ('Insulin', 'Poziom insuliny'),
            ('BMI', 'Wskaźnik masy ciała'),
            ('DiabetesPedigreeFunction', 'Funkcja rodowodu cukrzycy'),
            ('Age', 'Wiek')
        ]
    },
    'lung_cancer': {
        'features': [
            ('GENDER', 'Płeć (M/F)'),
            ('AGE', 'Wiek'),
            ('SMOKING', 'Palenie tytoniu (1-2)'),
            ('YELLOW_FINGERS', 'Żółte palce (1-2)'),
            ('ANXIETY', 'Niepokój (1-2)'),
            ('PEER_PRESSURE', 'Presja rówieśników (1-2)'),
            ('CHRONIC DISEASE', 'Choroba przewlekła (1-2)'),
            ('FATIGUE', 'Zmęczenie (1-2)'),
            ('ALLERGY', 'Alergia (1-2)'),
            ('WHEEZING', 'Świszczący oddech (1-2)'),
            ('ALCOHOL CONSUMING', 'Spożywanie alkoholu (1-2)'),
            ('COUGHING', 'Kaszel (1-2)'),
            ('SHORTNESS OF BREATH', 'Duszność (1-2)'),
            ('SWALLOWING DIFFICULTY', 'Trudności w połykaniu (1-2)'),
            ('CHEST PAIN', 'Ból w klatce piersiowej (1-2)')
        ]
    }
}

# Słownik na załadowane modele
loaded_models = {}

def load_models(dataset_name):
    """Ładuje modele dla wybranego zbioru danych"""
    if dataset_name not in loaded_models:
        models_dir = f'models/{dataset_name}'
        loaded_models[dataset_name] = {
            'rf': joblib.load(f'{models_dir}/rf_model.joblib'),
            'lr': joblib.load(f'{models_dir}/lr_model.joblib'),
            'dt': joblib.load(f'{models_dir}/dt_model.joblib'),
            'scaler': joblib.load(f'{models_dir}/scaler.joblib')
        }

def prepare_input_data(dataset_name, form_data):
    """Przygotowuje dane wejściowe do odpowiedniego formatu"""
    features = DATASETS_CONFIG[dataset_name]['features']
    input_data = []
    
    for feature_name, _ in features:
        value = form_data[feature_name]
        # Konwersja wartości dla lung_cancer
        if dataset_name == 'lung_cancer' and feature_name == 'GENDER':
            value = 1 if value.upper() == 'M' else 0
        input_data.append(float(value))
    
    return input_data

@app.route('/')
def index():
    return render_template('index.html', datasets=DATASETS_CONFIG.keys())

@app.route('/dataset/<dataset_name>')
def dataset_form(dataset_name):
    if dataset_name not in DATASETS_CONFIG:
        return "Nieznany zbiór danych", 404
    
    features = DATASETS_CONFIG[dataset_name]['features']
    return render_template('dataset_form.html', 
                         dataset_name=dataset_name, 
                         features=features)

@app.route('/predict/<dataset_name>', methods=['POST'])
def predict(dataset_name):
    try:
        if dataset_name not in DATASETS_CONFIG:
            return "Nieznany zbiór danych", 404

        # Załaduj modele jeśli jeszcze nie są załadowane
        if dataset_name not in loaded_models:
            load_models(dataset_name)

        # Pobierz modele i skaler
        models = loaded_models[dataset_name]
        
        # Przygotuj dane wejściowe
        input_data = prepare_input_data(dataset_name, request.form)
        
        # Sprawdź czy liczba cech się zgadza
        expected_features = len(DATASETS_CONFIG[dataset_name]['features'])
        if len(input_data) != expected_features:
            raise ValueError(f"Nieprawidłowa liczba cech. Oczekiwano {expected_features}, otrzymano {len(input_data)}")

        # Skalowanie danych
        input_scaled = models['scaler'].transform([input_data])

        # Dokonaj predykcji wszystkimi modelami
        predictions = {
            'Random Forest': {
                'prediction': int(models['rf'].predict(input_scaled)[0]),
                'probability': float(models['rf'].predict_proba(input_scaled)[0][1])
            },
            'Logistic Regression': {
                'prediction': int(models['lr'].predict(input_scaled)[0]),
                'probability': float(models['lr'].predict_proba(input_scaled)[0][1])
            },
            'Decision Tree': {
                'prediction': int(models['dt'].predict(input_scaled)[0]),
                'probability': float(models['dt'].predict_proba(input_scaled)[0][1])
            }
        }

        return render_template('result.html', 
                             dataset_name=dataset_name,
                             predictions=predictions)

    except Exception as e:
        return render_template('error.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
