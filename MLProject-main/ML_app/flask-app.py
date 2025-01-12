from flask import Flask, render_template, request, jsonify, redirect, url_for
import numpy as np
import pandas as pd
import joblib
import os
from models import db, HeartDiseasePrediction, DiabetesPrediction, LungCancerPrediction

# First define the DATASETS_CONFIG
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

# Then create the Flask app
app = Flask(__name__)

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///predictions.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

# Create tables within app context
with app.app_context():
    db.create_all()

# Dictionary for loaded models
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


@app.route('/delete_prediction/<dataset_name>/<int:prediction_id>')
def delete_prediction(dataset_name, prediction_id):
    try:
        if dataset_name == 'heart_disease':
            prediction = HeartDiseasePrediction.query.get_or_404(prediction_id)
        elif dataset_name == 'diabetes':
            prediction = DiabetesPrediction.query.get_or_404(prediction_id)
        else:  # lung_cancer
            prediction = LungCancerPrediction.query.get_or_404(prediction_id)

        db.session.delete(prediction)
        db.session.commit()
        return redirect(url_for('history', dataset_name=dataset_name))
    except Exception as e:
        return render_template('error.html', error=str(e))


@app.route('/history/<dataset_name>')
def history(dataset_name):
    try:
        if dataset_name == 'heart_disease':
            predictions = HeartDiseasePrediction.query.order_by(HeartDiseasePrediction.timestamp.desc()).all()
        elif dataset_name == 'diabetes':
            predictions = DiabetesPrediction.query.order_by(DiabetesPrediction.timestamp.desc()).all()
        else:  # lung_cancer
            predictions = LungCancerPrediction.query.order_by(LungCancerPrediction.timestamp.desc()).all()

        return render_template('history.html',
                               dataset_name=dataset_name,
                               predictions=predictions)
    except Exception as e:
        return render_template('error.html', error=str(e))


@app.route('/predict/<dataset_name>', methods=['POST'])
def predict(dataset_name):
    try:
        if dataset_name not in DATASETS_CONFIG:
            return "Nieznany zbiór danych", 404

        if dataset_name not in loaded_models:
            load_models(dataset_name)

        models = loaded_models[dataset_name]
        input_data = prepare_input_data(dataset_name, request.form)

        expected_features = len(DATASETS_CONFIG[dataset_name]['features'])
        if len(input_data) != expected_features:
            raise ValueError(f"Nieprawidłowa liczba cech. Oczekiwano {expected_features}, otrzymano {len(input_data)}")

        input_scaled = models['scaler'].transform([input_data])

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

        # Save predictions to database
        if dataset_name == 'heart_disease':
            prediction = HeartDiseasePrediction(
                sex=float(input_data[0]),
                age=float(input_data[1]),
                cp=float(input_data[2]),
                trestbps=float(input_data[3]),
                chol=float(input_data[4]),
                fbs=float(input_data[5]),
                restecg=float(input_data[6]),
                thalachh=float(input_data[7]),
                exng=float(input_data[8]),
                oldpeak=float(input_data[9]),
                slp=float(input_data[10]),
                caa=float(input_data[11]),
                thall=float(input_data[12]),
                rf_prediction=predictions['Random Forest']['prediction'],
                rf_probability=predictions['Random Forest']['probability'],
                lr_prediction=predictions['Logistic Regression']['prediction'],
                lr_probability=predictions['Logistic Regression']['probability'],
                dt_prediction=predictions['Decision Tree']['prediction'],
                dt_probability=predictions['Decision Tree']['probability']
            )
        elif dataset_name == 'diabetes':
            prediction = DiabetesPrediction(
                pregnancies=float(input_data[0]),
                glucose=float(input_data[1]),
                blood_pressure=float(input_data[2]),
                skin_thickness=float(input_data[3]),
                insulin=float(input_data[4]),
                bmi=float(input_data[5]),
                diabetes_pedigree_function=float(input_data[6]),
                age=float(input_data[7]),
                rf_prediction=predictions['Random Forest']['prediction'],
                rf_probability=predictions['Random Forest']['probability'],
                lr_prediction=predictions['Logistic Regression']['prediction'],
                lr_probability=predictions['Logistic Regression']['probability'],
                dt_prediction=predictions['Decision Tree']['prediction'],
                dt_probability=predictions['Decision Tree']['probability']
            )
        else:  # lung_cancer
            prediction = LungCancerPrediction(
                gender=request.form['GENDER'],
                age=float(input_data[1]),
                smoking=float(input_data[2]),
                yellow_fingers=float(input_data[3]),
                anxiety=float(input_data[4]),
                peer_pressure=float(input_data[5]),
                chronic_disease=float(input_data[6]),
                fatigue=float(input_data[7]),
                allergy=float(input_data[8]),
                wheezing=float(input_data[9]),
                alcohol_consuming=float(input_data[10]),
                coughing=float(input_data[11]),
                shortness_of_breath=float(input_data[12]),
                swallowing_difficulty=float(input_data[13]),
                chest_pain=float(input_data[14]),
                rf_prediction=predictions['Random Forest']['prediction'],
                rf_probability=predictions['Random Forest']['probability'],
                lr_prediction=predictions['Logistic Regression']['prediction'],
                lr_probability=predictions['Logistic Regression']['probability'],
                dt_prediction=predictions['Decision Tree']['prediction'],
                dt_probability=predictions['Decision Tree']['probability']
            )

        db.session.add(prediction)
        db.session.commit()

        return render_template('result.html',
                               dataset_name=dataset_name,
                               predictions=predictions)

    except Exception as e:
        return render_template('error.html', error=str(e))


if __name__ == '__main__':
    app.run(debug=True)