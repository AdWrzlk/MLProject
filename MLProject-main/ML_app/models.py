from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()


class HeartDiseasePrediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    # Input features
    sex = db.Column(db.Integer)
    age = db.Column(db.Integer)
    cp = db.Column(db.Integer)
    trestbps = db.Column(db.Float)
    chol = db.Column(db.Float)
    fbs = db.Column(db.Integer)
    restecg = db.Column(db.Integer)
    thalachh = db.Column(db.Float)
    exng = db.Column(db.Integer)
    oldpeak = db.Column(db.Float)
    slp = db.Column(db.Integer)
    caa = db.Column(db.Integer)
    thall = db.Column(db.Integer)

    # Predictions
    rf_prediction = db.Column(db.Integer)
    rf_probability = db.Column(db.Float)
    lr_prediction = db.Column(db.Integer)
    lr_probability = db.Column(db.Float)
    dt_prediction = db.Column(db.Integer)
    dt_probability = db.Column(db.Float)


class DiabetesPrediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    # Input features
    pregnancies = db.Column(db.Integer)
    glucose = db.Column(db.Float)
    blood_pressure = db.Column(db.Float)
    skin_thickness = db.Column(db.Float)
    insulin = db.Column(db.Float)
    bmi = db.Column(db.Float)
    diabetes_pedigree_function = db.Column(db.Float)
    age = db.Column(db.Integer)

    # Predictions
    rf_prediction = db.Column(db.Integer)
    rf_probability = db.Column(db.Float)
    lr_prediction = db.Column(db.Integer)
    lr_probability = db.Column(db.Float)
    dt_prediction = db.Column(db.Integer)
    dt_probability = db.Column(db.Float)


class LungCancerPrediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    # Input features
    gender = db.Column(db.String(1))
    age = db.Column(db.Integer)
    smoking = db.Column(db.Integer)
    yellow_fingers = db.Column(db.Integer)
    anxiety = db.Column(db.Integer)
    peer_pressure = db.Column(db.Integer)
    chronic_disease = db.Column(db.Integer)
    fatigue = db.Column(db.Integer)
    allergy = db.Column(db.Integer)
    wheezing = db.Column(db.Integer)
    alcohol_consuming = db.Column(db.Integer)
    coughing = db.Column(db.Integer)
    shortness_of_breath = db.Column(db.Integer)
    swallowing_difficulty = db.Column(db.Integer)
    chest_pain = db.Column(db.Integer)

    # Predictions
    rf_prediction = db.Column(db.Integer)
    rf_probability = db.Column(db.Float)
    lr_prediction = db.Column(db.Integer)
    lr_probability = db.Column(db.Float)
    dt_prediction = db.Column(db.Integer)
    dt_probability = db.Column(db.Float)