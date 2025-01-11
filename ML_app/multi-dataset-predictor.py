import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os


class MultiDatasetPredictor:
    DATASETS_CONFIG = {
        'heart_disease': {
            'path': 'datasets/heart-disease.csv',
            'target': 'target',
            'features': [
                ('age', 'Wiek pacjenta'),
                ('sex', 'Płeć pacjenta (0 = kobieta, 1 = mężczyzna)'),
                ('cp', 'Typ bólu w klatce piersiowej'),
                ('trestbps', 'Ciśnienie tętnicze krwi w spoczynku'),
                ('chol', 'Poziom cholesterolu'),
                ('fbs', 'Cukier we krwi na czczo'),
                ('restecg', 'Wyniki EKG spoczynkowego'),
                ('thalach', 'Maksymalne tętno'),  # Zmieniono z thalachh na thalach
                ('exang', 'Dławica wysiłkowa'),  # Zmieniono z exng na exang
                ('oldpeak', 'Obniżenie odcinka ST'),
                ('slope', 'Nachylenie odcinka ST'),  # Zmieniono z slp na slope
                ('ca', 'Liczba głównych naczyń wieńcowych'),  # Zmieniono z caa na ca
                ('thal', 'Wynik testu Thallium')  # Zmieniono z thall na thal
            ]
        },
        'diabetes': {
            'path': 'datasets/diabetes.csv',
            'target': 'Outcome',
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
            'path': 'datasets/survey-lung-cancer.csv',
            'target': 'LUNG_CANCER',
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

    def __init__(self):
        self.current_dataset = None
        self.models = {}
        self.scaler = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_dataset(self, dataset_name):
        """Wczytuje wybrany zbiór danych"""
        if dataset_name not in self.DATASETS_CONFIG:
            raise ValueError(f"Nieznany zbiór danych: {dataset_name}")

        config = self.DATASETS_CONFIG[dataset_name]
        self.current_dataset = dataset_name

        # Wczytanie danych
        data = pd.read_csv(config['path'])

        # Przygotowanie danych
        if dataset_name == 'lung_cancer':
            # Upewnij się, że wszystkie wymagane kolumny są obecne
            required_columns = [feature[0] for feature in config['features']] + [config['target']]
            missing_columns = set(required_columns) - set(data.columns)
            if missing_columns:
                raise ValueError(f"Brakujące kolumny w zbiorze danych: {missing_columns}")

            # Konwersja kolumny GENDER na wartości binarne
            data['GENDER'] = data['GENDER'].map({'M': 1, 'F': 0})
            # Konwersja kolumny LUNG_CANCER na wartości binarne
            data['LUNG_CANCER'] = data['LUNG_CANCER'].map({'YES': 1, 'NO': 0})

            # Upewnij się, że wszystkie kolumny są numeryczne
            for column in data.columns:
                if column != config['target']:
                    data[column] = pd.to_numeric(data[column], errors='raise')

        # Wybierz tylko kolumny zdefiniowane w konfiguracji
        feature_columns = [feature[0] for feature in config['features']]
        self.X = data[feature_columns]
        self.y = data[config['target']]

        # Sprawdź czy liczba cech się zgadza
        if len(self.X.columns) != len(config['features']):
            raise ValueError(
                f"Nieprawidłowa liczba cech. Oczekiwano {len(config['features'])}, otrzymano {len(self.X.columns)}")

        # Skalowanie danych
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(self.X)

        # Podział danych
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_scaled, self.y, test_size=0.2, random_state=42
        )

        print(f"\nZaładowano zbiór danych {dataset_name}:")
        print(f"Liczba cech: {len(self.X.columns)}")
        print(f"Liczba próbek: {len(self.X)}")
        print("Cechy:", ", ".join(self.X.columns))

    def train_models(self):
        """Trenuje modele dla aktualnego zbioru danych"""
        if self.current_dataset is None:
            raise ValueError("Najpierw wybierz zbiór danych!")

        print(f"\nRozpoczęto trenowanie modeli dla zbioru {self.current_dataset}...")

        # Trenowanie modeli
        self.models['rf'] = RandomForestClassifier(random_state=42)
        self.models['lr'] = LogisticRegression(random_state=42, max_iter=1000)
        self.models['dt'] = DecisionTreeClassifier(random_state=42)

        for name, model in self.models.items():
            print(f"\nTrenowanie modelu {name}...")
            model.fit(self.X_train, self.y_train)
            train_score = model.score(self.X_train, self.y_train)
            test_score = model.score(self.X_test, self.y_test)
            print(f"Dokładność na zbiorze treningowym: {train_score:.4f}")
            print(f"Dokładność na zbiorze testowym: {test_score:.4f}")

    def evaluate_models(self):
        """Ocenia wytrenowane modele"""
        if not self.models:
            print("Najpierw wytreniuj modele!")
            return

        print(f"\nOcena modeli dla zbioru {self.current_dataset}:")

        for model_name, model in self.models.items():
            print(f"\n=== {model_name.upper()} ===")
            print(f"Dokładność na zbiorze treningowym: {model.score(self.X_train, self.y_train):.4f}")
            print(f"Dokładność na zbiorze testowym: {model.score(self.X_test, self.y_test):.4f}")
            y_pred = model.predict(self.X_test)
            print("\nRaport klasyfikacji:")
            print(classification_report(self.y_test, y_pred))

    def save_models(self):
        """Zapisuje modele i skaler"""
        if not self.models:
            print("Najpierw wytreniuj modele!")
            return

        # Tworzenie katalogu dla modeli
        model_dir = f'models/{self.current_dataset}'
        os.makedirs(model_dir, exist_ok=True)

        # Zapisywanie modeli
        for model_name, model in self.models.items():
            model_path = f'{model_dir}/{model_name}_model.joblib'
            joblib.dump(model, model_path)
            print(f"Zapisano model {model_name} do {model_path}")

        # Zapisywanie skalera
        scaler_path = f'{model_dir}/scaler.joblib'
        joblib.dump(self.scaler, scaler_path)
        print(f"Zapisano skaler do {scaler_path}")

        print(f"\nWszystkie modele dla zbioru {self.current_dataset} zostały zapisane!")

    def get_features(self):
        """Zwraca listę cech dla aktualnego zbioru danych"""
        if self.current_dataset is None:
            raise ValueError("Najpierw wybierz zbiór danych!")
        return self.DATASETS_CONFIG[self.current_dataset]['features']


def main():
    predictor = MultiDatasetPredictor()

    while True:
        print("\n=== SYSTEM PREDYKCJI DLA WIELU ZBIORÓW DANYCH ===")
        print("1. Wybierz zbiór danych")
        print("2. Wytreniuj modele")
        print("3. Oceń modele")
        print("4. Zapisz modele")
        print("5. Wyjście")

        choice = input("\nWybierz opcję (1-5): ")

        if choice == '1':
            print("\nDostępne zbiory danych:")
            for idx, dataset in enumerate(predictor.DATASETS_CONFIG.keys(), 1):
                print(f"{idx}. {dataset}")

            try:
                dataset_choice = int(input("\nWybierz numer zbioru danych: ")) - 1
                dataset_name = list(predictor.DATASETS_CONFIG.keys())[dataset_choice]
                predictor.load_dataset(dataset_name)
            except Exception as e:
                print(f"\nBłąd: {str(e)}")

        elif choice == '2':
            try:
                predictor.train_models()
            except Exception as e:
                print(f"\nBłąd podczas trenowania modeli: {str(e)}")

        elif choice == '3':
            try:
                predictor.evaluate_models()
            except Exception as e:
                print(f"\nBłąd podczas oceny modeli: {str(e)}")

        elif choice == '4':
            try:
                predictor.save_models()
            except Exception as e:
                print(f"\nBłąd podczas zapisywania modeli: {str(e)}")

        elif choice == '5':
            print("\nDo widzenia!")
            break

        else:
            print("\nNieprawidłowy wybór. Spróbuj ponownie.")


if __name__ == "__main__":
    main()