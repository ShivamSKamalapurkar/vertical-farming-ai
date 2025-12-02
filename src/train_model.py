# src/train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(ROOT, "data", "plant_status.csv")
MODEL_DIR = os.path.join(ROOT, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

def main():
    # 1. Load data
    df = pd.read_csv(DATA_PATH)

    # 2. Features & target
    X = df[["temperature", "humidity", "light_hours", "ec", "pH"]]
    y = df["plant_status"]

    # 3. Encode labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # 4. Split
    X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )


    # 5. Model
    model = RandomForestClassifier(n_estimators=100, random_state=42)

    # 6. Train
    model.fit(X_train, y_train)

    # 7. Evaluate
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))

    # 8. Save artifacts
    joblib.dump(model, os.path.join(MODEL_DIR, "plant_health_model.joblib"))
    joblib.dump(le, os.path.join(MODEL_DIR, "label_encoder.joblib"))
    print("\nSaved: models/plant_health_model.joblib and label_encoder.joblib")

if __name__ == "__main__":
    main()
