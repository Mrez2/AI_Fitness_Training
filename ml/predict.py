# ML/predict.py

import joblib
import pandas as pd

from pathlib import Path

from ml.features import FEATURES


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "results"
    / "models"
    / "random_forest_model.pkl"
)


# ==========================================
# LOAD MODEL
# ==========================================

def load_model():

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}\n"
            "Run ML/train.py first."
        )

    return joblib.load(MODEL_PATH)


# ==========================================
# PREDICT
# ==========================================

def predict_exercise(features_dict):

    model = load_model()

    # Check required features
    missing = [
        feature
        for feature in FEATURES
        if feature not in features_dict
    ]

    if missing:

        raise ValueError(
            f"Missing prediction features: {missing}"
        )

    # Create input DataFrame
    input_data = pd.DataFrame([
        {
            feature: features_dict[feature]
            for feature in FEATURES
        }
    ])

    # Predict
    prediction = model.predict(
        input_data
    )[0]

    # Probabilities
    probabilities = model.predict_proba(
        input_data
    )[0]

    class_names = model.named_steps[
        "classifier"
    ].classes_

    confidence = float(
        probabilities[
            list(class_names).index(prediction)
        ] * 100
    )

    return {
        "prediction": str(prediction),
        "confidence": round(confidence, 2)
    }


# ==========================================
# EXAMPLE USAGE
# ==========================================

if __name__ == "__main__":

    example_features = {

        "elbow_angle_mean": 110,
        "elbow_angle_min": 45,
        "elbow_angle_max": 165,
        "elbow_angle_range": 120,

        "knee_angle_mean": 170,
        "knee_angle_min": 150,
        "knee_angle_max": 180,
        "knee_angle_range": 30,

        "hip_angle_mean": 160,
        "hip_angle_min": 130,
        "hip_angle_max": 175,
        "hip_angle_range": 45,

        "torso_angle_mean": 15,
        "torso_angle_max": 25,

        "clip_duration": 8.5,
        "movement_speed": 0.015,

        "exercise": "bicep_curl",
        "camera_view": "side"
    }

    result = predict_exercise(
        example_features
    )

    print("\nPrediction result")
    print("=" * 40)

    print(
        "Predicted form:",
        result["prediction"]
    )

    print(
        "Confidence:",
        result["confidence"],
        "%"
    )