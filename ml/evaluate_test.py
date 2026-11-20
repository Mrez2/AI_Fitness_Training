import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from ml.features import FEATURES, TARGET


# ==========================================
# PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "exercise_dataset.csv"
)

SPLIT_PATH = (
    PROJECT_ROOT
    / "results"
    / "subject_split.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "results"
    / "models"
    / "random_forest_model.pkl"
)

RESULTS_DIR = PROJECT_ROOT / "results"

TEST_RESULTS_DIR = RESULTS_DIR / "test_evaluation"

LABELS = ["good", "bad"]


# ==========================================
# LOAD FILES
# ==========================================

def load_files():

    for path in [DATASET_PATH, SPLIT_PATH, MODEL_PATH]:

        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found: {path}"
            )

    df = pd.read_csv(DATASET_PATH)

    manifest = pd.read_csv(SPLIT_PATH)

    model = joblib.load(MODEL_PATH)

    print("Dataset loaded:", df.shape)
    print("Split manifest loaded:", manifest.shape)
    print("Best Random Forest model loaded.")

    return df, manifest, model


# ==========================================
# GET TEST SUBJECTS ONLY
# ==========================================

def prepare_test_data(df, manifest):

    required_manifest_columns = [
        "subject_id",
        "video_path",
        "split"
    ]

    for column in required_manifest_columns:

        if column not in manifest.columns:
            raise ValueError(
                f"Missing manifest column: {column}"
            )

    if not {"subject_id", "video_path", TARGET}.issubset(df.columns):
        raise ValueError(
            "Dataset is missing subject_id, video_path, or label."
        )

    # Select only the original test partition.
    test_manifest = manifest[
        manifest["split"] == "test"
    ].copy()

    if test_manifest.empty:
        raise ValueError("No test videos found in split manifest.")

    # Check that the manifest contains no duplicate video records.
    if test_manifest.duplicated(
        subset=["subject_id", "video_path"]
    ).any():
        raise ValueError(
            "Duplicate videos found in the test manifest."
        )

    # Match dataset records to the saved test manifest.
    test_df = df.merge(
        test_manifest[
            ["subject_id", "video_path"]
        ],
        on=["subject_id", "video_path"],
        how="inner",
        validate="one_to_one"
    )

    if len(test_df) != len(test_manifest):
        raise ValueError(
            "Some test videos in the manifest could not be matched "
            "to the prepared dataset."
        )

    test_df = test_df.copy()

    test_df[TARGET] = (
        test_df[TARGET]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    if not set(test_df[TARGET].unique()).issubset(set(LABELS)):
        raise ValueError("Unexpected labels in test dataset.")

    if set(test_df[TARGET].unique()) != set(LABELS):
        raise ValueError(
            "The test set must contain both good and bad labels."
        )

    print("\nTest dataset information:")
    print("Test videos:", len(test_df))
    print("Test subjects:", test_df["subject_id"].nunique())

    print("\nTest class distribution:")
    print(test_df[TARGET].value_counts())

    print("\nTest exercise distribution:")
    print(test_df["exercise"].value_counts())

    return test_df


# ==========================================
# EVALUATE MODEL
# ==========================================

def evaluate(model, test_df):

    missing_features = [
        feature
        for feature in FEATURES
        if feature not in test_df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing model features: {missing_features}"
        )

    X_test = test_df[FEATURES]
    y_test = test_df[TARGET]

    predictions = model.predict(X_test)

    metrics = {
        "accuracy": float(
            accuracy_score(y_test, predictions)
        ),

        "precision_macro": float(
            precision_score(
                y_test,
                predictions,
                average="macro",
                zero_division=0
            )
        ),

        "recall_macro": float(
            recall_score(
                y_test,
                predictions,
                average="macro",
                zero_division=0
            )
        ),

        "f1_macro": float(
            f1_score(
                y_test,
                predictions,
                average="macro",
                zero_division=0
            )
        )
    }

    report = classification_report(
        y_test,
        predictions,
        labels=LABELS,
        target_names=["Good Form", "Bad Form"],
        output_dict=True,
        zero_division=0
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=LABELS
    )

    print("\n" + "=" * 55)
    print("FINAL RANDOM FOREST TEST RESULTS")
    print("=" * 55)

    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision_macro']:.4f}")
    print(f"Recall   : {metrics['recall_macro']:.4f}")
    print(f"F1-score : {metrics['f1_macro']:.4f}")

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            labels=LABELS,
            target_names=["Good Form", "Bad Form"],
            zero_division=0
        )
    )

    print("Confusion Matrix:")
    print(matrix)

    return predictions, metrics, report, matrix


# ==========================================
# SAVE RESULTS
# ==========================================

def save_results(
    test_df,
    predictions,
    metrics,
    report,
    matrix,
    model
):

    TEST_RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save individual predictions.
    prediction_df = test_df[
        [
            "subject_id",
            "video_path",
            "exercise",
            "camera_view",
            TARGET
        ]
    ].copy()

    prediction_df = prediction_df.rename(
        columns={TARGET: "actual_label"}
    )

    prediction_df["predicted_label"] = predictions

    probabilities = model.predict_proba(
        test_df[FEATURES]
    )

    classes = list(model.classes_)

    predicted_indices = [
        classes.index(prediction)
        for prediction in predictions
    ]

    prediction_df["confidence"] = [
        round(
            float(probabilities[i][predicted_indices[i]]) * 100,
            2
        )
        for i in range(len(predictions))
    ]

    prediction_df["correct_prediction"] = (
        prediction_df["actual_label"]
        == prediction_df["predicted_label"]
    )

    prediction_path = (
        TEST_RESULTS_DIR / "test_predictions.csv"
    )

    prediction_df.to_csv(
        prediction_path,
        index=False
    )

    # Save metrics and classification report.
    results = {
        "model": "Random Forest",
        "evaluation": "Final held-out test set",
        "test_videos": len(test_df),
        "test_subjects": int(
            test_df["subject_id"].nunique()
        ),
        "metrics": metrics,
        "classification_report": report,
        "test_evaluated": True
    }

    metrics_path = (
        TEST_RESULTS_DIR / "test_metrics.json"
    )

    with open(
        metrics_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results,
            file,
            indent=4
        )

    # Save confusion matrix as CSV.
    matrix_df = pd.DataFrame(
        matrix,
        index=["Actual Good", "Actual Bad"],
        columns=["Predicted Good", "Predicted Bad"]
    )

    matrix_df.to_csv(
        TEST_RESULTS_DIR / "confusion_matrix.csv"
    )

    # Save confusion matrix as an image.
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["Good", "Bad"]
    )

    display.plot(
        cmap="Blues",
        values_format="d"
    )

    plt.title("Random Forest - Final Test Confusion Matrix")
    plt.tight_layout()

    plt.savefig(
        TEST_RESULTS_DIR / "confusion_matrix.png",
        dpi=300
    )

    plt.close()

    print("\nResults saved in:")
    print(TEST_RESULTS_DIR)

    print("\nGenerated files:")
    print("- test_predictions.csv")
    print("- test_metrics.json")
    print("- confusion_matrix.csv")
    print("- confusion_matrix.png")


# ==========================================
# MAIN
# ==========================================

def main():

    print("=" * 60)
    print("AI FITNESS TRAINER - FINAL TEST EVALUATION")
    print("=" * 60)

    df, manifest, model = load_files()

    test_df = prepare_test_data(
        df,
        manifest
    )

    predictions, metrics, report, matrix = evaluate(
        model,
        test_df
    )

    save_results(
        test_df,
        predictions,
        metrics,
        report,
        matrix,
        model
    )

    print("\n" + "=" * 60)
    print("FINAL TEST EVALUATION COMPLETED")
    print("=" * 60)

    print("\nThe held-out test set has now been evaluated.")
    print("Do not use these results to tune the model.")


if __name__ == "__main__":
    main()
