# ML/train.py

import json
import joblib
import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from ml.baseline import create_random_forest
from ml.features import (
    FEATURES,
    TARGET,
    METADATA_COLUMNS
)


# ==========================================
# CONFIGURATION
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "exercise_dataset.csv"
)

RESULTS_DIR = PROJECT_ROOT / "results"

MODEL_DIR = RESULTS_DIR / "models"

SPLIT_FILE = RESULTS_DIR / "subject_split.csv"

SEEDS = [42, 123, 2026]

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15

# Test ratio is the remaining 15%.


# ==========================================
# LOAD DATASET
# ==========================================

def load_dataset():

    if not DATASET_PATH.exists():

        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}\n"
            "Run the extraction and preparation scripts first."
        )

    df = pd.read_csv(DATASET_PATH)

    required = (
        FEATURES
        + [TARGET]
        + METADATA_COLUMNS
    )

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:

        raise ValueError(
            f"Missing dataset columns: {missing}"
        )

    df = df.dropna(
        subset=["subject_id", TARGET]
    ).copy()

    df["subject_id"] = (
        df["subject_id"].astype(str)
    )

    df[TARGET] = (
        df[TARGET]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    if not set(df[TARGET].unique()).issubset(
        {"good", "bad"}
    ):

        raise ValueError(
            "Target labels must be good or bad."
        )

    return df


# ==========================================
# SUBJECT-BASED SPLIT
# ==========================================

def split_by_subject(df, seed=42):

    subjects = np.array(
        sorted(df["subject_id"].unique())
    )

    if len(subjects) < 6:

        raise ValueError(
            "At least 6 subjects are needed "
            "for a meaningful train/validation/test split."
        )

    rng = np.random.default_rng(seed)

    shuffled = rng.permutation(subjects)

    total = len(shuffled)

    train_count = int(
        total * TRAIN_RATIO
    )

    validation_count = int(
        total * VALIDATION_RATIO
    )

    # Ensure all three sets are non-empty
    train_count = max(1, train_count)
    validation_count = max(1, validation_count)

    if train_count + validation_count >= total:

        train_count = total - 2
        validation_count = 1

    train_subjects = shuffled[:train_count]

    validation_subjects = shuffled[
        train_count:
        train_count + validation_count
    ]

    test_subjects = shuffled[
        train_count + validation_count:
    ]

    train_df = df[
        df["subject_id"].isin(train_subjects)
    ].copy()

    validation_df = df[
        df["subject_id"].isin(validation_subjects)
    ].copy()

    test_df = df[
        df["subject_id"].isin(test_subjects)
    ].copy()

    # Check for subject overlap
    train_ids = set(train_df["subject_id"])
    validation_ids = set(validation_df["subject_id"])
    test_ids = set(test_df["subject_id"])

    assert train_ids.isdisjoint(validation_ids)
    assert train_ids.isdisjoint(test_ids)
    assert validation_ids.isdisjoint(test_ids)

    print("\nSubject-based split:")
    print("Training subjects:", len(train_ids))
    print("Validation subjects:", len(validation_ids))
    print("Testing subjects:", len(test_ids))

    print("\nVideo counts:")
    print("Training videos:", len(train_df))
    print("Validation videos:", len(validation_df))
    print("Testing videos:", len(test_df))

    return (
        train_df,
        validation_df,
        test_df
    )


# ==========================================
# SAVE SPLIT MANIFEST
# ==========================================

def save_split_manifest(
    train_df,
    validation_df,
    test_df
):

    manifests = []

    for name, subset in [
        ("train", train_df),
        ("validation", validation_df),
        ("test", test_df)
    ]:

        part = subset[
            ["subject_id", "video_path"]
        ].copy()

        part["split"] = name

        manifests.append(part)

    manifest = pd.concat(
        manifests,
        ignore_index=True
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    manifest.to_csv(
        SPLIT_FILE,
        index=False
    )

    print("\nSplit manifest saved:", SPLIT_FILE)


# ==========================================
# CHECK CLASS DISTRIBUTION
# ==========================================

def check_classes(train_df, validation_df):

    print("\nTraining class distribution:")
    print(train_df[TARGET].value_counts())

    print("\nValidation class distribution:")
    print(validation_df[TARGET].value_counts())

    expected = {"good", "bad"}

    if set(train_df[TARGET].unique()) != expected:

        raise ValueError(
            "Training data must contain both good and bad labels."
        )

    if set(validation_df[TARGET].unique()) != expected:

        raise ValueError(
            "Validation data does not contain both classes. "
            "Review the subject split before training."
        )


# ==========================================
# EVALUATION
# ==========================================

def evaluate_model(model, X, y):

    predictions = model.predict(X)

    metrics = {
        "accuracy": accuracy_score(
            y, predictions
        ),

        "precision_macro": precision_score(
            y,
            predictions,
            average="macro",
            zero_division=0
        ),

        "recall_macro": recall_score(
            y,
            predictions,
            average="macro",
            zero_division=0
        ),

        "f1_macro": f1_score(
            y,
            predictions,
            average="macro",
            zero_division=0
        )
    }

    print("\nClassification report:")

    print(
        classification_report(
            y,
            predictions,
            labels=["good", "bad"],
            zero_division=0
        )
    )

    print("Confusion matrix:")

    print(
        confusion_matrix(
            y,
            predictions,
            labels=["good", "bad"]
        )
    )

    return metrics


# ==========================================
# MAIN
# ==========================================

def main():

    print("=" * 60)
    print("AI FITNESS TRAINER - RANDOM FOREST BASELINE")
    print("=" * 60)

    df = load_dataset()

    print("\nDataset shape:", df.shape)

    (
        train_df,
        validation_df,
        test_df
    ) = split_by_subject(df)

    save_split_manifest(
        train_df,
        validation_df,
        test_df
    )

    check_classes(
        train_df,
        validation_df
    )

    X_train = train_df[FEATURES]
    y_train = train_df[TARGET]

    X_validation = validation_df[FEATURES]
    y_validation = validation_df[TARGET]

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    results = []

    print("\nStarting three-seed experiments...")

    for seed in SEEDS:

        print("\n" + "=" * 50)
        print(f"RANDOM FOREST - SEED {seed}")
        print("=" * 50)

        model = create_random_forest(
            random_state=seed
        )

        model.fit(
            X_train,
            y_train
        )

        # Validation only: test is untouched
        metrics = evaluate_model(
            model,
            X_validation,
            y_validation
        )

        metrics["seed"] = seed

        results.append(metrics)

        model_path = (
            MODEL_DIR
            / f"random_forest_seed_{seed}.pkl"
        )

        joblib.dump(
            model,
            model_path
        )

        print("\nSaved model:", model_path)

    # Results table
    results_df = pd.DataFrame(results)

    results_path = (
        RESULTS_DIR
        / "random_forest_validation_results.csv"
    )

    results_df.to_csv(
        results_path,
        index=False
    )

    print("\nValidation results:")
    print(results_df.to_string(index=False))

    metric_columns = [
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro"
    ]

    print("\nMean validation metrics:")
    print(
        results_df[metric_columns].mean()
    )

    print("\nStandard deviation:")
    print(
        results_df[metric_columns].std()
    )

    # Select the best model using validation F1 only
    best_index = results_df[
        "f1_macro"
    ].idxmax()

    best_seed = int(
        results_df.loc[best_index, "seed"]
    )

    best_model_path = (
        MODEL_DIR
        / f"random_forest_seed_{best_seed}.pkl"
    )

    best_model = joblib.load(
        best_model_path
    )

    joblib.dump(
        best_model,
        MODEL_DIR / "random_forest_model.pkl"
    )

    print("\nBest validation seed:", best_seed)

    print("\nBest model saved as:")
    print(MODEL_DIR / "random_forest_model.pkl")

    # Save experiment information
    experiment_info = {
        "model": "Random Forest",
        "seeds": SEEDS,
        "best_seed": best_seed,
        "features": FEATURES,
        "target": TARGET,
        "split_method": "subject-independent",
        "test_evaluated": False
    }

    with open(
        RESULTS_DIR / "experiment_info.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            experiment_info,
            file,
            indent=4
        )

    print("\nTraining completed.")
    print("The test subjects have NOT been evaluated.")


if __name__ == "__main__":
    main()