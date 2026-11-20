# ML/prepare_dataset.py

import pandas as pd

from pathlib import Path

from ml.features import (
    FEATURES,
    TARGET,
    METADATA_COLUMNS,
    EXERCISES
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_CSV = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "pose_features_raw.csv"
)

OUTPUT_CSV = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "exercise_dataset.csv"
)


def main():

    print("=" * 60)
    print("AI FITNESS TRAINER - DATASET PREPARATION")
    print("=" * 60)

    if not INPUT_CSV.exists():

        raise FileNotFoundError(
            f"Input CSV not found: {INPUT_CSV}\n"
            "Run extract_dataset.py first."
        )

    df = pd.read_csv(INPUT_CSV)

    print("\nOriginal dataset shape:", df.shape)

    required_columns = (
        FEATURES
        + [TARGET]
        + METADATA_COLUMNS
    )

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:

        raise ValueError(
            f"Missing required columns: {missing}"
        )

    # Normalize text columns
    for col in [
        "exercise",
        "label",
        "camera_view",
        "subject_id"
    ]:

        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .str.lower()
        )

    # Keep only the selected exercises
    df = df[
        df["exercise"].isin(EXERCISES)
    ].copy()

    # Keep valid labels
    df = df[
        df["label"].isin(["good", "bad"])
    ].copy()

    # Remove rows without participant IDs
    df = df[
        df["subject_id"].notna()
        & (df["subject_id"] != "")
    ].copy()

    # Remove duplicate video paths
    df = df.drop_duplicates(
        subset=["video_path"]
    )

    # Convert numerical columns
    from ml.features import NUMERIC_FEATURES

    for col in NUMERIC_FEATURES:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    # Remove videos with no usable pose measurements
    required_pose_features = [
        "elbow_angle_mean",
        "knee_angle_mean",
        "hip_angle_mean",
        "torso_angle_mean"
    ]

    df = df.dropna(
        subset=required_pose_features,
        how="all"
    )

    if df.empty:

        raise ValueError(
            "No usable rows remain after cleaning."
        )

    # Show dataset information
    print("\nClean dataset shape:", df.shape)

    print("\nExercises:")
    print(df["exercise"].value_counts())

    print("\nLabels:")
    print(df["label"].value_counts())

    print("\nParticipants:")
    print(df["subject_id"].nunique())

    print("\nExercise and label distribution:")
    print(
        pd.crosstab(
            df["exercise"],
            df["label"]
        )
    )

    print("\nCamera view distribution:")
    print(df["camera_view"].value_counts())

    # Save
    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_CSV,
        index=False
    )

    print("\nPrepared dataset saved to:")
    print(OUTPUT_CSV)

    print("\nDataset preparation completed.")


if __name__ == "__main__":
    main()