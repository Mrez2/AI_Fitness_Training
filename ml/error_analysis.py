
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ==========================================
# PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PREDICTIONS_PATH = (
    PROJECT_ROOT / "results" / "test_evaluation" / "test_predictions.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "results" / "error_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# LOAD DATA
# ==========================================

def load_predictions():

    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Predictions file not found: {PREDICTIONS_PATH}"
        )

    df = pd.read_csv(PREDICTIONS_PATH)

    required_columns = [
        "subject_id",
        "exercise",
        "camera_view",
        "actual_label",
        "predicted_label",
        "confidence",
        "correct_prediction"
    ]

    missing = [c for c in required_columns if c not in df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df["confidence"] = pd.to_numeric(
        df["confidence"], errors="coerce"
    )

    df["correct_prediction"] = (
        df["correct_prediction"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False})
    )

    if df["correct_prediction"].isna().any():
        raise ValueError("Invalid values in correct_prediction column.")

    print("=" * 55)
    print("RANDOM FOREST ERROR ANALYSIS")
    print("=" * 55)

    print(f"Total predictions: {len(df)}")
    print(f"Correct: {df['correct_prediction'].sum()}")
    print(f"Incorrect: {(~df['correct_prediction']).sum()}")

    return df


# ==========================================
# GROUP PERFORMANCE
# ==========================================

def analyze_group(df, column, filename):

    summary = (
        df.groupby(column, dropna=False)
        .agg(
            total_videos=("correct_prediction", "count"),
            correct=("correct_prediction", "sum"),
            mean_confidence=("confidence", "mean")
        )
        .reset_index()
    )

    summary["incorrect"] = (
        summary["total_videos"] - summary["correct"]
    )

    summary["accuracy"] = (
        summary["correct"] / summary["total_videos"] * 100
    )

    summary["accuracy"] = summary["accuracy"].round(2)
    summary["mean_confidence"] = (
        summary["mean_confidence"].round(2)
    )

    summary.to_csv(OUTPUT_DIR / filename, index=False)

    print(f"\nPerformance by {column}:")
    print(summary.to_string(index=False))

    return summary


# ==========================================
# CLASS-LEVEL ANALYSIS
# ==========================================

def analyze_classes(df):

    summary = (
        df.groupby(["exercise", "actual_label"])
        .agg(
            total=("correct_prediction", "count"),
            correct=("correct_prediction", "sum")
        )
        .reset_index()
    )

    summary["incorrect"] = summary["total"] - summary["correct"]

    summary["recall_percent"] = (
        summary["correct"] / summary["total"] * 100
    ).round(2)

    summary.to_csv(
        OUTPUT_DIR / "class_performance.csv",
        index=False
    )

    print("\nPerformance by exercise and actual class:")
    print(summary.to_string(index=False))

    return summary


# ==========================================
# CONFUSION COUNTS BY EXERCISE
# ==========================================

def analyze_confusion(df):

    summary = (
        df.groupby(["exercise", "actual_label", "predicted_label"])
        .size()
        .reset_index(name="count")
    )

    summary.to_csv(
        OUTPUT_DIR / "confusion_by_exercise.csv",
        index=False
    )

    print("\nConfusion counts by exercise:")
    print(summary.to_string(index=False))


# ==========================================
# ERROR ANALYSIS
# ==========================================

def analyze_errors(df):

    errors = df[
        df["correct_prediction"] == False
    ].copy()

    errors = errors.sort_values(
        by="confidence",
        ascending=False
    )

    errors.to_csv(
        OUTPUT_DIR / "incorrect_predictions.csv",
        index=False
    )

    print("\nIncorrect predictions:")
    print(f"Total errors: {len(errors)}")

    print("\nErrors by exercise:")
    print(errors["exercise"].value_counts())

    print("\nErrors by camera:")
    print(errors["camera_view"].value_counts())

    # High-confidence incorrect predictions
    high_confidence_errors = errors[
        errors["confidence"] >= 70
    ]

    high_confidence_errors.to_csv(
        OUTPUT_DIR / "high_confidence_errors.csv",
        index=False
    )

    print(
        "\nHigh-confidence incorrect predictions "
        "(confidence >= 70%):"
    )

    print(len(high_confidence_errors))

    return errors


# ==========================================
# CHART GENERATION
# ==========================================

def create_accuracy_chart(summary, category, filename):

    plt.figure(figsize=(9, 5))

    bars = plt.bar(
        summary[category].astype(str),
        summary["accuracy"]
    )

    plt.title(f"Random Forest Accuracy by {category}")
    plt.xlabel(category.replace("_", " ").title())
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 100)

    plt.grid(axis="y", alpha=0.3)

    for bar, value in zip(bars, summary["accuracy"]):

        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{value:.1f}%",
            ha="center"
        )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / filename,
        dpi=300
    )

    plt.close()


def create_class_chart(summary):

    plt.figure(figsize=(9, 5))

    labels = (
        summary["exercise"].astype(str)
        + " - "
        + summary["actual_label"].astype(str)
    )

    bars = plt.bar(
        labels,
        summary["recall_percent"]
    )

    plt.title("Recall by Exercise and Actual Class")
    plt.xlabel("Exercise and Actual Class")
    plt.ylabel("Recall (%)")
    plt.ylim(0, 100)
    plt.xticks(rotation=15)

    for bar, value in zip(bars, summary["recall_percent"]):

        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{value:.1f}%",
            ha="center"
        )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "class_recall.png",
        dpi=300
    )

    plt.close()


# ==========================================
# MAIN
# ==========================================

def main():

    df = load_predictions()

    exercise_summary = analyze_group(
        df,
        "exercise",
        "exercise_performance.csv"
    )

    camera_summary = analyze_group(
        df,
        "camera_view",
        "camera_performance.csv"
    )

    subject_summary = analyze_group(
        df,
        "subject_id",
        "subject_performance.csv"
    )

    class_summary = analyze_classes(df)

    analyze_confusion(df)

    analyze_errors(df)

    create_accuracy_chart(
        exercise_summary,
        "exercise",
        "accuracy_by_exercise.png"
    )

    create_accuracy_chart(
        camera_summary,
        "camera_view",
        "accuracy_by_camera.png"
    )

    create_accuracy_chart(
        subject_summary,
        "subject_id",
        "accuracy_by_subject.png"
    )

    create_class_chart(class_summary)

    print("\n" + "=" * 55)
    print("ERROR ANALYSIS COMPLETED")
    print("=" * 55)

    print(f"Results saved in:\n{OUTPUT_DIR}")


if __name__ == "__main__":
    main()
