
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd


# ==========================================
# CONFIGURATION
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = (
    PROJECT_ROOT / "data" / "processed" / "exercise_dataset.csv"
)

SPLIT_PATH = PROJECT_ROOT / "results" / "subject_split.csv"

OUTPUT_DIR = PROJECT_ROOT / "results" / "cnn_sequences"

SEQUENCE_LENGTH = 60
LANDMARK_COUNT = 33
VALUES_PER_LANDMARK = 4

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# LOAD DATASET AND ORIGINAL SPLIT
# ==========================================

def load_metadata():

    if not DATASET_PATH.exists():
        raise FileNotFoundError(DATASET_PATH)

    if not SPLIT_PATH.exists():
        raise FileNotFoundError(SPLIT_PATH)

    dataset = pd.read_csv(DATASET_PATH)
    split_manifest = pd.read_csv(SPLIT_PATH)

    required_dataset = [
        "subject_id",
        "video_path",
        "exercise",
        "camera_view",
        "label"
    ]

    required_split = [
        "subject_id",
        "video_path",
        "split"
    ]

    for column in required_dataset:
        if column not in dataset.columns:
            raise ValueError(
                f"Missing dataset column: {column}"
            )

    for column in required_split:
        if column not in split_manifest.columns:
            raise ValueError(
                f"Missing split column: {column}"
            )

    if dataset.duplicated(
        subset=["subject_id", "video_path"]
    ).any():
        raise ValueError("Duplicate videos in dataset.")

    if split_manifest.duplicated(
        subset=["subject_id", "video_path"]
    ).any():
        raise ValueError("Duplicate videos in split manifest.")

    merged = dataset.merge(
        split_manifest[
            ["subject_id", "video_path", "split"]
        ],
        on=["subject_id", "video_path"],
        how="inner",
        validate="one_to_one"
    )

    if len(merged) != len(dataset):
        raise ValueError(
            "Not every dataset video matched the original split."
        )

    if len(merged) != len(split_manifest):
        raise ValueError(
            "Dataset and split manifest contain different videos."
        )

    allowed_splits = {"train", "validation", "test"}

    if not set(merged["split"].unique()).issubset(allowed_splits):
        raise ValueError("Unexpected split name.")

    if not set(merged["label"].unique()).issubset({"good", "bad"}):
        raise ValueError("Unexpected class label.")

    print("=" * 60)
    print("POSE SEQUENCE EXTRACTION")
    print("=" * 60)

    print(f"Videos: {len(merged)}")
    print("\nOriginal split distribution:")
    print(merged["split"].value_counts())

    print("\nClass distribution:")
    print(merged["label"].value_counts())

    return merged


# ==========================================
# NORMALIZE POSE LANDMARKS
# ==========================================

def normalize_landmarks(landmarks):

    points = np.zeros(
        (LANDMARK_COUNT, VALUES_PER_LANDMARK),
        dtype=np.float32
    )

    for index, landmark in enumerate(landmarks):

        points[index] = [
            landmark.x,
            landmark.y,
            landmark.z,
            landmark.visibility
        ]

    # Hip center: landmarks 23 and 24
    hip_center = (
        points[23, :3] + points[24, :3]
    ) / 2.0

    # Shoulder center: landmarks 11 and 12
    shoulder_center = (
        points[11, :3] + points[12, :3]
    ) / 2.0

    # Estimate torso length
    torso_length = np.linalg.norm(
        shoulder_center[:2] - hip_center[:2]
    )

    if torso_length < 1e-6:
        torso_length = 1.0

    # Translate coordinates relative to hip center
    points[:, :3] -= hip_center

    # Normalize spatial coordinates
    points[:, :3] /= torso_length

    # Keep visibility unchanged
    return points.flatten()


# ==========================================
# EXTRACT ONE VIDEO
# ==========================================

def extract_video_sequence(video_path, pose):

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open video: {video_path}"
        )

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    if total_frames <= 0:
        cap.release()
        raise RuntimeError(
            f"Invalid video frame count: {video_path}"
        )

    # Sample frames uniformly across the entire video
    sample_indices = np.linspace(
        0,
        total_frames - 1,
        SEQUENCE_LENGTH
    ).round().astype(int)

    unique_indices, inverse_indices = np.unique(
        sample_indices,
        return_inverse=True
    )

    extracted_frames = {}

    frame_number = 0

    while frame_number <= unique_indices[-1]:

        success, frame = cap.read()

        if not success:
            break

        if frame_number in unique_indices:

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            result = pose.process(rgb)

            if result.pose_landmarks:

                normalized = normalize_landmarks(
                    result.pose_landmarks.landmark
                )

                extracted_frames[frame_number] = normalized

            else:
                extracted_frames[frame_number] = np.zeros(
                    LANDMARK_COUNT * VALUES_PER_LANDMARK,
                    dtype=np.float32
                )

        frame_number += 1

    cap.release()

    # Rebuild a fixed-length sequence.
    # Missing frames are represented by zeros.
    zero_frame = np.zeros(
        LANDMARK_COUNT * VALUES_PER_LANDMARK,
        dtype=np.float32
    )

    sequence = []

    for index in sample_indices:

        sequence.append(
            extracted_frames.get(
                index,
                zero_frame
            )
        )

    sequence = np.asarray(
        sequence,
        dtype=np.float32
    )

    valid_frames = sum(
        np.any(frame != 0)
        for frame in sequence
    )

    return sequence, valid_frames


# ==========================================
# MAIN EXTRACTION
# ==========================================

def main():

    metadata = load_metadata()

    all_sequences = []
    output_metadata = []

    failed_videos = []

    mp_pose = mp.solutions.pose

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:

        for index, row in metadata.iterrows():

            video_path = Path(row["video_path"])

            print(
                f"\n[{index + 1}/{len(metadata)}] "
                f"{row['subject_id']} | "
                f"{row['exercise']} | "
                f"{row['label']}"
            )

            try:

                sequence, valid_frames = extract_video_sequence(
                    video_path,
                    pose
                )

                if sequence.shape != (
                    SEQUENCE_LENGTH,
                    LANDMARK_COUNT * VALUES_PER_LANDMARK
                ):
                    raise ValueError(
                        f"Unexpected sequence shape: {sequence.shape}"
                    )

                if valid_frames < 10:
                    raise ValueError(
                        f"Too few detected pose frames: {valid_frames}"
                    )

                all_sequences.append(sequence)

                output_metadata.append({
                    "subject_id": row["subject_id"],
                    "video_path": row["video_path"],
                    "exercise": row["exercise"],
                    "camera_view": row["camera_view"],
                    "label": row["label"],
                    "split": row["split"],
                    "valid_pose_frames": valid_frames
                })

                print(
                    f"Pose frames: {valid_frames}/{SEQUENCE_LENGTH}"
                )

            except Exception as error:

                print(f"ERROR: {error}")

                failed_videos.append({
                    "video_path": row["video_path"],
                    "error": str(error)
                })

    # ======================================
    # SAVE RESULTS
    # ======================================

    if not all_sequences:
        raise RuntimeError(
            "No videos were successfully processed."
        )

    X = np.stack(
        all_sequences
    ).astype(np.float32)

    metadata_df = pd.DataFrame(
        output_metadata
    )

    np.save(
        OUTPUT_DIR / "pose_sequences.npy",
        X
    )

    metadata_df.to_csv(
        OUTPUT_DIR / "sequence_metadata.csv",
        index=False
    )

    pd.DataFrame(
        failed_videos
    ).to_csv(
        OUTPUT_DIR / "failed_videos.csv",
        index=False
    )

    print("\n" + "=" * 60)
    print("SEQUENCE EXTRACTION COMPLETED")
    print("=" * 60)

    print(f"Successful videos: {len(X)}")
    print(f"Failed videos: {len(failed_videos)}")
    print(f"Sequence shape: {X.shape}")

    print("\nSplit distribution:")
    print(metadata_df["split"].value_counts())

    print("\nClass distribution:")
    print(metadata_df["label"].value_counts())

    print("\nSaved files:")
    print(OUTPUT_DIR / "pose_sequences.npy")
    print(OUTPUT_DIR / "sequence_metadata.csv")
    print(OUTPUT_DIR / "failed_videos.csv")


if __name__ == "__main__":
    main()
