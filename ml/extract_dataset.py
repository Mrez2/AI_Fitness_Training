
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

from pathlib import Path

from ml.features import EXERCISES


# ==========================================
# CONFIGURATION
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_ROOT = PROJECT_ROOT / "data" / "fitness_dataset"

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

OUTPUT_CSV = OUTPUT_DIR / "pose_features_raw.csv"

FRAME_STRIDE = 3
MAX_FRAME_WIDTH = 640

mp_pose = mp.solutions.pose


# ==========================================
# FIND DATASET FOLDER
# ==========================================

def find_dataset_folder():

    matches = list(DATA_ROOT.rglob("Dataset Subject-wise"))

    if len(matches) == 0:
        raise FileNotFoundError(
            "\nDataset Subject-wise folder was not found.\n"
            f"Expected it somewhere inside: {DATA_ROOT}"
        )

    if len(matches) > 1:
        raise RuntimeError(
            "More than one Dataset Subject-wise folder found:\n"
            + "\n".join(str(path) for path in matches)
        )

    return matches[0]


# ==========================================
# CALCULATE JOINT ANGLE
# ==========================================

def calculate_angle(a, b, c):

    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    c = np.array(c, dtype=float)

    ba = a - b
    bc = c - b

    denominator = (
        np.linalg.norm(ba) * np.linalg.norm(bc)
    )

    if denominator < 1e-8:
        return None

    cosine = np.dot(ba, bc) / denominator

    cosine = np.clip(cosine, -1.0, 1.0)

    angle = np.degrees(np.arccos(cosine))

    return float(angle)


# ==========================================
# GET LANDMARK COORDINATES
# ==========================================

def get_point(landmarks, landmark_id):

    landmark = landmarks[landmark_id]

    if landmark.visibility < 0.5:
        return None

    return (
        landmark.x,
        landmark.y
    )


# ==========================================
# EXTRACT FRAME FEATURES
# ==========================================

def extract_frame_features(landmarks):

    P = mp_pose.PoseLandmark

    # Left side
    left_shoulder = get_point(
        landmarks, P.LEFT_SHOULDER.value
    )

    left_elbow = get_point(
        landmarks, P.LEFT_ELBOW.value
    )

    left_wrist = get_point(
        landmarks, P.LEFT_WRIST.value
    )

    left_hip = get_point(
        landmarks, P.LEFT_HIP.value
    )

    left_knee = get_point(
        landmarks, P.LEFT_KNEE.value
    )

    left_ankle = get_point(
        landmarks, P.LEFT_ANKLE.value
    )

    # Right side
    right_shoulder = get_point(
        landmarks, P.RIGHT_SHOULDER.value
    )

    right_elbow = get_point(
        landmarks, P.RIGHT_ELBOW.value
    )

    right_wrist = get_point(
        landmarks, P.RIGHT_WRIST.value
    )

    right_hip = get_point(
        landmarks, P.RIGHT_HIP.value
    )

    right_knee = get_point(
        landmarks, P.RIGHT_KNEE.value
    )

    right_ankle = get_point(
        landmarks, P.RIGHT_ANKLE.value
    )

    elbow_angles = []
    knee_angles = []
    hip_angles = []
    torso_angles = []

    # Left elbow
    if all([
        left_shoulder,
        left_elbow,
        left_wrist
    ]):

        angle = calculate_angle(
            left_shoulder,
            left_elbow,
            left_wrist
        )

        if angle is not None:
            elbow_angles.append(angle)

    # Right elbow
    if all([
        right_shoulder,
        right_elbow,
        right_wrist
    ]):

        angle = calculate_angle(
            right_shoulder,
            right_elbow,
            right_wrist
        )

        if angle is not None:
            elbow_angles.append(angle)

    # Left knee
    if all([
        left_hip,
        left_knee,
        left_ankle
    ]):

        angle = calculate_angle(
            left_hip,
            left_knee,
            left_ankle
        )

        if angle is not None:
            knee_angles.append(angle)

    # Right knee
    if all([
        right_hip,
        right_knee,
        right_ankle
    ]):

        angle = calculate_angle(
            right_hip,
            right_knee,
            right_ankle
        )

        if angle is not None:
            knee_angles.append(angle)

    # Left hip
    if all([
        left_shoulder,
        left_hip,
        left_knee
    ]):

        angle = calculate_angle(
            left_shoulder,
            left_hip,
            left_knee
        )

        if angle is not None:
            hip_angles.append(angle)

    # Right hip
    if all([
        right_shoulder,
        right_hip,
        right_knee
    ]):

        angle = calculate_angle(
            right_shoulder,
            right_hip,
            right_knee
        )

        if angle is not None:
            hip_angles.append(angle)

    # Torso inclination relative to vertical
    if all([
        left_shoulder,
        right_shoulder,
        left_hip,
        right_hip
    ]):

        shoulder_center = np.mean(
            [left_shoulder, right_shoulder],
            axis=0
        )

        hip_center = np.mean(
            [left_hip, right_hip],
            axis=0
        )

        dx = shoulder_center[0] - hip_center[0]
        dy = shoulder_center[1] - hip_center[1]

        torso_angle = np.degrees(
            np.arctan2(abs(dx), abs(dy))
        )

        torso_angles.append(float(torso_angle))

    # Store all visible normalized landmarks
    pose_vector = []

    for landmark in landmarks:

        if landmark.visibility >= 0.5:

            pose_vector.extend([
                landmark.x,
                landmark.y
            ])

        else:

            pose_vector.extend([
                np.nan,
                np.nan
            ])

    return {
        "elbow": elbow_angles,
        "knee": knee_angles,
        "hip": hip_angles,
        "torso": torso_angles,
        "pose_vector": np.array(
            pose_vector,
            dtype=float
        )
    }


# ==========================================
# STATISTICS
# ==========================================

def calculate_statistics(values, prefix):

    if not values:

        return {
            f"{prefix}_mean": np.nan,
            f"{prefix}_min": np.nan,
            f"{prefix}_max": np.nan,
            f"{prefix}_range": np.nan
        }

    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_max": float(np.max(values)),
        f"{prefix}_range": float(
            np.max(values) - np.min(values)
        )
    }


# ==========================================
# PROCESS ONE VIDEO
# ==========================================

def process_video(video_path):

    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():

        print(f"[SKIPPED] Cannot open: {video_path}")
        return None

    fps = capture.get(cv2.CAP_PROP_FPS)

    if not fps or fps <= 0:
        fps = 30.0

    frame_count = capture.get(
        cv2.CAP_PROP_FRAME_COUNT
    )

    clip_duration = frame_count / fps

    elbow_values = []
    knee_values = []
    hip_values = []
    torso_values = []

    previous_pose = None
    movement_values = []

    processed_frames = 0
    frame_index = 0

    try:

        with mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as pose:

            while True:

                success, frame = capture.read()

                if not success:
                    break

                if frame_index % FRAME_STRIDE != 0:

                    frame_index += 1
                    continue

                height, width = frame.shape[:2]

                if width > MAX_FRAME_WIDTH:

                    scale = MAX_FRAME_WIDTH / width

                    frame = cv2.resize(
                        frame,
                        (
                            MAX_FRAME_WIDTH,
                            int(height * scale)
                        )
                    )

                rgb = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                result = pose.process(rgb)

                if result.pose_landmarks:

                    landmarks = (
                        result.pose_landmarks.landmark
                    )

                    frame_features = (
                        extract_frame_features(landmarks)
                    )

                    elbow_values.extend(
                        frame_features["elbow"]
                    )

                    knee_values.extend(
                        frame_features["knee"]
                    )

                    hip_values.extend(
                        frame_features["hip"]
                    )

                    torso_values.extend(
                        frame_features["torso"]
                    )

                    current_pose = (
                        frame_features["pose_vector"]
                    )

                    # Average visible landmark displacement
                    if previous_pose is not None:

                        valid = (
                            np.isfinite(current_pose)
                            & np.isfinite(previous_pose)
                        )

                        if np.any(valid):

                            displacement = np.mean(
                                np.abs(
                                    current_pose[valid]
                                    - previous_pose[valid]
                                )
                            )

                            movement_values.append(
                                float(displacement)
                            )

                    previous_pose = current_pose

                    processed_frames += 1

                frame_index += 1

    except Exception as error:

        print(
            f"[ERROR] {video_path.name}: {error}"
        )

        capture.release()
        return None

    capture.release()

    if processed_frames == 0:

        print(
            f"[SKIPPED] No pose detected: {video_path}"
        )

        return None

    row = {}

    row.update(
        calculate_statistics(
            elbow_values,
            "elbow_angle"
        )
    )

    row.update(
        calculate_statistics(
            knee_values,
            "knee_angle"
        )
    )

    row.update(
        calculate_statistics(
            hip_values,
            "hip_angle"
        )
    )

    if torso_values:

        row["torso_angle_mean"] = float(
            np.mean(torso_values)
        )

        row["torso_angle_max"] = float(
            np.max(torso_values)
        )

    else:

        row["torso_angle_mean"] = np.nan
        row["torso_angle_max"] = np.nan

    row["clip_duration"] = float(clip_duration)

    row["movement_speed"] = (
        float(np.mean(movement_values))
        if movement_values else np.nan
    )

    row["processed_frames"] = processed_frames

    return row


# ==========================================
# READ VIDEO METADATA
# ==========================================

def read_video_metadata(video_path, subject_root):

    relative_path = video_path.relative_to(
        subject_root
    )

    parts = relative_path.parts

    # Expected:
    # subject_001 / bicep_curl / good / front / video.mp4

    if len(parts) < 5:
        return None

    subject_id = parts[0].lower()
    exercise = parts[1].lower()
    label = parts[2].lower()
    camera_view = parts[3].lower()

    # Validate folder names
    if exercise not in EXERCISES:
        return None

    if label not in ["good", "bad"]:
        return None

    if camera_view not in [
        "front",
        "side",
        "diagonal"
    ]:
        return None

    return {
        "subject_id": subject_id,
        "exercise": exercise,
        "label": label,
        "camera_view": camera_view
    }


# ==========================================
# MAIN
# ==========================================

def main():

    dataset_root = find_dataset_folder()

    print("=" * 60)
    print("AI FITNESS TRAINER - DATASET EXTRACTION")
    print("=" * 60)

    print("\nDataset folder:")
    print(dataset_root)

    video_files = list(
        dataset_root.rglob("*.mp4")
    )

    print("\nTotal MP4 files found:", len(video_files))

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    rows = []

    for index, video_path in enumerate(
        video_files,
        start=1
    ):

        metadata = read_video_metadata(
            video_path,
            dataset_root
        )

        if metadata is None:
            continue

        print(
            f"\n[{index}/{len(video_files)}] "
            f"{metadata['subject_id']} | "
            f"{metadata['exercise']} | "
            f"{metadata['label']} | "
            f"{metadata['camera_view']}"
        )

        features = process_video(video_path)

        if features is None:
            continue

        row = {
            **metadata,
            **features,
            "video_path": str(video_path)
        }

        rows.append(row)

    if not rows:

        raise RuntimeError(
            "No usable videos were processed. "
            "Check the actual folder structure."
        )

    df = pd.DataFrame(rows)

    df.to_csv(
        OUTPUT_CSV,
        index=False
    )

    print("\n" + "=" * 60)
    print("EXTRACTION COMPLETED")
    print("=" * 60)

    print("Videos processed:", len(df))
    print("Output:", OUTPUT_CSV)

    print("\nExercise distribution:")
    print(df["exercise"].value_counts())

    print("\nLabel distribution:")
    print(df["label"].value_counts())

    print("\nCamera distribution:")
    print(df["camera_view"].value_counts())


if __name__ == "__main__":
    main()
