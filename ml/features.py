# ML/features.py

# Features used by the Random Forest model.
# These are calculated from MediaPipe Pose landmarks.

NUMERIC_FEATURES = [
    "elbow_angle_mean",
    "elbow_angle_min",
    "elbow_angle_max",
    "elbow_angle_range",

    "knee_angle_mean",
    "knee_angle_min",
    "knee_angle_max",
    "knee_angle_range",

    "hip_angle_mean",
    "hip_angle_min",
    "hip_angle_max",
    "hip_angle_range",

    "torso_angle_mean",
    "torso_angle_max",

    "clip_duration",
    "movement_speed"
]

CATEGORICAL_FEATURES = [
    "exercise",
    "camera_view"
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

TARGET = "label"

# Metadata columns. These must never be used as model features.
METADATA_COLUMNS = [
    "subject_id",
    "video_path"
]

CLASSES = [
    "good",
    "bad"
]

EXERCISES = [
    "bicep_curl",
    "squat"
]

CAMERA_VIEWS = [
    "front",
    "side",
    "diagonal"
]