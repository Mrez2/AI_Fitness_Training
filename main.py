import cv2
import mediapipe as mp
from utils import calculate_angle

from exercises.biceps import BicepsCurl
from exercises.squat import Squat

MODE = "biceps"   # "biceps" or "squat"

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

if MODE == "biceps":
    exercise = BicepsCurl()
    video_path = "data/test_videos/biceps_bad.mp4"
else:
    exercise = Squat()
    video_path = "data/test_videos/squat_good.mp4"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("ERROR: Could not open video")
    exit()

with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as pose:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        counter = exercise.counter
        stage = exercise.stage if exercise.stage is not None else "-"
        feedback = "No body detected"
        form_status = "-"
        angle = 0
        back_angle_debug = 0

        try:
            if results.pose_landmarks is not None:
                landmarks = results.pose_landmarks.landmark

                if MODE == "biceps":
                    angle, counter, stage, feedback = exercise.process(landmarks, mp_pose)
                    form_status = "GOOD" if feedback == "Good" else "BAD"

                elif MODE == "squat":
                    angle, counter, stage, feedback, form_status = exercise.process(landmarks, mp_pose)

                    shoulder = [
                        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y
                    ]
                    hip = [
                        landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y
                    ]
                    knee = [
                        landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y
                    ]

                    back_angle_debug = calculate_angle(shoulder, hip, knee)

        except Exception as e:
            print("REAL ERROR:", e)
            feedback = "Processing error"
            form_status = "BAD"

        cv2.rectangle(image, (0, 0), (620, 190), (0, 0, 0), -1) #-1 to fill the rectangle with color if 2 يبقي frame بس

        cv2.putText(image, f'REPS: {counter}', (10, 35), #x,y coordinates of text
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2) # 1 size of text, 2 thickness of text, (0,255,0) color green for reps

        cv2.putText(image, f'STAGE: {stage}', (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.putText(image, f'FORM: {feedback}', (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        status_color = (0, 255, 0) if form_status == "GOOD" else (0, 0, 255)
        cv2.putText(image, f'STATUS: {form_status}', (300, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 2)

        cv2.putText(image, f'Knee Angle: {int(angle)}', (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if MODE == "squat":
            cv2.putText(image, f'Back Angle: {int(back_angle_debug)}', (300, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if results.pose_landmarks is not None:
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

        cv2.imshow("AI Fitness Trainer", image)

        if cv2.waitKey(25) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()