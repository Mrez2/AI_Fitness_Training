import cv2
import mediapipe as mp
import tempfile
import streamlit as st

from exercises.biceps import BicepsCurl
from exercises.squat import Squat
from utils import calculate_angle

st.set_page_config(
    page_title="AI Fitness Trainer",
    page_icon="🏋️",
    layout="wide"
)


total_frames = 0
correct_frames = 0
accuracy = 0


st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: 'Arial', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0f0f0f, #1a1a1a);
    color: white;
}

.main-title {
    font-size: 42px;
    font-weight: 900;
    color: #ff4b2b;
    text-align: center;
    margin-bottom: 8px;
    letter-spacing: 1px;
}

.sub-title {
    text-align: center;
    color: #d9d9d9;
    font-size: 18px;
    margin-bottom: 30px;
}

.metric-card {
    background: #1f1f1f;
    border: 1px solid #333;
    border-radius: 18px;
    padding: 18px;
    text-align: center;
    box-shadow: 0 4px 15px rgba(0,0,0,0.4);
}

.metric-title {
    color: #aaaaaa;
    font-size: 16px;
    margin-bottom: 8px;
}

.metric-value {
    color: #ff4b2b;
    font-size: 30px;
    font-weight: bold;
}

.status-good {
    color: #00ff88;
    font-weight: bold;
    font-size: 22px;
}

.status-bad {
    color: #ff3b3b;
    font-weight: bold;
    font-size: 22px;
}

.block-container {
    padding-top: 2rem;
}

section[data-testid="stSidebar"] {
    background: #111111;
    border-right: 1px solid #2a2a2a;
}

.sidebar-title {
    color: #ff4b2b;
    font-size: 24px;
    font-weight: bold;
    margin-bottom: 15px;
}

.video-box {
    background: #111;
    padding: 10px;
    border-radius: 18px;
    border: 1px solid #333;
}

.feedback-box {
    background: #1b1b1b;
    padding: 20px;
    border-radius: 18px;
    border-left: 6px solid #ff4b2b;
    margin-top: 10px;
}

hr {
    border: none;
    border-top: 1px solid #333;
    margin: 20px 0;
}
</style>
""", unsafe_allow_html=True)


st.markdown('<div class="main-title">AI FITNESS TRAINER</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Real-time Exercise Form Analysis and Repetition Counting</div>',
    unsafe_allow_html=True
)


st.sidebar.markdown('<div class="sidebar-title">GYM CONTROL PANEL</div>', unsafe_allow_html=True)

mode = st.sidebar.selectbox("Choose Exercise", ["biceps", "squat"])
input_type = st.sidebar.radio("Input Source", ["Upload Video", "Webcam"])

uploaded_file = None
if input_type == "Upload Video":
    uploaded_file = st.sidebar.file_uploader("Upload Workout Video", type=["mp4", "mov", "avi"])

run_button = st.sidebar.button("START ANALYSIS")


mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

if mode == "biceps":
    exercise = BicepsCurl()
else:
    exercise = Squat()


left_col, right_col = st.columns([2.2, 1])

with left_col:
    st.markdown('<div class="video-box">', unsafe_allow_html=True)
    frame_placeholder = st.empty()
    st.markdown('</div>', unsafe_allow_html=True)

with right_col:
    reps_box = st.empty()
    stage_box = st.empty()
    angle_box = st.empty()
    acc_box = st.empty()  
    status_box = st.empty()
    feedback_box = st.empty()

def update_metrics(counter, stage, angle, form_status, feedback, accuracy):
    reps_box.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">REPS</div>
        <div class="metric-value">{counter}</div>
    </div>
    """, unsafe_allow_html=True)

    stage_box.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">STAGE</div>
        <div class="metric-value">{stage}</div>
    </div>
    """, unsafe_allow_html=True)

    angle_box.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">ANGLE</div>
        <div class="metric-value">{int(angle)}</div>
    </div>
    """, unsafe_allow_html=True)

    
    acc_box.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">ACCURACY</div>
        <div class="metric-value">{accuracy:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

    status_class = "status-good" if form_status == "GOOD" else "status-bad"
    status_box.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">FORM STATUS</div>
        <div class="{status_class}">{form_status}</div>
    </div>
    """, unsafe_allow_html=True)

    feedback_box.markdown(f"""
    <div class="feedback-box">
        <h4 style="color:#ff4b2b; margin-top:0;">Coach Feedback</h4>
        <p style="font-size:18px; color:white;">{feedback}</p>
    </div>
    """, unsafe_allow_html=True)

def process_frame(frame, results, exercise, mode, mp_pose):
    image = frame.copy()

    counter = exercise.counter
    stage = exercise.stage if exercise.stage is not None else "-"
    feedback = "No body detected"
    form_status = "-"
    angle = 0
    back_angle_debug = 0

    try:
        if results.pose_landmarks is not None:
            landmarks = results.pose_landmarks.landmark

            if mode == "biceps":
                angle, counter, stage, feedback = exercise.process(landmarks, mp_pose)
                form_status = "GOOD" if feedback in ["Arm fully lowered", "Curling", "Top position", "Good"] else "BAD"

            elif mode == "squat":
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
        feedback = f"Processing error: {e}"
        form_status = "BAD"

    cv2.rectangle(image, (0, 0), (700, 170), (15, 15, 15), -1)

    cv2.putText(image, f'REPS: {counter}', (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.putText(image, f'STAGE: {stage}', (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.putText(image, f'FORM: {feedback}', (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    status_color = (0, 255, 0) if form_status == "GOOD" else (0, 0, 255)
    cv2.putText(image, f'STATUS: {form_status}', (360, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 2)

    cv2.putText(image, f'ANGLE: {int(angle)}', (360, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    if mode == "squat":
        cv2.putText(image, f'BACK ANGLE: {int(back_angle_debug)}', (360, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    if results.pose_landmarks is not None:
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

    return image, counter, stage, feedback, form_status, angle

if run_button:
    if input_type == "Upload Video":
        if uploaded_file is None:
            st.warning("Please upload a video first.")
        else:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            video_path = tfile.name

            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                st.error("Could not open uploaded video.")
            else:
                with mp_pose.Pose(min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5) as pose:
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break

                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        results = pose.process(rgb)

                        processed_frame, counter, stage, feedback, form_status, angle = process_frame(
                            frame, results, exercise, mode, mp_pose
                        )

                        
                        total_frames += 1
                        if form_status == "GOOD":
                            correct_frames += 1
                        accuracy = (correct_frames / total_frames) * 100 if total_frames > 0 else 0

                        frame_placeholder.image(
                            cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB),
                            channels="RGB",
                            use_container_width=True
                        )

                        update_metrics(counter, stage, angle, form_status, feedback, accuracy)

                cap.release()

    elif input_type == "Webcam":
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            st.error("Could not access webcam.")
        else:
            with mp_pose.Pose(min_detection_confidence=0.5,
                              min_tracking_confidence=0.5) as pose:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = pose.process(rgb)

                    processed_frame, counter, stage, feedback, form_status, angle = process_frame(
                        frame, results, exercise, mode, mp_pose
                    )

                    
                    total_frames += 1
                    if form_status == "GOOD":
                        correct_frames += 1
                    accuracy = (correct_frames / total_frames) * 100 if total_frames > 0 else 0

                    frame_placeholder.image(
                        cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB),
                        channels="RGB",
                        use_container_width=True
                    )

                    update_metrics(counter, stage, angle, form_status, feedback, accuracy)

                cap.release()