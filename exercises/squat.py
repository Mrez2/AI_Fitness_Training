from utils import calculate_angle

class Squat:
    def __init__(self):
        self.counter = 0
        self.stage = "up"

    def process(self, landmarks, mp_pose):
        shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                    landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]

        hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
               landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]

        knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]

        ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                 landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

        knee_angle = calculate_angle(hip, knee, ankle)
        back_angle = calculate_angle(shoulder, hip, knee)

        # Rep count
        if knee_angle > 150:
            if self.stage == "down":
                self.counter += 1
            self.stage = "up"

        elif knee_angle < 110:
            self.stage = "down"

        # Form check
        form_status = "GOOD"
        feedback = "Good squat"

        if knee_angle > 130 and self.stage == "down":
            form_status = "BAD"
            feedback = "Go lower"
        elif knee_angle < 50:
            form_status = "BAD"
            feedback = "Too deep"
        elif back_angle < 130:
            form_status = "BAD"
            feedback = "Lean less forward"

        return knee_angle, self.counter, self.stage, feedback, form_status