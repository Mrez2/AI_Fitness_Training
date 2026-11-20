from utils import calculate_angle

class BicepsCurl:
    def __init__(self):
        self.counter = 0
        self.stage = "down"
        self.last_angle = 180
        self.direction = None
        self.form_feedback = "Ready"

    def process(self, landmarks, mp_pose):
        shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                    landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]

        elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                 landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]

        wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                 landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

        angle = calculate_angle(shoulder, elbow, wrist)

        # Detect movement direction
        if angle < self.last_angle:
            self.direction = "up"
        elif angle > self.last_angle:
            self.direction = "down"

        # Better stage logic
        if angle > 145:
            self.stage = "down"

        if angle < 65 and self.stage == "down" and self.direction == "up":
            self.stage = "up"
            self.counter += 1

        # Better feedback
        if angle > 160:
            feedback = "Arm fully lowered"
        elif 65 <= angle <= 145:
            feedback = "Curling"
        elif angle < 65:
            feedback = "Top position"
        else:
            feedback = "Good"

        self.last_angle = angle

        return angle, self.counter, self.stage, feedback