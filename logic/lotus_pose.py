import cv2
import numpy as np
import mediapipe as mp

class PadmasanDistanceAngleChecker:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        # Key landmarks
        self.keypoint_indices = {
            'left_shoulder': 11,
            'right_shoulder': 12,
            'left_hip': 23,
            'right_hip': 24,
            'left_knee': 25,
            'right_knee': 26,
            'left_ankle': 27,
            'right_ankle': 28
        }
        # Ideal / reference values
        # Now ideal knee angle is set to 37.5° (midpoint of 30-45° range)
        self.ideal_knee_angle = 37.5
        self.acceptable_knee_deviation = 7.5  # Allowable deviation to keep the angle between 30° and 45°
        self.ideal_ankle_distance = 0.05       # normalized distance between ankles
        self.acceptable_ankle_range = 0.05     # tolerance around ideal ankle distance
        self.shoulder_alignment_threshold = 0.03

    def process_frame(self, frame):
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)
        if not results.pose_landmarks:
            return None, None
        user_keypoints = {}
        for name, idx in self.keypoint_indices.items():
            lm = results.pose_landmarks.landmark[idx]
            user_keypoints[name] = [lm.x, lm.y]  # normalized coordinates
        return user_keypoints, results.pose_landmarks

    def angle_between_points(self, a, b, c):
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        ba = a - b
        bc = c - b
        dot_prod = np.dot(ba, bc)
        mag_ba = np.linalg.norm(ba)
        mag_bc = np.linalg.norm(bc)
        if mag_ba == 0 or mag_bc == 0:
            return 0
        cos_angle = dot_prod / (mag_ba * mag_bc)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return np.degrees(np.arccos(cos_angle))

    def distance_2d(self, p1, p2):
        p1 = np.array(p1)
        p2 = np.array(p2)
        return np.linalg.norm(p1 - p2)

    def analyze_pose(self, keypoints):
        left_knee_angle = self.angle_between_points(
            keypoints['left_hip'],
            keypoints['left_knee'],
            keypoints['left_ankle']
        )
        right_knee_angle = self.angle_between_points(
            keypoints['right_hip'],
            keypoints['right_knee'],
            keypoints['right_ankle']
        )
        ankle_dist = self.distance_2d(keypoints['left_ankle'], keypoints['right_ankle'])
        shoulder_diff_y = abs(keypoints['left_shoulder'][1] - keypoints['right_shoulder'][1])
        return left_knee_angle, right_knee_angle, ankle_dist, shoulder_diff_y

    def compute_similarity(self, left_knee_angle, right_knee_angle, ankle_dist, shoulder_diff_y):
        left_diff = abs(left_knee_angle - self.ideal_knee_angle)
        right_diff = abs(right_knee_angle - self.ideal_knee_angle)
        left_knee_sim = 1 - min(left_diff, 180) / 180.0
        right_knee_sim = 1 - min(right_diff, 180) / 180.0

        ideal_min = self.ideal_ankle_distance - self.acceptable_ankle_range
        ideal_max = self.ideal_ankle_distance + self.acceptable_ankle_range
        if ankle_dist < ideal_min:
            diff_ratio = (ideal_min - ankle_dist) / ideal_min
            ankle_sim = max(0.0, 1 - diff_ratio)
        elif ankle_dist > ideal_max:
            diff_ratio = (ankle_dist - ideal_max) / ideal_max
            ankle_sim = max(0.0, 1 - diff_ratio)
        else:
            ankle_sim = 1.0

        if shoulder_diff_y <= self.shoulder_alignment_threshold:
            shoulder_sim = 1.0
        else:
            diff_ratio = (shoulder_diff_y - self.shoulder_alignment_threshold) / self.shoulder_alignment_threshold
            shoulder_sim = max(0.0, 1 - diff_ratio)

        similarities = [left_knee_sim, right_knee_sim, ankle_sim, shoulder_sim]
        overall_sim = sum(similarities) / len(similarities)
        return overall_sim, {
            'left_knee': left_knee_sim,
            'right_knee': right_knee_sim,
            'ankle_distance': ankle_sim,
            'shoulders': shoulder_sim
        }

    def generate_feedback(self, left_knee_angle, right_knee_angle, ankle_dist, shoulder_diff_y):
        feedback = []
        if abs(left_knee_angle - self.ideal_knee_angle) > self.acceptable_knee_deviation:
            feedback.append("Adjust LEFT knee angle to be between 30° and 45°.")
        if abs(right_knee_angle - self.ideal_knee_angle) > self.acceptable_knee_deviation:
            feedback.append("Adjust RIGHT knee angle to be between 30° and 45°.")

        ideal_min = self.ideal_ankle_distance - self.acceptable_ankle_range
        ideal_max = self.ideal_ankle_distance + self.acceptable_ankle_range
        if not (ideal_min <= ankle_dist <= ideal_max):
            feedback.append("Adjust ankles for a comfortable lotus cross (distance off).")

        if shoulder_diff_y > self.shoulder_alignment_threshold:
            feedback.append("Level your shoulders for a relaxed, upright posture.")

        if not feedback:
            feedback.append("Great Padmasan alignment! Keep shoulders relaxed and knees folded comfortably.")
        return feedback

def main():
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    checker = PadmasanDistanceAngleChecker()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output_PadmasanDistanceAngle.avi', fourcc, 20.0, (frame_width, frame_height))

    cv2.namedWindow("Padmasan", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Padmasan", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break
        frame = cv2.flip(frame, 1)

        user_keypoints, pose_landmarks = checker.process_frame(frame)

        if user_keypoints:
            left_knee_angle, right_knee_angle, ankle_dist, shoulder_diff_y = checker.analyze_pose(user_keypoints)
            overall_sim, sims = checker.compute_similarity(
                left_knee_angle, right_knee_angle, ankle_dist, shoulder_diff_y
            )
            feedback_lines = checker.generate_feedback(
                left_knee_angle, right_knee_angle, ankle_dist, shoulder_diff_y
            )

            is_correct = (overall_sim >= 0.8)
            text_color = (0, 255, 0) if is_correct else (0, 0, 255)

            mp_drawing.draw_landmarks(
                frame,
                pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2)
            )

            similarity_text = f"Similarity: {overall_sim * 100:.2f}%"
            cv2.putText(frame, similarity_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)

            y_offset = 60
            for line in feedback_lines:
                cv2.putText(frame, line, (10, y_offset), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
                y_offset += 30

        else:
            cv2.putText(frame, "No pose detected. Please stay in frame.", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        out.write(frame)
        cv2.imshow("Padmasan", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
