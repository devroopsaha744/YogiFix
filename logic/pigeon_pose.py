import cv2
import numpy as np
import mediapipe as mp

class PigeonPoseAngleChecker:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.keypoint_indices = {
            'nose': 0,
            'left_shoulder': 11,
            'right_shoulder': 12,
            'left_elbow': 13,
            'right_elbow': 14,
            'left_wrist': 15,
            'right_wrist': 16,
            'left_hip': 23,
            'right_hip': 24,
            'left_knee': 25,
            'right_knee': 26,
            'left_ankle': 27,
            'right_ankle': 28
        }
        self.ideal_pose = {
            'nose': [0.5, 0.2],
            'left_shoulder': [0.45, 0.35],
            'right_shoulder': [0.55, 0.35],
            'left_elbow': [0.44, 0.45],
            'right_elbow': [0.56, 0.45],
            'left_wrist': [0.43, 0.55],
            'right_wrist': [0.57, 0.55],
            'left_hip': [0.45, 0.6],
            'right_hip': [0.55, 0.6],
            'left_knee': [0.4, 0.75],
            'right_knee': [0.7, 0.75],
            'left_ankle': [0.35, 0.85],
            'right_ankle': [0.85, 0.85]
        }
        self.angle_definitions = {
            'left_knee': ('left_hip', 'left_knee', 'left_ankle'),
            'right_knee': ('right_hip', 'right_knee', 'right_ankle'),
            'left_hip': ('left_shoulder', 'left_hip', 'left_knee'),
            'right_hip': ('right_shoulder', 'right_hip', 'right_knee'),
            'left_shoulder': ('left_hip', 'left_shoulder', 'left_elbow'),
            'right_shoulder': ('right_hip', 'right_shoulder', 'right_elbow')
        }
        self.ideal_angles = self._calculate_joint_angles(self.ideal_pose)

    def process_frame(self, frame):
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)
        if not results.pose_landmarks:
            return None, None
        user_keypoints = {}
        for name, idx in self.keypoint_indices.items():
            landmark = results.pose_landmarks.landmark[idx]
            user_keypoints[name] = [landmark.x, landmark.y]
        return user_keypoints, results.pose_landmarks

    def _calculate_joint_angles(self, keypoints):
        angles = {}
        for joint, (p1, p2, p3) in self.angle_definitions.items():
            if p1 in keypoints and p2 in keypoints and p3 in keypoints:
                angles[joint] = self._angle_between_points(keypoints[p1], keypoints[p2], keypoints[p3])
        return angles

    def _angle_between_points(self, a, b, c):
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

    def compute_pose_similarity(self, user_keypoints):
        user_angles = self._calculate_joint_angles(user_keypoints)
        if not user_angles:
            return 0.0, {}
        total_similarity = 0.0
        valid_joints = 0
        joint_similarities = {}
        for joint, ideal_angle in self.ideal_angles.items():
            if joint in user_angles:
                user_angle = user_angles[joint]
                diff = abs(user_angle - ideal_angle)
                if diff > 180:
                    diff = 360 - diff
                similarity = 1 - (diff / 180.0)
                joint_similarities[joint] = similarity
                total_similarity += similarity
                valid_joints += 1
        overall_similarity = (total_similarity / valid_joints) if valid_joints else 0
        return overall_similarity, joint_similarities

    def generate_feedback(self, overall_similarity, joint_similarities):
        feedback = []
        if overall_similarity < 0.1:
            return ["Pose not detected or extremely off."]
        for joint, sim in joint_similarities.items():
            if sim < 0.7:
                if "knee" in joint:
                    feedback.append("Adjust the front or back knee alignment.")
                elif "hip" in joint:
                    feedback.append("Square the hips and keep the pelvis stable.")
                elif "shoulder" in joint:
                    feedback.append("Open your chest and keep shoulders relaxed.")
        if not feedback and overall_similarity < 0.8:
            feedback.append("Almost there. Slightly refine your alignment.")
        if not feedback:
            feedback.append("Great job! Your Pigeon Pose looks well aligned.")
        return feedback

def main():
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    checker = PigeonPoseAngleChecker()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output_PigeonPose.avi', fourcc, 20.0, (frame_width, frame_height))
    cv2.namedWindow("Pigeon_Pose", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Pigeon_Pose", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame from webcam.")
            break
        frame = cv2.flip(frame, 1)
        user_keypoints, pose_landmarks = checker.process_frame(frame)
        if user_keypoints:
            overall_sim, joint_sims = checker.compute_pose_similarity(user_keypoints)
            feedback_lines = checker.generate_feedback(overall_sim, joint_sims)
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
            cv2.putText(frame, similarity_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
            y_offset = 60
            for line in feedback_lines:
                cv2.putText(frame, line, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
                y_offset += 30
        else:
            cv2.putText(frame, "No pose detected. Please stay in the frame.", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        out.write(frame)
        cv2.imshow("Pigeon_Pose", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
