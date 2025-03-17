import cv2
import numpy as np
import mediapipe as mp

class CobraPoseAngleChecker:
    """
    This class encapsulates angle-based logic for Cobra Pose (Bhujangasana).
    Assumes a side-view perspective: The body is lying face-down, with legs extended
    and the chest lifted. Arms are partially bent, supporting the torso lift.
    """
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        # Key landmark indices for Mediapipe
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
        
        # Approximate "ideal" Cobra Pose in normalized coordinates (side view).
        # The hips are on/near the ground (y ~ 0.70), shoulders significantly higher
        # (y ~ 0.45), elbows bent near the ribcage, legs extended behind.
        self.ideal_pose = {
            # Legs extended
            'left_ankle':    [0.50, 0.90],
            'right_ankle':   [0.50, 0.90],
            'left_knee':     [0.50, 0.80],
            'right_knee':    [0.50, 0.80],
            'left_hip':      [0.50, 0.70],
            'right_hip':     [0.50, 0.70],
            # Torso lifted
            'left_shoulder': [0.50, 0.45],
            'right_shoulder':[0.50, 0.45],
            # Arms supporting
            'left_elbow':    [0.52, 0.55],
            'right_elbow':   [0.48, 0.55],
            'left_wrist':    [0.52, 0.60],
            'right_wrist':   [0.48, 0.60]
        }
        
        # Define which joints' angles we care about:
        # (p1, center, p2) => angle at `center`.
        # For Cobra Pose:
        # - Knees ~ 180° (legs mostly straight).
        # - Hips ~ 150-160° extension from knees to hips to shoulders (depending on how high chest lifts).
        # - Shoulders ~ 90-120° from elbows to shoulders to hips (arms partially bent).
        self.angle_definitions = {
            'left_knee':     ('left_ankle',  'left_knee',   'left_hip'),
            'right_knee':    ('right_ankle', 'right_knee',  'right_hip'),
            'left_hip':      ('left_knee',   'left_hip',    'left_shoulder'),
            'right_hip':     ('right_knee',  'right_hip',   'right_shoulder'),
            'left_shoulder': ('left_elbow',  'left_shoulder','left_hip'),
            'right_shoulder':('right_elbow', 'right_shoulder','right_hip'),
            'left_elbow':    ('left_wrist',  'left_elbow',  'left_shoulder'),
            'right_elbow':   ('right_wrist', 'right_elbow', 'right_shoulder')
        }
        
        # Precompute "ideal" angles from the approximate ideal_pose
        self.ideal_angles = self._calculate_joint_angles(self.ideal_pose)

    def process_frame(self, frame):
        """
        Runs Mediapipe Pose detection on a frame (BGR image),
        returns the normalized keypoints (dict) and the raw landmarks.
        """
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)
        if not results.pose_landmarks:
            return None, None
        
        # Extract normalized [x,y] for the key joints
        user_keypoints = {}
        for name, idx in self.keypoint_indices.items():
            landmark = results.pose_landmarks.landmark[idx]
            user_keypoints[name] = [landmark.x, landmark.y]
        
        return user_keypoints, results.pose_landmarks

    def _calculate_joint_angles(self, keypoints):
        """
        Calculate angles (in degrees) for each joint in self.angle_definitions.
        keypoints: {joint_name: [x, y]} in normalized coordinates
        Returns a dict of {joint_name: angle_degrees}
        """
        angles = {}
        for joint, (p1, p2, p3) in self.angle_definitions.items():
            if p1 in keypoints and p2 in keypoints and p3 in keypoints:
                angle = self._angle_between_points(
                    keypoints[p1], keypoints[p2], keypoints[p3]
                )
                angles[joint] = angle
        return angles

    def _angle_between_points(self, a, b, c):
        """
        Compute the angle (in degrees) at point b, formed by points a-b-c.
        Each is [x, y].
        """
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
        angle = np.degrees(np.arccos(cos_angle))
        return angle

    def compute_pose_similarity(self, user_keypoints):
        """
        Compare user angles to the "ideal" Cobra Pose angles.
        Returns (overall_similarity, dict_of_joint_similarities) in [0..1].
        """
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
                # For angles, difference can be up to 180
                if diff > 180:
                    diff = 360 - diff
                similarity = 1 - (diff / 180.0)  # 1 if diff=0, 0 if diff=180
                joint_similarities[joint] = similarity
                total_similarity += similarity
                valid_joints += 1
        
        overall_similarity = total_similarity / valid_joints if valid_joints else 0
        return overall_similarity, joint_similarities

    def generate_feedback(self, overall_similarity, joint_similarities):
        """
        Provide textual feedback based on how close angles are to the ideal Cobra Pose.
        """
        feedback = []
        
        if overall_similarity < 0.1:
            return ["Pose not detected or extremely off."]
        
        for joint, sim in joint_similarities.items():
            if sim < 0.7:  # <70% match for that joint
                if "knee" in joint:
                    feedback.append("Legs should remain extended and relaxed on the floor.")
                elif "hip" in joint:
                    feedback.append("Lift your chest more while keeping hips grounded.")
                elif "shoulder" in joint:
                    feedback.append("Open your chest and roll shoulders back.")
                elif "elbow" in joint:
                    feedback.append("Keep your elbows close to your sides, slightly bent.")
        
        # If no specific feedback, but overall <80%, general comment
        if not feedback and overall_similarity < 0.8:
            feedback.append("Close! Slightly adjust your posture for a perfect Cobra Pose.")
        
        if not feedback:
            feedback.append("Great job! Your Cobra Pose looks accurate.")
        
        return feedback

def main():
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    checker = CobraPoseAngleChecker()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    # Set up video recording
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output_cobra.avi', fourcc, 20.0, (frame_width, frame_height))

    # Optional: make the window full-screen
    cv2.namedWindow("cobra_Pose", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("cobra_Pose", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame from webcam.")
            break
        
        frame = cv2.flip(frame, 1)  # mirror-like view
        user_keypoints, pose_landmarks = checker.process_frame(frame)

        if user_keypoints:
            overall_sim, joint_sims = checker.compute_pose_similarity(user_keypoints)
            feedback_lines = checker.generate_feedback(overall_sim, joint_sims)
            
            # Decide text color (green if fairly correct, else red)
            is_correct = (overall_sim >= 0.8)
            text_color = (0, 255, 0) if is_correct else (0, 0, 255)
            
            # Draw the pose skeleton
            mp_drawing.draw_landmarks(
                frame,
                pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2)
            )
            
            # Show similarity percentage
            similarity_text = f"Similarity: {overall_sim * 100:.2f}%"
            cv2.putText(frame, similarity_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
            
            # Show feedback lines
            y_offset = 60
            for line in feedback_lines:
                cv2.putText(frame, line, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
                y_offset += 30
        else:
            # No pose detected
            cv2.putText(frame, "No pose detected. Please stand in the frame.", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Write the processed frame to the video
        out.write(frame)

        cv2.imshow("cobra_Pose", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()