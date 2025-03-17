import cv2
import numpy as np
import mediapipe as mp

import cv2
import numpy as np
import mediapipe as mp

class CatCowPoseAngleChecker:
    """
    Angle-based logic for Cat-Cow Pose (Marjaryasana-Bitilasana).
    Because Cat and Cow are two different shapes, we define two sets of ideal coordinates:
      - Cat: spine rounded upward, nose above shoulders, hips slightly above knees
      - Cow: spine arched downward, chest lifted, nose in line or below shoulders

    This class returns whichever pose the user is closer to (Cat or Cow),
    along with feedback about how to improve that pose.
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

        # ---------- 1) Define "ideal" Cat Pose in normalized coordinates (approx. side view) ----------
        # Knees under hips, wrists under shoulders, spine arched upward, nose above shoulders.
        self.cat_ideal_pose = {
            'left_ankle':    [0.50, 0.90],
            'right_ankle':   [0.50, 0.90],
            'left_knee':     [0.45, 0.75],
            'right_knee':    [0.55, 0.75],
            'left_hip':      [0.45, 0.60],
            'right_hip':     [0.55, 0.60],
            'left_shoulder': [0.45, 0.40],
            'right_shoulder':[0.55, 0.40],
            'left_elbow':    [0.45, 0.45],
            'right_elbow':   [0.55, 0.45],
            'left_wrist':    [0.45, 0.50],
            'right_wrist':   [0.55, 0.50],
            'nose':          [0.50, 0.35]
        }

        # ---------- 2) Define "ideal" Cow Pose in normalized coordinates (approx. side view) ----------
        # Knees under hips, wrists under shoulders, spine arched downward, chest lifted, nose in line/below shoulders.
        self.cow_ideal_pose = {
            'left_ankle':    [0.50, 0.90],
            'right_ankle':   [0.50, 0.90],
            'left_knee':     [0.45, 0.75],
            'right_knee':    [0.55, 0.75],
            'left_hip':      [0.45, 0.60],
            'right_hip':     [0.55, 0.60],
            'left_shoulder': [0.45, 0.45],
            'right_shoulder':[0.55, 0.45],
            'left_elbow':    [0.45, 0.40],
            'right_elbow':   [0.55, 0.40],
            'left_wrist':    [0.45, 0.50],
            'right_wrist':   [0.55, 0.50],
            'nose':          [0.50, 0.48]
        }

        # ---------- 3) Define angle definitions (the same for Cat and Cow) ----------
        # (p1, center, p2) => angle at `center`
        self.angle_definitions = {
            'left_knee':     ('left_ankle',  'left_knee',   'left_hip'),
            'right_knee':    ('right_ankle', 'right_knee',  'right_hip'),
            'left_hip':      ('left_knee',   'left_hip',    'left_shoulder'),
            'right_hip':     ('right_knee',  'right_hip',   'right_shoulder'),
            'left_shoulder': ('left_elbow',  'left_shoulder','left_hip'),
            'right_shoulder':('right_elbow', 'right_shoulder','right_hip'),
            'left_elbow':    ('left_wrist',  'left_elbow',  'left_shoulder'),
            'right_elbow':   ('right_wrist', 'right_elbow', 'right_shoulder'),
            'nose':          ('left_shoulder','nose','right_shoulder')  # optional angle for head position
        }

        # Precompute angles for both cat and cow
        self.cat_ideal_angles = self._calculate_joint_angles(self.cat_ideal_pose)
        self.cow_ideal_angles = self._calculate_joint_angles(self.cow_ideal_pose)

    def process_frame(self, frame):
        """
        Runs Mediapipe Pose detection on a frame (BGR image),
        returns the normalized keypoints (dict) and the raw landmarks.
        """
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

    def _compute_single_similarity(self, user_angles, ideal_angles):
        """
        Compare user_angles to one set of ideal_angles, returning (overall_similarity, joint_similarities).
        overall_similarity in [0..1].
        """
        total_similarity = 0.0
        valid_joints = 0
        joint_similarities = {}

        for joint, ideal_angle in ideal_angles.items():
            if joint in user_angles:
                user_angle = user_angles[joint]
                diff = abs(user_angle - ideal_angle)
                # Angle difference up to 180 => convert to similarity in [0..1]
                if diff > 180:
                    diff = 360 - diff
                similarity = 1 - (diff / 180.0)
                joint_similarities[joint] = similarity
                total_similarity += similarity
                valid_joints += 1
        
        if valid_joints == 0:
            return 0.0, {}
        overall = total_similarity / valid_joints
        return overall, joint_similarities

    def compute_pose_similarity(self, user_keypoints):
        """
        Compare user angles to BOTH cat and cow angles, pick whichever is higher.
        Returns (overall_similarity, joint_similarities, label) where label is "Cat" or "Cow".
        """
        user_angles = self._calculate_joint_angles(user_keypoints)
        if not user_angles:
            return (0.0, {}, "Unknown")

        # Compare to Cat
        cat_overall, cat_joints = self._compute_single_similarity(user_angles, self.cat_ideal_angles)
        # Compare to Cow
        cow_overall, cow_joints = self._compute_single_similarity(user_angles, self.cow_ideal_angles)

        if cat_overall > cow_overall:
            return (cat_overall, cat_joints, "Cat")
        else:
            return (cow_overall, cow_joints, "Cow")

    def generate_feedback(self, overall_similarity, joint_similarities, label):
        """
        Provide textual feedback based on which pose we decided (Cat or Cow),
        plus how close angles are to the ideal pose for that label.
        """
        feedback = []
        if label not in ["Cat", "Cow"]:
            return ["Pose not detected clearly."]

        if overall_similarity < 0.1:
            return [f"{label} Pose not detected or extremely off."]

        # For each joint that is significantly off, add feedback
        for joint, sim in joint_similarities.items():
            if sim < 0.7:  # <70% match for that joint
                if "knee" in joint:
                    feedback.append("Align knees under hips; keep them stable.")
                elif "hip" in joint:
                    if label == "Cat":
                        feedback.append("Round your back more, tuck your hips slightly.")
                    else:
                        feedback.append("Drop your belly more, lift hips slightly for Cow.")
                elif "shoulder" in joint:
                    feedback.append("Stack shoulders over wrists, engage your core.")
                elif "elbow" in joint:
                    feedback.append("Arms should be straight but not locked, shoulders relaxed.")
                elif "nose" in joint:
                    if label == "Cat":
                        feedback.append("Tuck your chin more for a full Cat stretch.")
                    else:
                        feedback.append("Lift your head for Cow pose, open the chest.")

        # If no specific feedback, but overall <80%, general comment
        if not feedback and overall_similarity < 0.8:
            feedback.append(f"Close! Slightly adjust your posture for a perfect {label} Pose.")
        
        if not feedback:
            feedback.append(f"Great job! Your {label} Pose looks accurate.")
        
        return feedback

def main():
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    checker = CatCowPoseAngleChecker()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    # Set up video recording
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output_Cat_Cow.avi', fourcc, 20.0, (frame_width, frame_height))

    # Optional: make the window full-screen
    cv2.namedWindow("Cat_Cow_Pose", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Cat_Cow_Pose", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame from webcam.")
            break
        
        frame = cv2.flip(frame, 1)  # mirror-like view
        user_keypoints, pose_landmarks = checker.process_frame(frame)

        if user_keypoints:
            # UNPACK ALL THREE
            overall_sim, joint_sims, label = checker.compute_pose_similarity(user_keypoints)
            feedback_lines = checker.generate_feedback(overall_sim, joint_sims, label)
            
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

            # Show which pose we think it is (Cat or Cow)
            label_text = f"Detected Pose: {label}"
            cv2.putText(frame, label_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
            
            # Show feedback lines
            y_offset = 90
            for line in feedback_lines:
                cv2.putText(frame, line, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
                y_offset += 30
        else:
            # No pose detected
            cv2.putText(frame, "No pose detected. Please stand in the frame.", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Write the processed frame to the video
        out.write(frame)

        cv2.imshow("Cat_Cow_Pose", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()