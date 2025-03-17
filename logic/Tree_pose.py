<<<<<<< HEAD
import cv2
import numpy as np
import mediapipe as mp

class TreePoseAngleChecker:
    """
    This class encapsulates angle-based logic for Tree Pose (Vrksasana) detection and feedback.
    Assumes LEFT foot is on the ground, RIGHT foot is lifted and placed on the inside of the left thigh.
    Arms overhead.
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
        
        # Approximate "ideal" Tree Pose in normalized coordinates (left foot on ground, right foot lifted).
        # The arms are overhead, palms together or close.
        self.ideal_pose = {
            'left_ankle':    [0.50, 0.90],  # supporting foot near center
            'left_knee':     [0.50, 0.70],  # supporting knee fairly straight
            'left_hip':      [0.50, 0.55],  # supporting hip
            'right_hip':     [0.52, 0.55],  # slightly outward for the lifted leg
            'right_knee':    [0.60, 0.45],  # bent, foot high on inner thigh
            'right_ankle':   [0.55, 0.35],  # near left thigh region
            
            'left_shoulder': [0.48, 0.25],
            'right_shoulder':[0.52, 0.25],
            'left_elbow':    [0.48, 0.15],
            'right_elbow':   [0.52, 0.15],
            'left_wrist':    [0.48, 0.05],
            'right_wrist':   [0.52, 0.05]
        }
        
        # Define which joints' angles we care about:
        # (p1, center, p2) => angle at `center`
        # For Tree Pose, we often check: 
        #  - supporting knee (straight), 
        #  - lifted knee (bent ~90-100° outward),
        #  - hips and shoulders aligned, 
        #  - arms overhead
        self.angle_definitions = {
            'left_knee':      ('left_ankle',  'left_knee',   'left_hip'),      # supporting knee
            'right_knee':     ('right_ankle', 'right_knee',  'right_hip'),     # bent outward
            'left_shoulder':  ('left_elbow',  'left_shoulder','left_hip'),     # arms overhead
            'right_shoulder': ('right_elbow', 'right_shoulder','right_hip'),
            'left_hip':       ('left_knee',   'left_hip',    'left_shoulder'), # upright posture
            'right_hip':      ('right_knee',  'right_hip',   'right_shoulder')
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
        Compare user angles to the "ideal" Tree Pose angles.
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
        Provide textual feedback based on how close angles are to the ideal Tree Pose.
        """
        feedback = []
        
        if overall_similarity < 0.1:
            return ["Pose not detected or extremely off."]
        
        # For each joint that is significantly off, add feedback
        for joint, sim in joint_similarities.items():
            if sim < 0.7:  # <70% match for that joint
                if "knee" in joint:
                    if "left_knee" in joint:
                        feedback.append("Straighten your standing (left) knee.")
                    else:
                        feedback.append("Lift your right knee higher and open it outward.")
                elif "shoulder" in joint:
                    feedback.append("Extend your arms overhead more (shoulders).")
                elif "hip" in joint:
                    feedback.append("Keep your hips level and torso upright.")
        
        # If no specific feedback, but overall <80%, give general comment
        if not feedback and overall_similarity < 0.8:
            feedback.append("Close! Slightly adjust your posture for a perfect Tree Pose.")
        
        if not feedback:
            feedback.append("Great job! Your Tree Pose looks accurate.")
        
        return feedback


# def main():
#     mp_drawing = mp.solutions.drawing_utils
#     mp_pose = mp.solutions.pose
#     tpose_checker = TreePoseAngleChecker()

#     cap = cv2.VideoCapture(0)
#     if not cap.isOpened():
#         print("Error: Could not open webcam.")
#         return

#     # Set up video recording: get frame dimensions and create VideoWriter.
#     frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fourcc = cv2.VideoWriter_fourcc(*'XVID')
#     out = cv2.VideoWriter('output_tree_pose.avi', fourcc, 20.0, (frame_width, frame_height))

#     # Optional: display full-screen window
#     cv2.namedWindow("Tree-Pose Angle Feedback", cv2.WND_PROP_FULLSCREEN)
#     cv2.setWindowProperty("Tree-Pose Angle Feedback", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("Failed to grab frame from webcam.")
#             break
        
#         # Mirror view for intuitive feedback
#         frame = cv2.flip(frame, 1)
#         user_keypoints, pose_landmarks = tpose_checker.process_frame(frame)

#         if user_keypoints:
#             overall_sim, joint_sims = tpose_checker.compute_pose_similarity(user_keypoints)
#             feedback_lines = tpose_checker.generate_feedback(overall_sim, joint_sims)
#             is_correct = (overall_sim >= 0.8)
#             text_color = (0, 255, 0) if is_correct else (0, 0, 255)
#             mp_drawing.draw_landmarks(
#                 frame, 
#                 pose_landmarks, 
#                 mp_pose.POSE_CONNECTIONS,
#                 mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
#                 mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2)
#             )
#             similarity_text = f"Similarity: {overall_sim * 100:.2f}%"
#             cv2.putText(frame, similarity_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
#             y_offset = 60
#             for line in feedback_lines:
#                 cv2.putText(frame, line, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
#                 y_offset += 30
#         else:
#             cv2.putText(frame, "No pose detected. Please stand in the frame.", (10, 30),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

#         # Write the processed frame to the output video file
#         out.write(frame)
#         cv2.imshow("Tree-Pose Angle Feedback", frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

# if __name__ == "__main__":
#     main()
=======
import cv2
import numpy as np
import mediapipe as mp

class TreePoseAngleChecker:
    """
    This class encapsulates angle-based logic for Tree Pose (Vrksasana) detection and feedback.
    Assumes LEFT foot is on the ground, RIGHT foot is lifted and placed on the inside of the left thigh.
    Arms overhead.
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
        
        # Approximate "ideal" Tree Pose in normalized coordinates (left foot on ground, right foot lifted).
        # The arms are overhead, palms together or close.
        self.ideal_pose = {
            'left_ankle':    [0.50, 0.90],  # supporting foot near center
            'left_knee':     [0.50, 0.70],  # supporting knee fairly straight
            'left_hip':      [0.50, 0.55],  # supporting hip
            'right_hip':     [0.52, 0.55],  # slightly outward for the lifted leg
            'right_knee':    [0.60, 0.45],  # bent, foot high on inner thigh
            'right_ankle':   [0.55, 0.35],  # near left thigh region
            
            'left_shoulder': [0.48, 0.25],
            'right_shoulder':[0.52, 0.25],
            'left_elbow':    [0.48, 0.15],
            'right_elbow':   [0.52, 0.15],
            'left_wrist':    [0.48, 0.05],
            'right_wrist':   [0.52, 0.05]
        }
        
        # Define which joints' angles we care about:
        # (p1, center, p2) => angle at `center`
        # For Tree Pose, we often check: 
        #  - supporting knee (straight), 
        #  - lifted knee (bent ~90-100° outward),
        #  - hips and shoulders aligned, 
        #  - arms overhead
        self.angle_definitions = {
            'left_knee':      ('left_ankle',  'left_knee',   'left_hip'),      # supporting knee
            'right_knee':     ('right_ankle', 'right_knee',  'right_hip'),     # bent outward
            'left_shoulder':  ('left_elbow',  'left_shoulder','left_hip'),     # arms overhead
            'right_shoulder': ('right_elbow', 'right_shoulder','right_hip'),
            'left_hip':       ('left_knee',   'left_hip',    'left_shoulder'), # upright posture
            'right_hip':      ('right_knee',  'right_hip',   'right_shoulder')
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
        Compare user angles to the "ideal" Tree Pose angles.
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
        Provide textual feedback based on how close angles are to the ideal Tree Pose.
        """
        feedback = []
        
        if overall_similarity < 0.1:
            return ["Pose not detected or extremely off."]
        
        # For each joint that is significantly off, add feedback
        for joint, sim in joint_similarities.items():
            if sim < 0.7:  # <70% match for that joint
                if "knee" in joint:
                    if "left_knee" in joint:
                        feedback.append("Straighten your standing (left) knee.")
                    else:
                        feedback.append("Lift your right knee higher and open it outward.")
                elif "shoulder" in joint:
                    feedback.append("Extend your arms overhead more (shoulders).")
                elif "hip" in joint:
                    feedback.append("Keep your hips level and torso upright.")
        
        # If no specific feedback, but overall <80%, give general comment
        if not feedback and overall_similarity < 0.8:
            feedback.append("Close! Slightly adjust your posture for a perfect Tree Pose.")
        
        if not feedback:
            feedback.append("Great job! Your Tree Pose looks accurate.")
        
        return feedback


# def main():
#     mp_drawing = mp.solutions.drawing_utils
#     mp_pose = mp.solutions.pose
#     tpose_checker = TreePoseAngleChecker()

#     cap = cv2.VideoCapture(0)
#     if not cap.isOpened():
#         print("Error: Could not open webcam.")
#         return

#     # Set up video recording: get frame dimensions and create VideoWriter.
#     frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fourcc = cv2.VideoWriter_fourcc(*'XVID')
#     out = cv2.VideoWriter('output_tree_pose.avi', fourcc, 20.0, (frame_width, frame_height))

#     # Optional: display full-screen window
#     cv2.namedWindow("Tree-Pose Angle Feedback", cv2.WND_PROP_FULLSCREEN)
#     cv2.setWindowProperty("Tree-Pose Angle Feedback", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("Failed to grab frame from webcam.")
#             break
        
#         # Mirror view for intuitive feedback
#         frame = cv2.flip(frame, 1)
#         user_keypoints, pose_landmarks = tpose_checker.process_frame(frame)

#         if user_keypoints:
#             overall_sim, joint_sims = tpose_checker.compute_pose_similarity(user_keypoints)
#             feedback_lines = tpose_checker.generate_feedback(overall_sim, joint_sims)
#             is_correct = (overall_sim >= 0.8)
#             text_color = (0, 255, 0) if is_correct else (0, 0, 255)
#             mp_drawing.draw_landmarks(
#                 frame, 
#                 pose_landmarks, 
#                 mp_pose.POSE_CONNECTIONS,
#                 mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
#                 mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2)
#             )
#             similarity_text = f"Similarity: {overall_sim * 100:.2f}%"
#             cv2.putText(frame, similarity_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
#             y_offset = 60
#             for line in feedback_lines:
#                 cv2.putText(frame, line, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
#                 y_offset += 30
#         else:
#             cv2.putText(frame, "No pose detected. Please stand in the frame.", (10, 30),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

#         # Write the processed frame to the output video file
#         out.write(frame)
#         cv2.imshow("Tree-Pose Angle Feedback", frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

# if __name__ == "__main__":
#     main()
>>>>>>> 561effd (Adding MORE Yoga Poses, working fine locally)
