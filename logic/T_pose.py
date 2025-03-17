<<<<<<< HEAD
import cv2
import numpy as np
import mediapipe as mp

class TPoseAngleChecker:
    """
    This class encapsulates angle-based logic for T-pose detection and feedback.
    """
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        # Use real-time mode (static_image_mode=False) for continuous detection/tracking
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
        # Define "ideal" T-pose keypoints in normalized coordinates (rough approximation)
        # We'll compute ideal angles from these points
        self.ideal_pose = {
            'nose':          [0.50, 0.20],
            'left_shoulder': [0.35, 0.30],
            'right_shoulder':[0.65, 0.30],
            'left_elbow':    [0.20, 0.30],
            'right_elbow':   [0.80, 0.30],
            'left_wrist':    [0.05, 0.30],
            'right_wrist':   [0.95, 0.30],
            'left_hip':      [0.40, 0.60],
            'right_hip':     [0.60, 0.60],
            'left_knee':     [0.40, 0.75],
            'right_knee':    [0.60, 0.75],
            'left_ankle':    [0.40, 0.90],
            'right_ankle':   [0.60, 0.90]
        }
        
        # Specify joints to calculate angles for:
        # (point1, center_point, point2) => angle at center_point
        self.angle_definitions = {
            'right_shoulder': ('right_elbow', 'right_shoulder', 'right_hip'),
            'left_shoulder':  ('left_elbow',  'left_shoulder',  'left_hip'),
            'right_elbow':    ('right_wrist', 'right_elbow',    'right_shoulder'),
            'left_elbow':     ('left_wrist',  'left_elbow',     'left_shoulder'),
            'right_hip':      ('right_knee',  'right_hip',      'right_shoulder'),
            'left_hip':       ('left_knee',   'left_hip',       'left_shoulder'),
            'right_knee':     ('right_ankle', 'right_knee',     'right_hip'),
            'left_knee':      ('left_ankle',  'left_knee',      'left_hip')
        }
        
        # Precompute ideal angles from the "ideal_pose"
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
        
        # Extract normalized landmarks for the key joints we care about
        user_keypoints = {}
        for name, idx in self.keypoint_indices.items():
            landmark = results.pose_landmarks.landmark[idx]
            # Normalized x, y in [0,1]
            user_keypoints[name] = [landmark.x, landmark.y]
        
        return user_keypoints, results.pose_landmarks

    def _calculate_joint_angles(self, keypoints):
        """
        Calculate angles (in degrees) at specified joints.
        keypoints: dict of {joint_name: [x, y]} in normalized coordinates
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
        # Clip to avoid floating precision issues
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_angle))
        return angle

    def compute_pose_similarity(self, user_keypoints):
        """
        Compare user angles with the ideal T-pose angles.
        Returns (overall_similarity, dict_of_joint_similarities).
        overall_similarity in [0..1].
        """
        user_angles = self._calculate_joint_angles(user_keypoints)
        
        # If no angles computed, return 0
        if not user_angles:
            return 0.0, {}
        
        total_similarity = 0.0
        joint_similarities = {}
        valid_joints = 0
        
        # Compare angle differences
        for joint, ideal_angle in self.ideal_angles.items():
            if joint in user_angles:
                user_angle = user_angles[joint]
                diff = abs(user_angle - ideal_angle)
                # Angle difference up to 180 => convert to similarity in [0,1]
                if diff > 180:
                    diff = 360 - diff
                similarity = 1 - (diff / 180.0)  # 0 if diff=180, 1 if diff=0
                joint_similarities[joint] = similarity
                total_similarity += similarity
                valid_joints += 1
        
        overall_similarity = (total_similarity / valid_joints) if valid_joints else 0
        return overall_similarity, joint_similarities

    def generate_feedback(self, overall_similarity, joint_similarities):
        """
        Provide textual feedback based on how close the angles are to ideal.
        Returns a list of feedback lines.
        """
        feedback = []
        
        # If we have no joints or extremely low overall similarity
        if overall_similarity < 0.1:
            return ["Pose not detected or extremely off."]

        # Check each joint's similarity
        for joint, sim in joint_similarities.items():
            if sim < 0.7:
                # Provide some suggestions based on which joint is off
                if 'shoulder' in joint:
                    feedback.append(f"Adjust your {joint}. Arms should be straight out to sides.")
                elif 'elbow' in joint:
                    feedback.append(f"Straighten your {joint}. Keep arms extended for T-pose.")
                elif 'hip' in joint:
                    feedback.append(f"Align your {joint} under shoulders. Keep torso upright.")
                elif 'knee' in joint:
                    feedback.append(f"Straighten your {joint}. Legs should be straight in T-pose.")
        
        # If no specific feedback lines but overall similarity is below perfect
        if not feedback and overall_similarity < 0.8:
            feedback.append("Overall pose needs minor adjustment. Aim for a perfect 'T' shape.")
        
        if not feedback:
            # If there's no feedback at all, we consider it a good T-pose
            feedback.append("Great job! Your T-pose looks accurate.")
        
        return feedback


# def main():
#     mp_drawing = mp.solutions.drawing_utils
#     mp_pose = mp.solutions.pose
    
#     # Instantiate the T-pose angle checker
#     tpose_checker = TPoseAngleChecker()

#     cap = cv2.VideoCapture(0)
#     if not cap.isOpened():
#         print("Error: Could not open webcam.")
#         return
    
#     cv2.namedWindow("T Pose", cv2.WND_PROP_FULLSCREEN)
#     cv2.setWindowProperty("T Pose", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("Failed to grab frame from webcam.")
#             break
        
#         # Flip for a mirror-like view
#         frame = cv2.flip(frame, 1)
#         height, width = frame.shape[:2]

#         # 1) Extract keypoints
#         user_keypoints, pose_landmarks = tpose_checker.process_frame(frame)

#         # 2) If we have landmarks, compute angles and feedback
#         if user_keypoints:
#             overall_sim, joint_sims = tpose_checker.compute_pose_similarity(user_keypoints)
#             feedback_lines = tpose_checker.generate_feedback(overall_sim, joint_sims)
            
#             # Decide if we are "correct" or "incorrect" based on overall similarity
#             is_correct = (overall_sim >= 0.8)
            
#             # Draw the Mediapipe pose skeleton
#             mp_drawing.draw_landmarks(
#                 frame, 
#                 pose_landmarks, 
#                 mp_pose.POSE_CONNECTIONS,
#                 mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
#                 mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2)
#             )
            
#             # 3) Display feedback
#             #    If correct => green text, otherwise => red text
#             text_color = (0, 255, 0) if is_correct else (0, 0, 255)
            
#             # Show similarity percentage
#             similarity_percent = overall_sim * 100
#             similarity_text = f"Similarity: {similarity_percent:.2f}%"
#             cv2.putText(frame, similarity_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
            
#             # Then show feedback lines
#             y_offset = 60
#             for line in feedback_lines:
#                 cv2.putText(frame, line, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
#                 y_offset += 30
#         else:
#             # No pose detected
#             cv2.putText(frame, "No pose detected. Please stand in the frame.", (10, 30),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            
#         cv2.imshow("T-Pose Angle Feedback", frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
    
#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()
=======
import cv2
import numpy as np
import mediapipe as mp

class TPoseAngleChecker:
    """
    This class encapsulates angle-based logic for T-pose detection and feedback.
    """
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        # Use real-time mode (static_image_mode=False) for continuous detection/tracking
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
        # Define "ideal" T-pose keypoints in normalized coordinates (rough approximation)
        # We'll compute ideal angles from these points
        self.ideal_pose = {
            'nose':          [0.50, 0.20],
            'left_shoulder': [0.35, 0.30],
            'right_shoulder':[0.65, 0.30],
            'left_elbow':    [0.20, 0.30],
            'right_elbow':   [0.80, 0.30],
            'left_wrist':    [0.05, 0.30],
            'right_wrist':   [0.95, 0.30],
            'left_hip':      [0.40, 0.60],
            'right_hip':     [0.60, 0.60],
            'left_knee':     [0.40, 0.75],
            'right_knee':    [0.60, 0.75],
            'left_ankle':    [0.40, 0.90],
            'right_ankle':   [0.60, 0.90]
        }
        
        # Specify joints to calculate angles for:
        # (point1, center_point, point2) => angle at center_point
        self.angle_definitions = {
            'right_shoulder': ('right_elbow', 'right_shoulder', 'right_hip'),
            'left_shoulder':  ('left_elbow',  'left_shoulder',  'left_hip'),
            'right_elbow':    ('right_wrist', 'right_elbow',    'right_shoulder'),
            'left_elbow':     ('left_wrist',  'left_elbow',     'left_shoulder'),
            'right_hip':      ('right_knee',  'right_hip',      'right_shoulder'),
            'left_hip':       ('left_knee',   'left_hip',       'left_shoulder'),
            'right_knee':     ('right_ankle', 'right_knee',     'right_hip'),
            'left_knee':      ('left_ankle',  'left_knee',      'left_hip')
        }
        
        # Precompute ideal angles from the "ideal_pose"
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
        
        # Extract normalized landmarks for the key joints we care about
        user_keypoints = {}
        for name, idx in self.keypoint_indices.items():
            landmark = results.pose_landmarks.landmark[idx]
            # Normalized x, y in [0,1]
            user_keypoints[name] = [landmark.x, landmark.y]
        
        return user_keypoints, results.pose_landmarks

    def _calculate_joint_angles(self, keypoints):
        """
        Calculate angles (in degrees) at specified joints.
        keypoints: dict of {joint_name: [x, y]} in normalized coordinates
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
        # Clip to avoid floating precision issues
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_angle))
        return angle

    def compute_pose_similarity(self, user_keypoints):
        """
        Compare user angles with the ideal T-pose angles.
        Returns (overall_similarity, dict_of_joint_similarities).
        overall_similarity in [0..1].
        """
        user_angles = self._calculate_joint_angles(user_keypoints)
        
        # If no angles computed, return 0
        if not user_angles:
            return 0.0, {}
        
        total_similarity = 0.0
        joint_similarities = {}
        valid_joints = 0
        
        # Compare angle differences
        for joint, ideal_angle in self.ideal_angles.items():
            if joint in user_angles:
                user_angle = user_angles[joint]
                diff = abs(user_angle - ideal_angle)
                # Angle difference up to 180 => convert to similarity in [0,1]
                if diff > 180:
                    diff = 360 - diff
                similarity = 1 - (diff / 180.0)  # 0 if diff=180, 1 if diff=0
                joint_similarities[joint] = similarity
                total_similarity += similarity
                valid_joints += 1
        
        overall_similarity = (total_similarity / valid_joints) if valid_joints else 0
        return overall_similarity, joint_similarities

    def generate_feedback(self, overall_similarity, joint_similarities):
        """
        Provide textual feedback based on how close the angles are to ideal.
        Returns a list of feedback lines.
        """
        feedback = []
        
        # If we have no joints or extremely low overall similarity
        if overall_similarity < 0.1:
            return ["Pose not detected or extremely off."]

        # Check each joint's similarity
        for joint, sim in joint_similarities.items():
            if sim < 0.7:
                # Provide some suggestions based on which joint is off
                if 'shoulder' in joint:
                    feedback.append(f"Adjust your {joint}. Arms should be straight out to sides.")
                elif 'elbow' in joint:
                    feedback.append(f"Straighten your {joint}. Keep arms extended for T-pose.")
                elif 'hip' in joint:
                    feedback.append(f"Align your {joint} under shoulders. Keep torso upright.")
                elif 'knee' in joint:
                    feedback.append(f"Straighten your {joint}. Legs should be straight in T-pose.")
        
        # If no specific feedback lines but overall similarity is below perfect
        if not feedback and overall_similarity < 0.8:
            feedback.append("Overall pose needs minor adjustment. Aim for a perfect 'T' shape.")
        
        if not feedback:
            # If there's no feedback at all, we consider it a good T-pose
            feedback.append("Great job! Your T-pose looks accurate.")
        
        return feedback


# def main():
#     mp_drawing = mp.solutions.drawing_utils
#     mp_pose = mp.solutions.pose
    
#     # Instantiate the T-pose angle checker
#     tpose_checker = TPoseAngleChecker()

#     cap = cv2.VideoCapture(0)
#     if not cap.isOpened():
#         print("Error: Could not open webcam.")
#         return
    
#     cv2.namedWindow("T Pose", cv2.WND_PROP_FULLSCREEN)
#     cv2.setWindowProperty("T Pose", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("Failed to grab frame from webcam.")
#             break
        
#         # Flip for a mirror-like view
#         frame = cv2.flip(frame, 1)
#         height, width = frame.shape[:2]

#         # 1) Extract keypoints
#         user_keypoints, pose_landmarks = tpose_checker.process_frame(frame)

#         # 2) If we have landmarks, compute angles and feedback
#         if user_keypoints:
#             overall_sim, joint_sims = tpose_checker.compute_pose_similarity(user_keypoints)
#             feedback_lines = tpose_checker.generate_feedback(overall_sim, joint_sims)
            
#             # Decide if we are "correct" or "incorrect" based on overall similarity
#             is_correct = (overall_sim >= 0.8)
            
#             # Draw the Mediapipe pose skeleton
#             mp_drawing.draw_landmarks(
#                 frame, 
#                 pose_landmarks, 
#                 mp_pose.POSE_CONNECTIONS,
#                 mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
#                 mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2)
#             )
            
#             # 3) Display feedback
#             #    If correct => green text, otherwise => red text
#             text_color = (0, 255, 0) if is_correct else (0, 0, 255)
            
#             # Show similarity percentage
#             similarity_percent = overall_sim * 100
#             similarity_text = f"Similarity: {similarity_percent:.2f}%"
#             cv2.putText(frame, similarity_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
            
#             # Then show feedback lines
#             y_offset = 60
#             for line in feedback_lines:
#                 cv2.putText(frame, line, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
#                 y_offset += 30
#         else:
#             # No pose detected
#             cv2.putText(frame, "No pose detected. Please stand in the frame.", (10, 30),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            
#         cv2.imshow("T-Pose Angle Feedback", frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
    
#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()
>>>>>>> 561effd (Adding MORE Yoga Poses, working fine locally)
