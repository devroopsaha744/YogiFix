# web-app/app.py

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import time
import threading
import sys
import os
import json
import base64
import asyncio
from typing import List, Dict, Any

from fastapi.middleware.cors import CORSMiddleware

# Append parent directory so that 'logic' can be found.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import your pose-checker classes from your logic package
from logic.T_pose import TPoseAngleChecker
from logic.traingle_pose import TrianglePoseAngleChecker
from logic.Tree_pose import TreePoseAngleChecker
from logic.Crescent_lunge_pose import CrescentLungeAngleChecker
from logic.warrior_pose import WarriorPoseAngleChecker
from logic.mountain_pose import MountainPoseAngleChecker
from logic.bridge_pose import BridgePoseAngleChecker
from logic.cat_pose import CatCowPoseAngleChecker
from logic.cobra_pose import CobraPoseAngleChecker
from logic.downward_dog_pose import DownwardDogPoseAngleChecker
from logic.legs_wall_pose import LegsUpTheWallPoseAngleChecker
from logic.pigeon_pose import PigeonPoseAngleChecker
from logic.lotus_pose import PadmasanDistanceAngleChecker
from logic.seated_forward_bent import SeatedForwardBendAngleChecker
from logic.standing_forward_bent_pose import StandingForwardFoldAngleChecker

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create a dictionary mapping pose names to their checker instances
pose_checkers = {
    "Triangle": TrianglePoseAngleChecker(),
    "Tree": TreePoseAngleChecker(),
    "T": TPoseAngleChecker(),
    "Crescent_lunge": CrescentLungeAngleChecker(),
    "Warrior": WarriorPoseAngleChecker(),
    "Mountain": MountainPoseAngleChecker(),
    "Bridge" :BridgePoseAngleChecker(),
    "Cat-Cow": CatCowPoseAngleChecker(),
    "Cobra": CobraPoseAngleChecker(),
    "Seated": SeatedForwardBendAngleChecker(),
    "Standing": StandingForwardFoldAngleChecker(),
    "Downward Dog": DownwardDogPoseAngleChecker(),
    "Lotus" : PadmasanDistanceAngleChecker(),
    "Pigeon": PigeonPoseAngleChecker(),
    "Legs-Up-The-Wall": LegsUpTheWallPoseAngleChecker()
}

# Connection manager for WebSockets
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.processing_tasks = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        if client_id in self.processing_tasks:
            self.processing_tasks[client_id].cancel()
            del self.processing_tasks[client_id]

    async def start_processing(self, client_id: str, pose_type: str):
        if client_id in self.processing_tasks:
            self.processing_tasks[client_id].cancel()
        
        task = asyncio.create_task(self.process_frames(client_id, pose_type))
        self.processing_tasks[client_id] = task

    async def process_frames(self, client_id: str, pose_type: str):
        """Process frames for a specific client using the specified pose checker"""
        if client_id not in self.active_connections:
            return

        websocket = self.active_connections[client_id]
        checker = pose_checkers.get(pose_type, TPoseAngleChecker())
        
        # Open a webcam for this client
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            await websocket.send_json({"error": "Could not open webcam"})
            return
        
        try:
            while client_id in self.active_connections:
                ret, frame = cap.read()
                if not ret:
                    await asyncio.sleep(0.01)
                    continue

                # Flip frame horizontally for a mirror view
                frame = cv2.flip(frame, 1)
                
                # Process the frame using the pose-checker
                user_keypoints, landmarks = checker.process_frame(frame)
                
                if user_keypoints is None:
                    feedback_text = "No pose detected."
                    overall_sim = 0.0
                    joint_sims = {}
                else:
                    overall_sim, joint_sims = checker.compute_pose_similarity(user_keypoints)
                    feedback_lines = checker.generate_feedback(overall_sim, joint_sims)
                    feedback_text = f"Similarity: {overall_sim*100:.2f}%\n" + "\n".join(feedback_lines)
                    
                    # Draw landmarks on frame
                    frame = default_annotate(frame, landmarks, checker)
                
                # Convert frame to base64 for sending over WebSocket
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                frame_base64 = base64.b64encode(buffer).decode('utf-8')
                
                # Send frame and feedback together
                await websocket.send_json({
                    "frame": frame_base64,
                    "feedback": {
                        "similarity": float(overall_sim),
                        "feedback_text": feedback_text,
                        "joint_similarities": joint_sims if isinstance(joint_sims, dict) else {}
                    }
                })
                
                # Small delay to control frame rate
                await asyncio.sleep(0.03)
        
        except Exception as e:
            print(f"Error processing frames: {str(e)}")
        finally:
            cap.release()


def default_annotate(frame, landmarks, checker):
    """
    Draw the detected landmarks on the frame.
    """
    # If landmarks were detected, draw them
    if landmarks is not None:
        import mediapipe as mp
        mp_drawing = mp.solutions.drawing_utils
        # Use the checker's mp_pose if available; otherwise, import directly.
        mp_pose = checker.mp_pose if hasattr(checker, "mp_pose") else mp.solutions.pose
        mp_drawing.draw_landmarks(
            frame, 
            landmarks, 
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
            mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2)
        )
    return frame


manager = ConnectionManager()


@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id)
    try:
        while True:
            data = await websocket.receive_text()
            try:
                json_data = json.loads(data)
                if "pose_type" in json_data:
                    pose_type = json_data["pose_type"]
                    await manager.start_processing(client_id, pose_type)
                elif "command" in json_data and json_data["command"] == "stop":
                    manager.disconnect(client_id)
                    break
            except json.JSONDecodeError:
                pass
    except WebSocketDisconnect:
        manager.disconnect(client_id)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)