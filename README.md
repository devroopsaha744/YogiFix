# YogiFix

YogiFix is a  **real-time** Yoga Pose detection and feedback system built using **Python**, **OpenCV**, and **Mediapipe** for pose estimation. The system is served via a **FastAPI** backend that captures webcam frames server-side, processes them to detect poses, and provides real-time feedback over WebSocket connections.

> **Important:** This version relies on **server-side webcam access** (using `cv2.VideoCapture(0)`). It must be deployed on hardware with an attached webcam (e.g., a local machine or a dedicated server/VPS with USB passthrough). Cloud platforms like Render or similar PaaS do not provide direct hardware access.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Directory Structure](#directory-structure)
- [How It Works](#how-it-works)
- [API Endpoints](#api-endpoints)
- [Running Locally](#running-locally)
- [Performance & Concurrency Considerations](#performance--concurrency-considerations)
- [Future Enhancements](#future-enhancements)
- [License](#license)

---

## Overview

This module provides real-time feedback for yoga poses by:
- Capturing webcam video directly on the server.
- Processing each frame with a pose detection algorithm.
- Comparing the user's pose to predefined “ideal” poses.
- Returning annotated frames and detailed feedback over WebSockets.

The system is designed for scenarios where server-side processing is viable (e.g., dedicated hardware) and offers low-latency feedback for enhanced user interaction.

---

## Features

- **Real-Time Processing:** Captures and processes frames from a physical webcam attached to the server.
- **Multiple Pose Detection:** Supports a variety of yoga poses, including:
  - T Pose
  - Tree Pose
  - Warrior 3 Pose
  - Bridge Pose
  - Cat Pose
  - Cobra Pose
  - Crescent Lunge Pose
  - Downward Facing Dog Pose
  - Leg-Up-The-Wall Pose
  - Mountain Pose
  - Padmasana (Lotus Pose)
  - Pigeon Pose
  - Seated Forward Bend
  - Standing Forward Bend
  - Triangle Pose
  - Warrior Pose
- **Detailed Feedback:** Computes similarity scores based on joint angles and generates corrective feedback.
- **WebSocket Communication:** Uses FastAPI’s asynchronous WebSocket support for real-time bi-directional communication.
- **CORS Enabled:** Easily integrates with separate front-end applications.

---

## Tech Stack

- **Programming Language:** Python 3.x
- **Backend Framework:** FastAPI
- **WebSocket Server:** Uvicorn (ASGI server)
- **Computer Vision:** OpenCV
- **Pose Estimation:** Mediapipe
- **Data Processing:** NumPy
- **Asynchronous Programming:** asyncio

---

## Directory Structure

```
YogaModule/
├─ api/
│  └─ main.py               # FastAPI application with server-side webcam processing
├─ logic/
│  ├─ __init__.py
│  ├─ T_pose.py           # T Pose detection logic
│  ├─ traingle_pose.py    # Triangle Pose detection logic
│  ├─ Tree_pose.py        # Tree Pose detection logic
│  ├─ Crescent_lunge_pose.py  # Crescent Lunge detection logic
│  ├─ warrior_pose.py     # Warrior Pose detection logic
│  └─ mountain_pose.py    # Mountain Pose detection logic
├─ tests/
|    └─index.htm          # Client side code to test the API
└─ README.md              # This README file
```

- **`web-app/app.py`**: Contains the FastAPI backend which captures frames from a server-side webcam, processes them, and sends back annotated frames and feedback.
- **`logic/`**: Contains the pose checker classes that perform frame processing, angle calculations, and generate feedback.

---

## How It Works

1. **Webcam Capture:**  
   The API opens a connection to a physical webcam using `cv2.VideoCapture(0)`.

2. **Frame Processing:**  
   - Each frame is read, flipped for a mirror view, and passed to a selected pose checker.
   - The pose checker uses Mediapipe to extract landmarks and compute joint angles.
   - A similarity score is calculated by comparing the user's pose with the ideal pose.
   - Annotated frames are generated by drawing landmarks using Mediapipe’s drawing utilities.

3. **WebSocket Communication:**  
   - The processed frame (encoded as a JPEG and then base64) and the feedback (similarity score, joint details, textual corrections) are sent back to the client via a WebSocket connection.
   - A connection manager handles multiple clients and processing tasks concurrently.

---

## API Endpoints

### WebSocket Endpoint

- **URL:** `/ws/{client_id}`
- **Method:** WebSocket
- **Description:**  
  When a client connects and sends a JSON message containing a `"pose_type"`, the API starts processing frames from the server-side webcam. It continuously sends back a JSON response containing:
  - **`frame`**: Base64-encoded annotated JPEG image.
  - **`feedback`**: An object with:
    - `similarity`: A float value representing overall pose similarity.
    - `feedback_text`: A textual description of the feedback.
    - `joint_similarities`: Detailed feedback per joint (if applicable).

- **Stop Command:**  
  Clients can send `{"command": "stop"}` to disconnect and stop processing.

### Health Check Endpoint

- **URL:** `/health`
- **Method:** GET
- **Description:**  
  Returns a JSON response indicating the server status.
  ```json
  {
    "status": "healthy"
  }
  ```

---

## Running Locally

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/YogaModule.git
   cd YogaModule
   ```

2. **Set Up a Virtual Environment (Optional but Recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # For Linux/Mac
   # or venv\Scripts\activate   # For Windows
   ```

3. **Install Dependencies:**
   ```bash
   pip install fastapi uvicorn opencv-python mediapipe numpy
   ```
   > If a `requirements.txt` is available, run:  
   > `pip install -r requirements.txt`

4. **Run the FastAPI Server:**
   ```bash
   cd web-app
   uvicorn app:app --reload --host 0.0.0.0 --port 8000
   ```
   The server will start at **http://localhost:8000**.

5. **Connect a WebSocket Client:**
   Use a WebSocket client or a browser-based front-end to connect to `ws://localhost:8000/ws/{client_id}` and send JSON messages as described.

> **Note:** Ensure that the machine running the server has a webcam attached. If `cv2.VideoCapture(0)` fails, verify the webcam index or hardware permissions.

---

## Performance & Concurrency Considerations

- **CPU-Intensive Processing:**  
  Frame processing (especially with OpenCV and Mediapipe) is CPU-bound. For multiple concurrent connections, consider:
  - Offloading heavy computations to separate worker threads or processes.
  - Horizontal scaling (running multiple instances) if using dedicated hardware.

- **Vertical Scaling:**  
  Since server-side webcam processing avoids network transmission delays and base64 overhead from the client, it can offer faster processing. However, vertical scaling (upgrading CPU/RAM) is crucial if many clients connect concurrently.

- **Hardware Constraints:**  
  This approach requires a physical webcam. In cloud environments, server-side webcam access is typically not available, so this setup is best suited for dedicated hardware or on-premise servers.

---

## Demo Videos
Demo of the project can be seen in this [playlist](https://www.youtube.com/playlist?list=PLevupJ4B1q4Mn3YHLnyD1Q8_3HtMfyegf)

## Future Enhancements

- **GPU Acceleration:**  
  Integrate CUDA/TensorRT to speed up pose estimation on GPUs.

- **Asynchronous Processing:**  
  Use thread pools or asynchronous libraries to better handle CPU-bound tasks without blocking the event loop.

- **Client-Side Integration:**  
  Develop a web or mobile front-end that dynamically connects via WebSockets for real-time feedback.

- **Support for Multiple Cameras:**  
  Extend the module to support multiple simultaneous camera inputs or multiple users.

---

## License

This project is licensed under the Apache 2.0 License. See the [LICENSE](LICENSE) file for details.
