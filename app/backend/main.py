from fastapi import FastAPI, WebSocket
import cv2
import numpy as np
import base64
from cvzone.PoseModule import PoseDetector

app = FastAPI()

detector = PoseDetector()

@app.websocket("/ws/pose")
async def pose_ws(websocket: WebSocket):
    await websocket.accept()

    while True:
        data = await websocket.receive_text()

        # base64 to image
        img_bytes = base64.b64decode(data)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            continue

        # pose detection
        frame = detector.findPose(frame, draw=False)
        lmList, bboxInfo = detector.findPosition(frame, draw=False)

        landmarks = []

        if lmList:
            for i, lm in enumerate(lmList):

                # cvzone real output: [x, y, z]
                x, y, z = lm

                landmarks.append({
                    "id": i,
                    "x": float(x),
                    "y": float(y),
                    "z": float(z)
                })

        await websocket.send_json({
            "landmarks": landmarks
        })
