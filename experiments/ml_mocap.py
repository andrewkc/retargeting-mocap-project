import cv2
import c3d
import json
import os
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp

POSE_CONNECTIONS = [
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (11, 23), (12, 24),
    (23, 24),
    (23, 25), (25, 27),
    (24, 26), (26, 28),
]

VIDEO_PATH = r"C:\projects\retargeting-mocap-project\data\test-video-2.mp4"

MODEL_PATH = r"C:\projects\retargeting-mocap-project\models\pose_landmarker_full.task"

TXT_OUTPUT = r"C:\projects\retargeting-mocap-project\outputs\others\AnimationFile.txt"

NPY_OUTPUT = r"C:\projects\retargeting-mocap-project\outputs\others\landmarks.npy"

C3D_OUTPUT = r"C:\projects\retargeting-mocap-project\outputs\others\landmarks.c3d"

VIDEO_OUTPUT = r"C:\projects\retargeting-mocap-project\outputs\others\pose_overlay.mp4"

JSON_OUTPUT = r"C:\projects\retargeting-mocap-project\outputs\others\landmarks_sync.json"

#####################################################################################

def save_json_sync(points, output_path, fps=30):

    data = {
        "fps": fps,
        "num_frames": len(points),
        "frames": []
    }

    for i, frame in enumerate(points):

        data["frames"].append({
            "frame_id": i,
            "landmarks": frame
        })

    with open(output_path, "w") as f:
        json.dump(data, f)

    print("JSON saved:", output_path)
    print("frames:", len(points))


def draw_skeleton(img, landmarks):

    h, w, _ = img.shape

    for a, b in POSE_CONNECTIONS:

        xa = int(landmarks[a].x * w)
        ya = int(landmarks[a].y * h)

        xb = int(landmarks[b].x * w)
        yb = int(landmarks[b].y * h)

        cv2.line(
            img,
            (xa, ya),
            (xb, yb),
            (0, 255, 0),
            2
        )

def extract_frame_landmarks(world_landmarks):

    frame_points = []

    for lm in world_landmarks:

        x = lm.x * 1000
        y = lm.y * 1000
        z = lm.z * 1000

        frame_points.append([
            x,
            z,
            -y
        ])

    return frame_points


def save_txt(points, output_path):

    with open(output_path, "w") as f:

        for frame in points:

            line = ""

            for x, y, z in frame:

                line += f"{x},{y},{z},"

            f.write(line + "\n")
            
def save_npy(points, output_path):

    points = np.array(points)

    np.save(
        output_path,
        points
    )

    print(
        "NPY saved:",
        output_path
    )

    print(
        "shape:",
        points.shape
    )

def save_c3d(points, output_path):

    writer = c3d.Writer()

    for frame in points:

        frame = np.array(frame)

        c3d_points = np.zeros(
            (frame.shape[0], 5)
        )

        c3d_points[:, :3] = frame

        writer.add_frames(
            [(c3d_points, np.empty((0,)))]
        )

    with open(output_path, "wb") as h:

        writer.write(h)

    print(
        "C3D saved:",
        output_path
    )

def process_video(video_path, output_video_path=None):

    cap = cv2.VideoCapture(video_path)

    base_options = python.BaseOptions(
        model_asset_path=MODEL_PATH
    )

    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO
    )

    landmarker = vision.PoseLandmarker.create_from_options(
        options
    )

    all_frames = []

    # ===== VideoWriter =====
    writer = None

    if output_video_path is not None:

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        writer = cv2.VideoWriter(
            output_video_path,
            fourcc,
            fps,
            (width, height)
        )

    while True:

        success, img = cap.read()

        if not success:
            break

        rgb = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        timestamp_ms = int(
            cap.get(cv2.CAP_PROP_POS_MSEC)
        )

        result = landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )

        if len(result.pose_world_landmarks) > 0:

            draw_skeleton(
                img,
                result.pose_landmarks[0]
            )

            frame_points = extract_frame_landmarks(
                result.pose_world_landmarks[0]
            )

            all_frames.append(frame_points)
        
        # ====================
        if writer is not None:
            writer.write(img)

        cv2.imshow("Image", img)

        key = cv2.waitKey(1)

        if key == 27:
            break

    cap.release()

    if writer is not None:
        writer.release()

    cv2.destroyAllWindows()

    return all_frames


if __name__ == "__main__":

    points = process_video(
        VIDEO_PATH,
        VIDEO_OUTPUT
    )
    
    print(points[0][0])
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30
    cap.release()

    save_json_sync(
        points,
        JSON_OUTPUT,
        fps=fps
    )

    save_txt(
        points,
        TXT_OUTPUT
    )

    save_npy(
        points,
        NPY_OUTPUT
    )

    save_c3d(
        points,
        C3D_OUTPUT
    )

    print(
        "Processed frames",
        len(points)
    )