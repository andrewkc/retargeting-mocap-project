import cv2
import c3d 
import numpy as np
from cvzone.PoseModule import PoseDetector

POSE_CONNECTIONS = [
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (11, 23), (12, 24),
    (23, 24),
    (23, 25), (25, 27),
    (24, 26), (26, 28),
]

VIDEO_PATH = r"C:\projects\retargeting-mocap-project\data\test-video.mp4"

TXT_OUTPUT = r"C:\projects\retargeting-mocap-project\outputs\others\AnimationFile.txt"

NPY_OUTPUT = r"C:\projects\retargeting-mocap-project\outputs\others\landmarks.npy"

C3D_OUTPUT = r"C:\projects\retargeting-mocap-project\outputs\others\landmarks.c3d"

#####################################################################################

def draw_skeleton(img, lmList):

    for a, b in POSE_CONNECTIONS:

        xa, ya, za = lmList[a]
        xb, yb, zb = lmList[b]

        cv2.line(
            img,
            (xa, ya),
            (xb, yb),
            (0, 255, 0),
            2
        )
        
def extract_frame_landmarks(lmList, image_height):

    frame_points = []

    for lm in lmList:

        x = lm[0]

        y = image_height - lm[1]

        z = lm[2]

        frame_points.append(
            [x, y, z]
        )

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
    

def process_video(video_path):

    cap = cv2.VideoCapture(video_path)

    detector = PoseDetector()

    all_frames = []

    while True:

        success, img = cap.read()

        if not success:
            break

        img = detector.findPose(
            img,
            draw=True
        )

        lmList, bboxInfo = detector.findPosition(
            img,
            draw=True
        )

        if bboxInfo:

            draw_skeleton(
                img,
                lmList
            )

            frame_points = extract_frame_landmarks(
                lmList,
                img.shape[0]
            )

            all_frames.append(
                frame_points
            )

        cv2.imshow(
            "Image",
            img
        )

        key = cv2.waitKey(1)

        if key == 27:
            break

    cap.release()

    cv2.destroyAllWindows()

    return all_frames

if __name__ == "__main__":

    points = process_video(
        VIDEO_PATH
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