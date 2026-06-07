import numpy as np


def load_calibration(path):
    data = np.load(path)
    K = data["K"]
    R = data["R"]
    t = data["t"].reshape(3, 1)

    P = K @ np.hstack([R, t])
    return P


def triangulate_point(points_2d, projection_matrices):


    A = []

    for (x, y), P in zip(points_2d, projection_matrices):
        if np.isnan(x) or np.isnan(y):
            continue

        A.append(x * P[2] - P[0])
        A.append(y * P[2] - P[1])

    if len(A) < 4:
        return np.array([np.nan, np.nan, np.nan])

    A = np.array(A)

    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    X = X / X[3]

    return X[:3]


def triangulate_frame(keypoints_2d, projection_matrices):


    num_keypoints = keypoints_2d.shape[1]

    keypoints_3d = []

    for k in range(num_keypoints):
        points_k = keypoints_2d[:, k, :]
        point_3d = triangulate_point(points_k, projection_matrices)
        keypoints_3d.append(point_3d)

    return np.array(keypoints_3d)


def triangulate_sequence(all_keypoints_2d, projection_matrices):


    trajectories_3d = []

    for f in range(all_keypoints_2d.shape[0]):
        frame_3d = triangulate_frame(
            all_keypoints_2d[f],
            projection_matrices
        )

        trajectories_3d.append(frame_3d)

    return np.array(trajectories_3d)


calibration_paths = [
    "calibration/cam0.npz",
    "calibration/cam1.npz",
    "calibration/cam2.npz",
    "calibration/cam3.npz",
]

projection_matrices = [
    load_calibration(path)
    for path in calibration_paths
]

all_keypoints_2d = np.load("keypoints_2d.npy")


trajectories_3d = triangulate_sequence(
    all_keypoints_2d,
    projection_matrices
)

np.save("trajectories_3d.npy", trajectories_3d)

print(trajectories_3d.shape)