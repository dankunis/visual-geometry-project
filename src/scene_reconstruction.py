import cv2
import numpy as np


class Camera:
    def __init__(self, R, t):
        self.R = R
        self.t = t

    def P(self):
        return np.c_[self.R, self.t]


def apply_mask(arr, mask):
    return np.reshape(arr[mask.nonzero()], (1, len(arr), 2))


def keypoints_coordinates(keypoints):
    kp = []
    for trace in keypoints:
        kp.append(np.asarray([x[0].pt for x in trace['2d_points']], dtype=np.float))
    return np.asarray(kp)


def get_projection_camera(pts1, pts2, K):
    E, mask = cv2.findEssentialMat(pts1, pts2, method=cv2.RANSAC, prob=0.999, threshold=0.1, cameraMatrix=K)

    pts1 = apply_mask(pts1, mask)
    pts2 = apply_mask(pts2, mask)

    pts1, pts2 = cv2.correctMatches(E, pts1, pts2)

    _, R, t, _ = cv2.recoverPose(E, pts1, pts2, K)

    return Camera(R, t)


def triangulate_points(P1, pts1, P2, pts2, K, distortion):
    pts1 = cv2.undistortPoints(np.expand_dims(pts1, axis=1).astype(dtype=np.float32), cameraMatrix=K,
                               distCoeffs=distortion)
    pts2 = cv2.undistortPoints(np.expand_dims(pts2, axis=1).astype(dtype=np.float32), cameraMatrix=K,
                               distCoeffs=distortion)

    pts_3d = cv2.triangulatePoints(P1, P2, pts1, pts2)

    # from homogeneous to normal coordinates
    pts_3d /= pts_3d[3]
    pts_3d = pts_3d[:-1]

    pts_3d = pts_3d.transpose()

    return pts_3d


def resection_camera(points_2d, points_3d, K, distorions):
    _, R, t, _ = cv2.solvePnPRansac(points_3d, points_2d, K, distorions, reprojectionError=2.0)
    return Camera(R, t)


def stereo_reconstruction(all_frames, keyframes, points_2d, intermediate_frames_matches, K, distortions):
    cameras = [None] * len(all_frames)

    points_3d = np.empty_like(points_2d, dtype=np.float)

    # initial reconstruction
    cameras[keyframes[0]] = Camera(np.identity(3), np.asarray([0, 0, 0], dtype=float))
    cameras[keyframes[-1]] = get_projection_camera(points_2d[:, 0], points_2d[:, -1], K)

    mid_idx = keyframes[int(len(keyframes) / 2)]
    points_3d[:, keyframes[mid_idx]] = triangulate_points(cameras[0].P(), points_2d[:, 0],
                                                          cameras[-1].P(), points_2d[:, -1],
                                                          K, distortions)

    cameras[keyframes[mid_idx]] = resection_camera(points_2d[:, keyframes[mid_idx]], points_3d, K, distortions)

    # resection other
    for i in range(1, len(keyframes) - 1):
        cameras[keyframes[i]] = resection_camera(points_2d[keyframes[i]], points_3d, K, distortions)

    # recover intermediate cameras
    prev_keyframe = 0
    for i in range(1, keyframes[-1]):
        if cameras[i] is not None:
            prev_keyframe += 1
            continue

        matches = intermediate_frames_matches[i]
        interm_points_3d = triangulate_points(cameras[keyframes[prev_keyframe]].P(), matches['prev_keyframe_pts'],
                                              cameras[keyframes[prev_keyframe + 1]].P(), matches['next_keyframe_pts'],
                                              K, distortions)

        cameras[i] = resection_camera(matches['curr_frame_pts'], interm_points_3d, K, distortions)

    return cameras
