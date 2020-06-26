#!/usr/bin/env python3.7.2
# vim: set ts=4 et:
# -*- indent-tabs-mode: t; tab-width: 4 -*-
#
# @brief   Class camera, stereo functions
# @details All methods related to camera poses, world poses can be found here
# @author  Simon Rueba <simon.rueba@student.uibk.ac.at>
#          Daniel Kunis <daniil.kunis@student.uibk.ac>
#          Florian Maier <florian.Maier@student.uibk.ac>



import cv2
import numpy as np


class Camera:
    def __init__(self, R, t, K):
        if R.shape[1] == 1:
            self.R, _ = cv2.Rodrigues(R)
        else:
            self.R = R
        self.t = t
        self.P = np.c_[self.R, self.t]

    def R_vec(self):
        R_vec, _ = cv2.Rodrigues(self.R)
        return R_vec


def apply_mask(arr, mask):
    '''

    :param arr:
    :param mask:
    :return:
    '''
    return np.array([[x for (x, m) in zip(arr, mask) if m[0] == 1]])


def keypoints_coordinates(keypoints):
    '''

    :param keypoints:
    :return:
    '''

    kp = []
    for trace in keypoints:
        kp.append(np.float32([x[0].pt for x in trace['2d_points']]))
    return np.asarray(kp)


def get_projection_camera(pts1, pts2, K):
    '''

    :param pts1:
    :param pts2:
    :param K:
    :return:
    '''

    E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=0.1)

    pts1 = apply_mask(pts1, mask)
    pts2 = apply_mask(pts2, mask)

    pts1, pts2 = cv2.correctMatches(E, pts1, pts2)

    _, R, t, _ = cv2.recoverPose(E, pts1, pts2, K)

    return Camera(R, t, K)


def triangulate_points(P1, pts1, P2, pts2, K, distortion):
    '''

    :param P1:
    :param pts1:
    :param P2:
    :param pts2:
    :param K:
    :param distortion:
    :return:
    '''

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
    '''

    :param points_2d:
    :param points_3d:
    :param K:
    :param distorions:
    :return:
    '''
    _, R, t, _ = cv2.solvePnPRansac(points_3d, points_2d, K, distorions, reprojectionError=2.0)
    return Camera(R, t, K)


def stereo_reconstruction(all_frames, keyframes, points_2d, intermediate_frames_matches, K, distortions):
    '''

    :param all_frames:
    :param keyframes:
    :param points_2d:
    :param intermediate_frames_matches:
    :param K:
    :param distortions:
    :return:
    '''

    cameras = [None] * len(all_frames)

    # points_3d = np.empty((points_2d.shape[0], points_2d.shape[1], 3), dtype=np.float)

    # initial reconstruction
    cameras[keyframes[0]] = Camera(np.identity(3), np.asarray([0, 0, 0], dtype=float), K)
    cameras[keyframes[-1]] = get_projection_camera(points_2d[:, 0, :], points_2d[:, -1, :], K)

    mid_idx = int(len(keyframes) / 2)
    points_3d = triangulate_points(cameras[keyframes[0]].P, points_2d[:, 0, :],
                                   cameras[keyframes[-1]].P, points_2d[:, -1, :],
                                   K, distortions)

    cameras[keyframes[mid_idx]] = resection_camera(points_2d[:, mid_idx, :], points_3d, K, distortions)

    # resection other
    for i in range(1, len(keyframes) - 1):
        cameras[keyframes[i]] = resection_camera(points_2d[:, i, :], points_3d, K, distortions)

    # recover intermediate cameras
    prev_keyframe = 0
    for i in range(1, keyframes[-1]):
        if cameras[i] is not None:
            prev_keyframe += 1
            continue

        matches = intermediate_frames_matches[i]
        interm_points_3d = triangulate_points(cameras[keyframes[prev_keyframe]].P, matches['prev_keyframe_pts'],
                                              cameras[keyframes[prev_keyframe + 1]].P, matches['next_keyframe_pts'],
                                              K, distortions)

        cameras[i] = resection_camera(matches['curr_frame_pts'], interm_points_3d, K, distortions)

    return cameras
