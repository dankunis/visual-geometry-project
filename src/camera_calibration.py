#!/usr/bin/env python3.7.2
# vim: set ts=4 et:
# -*- indent-tabs-mode: t; tab-width: 4 -*-
#
# @brief   Camera Calibration function
# @details In this function the camera calibration will be calculated from a given chessboard_images
# @author  Simon Rueba <simon.rueba@student.uibk.ac.at>
#          Daniel Kunis <daniil.kunis@student.uibk.ac>
#          Florian Maier <florian.Maier@student.uibk.ac>

import glob
import os

import cv2
import numpy as np
import yaml
from tqdm import tqdm

from utils import image_resize


def calc_camera_calibration(chessboard_size, termination_criteria, calibration_img_path, calibration_config_path):
    """
    Calculates the camera calibration from a given chessboard_images
    :param chessboard_size: Size of the chessboard_images
    :param termination_criteria: number of iterations and/or the accuracy
    :param calibration_img_path: Path to the chessboard_images
    :param calibration_config_path: Path on where to store the calibration results
    :return: None
    """
    print("[CALIBRATION] : Calculating camera calibration...")

    chessboard_x, chessboard_y = chessboard_size

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((chessboard_y * chessboard_x, 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_x, 0:chessboard_y].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    obj_points = []  # 3d point in real world space
    img_points = []  # 2d points in image plane.

    counter = 0
    number_of_images = len(os.listdir(calibration_img_path)) - 1
    with tqdm(total=number_of_images) as pbar:
        for fname in glob.glob(os.path.join(calibration_img_path, '*')):
            counter += 1
            pbar.update(1)

            if counter == number_of_images:
                break

            img = cv2.imread(fname)

            # resize the image so it would be no bigger than 1920x1080
            height, width = img.shape[:2]
            if max(width, height) > 2000:
                if height > width:
                    new_width = 1080
                else:
                    new_width = 1920

                img = image_resize(img, width=new_width)

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, (chessboard_x, chessboard_y), None)

            # If found, add object points, image points (after refining them)
            if ret:
                obj_points.append(objp)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), termination_criteria)
                img_points.append(corners2)

    cv2.destroyAllWindows()

    _, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)

    reprojection_error = 0
    for i in range(len(obj_points)):
        imgpoints2, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(img_points[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        reprojection_error += error

    reprojection_error /= len(obj_points)

    print("[CALIBRATION] : Done with calculating. Saving file in: " + calibration_config_path)

    # It's very important to transform the matrix to list.
    data = {
        'camera_matrix': np.asarray(mtx).tolist(),
        'dist_coeff': np.asarray(dist).tolist(),
        'reprojection_error': reprojection_error
    }
    with open(calibration_config_path, "w") as f:
        yaml.dump(data, f)

    print("[CALIBRATION] : File saved!")
