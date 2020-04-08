import glob
import os

import cv2
import numpy as np
import yaml
from tqdm import tqdm


def calc_camera_calibration(chessboard_size, termination_criteria, calibration_img_path, calibration_config_path):
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
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, (chessboard_x, chessboard_y), None)

            # If found, add object points, image points (after refining them)
            if ret:
                obj_points.append(objp)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), termination_criteria)
                img_points.append(corners2)

    cv2.destroyAllWindows()

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)

    print("[CALIBRATION] : Done with calculating. Saving file in: " + calibration_config_path)
    # It's very important to transform the matrix to list.
    data = {'camera_matrix': np.asarray(mtx).tolist(), 'dist_coeff': np.asarray(dist).tolist()}
    with open(calibration_config_path, "w") as f:
        yaml.dump(data, f)

    print("[CALIBRATION] : File saved!")
