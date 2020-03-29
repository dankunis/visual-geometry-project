#!/usr/bin/env python3.7.2
# vim: set ts=4 et:
# -*- indent-tabs-mode: t; tab-width: 4 -*-
#
# @brief   Visual Geometry project
# @details This application feautures the project/homework for
#          VO Visual Geoemtry
# @author  Simon Rueba <simon.rueba@student.uibk.ac.at>


import sys
import cv2
import numpy as np
import glob
import os

VIDEO_PATH = "resources/testvid.MOV" # TODO replace with real video path
CHESSBOARD_PATH = "resources/chessboard/"

CHESSBOARD_X = 9
CHESSBOARD_Y = 6

def main():
    check_version()
    video = load_video(VIDEO_PATH)
    show_video(video)
    chessboards()


def chessboards():

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((CHESSBOARD_Y * CHESSBOARD_X, 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHESSBOARD_X, 0:CHESSBOARD_Y].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    images = glob.glob(CHESSBOARD_PATH +  '*.JPG')
    counter = 0

    for fname in images:
        img = cv2.imread(fname)
        print(img.shape)
        img = cv2.resize(img, (0,0), fx=0.3, fy=0.3)
        print(img.shape)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        counter = counter + 1
        if(counter > 12):
            break

        print(counter)
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (CHESSBOARD_X, CHESSBOARD_Y), None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (CHESSBOARD_X, CHESSBOARD_Y), corners2, ret)
            cv2.imshow("test", img)
            cv2.waitKey(100)

    print("destroying")
    cv2.destroyAllWindows()
    print("calc")
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    print("idk")
    # refine the camera matrix based on a free scaling parameter. If the scaling parameter alpha=0,
    # it returns undistorted image with minimum unwanted pixels. So it may even remove some pixels at
    # image corners. If alpha=1, all pixels are retained with some extra black images.
    img = cv2.imread("resources/IMG_5983.JPG")
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    # undistort
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

    # crop the image, output saved as calibresult.png in the pics folder. All edges are straight.
    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]
    print(os.path.sep)
    cv2.imwrite("resources/calibresult.png", dst)

    # Now you can store the camera matrix and distortion coefficients using write functions in Numpy (np.savez, np.savetxt etc) for future uses.
    # Tutorial from https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html

    mean_error = 0
    for i in range(len(objpoints)):
        # transform the object point to image point
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error

    print("total error: ", mean_error / len(objpoints))


def show_video(cap):
    """Open a new windows and display the video

        Parameters
        ----------
        cap: video
            The video to be displayed

        Returns
        -------
        None
        """

    # Read until video is completed
    while (cap.isOpened()):

        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:

            # Display the resulting frame
            cv2.imshow('Frame', frame)

            # Get the frames per second
            length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print("FPS: " + str(length))

            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        # Break the loop
        else:
            break

    # When everything done, release
    # the video capture object
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()


def load_video(path):
    """Tries to load the demo video and returns it if successfull

        Parameters
        ----------
        path : str
            path to the video. Local in this project

        Returns
        -------
        video
            return the video if succesfull, otherwise return none and exit
        """
    video = cv2.VideoCapture(path)

    if(video.isOpened() == False):
        print("Error loading video file from path: " + path)
        sys.exit(-1)
    else:
        print("Succesfully loaded video from path: " + path)

    return video


def check_version():
    """Checks if current python version is compatible

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
    if ("3" not in sys.version):
        print("You are running python version " + sys.version + ". We recommend python 3.7.2 for expected performance.")

    print("Open CV version: " + cv2.__version__)


if __name__ == "__main__":
    main()
