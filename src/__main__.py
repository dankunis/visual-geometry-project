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
from os.path import isfile, join
import os
import yaml
from tqdm import tqdm

VIDEO_INPUT_PATH = "resources/videos/chessboard.MOV" # TODO replace with real video path
VIDEO_OUTPUT_PATH = "resources/vid_to_img/"
IMG_WITH_DRAWING_PATH = "resources/img_with_drawings/"
CHESSBOARD_PATH = "resources/chessboard/"
CALIBRATION_PATH = "resources/calibration/calibration.yaml"

CHESSBOARD_X = 9
CHESSBOARD_Y = 6

FPS = 60.0

def main():
    check_version()

    if (input("Do you want to calibrate the camera coefficients? This may take some time. (yes, no)") == "yes"):
        calc_camera_calibration()
    else:
        chessboards()

def calc_camera_calibration():
    print("[CALIBRATION] : Calculating camera calibration...")

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((CHESSBOARD_Y * CHESSBOARD_X, 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHESSBOARD_X, 0:CHESSBOARD_Y].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    counter = 0
    number_of_images = len(os.listdir(VIDEO_OUTPUT_PATH)) - 1
    with tqdm(total=number_of_images) as pbar:
        for fname in glob.glob(CHESSBOARD_PATH + '*.JPG'):
            counter = counter + 1
            if (counter == number_of_images):
                pbar.update(1)
                break

            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, (CHESSBOARD_X, CHESSBOARD_Y), None)

            # If found, add object points, image points (after refining them)
            if ret == True:
                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)

            pbar.update(1)

    cv2.destroyAllWindows()

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    print("[CALIBRATION] : Done with calculating. Saveing file in: " + CALIBRATION_PATH)
    # It's very important to transform the matrix to list.
    data = {'camera_matrix': np.asarray(mtx).tolist(), 'dist_coeff': np.asarray(dist).tolist()}
    with open(CALIBRATION_PATH, "w") as f:
        yaml.dump(data, f)

    print("[CALIBRATION] : File saved!")

def get_camera_calibration():
    print("[CALIBRATION] : Loading camera calibration file from: " + CALIBRATION_PATH)
    with open(CALIBRATION_PATH) as f:
        loadeddict = yaml.load(f, Loader=yaml.FullLoader)

    mtxloaded = loadeddict.get('camera_matrix')
    distloaded = loadeddict.get('dist_coeff')

    print("[CALIBRATION] : Done loading.")

    return mtxloaded, distloaded


def chessboards():

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((CHESSBOARD_Y * CHESSBOARD_X, 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHESSBOARD_X, 0:CHESSBOARD_Y].T.reshape(-1, 2)

    mtx, dist = get_camera_calibration()
    axis = get_axis("cube")

    video_to_frames(VIDEO_INPUT_PATH, VIDEO_OUTPUT_PATH)
    counter = 0

    print("[DRAWING] : Drawing shape in every frame.")
    for fname in tqdm(range(len(os.listdir(VIDEO_OUTPUT_PATH)) - 1)):
        counter = counter + 1
        img = cv2.imread(VIDEO_OUTPUT_PATH + str(counter) + '.png')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (CHESSBOARD_X, CHESSBOARD_Y), None)

        if ret == True:
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            _, rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, corners2, np.float32(mtx), np.float32(dist))
            imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, np.float32(mtx), np.float32(dist))
            img = draw_cube(img, corners2, imgpts)
            cv2.imwrite(IMG_WITH_DRAWING_PATH + "{:05d}.png".format(counter), img)

    print("[DRAWING] : Done.")
    convert_frames_to_video(IMG_WITH_DRAWING_PATH, "resources/output.avi", FPS)
    cv2.destroyAllWindows()


def convert_frames_to_video(pathIn,pathOut,fps):
    frame_array = []
    files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]

    #for sorting the file names properly
    files.sort(key = lambda x: x[:-3])

    print("[CONVERSION] : Converting " + str(len(files)) + " images to video.")

    for i in tqdm(range(len(files))):
        filename=pathIn + files[i]
        img = cv2.imread(filename)
        height, width, layers = img.shape
        frame_array.append(img)

    # fourcc = cv2.cv.CV_FOURCC(*'mp4v')
    #fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    videowriter = cv2.VideoWriter("output.avi", fourcc, 30.0, (width, height))

    for i in range(0, len(frame_array) - 1):
        videowriter.write(frame_array[i])
    videowriter.release()

    print("[CONVERSION] : Video saved in: " + pathOut)


def get_axis(desc):

    if(desc == 'cube'):
        return np.float32([[0, 0, 0], [0, 3, 0], [3, 3, 0], [3, 0, 0], [0, 0, -3], [0, 3, -3], [3, 3, -3], [3, 0, -3]])
    elif(desc == 'coord'):
        return np.float32([[3, 0, 0], [0, 3, 0], [0, 0, -3]]).reshape(-1, 3)
    else:
        print(desc + " is not valid, only cube or coord are possible.")


def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255, 0, 0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0, 255, 0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0, 0, 255), 5)
    return img


def draw_cube(img, corners, imgpts):
    imgpts = np.int32(imgpts).reshape(-1, 2)
    img = cv2.drawContours(img, [imgpts[:4]], -1, (0,255,0), -3)

    for i,j in zip(range(4), range(4,8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255),3)

    img = cv2.drawContours(img, [imgpts[4:]],-1,(0,0,255),3)

    return img


def video_to_frames(video, path_output_dir):
    # extract frames from a video and save to directory as 'x.png' where
    # x is the frame index
    vidcap = load_video(VIDEO_INPUT_PATH)
    count = 0

    while vidcap.isOpened():
        success, image = vidcap.read()
        if success:
            cv2.imwrite(os.path.join(path_output_dir, '%d.png') % count, image)
            count += 1
        else:
            break
    cv2.destroyAllWindows()
    vidcap.release()


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
        print("[VIDEO] : Succesfully loaded video from path: " + path)

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
