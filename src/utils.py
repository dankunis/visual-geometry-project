#!/usr/bin/env python3.7.2
# vim: set ts=4 et:
# -*- indent-tabs-mode: t; tab-width: 4 -*-
#
# @brief   Various support functions for the project
# @details Some support functions e.g. resize, read, write files can be found here
# @author  Simon Rueba <simon.rueba@student.uibk.ac.at>
#          Daniel Kunis <daniil.kunis@student.uibk.ac>
#          Florian Maier <florian.Maier@student.uibk.ac>

import fnmatch
import os
import pickle
import sys
from os.path import isfile, join
import numpy as np
import yaml

import cv2
from tqdm import tqdm


def convert_frames_to_video(path_in, path_out, fps):
    """
    Converts a number of images to a .avi video
    :param path_in: Path from where to read the images
    :param path_out: Path where to store the .avi
    :param fps: Number of frames per second in the video
    :return: None
    """
    frame_array = []
    files = [f for f in os.listdir(path_in) if isfile(join(path_in, f))]

    # for sorting the file names properly
    files.sort(key=lambda x: x[:-3])

    print("[CONVERSION] : Converting " + str(len(files)) + " images to video.")

    width = height = 0

    for i in tqdm(range(len(files))):
        filename = path_in + files[i]
        img = cv2.imread(filename)
        height, width, _ = img.shape
        frame_array.append(img)

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    video_writer = cv2.VideoWriter(path_out, fourcc, fps, (width, height))

    for i in range(0, len(frame_array) - 1):
        video_writer.write(frame_array[i])
    video_writer.release()

    print("[CONVERSION] : Video saved in: " + path_out)


def get_camera_calibration(calibration_config_path):
    """
    Loads the camera calibration file if it already exists, else recalculate from the chessboard_images
    :param calibration_config_path: Path on where to read the calibration file from
    :return: calibration parameters, camera matrix and distortion coefficient
    """
    print("[CALIBRATION] : Loading camera calibration file from: " + calibration_config_path)
    with open(calibration_config_path) as f:
        loaded_dict = yaml.load(f, Loader=yaml.FullLoader)

    mtx_loaded = np.float32(loaded_dict.get('camera_matrix'))
    dist_loaded = np.float32(loaded_dict.get('dist_coeff'))

    print("[CALIBRATION] : Done loading.")

    return mtx_loaded, dist_loaded


def create_directories(*args):
    """
    Creates a directory if it does not exist yet
    :param args: Path/Name of the directory
    :return: None
    """
    for dir in args:
        os.makedirs(dir, exist_ok=True)


def check_version():
    """
    Checks if the python and OpenCV version are compatible and prints version as information
    :return: None
    """
    if "3" not in sys.version:
        print("[Info] : You are running python version " + sys.version
              + ". We recommend python 3.7.2 for expected performance.")
    else:
        print("[Info] : You are running python version " + sys.version + ".")

    if "3" not in cv2.__version__:
        print("[Info] : You are running cv2 version " + cv2.__version__
              + ". We recommend cv2 3.4.2 for expected performance.")
    else:
        print("[Info] : You are running cv2 version " + cv2.__version__ + ".")


def video_to_frames(video, path_output_dir, save_every_nth=1):
    """
    Extracts every nth frame from a video and saves is as a png
    :param video: Video from where to take the frames
    :param path_output_dir: Path on where to save the frames as .png
    :param save_every_nth: number that determines nth frame should be saved
    :return: None
    """
    # extract frames from a video and save to directory as 'x.png' where
    # x is the frame index
    vidcap = load_video(video)
    count = 0
    iteration = 0

    while vidcap.isOpened():
        success, image = vidcap.read()

        if iteration % save_every_nth != 0:
            iteration += 1
            continue

        if success:
            cv2.imwrite(os.path.join(path_output_dir, '%d.png') % count, image)
            count += 1
        else:
            break
        iteration += 1

    print(os.path.abspath(path_output_dir))
    cv2.destroyAllWindows()
    vidcap.release()


def read_all_frames(frames_path, frame_transform=lambda x: x):
    """
    Reads all frames from a given path and sorts them by name
    :param frames_path: Path from where to take the frames
    :param frame_transform: sorting method
    :return: returns sorted frames
    """
    file_names = sorted(fnmatch.filter(os.listdir(frames_path), '*.png'), key=lambda x: int(x[:-4]))
    frames = []

    for name in file_names:
        frames.append(frame_transform(cv2.imread(os.path.join(frames_path, name), 1)))

    return frames


def load_video(path):
    """
    Tries to load the demo video and returns it if successful
    :param path: path to the video. Local in this project
    :return: returns the video if successful, otherwise return none and exit
    """
    video = cv2.VideoCapture(path)

    if not video.isOpened():
        print("Error loading video file from path: " + path)
        sys.exit(-1)
    else:
        print("[VIDEO] : Successfully loaded video from path: " + path)

    return video


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    """
    Resizes the given image with cv2 to given width and height
    :param image: image to resize
    :param width: new width of the image
    :param height: new height of the image
    :param inter: Method on how to resize
    :return: returns the resized image
    """
    h, w = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    if width is None:
        ratio = height / float(h)
        dim = (int(w * ratio), height)
    else:
        ratio = width / float(w)
        dim = (width, int(h * ratio))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


def resize_to_dims(image, dims=(1920, 1080)):
    """
    Resize the image to dimensions
    :param image: image to resize
    :param dims: dimensions of new image
    :return: resized image
    """

    height, width = image.shape[:2]
    new_width = width
    if max(width, height) > max(dims):
        if height > width:
            new_width = dims[1]
        else:
            new_width = dims[0]

    return image_resize(image, width=new_width)


def save_object(obj, filepath):
    """
    Save data to a file
    :param obj: data to save
    :param filepath: path on where to save the file
    :return:
    """
    with open(filepath, "wb") as fp:
        pickle.dump(obj, fp)


def read_object(filepath):
    """
    Reads data from a file
    :param filepath: path on where to read the data from
    :return: read data
    """
    with open(filepath, "rb") as fp:
        return pickle.load(fp)
