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

VIDEO_PATH = "resources/cube.mp4" # TODO replace with real video path

def main():
    check_version()
    video = load_video(VIDEO_PATH)
    show_video(video)
    print(sys.version)


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


if __name__ == "__main__":
    main()
