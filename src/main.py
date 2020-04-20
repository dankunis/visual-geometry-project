#!/usr/bin/env python3.7.2
# vim: set ts=4 et:
# -*- indent-tabs-mode: t; tab-width: 4 -*-
#
# @brief   Visual Geometry project
# @details This application feautures the project/homework for
#          VO Visual Geoemtry
# @author  Simon Rueba <simon.rueba@student.uibk.ac.at>


from camera_calibration import *
from draw_cube_chessboard import *
from point_correspondences import *

FRAMES_VIDEO_FRAMES_PATH = "../resources/frames/"
HCD_OUTPUT = "../resources/hcd/"
VIDEO_INPUT_PATH = "../resources/videos/chessboard.MOV"  # TODO replace with real video path
VIDEO_INPUT_FRAMES_PATH = "../resources/vid_to_img/"
VIDEO_OUTPUT_FRAMES_PATH = "../resources/img_with_drawings/"
VIDEO_OUTPUT_PATH = "../resources/output.avi"
CHESSBOARD_PATH = "../resources/chessboard/horizontal/"
CALIBRATION_PATH = "../resources/calibration/calibration_horizontal.yaml"

CHESSBOARD_SIZE = (9, 6)

FPS = 60.0


def main():
    check_version()

    os.makedirs(VIDEO_INPUT_FRAMES_PATH, exist_ok=True)
    os.makedirs(VIDEO_OUTPUT_FRAMES_PATH, exist_ok=True)

    termination_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    if not os.path.exists(CALIBRATION_PATH):
        calc_camera_calibration(CHESSBOARD_SIZE, termination_criteria, CHESSBOARD_PATH, CALIBRATION_PATH)

    camera_params = tuple(get_camera_calibration(CALIBRATION_PATH))

    get_key_points(FRAMES_VIDEO_FRAMES_PATH, HCD_OUTPUT)

    #draw_cube_on_chessboard(CHESSBOARD_SIZE,
    #                        termination_criteria,
    #                        camera_params,
    #                        VIDEO_INPUT_PATH,
    #                        VIDEO_INPUT_FRAMES_PATH,
    #                        VIDEO_OUTPUT_FRAMES_PATH,
    #                        VIDEO_OUTPUT_PATH,
    #                        FPS)


def get_camera_calibration(calibration_config_path):
    print("[CALIBRATION] : Loading camera calibration file from: " + calibration_config_path)
    with open(calibration_config_path) as f:
        loaded_dict = yaml.load(f, Loader=yaml.FullLoader)

    mtx_loaded = loaded_dict.get('camera_matrix')
    dist_loaded = loaded_dict.get('dist_coeff')

    print("[CALIBRATION] : Done loading.")

    return mtx_loaded, dist_loaded


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
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret:
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


def check_version():
    """Checks if current python version is compatible

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
    if "3" not in sys.version:
        print("You are running python version " + sys.version + ". We recommend python 3.7.2 for expected performance.")

    print("Open CV version: " + cv2.__version__)


if __name__ == "__main__":
    main()
