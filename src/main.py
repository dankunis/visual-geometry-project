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
from scene_reconstruction import *

from utils import *

FEATURE_MATCHING_OUTPUT = "../resources/featureMatchingOutput/"
FRAMES_VIDEO_FRAMES_PATH = "../resources/frames/"
SIFT_OUTPUT = "../resources/sift/"
HCD_OUTPUT = "../resources/hcd/"
VIDEO_INPUT_PATH = "../resources/videos/boxes.MOV"
VIDEO_INPUT_FRAMES_PATH = "../resources/vid_to_img/"
VIDEO_OUTPUT_FRAMES_PATH = "../resources/img_with_drawings/"
VIDEO_OUTPUT_PATH = "../resources/output.avi"
CHESSBOARD_PATH = "../resources/chessboard/horizontal/"
CALIBRATION_PATH = "../resources/calibration/calibration_horizontal.yaml"
MATCHED_KEYFRAMES_PATH = "../resources/tmp/matched_keyframes.pickle"
MATCHED_INTERMEDIATE_FRAMES_PATH = "../resources/tmp/matched_intermediate.pickle"
KEYFRAMES_IDX_PATH = "../resources/tmp/keyframes_idx.pickle"

CHESSBOARD_SIZE = (9, 6)

FPS = 60.0


def main():
    check_version()
    create_directories(VIDEO_INPUT_FRAMES_PATH,
                       VIDEO_OUTPUT_FRAMES_PATH,
                       SIFT_OUTPUT,
                       HCD_OUTPUT,
                       FEATURE_MATCHING_OUTPUT,
                       os.path.dirname(CALIBRATION_PATH),
                       os.path.dirname(MATCHED_KEYFRAMES_PATH),
                       os.path.dirname(MATCHED_INTERMEDIATE_FRAMES_PATH),
                       os.path.dirname(KEYFRAMES_IDX_PATH))

    termination_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    if not os.path.exists(CALIBRATION_PATH):
        calc_camera_calibration(CHESSBOARD_SIZE, termination_criteria, CHESSBOARD_PATH, CALIBRATION_PATH)

    K, distortion = get_camera_calibration(CALIBRATION_PATH)
    # camera_params = tuple(K, distortions)

    #choose sift or harrison corner detection
    #get_SIFT_key_points(FRAMES_VIDEO_FRAMES_PATH, SIFT_OUTPUT)
    #get_key_points(FRAMES_VIDEO_FRAMES_PATH, HCD_OUTPUT)

    # increase the last arg to reduce number of frames and speed it up
    if not os.listdir(VIDEO_INPUT_FRAMES_PATH):
        video_to_frames(VIDEO_INPUT_PATH, VIDEO_INPUT_FRAMES_PATH, 6)

    all_frames = read_all_frames(VIDEO_INPUT_FRAMES_PATH, frame_transform=resize_to_dims)

    if not os.path.exists(MATCHED_KEYFRAMES_PATH) \
            or not os.path.exists(MATCHED_INTERMEDIATE_FRAMES_PATH) \
            or not os.path.exists(KEYFRAMES_IDX_PATH):
        keyframes, keypoints, intermediate_frames_matches = feature_matching(all_frames, FEATURE_MATCHING_OUTPUT)

        points_2d = keypoints_coordinates(keypoints)
        save_object(keyframes, KEYFRAMES_IDX_PATH)
        save_object(points_2d, MATCHED_KEYFRAMES_PATH)
        save_object(intermediate_frames_matches, MATCHED_INTERMEDIATE_FRAMES_PATH)
    else:
        keyframes = read_object(KEYFRAMES_IDX_PATH)
        points_2d = read_object(MATCHED_KEYFRAMES_PATH)
        intermediate_frames_matches = read_object(MATCHED_INTERMEDIATE_FRAMES_PATH)

    cameras = stereo_reconstruction(all_frames, keyframes, points_2d, intermediate_frames_matches, K, distortion)

    # TODO: bundle adjustment (after we learn how to draw)

    # TODO: after getting the cameras pick the first and last keyframe, manually choose same coordinates there
    # example, corners of the middle box and triangulate them (get 3d points). than iterate over all frames and
    # project these 3d points using the estimated cameras

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

    mtx_loaded = np.float32(loaded_dict.get('camera_matrix'))
    dist_loaded = np.float32(loaded_dict.get('dist_coeff'))

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


def create_directories(*args):
    for dir in args:
        os.makedirs(dir, exist_ok=True)


if __name__ == "__main__":
    main()
