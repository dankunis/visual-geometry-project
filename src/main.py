#!/usr/bin/env python3.7.2
# vim: set ts=4 et:
# -*- indent-tabs-mode: t; tab-width: 4 -*-
#
# @brief   Visual Geometry project
# @details This application features the project/homework for
#          VO Visual Geometry. More information can be found in the report
# @author  Simon Rueba <simon.rueba@student.uibk.ac.at>
#          Daniel Kunis <daniil.kunis@student.uibk.ac>
#          Florian Maier <florian.Maier@student.uibk.ac>


from camera_calibration import *
from draw_cube_chessboard import *
from point_correspondences import *
from scene_reconstruction import *

from utils import *

from tqdm import tqdm

FEATURE_MATCHING_OUTPUT = "../resources/feature_matching_output/"
FRAMES_VIDEO_FRAMES_PATH = "../resources/frames/"
SIFT_OUTPUT = "../resources/sift/"
HCD_OUTPUT = "../resources/hcd/"
VIDEO_INPUT_PATH = "../resources/videos/boxes.MOV"
VIDEO_INPUT_FRAMES_PATH = "../resources/vid_to_img/"
VIDEO_OUTPUT_FRAMES_PATH = "../resources/img_with_drawings/"
VIDEO_OUTPUT_PATH = "../resources/output.avi"
CHESSBOARD_PATH = "../resources/chessboard/"
CALIBRATION_PATH = "../resources/calibration/calibration_horizontal.yaml"
MATCHED_KEYFRAMES_PATH = "../resources/tmp/matched_keyframes.pickle"
MATCHED_INTERMEDIATE_FRAMES_PATH = "../resources/tmp/matched_intermediate.pickle"
KEYFRAMES_IDX_PATH = "../resources/tmp/keyframes_idx.pickle"

CHESSBOARD_SIZE = (9, 6)

FPS = 60.0


def main():

    # Check if version is compatible
    check_version()

    # Create all necessary directories if the dont't exist yet
    create_directories(VIDEO_INPUT_FRAMES_PATH,
                       VIDEO_OUTPUT_FRAMES_PATH,
                       SIFT_OUTPUT,
                       HCD_OUTPUT,
                       FEATURE_MATCHING_OUTPUT,
                       os.path.dirname(CALIBRATION_PATH),
                       os.path.dirname(MATCHED_KEYFRAMES_PATH),
                       os.path.dirname(MATCHED_INTERMEDIATE_FRAMES_PATH),
                       os.path.dirname(KEYFRAMES_IDX_PATH))

    # Create termination criteria for camera calibration
    termination_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Check if calibration yaml file exists
    if not os.path.exists(CALIBRATION_PATH):
        calc_camera_calibration(CHESSBOARD_SIZE, termination_criteria, CHESSBOARD_PATH, CALIBRATION_PATH)

    # Get calibration paramaters (Camera matrix and distortion)
    K, distortion = get_camera_calibration(CALIBRATION_PATH)

    # increase the last arg to reduce number of frames and speed it up
    if not os.listdir(VIDEO_INPUT_FRAMES_PATH):
        video_to_frames(VIDEO_INPUT_PATH, VIDEO_INPUT_FRAMES_PATH, 6)

    # Get all frames sorted by name (number)
    all_frames = read_all_frames(VIDEO_INPUT_FRAMES_PATH, frame_transform=resize_to_dims)

    # Get keyframes, keypoints, intermediate_frames_matches by feature matching consecutive frames if they don't exist saved in a file yet
    if not os.path.exists(MATCHED_KEYFRAMES_PATH) \
            or not os.path.exists(MATCHED_INTERMEDIATE_FRAMES_PATH) \
            or not os.path.exists(KEYFRAMES_IDX_PATH):
        keyframes, keypoints, intermediate_frames_matches = feature_matching(all_frames, FEATURE_MATCHING_OUTPUT)

        points_2D = keypoints_coordinates(keypoints)
        save_object(keyframes, KEYFRAMES_IDX_PATH)
        save_object(points_2D, MATCHED_KEYFRAMES_PATH)
        save_object(intermediate_frames_matches, MATCHED_INTERMEDIATE_FRAMES_PATH)
    else:
        keyframes = read_object(KEYFRAMES_IDX_PATH)
        points_2D = read_object(MATCHED_KEYFRAMES_PATH)
        intermediate_frames_matches = read_object(MATCHED_INTERMEDIATE_FRAMES_PATH)

    # Caluclate keyframes from stereo reconstruction
    cameras = stereo_reconstruction(all_frames, keyframes, points_2D, intermediate_frames_matches, K, distortion)


    # Get keyframes from camera
    print("[INFO] : We have %s frames, %s of them are keyframes. The first keyframe is frame %s and the last keyframe is frame %s." % (len(all_frames), len(keyframes), keyframes[0], keyframes[-1]))
    cameraOfFirstKeyframe = cameras[keyframes[0]].P
    cameraOfLastKeyframe = cameras[keyframes[-1]].P

    # Calculate the corners in image coordinates and the length of the sides of the cube to inherit the other ones
    point5 = triangulate_points(cameraOfFirstKeyframe, np.asarray([np.asarray([794.2, 744.2])]), cameraOfLastKeyframe, np.asarray([np.asarray([683.2, 592.8])]), K, distortion) #0,0,-
    point1 = triangulate_points(cameraOfFirstKeyframe, np.asarray([np.asarray([814, 501.5])]), cameraOfLastKeyframe, np.asarray([np.asarray([949, 429.5])]), K, distortion) #0,0,0
    sideLen = point1[0][1] - point5[0][1]

    point2 = triangulate_points(cameraOfFirstKeyframe, np.asarray([np.asarray([1072.5, 500])]), cameraOfLastKeyframe, np.asarray([np.asarray([1141, 566.8])]), K, distortion)
    point3 = np.asarray([[point1[0][0] - sideLen, point1[0][1] + sideLen, point1[0][2] + sideLen*1.15]]) #-,+,0
    point4 = np.asarray([[point1[0][0], point1[0][1] + sideLen, point1[0][2] + sideLen*1.15]]) #0,+,0
    point6 = triangulate_points(cameraOfFirstKeyframe, np.asarray([np.asarray([1092.2, 741])]), cameraOfLastKeyframe, np.asarray([np.asarray([874.8, 757.2])]), K, distortion)
    point7 = np.asarray([[point1[0][0] - sideLen, point1[0][1], point1[0][2] + sideLen*2]]) #-,0,+
    point8 = np.asarray([[point1[0][0], point1[0][1], point1[0][2] + sideLen*2]]) #0,0,+


    # Pass projected Points to draw the wireframe cube in the image
    for i in tqdm(range(keyframes[-1]+1)):

        # Array to store cube corners
        cube = []

        # Transform points from 3D World coordinates to 2D image coordinates and add them to array
        point1_2D, _ = cv2.projectPoints(point1, cameras[i].R_vec(), cameras[i].t, K, distortion)
        point2_2D, _ = cv2.projectPoints(point2, cameras[i].R_vec(), cameras[i].t, K, distortion)
        point3_2D, _ = cv2.projectPoints(point3, cameras[i].R_vec(), cameras[i].t, K, distortion)
        point4_2D, _ = cv2.projectPoints(point4, cameras[i].R_vec(), cameras[i].t, K, distortion)
        point5_2D, _ = cv2.projectPoints(point5, cameras[i].R_vec(), cameras[i].t, K, distortion)
        point6_2D, _ = cv2.projectPoints(point6, cameras[i].R_vec(), cameras[i].t, K, distortion)
        point7_2D, _ = cv2.projectPoints(point7, cameras[i].R_vec(), cameras[i].t, K, distortion)
        point8_2D, _ = cv2.projectPoints(point8, cameras[i].R_vec(), cameras[i].t, K, distortion)
        cube.append(point1_2D)
        cube.append(point2_2D)
        cube.append(point3_2D)
        cube.append(point4_2D)
        cube.append(point5_2D)
        cube.append(point6_2D)
        cube.append(point7_2D)
        cube.append(point8_2D)

        # Get the image with a wireframe cube in it
        img = draw_wireframe_cube(all_frames[i], cube)

        # Save the image in directory for later to convert it to video
        cv2.imwrite(os.path.join(VIDEO_OUTPUT_FRAMES_PATH + "{:05d}.png".format(i)), img)

    # Convert all frames to generate a output video
    convert_frames_to_video(VIDEO_OUTPUT_FRAMES_PATH, VIDEO_OUTPUT_PATH, 15)



if __name__ == "__main__":
    main()
