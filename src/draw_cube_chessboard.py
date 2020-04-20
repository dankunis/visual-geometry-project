from time import sleep

import numpy as np

from utils import *


def get_axis_points(object_to_draw):
    if object_to_draw == 'cube':
        return np.float32([[0, 0, 0], [0, 3, 0], [3, 3, 0], [3, 0, 0], [0, 0, -3], [0, 3, -3], [3, 3, -3], [3, 0, -3]])
    elif object_to_draw == 'coord':
        return np.float32([[3, 0, 0], [0, 3, 0], [0, 0, -3]]).reshape(-1, 3)
    else:
        print(object_to_draw + " is not valid, only cube or coord are possible.")


def draw_coord(img, corners, img_pts):
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(img_pts[0].ravel()), (255, 0, 0), 5)
    img = cv2.line(img, corner, tuple(img_pts[1].ravel()), (0, 255, 0), 5)
    img = cv2.line(img, corner, tuple(img_pts[2].ravel()), (0, 0, 255), 5)
    return img


def draw_cube(img, corners):
    corners = np.int32(corners).reshape(-1, 2)
    img = cv2.drawContours(img, [corners[:4]], -1, (0, 255, 0), -3)

    for i, j in zip(range(4), range(4, 8)):
        img = cv2.line(img, tuple(corners[i]), tuple(corners[j]), 255, 3)

    img = cv2.drawContours(img, [corners[4:]], -1, (0, 0, 255), 3)

    return img


def draw_cube_on_chessboard(chessboard_size,
                            termination_criteria,
                            camera_params,
                            input_video,
                            input_frames,
                            output_video_frames,
                            output_video,
                            fps):
    chessboard_x, chessboard_y = chessboard_size

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((chessboard_y * chessboard_x, 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_x, 0:chessboard_y].T.reshape(-1, 2)

    mtx, dist = camera_params
    axis = get_axis_points("cube")

    video_to_frames(input_video, input_frames)

    print("[DRAWING] : Drawing shape in every frame.")

    for counter in tqdm(range(len(os.listdir(input_frames)) - 1)):
        img = cv2.imread(os.path.join(input_frames, str(counter) + '.png'))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

        if ret:
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), termination_criteria)
            _, rvecs, tvecs, _ = cv2.solvePnPRansac(objp, corners, np.float32(mtx), np.float32(dist))
            imgpts, _ = cv2.projectPoints(axis, rvecs, tvecs, np.float32(mtx), np.float32(dist))
            img = draw_cube(img, imgpts)
            cv2.imwrite(os.path.join(output_video_frames + "{:05d}.png".format(counter)), img)

    # avoid corrupt output because of async tqdm
    sleep(0.001)
    print("[DRAWING] : Done.")

    convert_frames_to_video(output_video_frames, output_video, fps)
    cv2.destroyAllWindows()
