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
                            fps, corners):

    img = cv2.imread(os.path.join(input_frames, str(counter) + '.png'))
    img = draw_cube(img, corners)
    cv2.imwrite(os.path.join(output_video_frames + "{:05d}.png".format(counter)), img)

    # avoid corrupt output because of async tqdm
    sleep(0.001)
    print("[DRAWING] : Done.")

    convert_frames_to_video(output_video_frames, output_video, fps)
    cv2.destroyAllWindows()
