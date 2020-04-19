import os
import sys
from os.path import isfile, join

import cv2
from tqdm import tqdm


def convert_frames_to_video(path_in, path_out, fps):
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


def video_to_frames(video, path_output_dir):
    # extract frames from a video and save to directory as 'x.png' where
    # x is the frame index
    vidcap = load_video(video)
    count = 0

    while vidcap.isOpened():
        success, image = vidcap.read()
        if success:
            cv2.imwrite(os.path.join(path_output_dir, '%d.png') % count, image)
            count += 1
        else:
            break
    print(os.path.abspath(path_output_dir))
    cv2.destroyAllWindows()
    vidcap.release()


def load_video(path):
    """Tries to load the demo video and returns it if successful

        Parameters
        ----------
        path : str
            path to the video. Local in this project

        Returns
        -------
        video
            return the video if successful, otherwise return none and exit
        """
    video = cv2.VideoCapture(path)

    if not video.isOpened():
        print("Error loading video file from path: " + path)
        sys.exit(-1)
    else:
        print("[VIDEO] : Successfully loaded video from path: " + path)

    return video


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
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
