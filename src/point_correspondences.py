#!/usr/bin/env python3.7.2
# vim: set ts=4 et:
# -*- indent-tabs-mode: t; tab-width: 4 -*-
#
# @brief   Point correspondences functios
# @details In this file all functions related to feature matching, matching keyframes and drawing them can be found
# @author  Simon Rueba <simon.rueba@student.uibk.ac.at>
#          Daniel Kunis <daniil.kunis@student.uibk.ac>
#          Florian Maier <florian.Maier@student.uibk.ac>

from statistics import mean

import cv2
import numpy as np
from matplotlib import pyplot as plt
from utils import *

MIN_MATCHES = 25


def feature_matching(input_frames, output_folder=None):
    '''
    Matches features between frames
    :param input_frames: directory from where to read frames
    :param output_folder: directory on where to store frames with matched keypoints drawn
    :return: keyframes, point_tracks, intermediate_frames_matches
    '''
    print("[FEATURE MATCHING] : start matching. This will take some time...")
    min_baseline_dist = max(input_frames[0].shape[:2]) / 17

    curr_keyframe = 0

    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    init_pts, init_des = sift.detectAndCompute(input_frames[curr_keyframe], None)

    keyframes = [curr_keyframe]

    # modified data structure from https://jivp-eurasipjournals.springeropen.com/track/pdf/10.1186/s13640-017-0168-3
    point_tracks = np.asarray([{
        '2d_points': [(pt, des)],
    } for pt, des in zip(init_pts, init_des)])

    intermediate_frames_matches = {}

    while curr_keyframe < len(input_frames) - 1:
        curr_pts = [trace['2d_points'][-1] for trace in point_tracks]
        kp1, des1 = zip(*curr_pts)

        intermidiate_matches = []

        # search for next keyframe
        next_frame = curr_keyframe + 1
        while next_frame < len(input_frames) - 1:
            kp2, des2 = sift.detectAndCompute(input_frames[next_frame], None)

            # match
            FLANN_INDEX_KDTREE = 0
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)

            flann = cv2.FlannBasedMatcher(index_params, search_params)
            all_matches = flann.knnMatch(np.asarray(des1), des2, k=2)

            # filter using Lowe's ratio test
            matches = [x[0] for x in list(filter(lambda x: x[0].distance < 0.7 * x[1].distance, all_matches))]

            # draw matches (optional)
            if output_folder is not None and len(matches) > 0:
                draw_matches(input_frames[curr_keyframe], kp1, input_frames[next_frame],
                             kp2, matches, (curr_keyframe, next_frame), output_folder)

            # check keyframe criteria (distant enough and doesn't cut off more than 50% of matches)
            if len(point_tracks) > 1000:
                drop_rate = 0.8
            elif len(point_tracks) > 200:
                drop_rate = 0.6
            else:
                drop_rate = 0.5

            is_normal_frame = 1 - len(matches) / len(point_tracks) < drop_rate
            mean_dist = mean([x.distance for x in matches])
            is_keyframe = mean_dist > min_baseline_dist and is_normal_frame and len(matches) >= MIN_MATCHES

            # edge case when no frames matched the criteria, but there were really good candidates
            if not is_keyframe and next_frame == len(input_frames) - 1:
                normal_intermediate_frames = [i for i, x in enumerate(intermidiate_matches) if x['is_normal']]
                mean_dists = [intermidiate_matches[idx]['mean_dist'] for idx in normal_intermediate_frames]
                best_candidate = mean_dists.index(max(mean_dists))
                if mean_dists[best_candidate] >= min_baseline_dist * 0.9:
                    is_keyframe = True
                    kp2 = intermidiate_matches[best_candidate]['kp']
                    des2 = intermidiate_matches[best_candidate]['des']
                    matches = intermidiate_matches[best_candidate]['matches']
                    next_frame = intermidiate_matches[best_candidate]['frame']
                    intermidiate_matches = intermidiate_matches[:best_candidate]

            if not is_keyframe:
                intermidiate_matches.append({
                    'frame': next_frame,
                    'matches': matches,
                    'mean_dist': mean_dist,
                    'kp': kp2,
                    'des': des2,
                    'is_normal': is_normal_frame
                })
                next_frame += 1
                continue

            # update existing point traces and save keyframe
            keyframes.append(next_frame)
            matched_kp1_idx = []
            for m in matches:
                point_tracks[m.queryIdx]['2d_points'].append((kp2[m.trainIdx], des2[m.trainIdx]))
                matched_kp1_idx.append(m.queryIdx)

            point_tracks = np.asarray([pt for pt in point_tracks if len(pt['2d_points']) == len(keyframes)])

            # match frames in-between the last two keyframes
            match_frames_between_keyframes(curr_keyframe, kp1, next_frame, kp2,
                                           intermidiate_matches, matches, intermediate_frames_matches)

            curr_keyframe = next_frame
            break

        if next_frame == len(input_frames) - 1:
            break

    print("[FEATURE MATCHING] : finished matching. {} keyframes found".format(len(keyframes)))
    return keyframes, point_tracks, intermediate_frames_matches


def match_frames_between_keyframes(prev_keyframe, kp1, next_keyframe, kp2, intermediate_matches,
                                   kf_matches, all_intermediate_frames_matches):
    '''
    Matches keypoints between keyframes
    :param prev_keyframe: Keyframe from where to take keypoints
    :param kp1: keypoints of frame 1
    :param next_keyframe: Keyrame2 from where to take keypoints
    :param kp2: keypoints2 of frame 2
    :param intermediate_matches: intermediate keyframe matches
    :param kf_matches: keyframe matches
    :param all_intermediate_frames_matches:
    :return:
    '''
    for i in range(1, next_keyframe - prev_keyframe):
        curr_frame = prev_keyframe + i

        # find common matches between the 3 frames
        common_matches = {}
        for m in intermediate_matches[i - 1]['matches']:
            common_matches[m.queryIdx] = {
                'curr_frame_match_idx': m.trainIdx
            }

        common_matches_with_prev = []
        common_matches_with_next = []
        curr_kp = []
        for m in kf_matches:
            if m.queryIdx not in common_matches.keys():
                continue
            common_matches_with_prev.append(kp1[m.queryIdx].pt)
            common_matches_with_next.append(kp2[m.trainIdx].pt)
            curr_frame_match_idx = common_matches[m.queryIdx]['curr_frame_match_idx']
            curr_kp.append(intermediate_matches[i - 1]['kp'][curr_frame_match_idx].pt)

        all_intermediate_frames_matches[curr_frame] = {
            'prev_keyframe_pts': np.asarray(common_matches_with_prev),
            'next_keyframe_pts': np.asarray(common_matches_with_next),
            'curr_frame_pts': np.asarray(curr_kp)
        }

    return all_intermediate_frames_matches


def draw_matches(img1, kp1, img2, kp2, matches, pos, output_folder):
    '''
        Draws green colored lines between to frames based on detected keypoints
        :param img1: Image 1 from where to take keypoints
        :param kp1: keypoints of image one
        :param img2: Image 2 from where to take keypoints
        :param kp2: keypoints of image two
        :param matches: List of matches
        :param pos: tuple of current keyframe and  next_frame
        :param output_folder: path on where to save the result
        :return: None
        '''
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()

    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       singlePointColor=None,
                       matchesMask=matchesMask,  # draw only inliers
                       flags=2)

    img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, **draw_params)

    plt.imshow(img3, 'gray'), plt.savefig(output_folder + "img%s+%s" % pos)