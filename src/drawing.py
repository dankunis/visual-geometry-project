#!/usr/bin/env python3.7.2
# vim: set ts=4 et:
# -*- indent-tabs-mode: t; tab-width: 4 -*-
#
# @brief   Wireframe cube drawing function
# @details All methods related to drawing the cube can be found here
# @author  Simon Rueba <simon.rueba@student.uibk.ac.at>
#          Daniel Kunis <daniil.kunis@student.uibk.ac>
#          Florian Maier <florian.Maier@student.uibk.ac>

import numpy as np
from utils import *


def draw_wireframe_cube(image, corners):
    """
    Draws a colored wireframe cube in the image at given positions
    :param image: Image on which to draw the cube
    :param corners: 8 corners of the cube in xy coordinates
    :return: Image with a cube drawn at position corners
    """
    corners = np.int32(corners).reshape(-1, 2)
    image = cv2.drawContours(image, [corners[:4]], -1, (0, 255, 0), -3)

    for i, j in zip(range(4), range(4, 8)):
        image = cv2.line(image, tuple(corners[i]), tuple(corners[j]), 255, 3)

    image = cv2.drawContours(image, [corners[4:]], -1, (0, 0, 255), 3)

    return image
