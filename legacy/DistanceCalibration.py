""" DistanceCalibration.py

    This file contains all the functions that are used to calibrate the distance
    stimuli.

"""

import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import pandas as pd


def get_contour(img, threshold, approx=False, single_region=False, show=False):
    """ This function takes an image and returns the contours of the image.
    """

    # Convert the image to greyscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply a threshold to the image
    _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    # Approx will find the contour with less points
    if approx:
        method = cv2.CHAIN_APPROX_SIMPLE
    else:
        method = cv2.CHAIN_APPROX_NONE

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, method)

    # check area of contours
    cnt = []
    for i in range(len(contours)):
        if 0.002 * img.shape[0] * img.shape[1] < cv2.contourArea(contours[i]) < 0.1 * img.shape[0] * img.shape[1]:
            cnt.append(contours[i])

    pts_region = []
    for single_cnt in cnt:
        single_cnt = np.squeeze(single_cnt, axis=1)
        pts_region.append(single_cnt)

    if single_region and len(pts_region) > 1:
        pts_region = clean_contour(pts_region)
        # pts_region = get_points(pts_region)

    elif single_region and len(pts_region) == 1:
        pts_region = get_points(pts_region)

    if show:
        img_to_show = img.copy()
        cv2.drawContours(img_to_show, cnt, -1, (0, 0, 0), 3)
        cv2.imshow("Contours", img_to_show)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if single_region:

        return np.array(pts_region)
    else:
        return pts_region


def clean_contour(pts_region, disk_size=200):
    # calculate x, y of the center of the point for every region
    mx = []
    my = []
    for pts in pts_region:
        m = cv2.moments(pts)
        mx.append(int(m["m10"] / m["m00"]))
        my.append(int(m["m01"] / m["m00"]))

    # get the one with lower y
    my_sorted = np.sort(my)
    idxs_sorted = np.argsort(my)

    if len(my) > 1 and (my_sorted[1] - my_sorted[0]) < disk_size:

        idxs_low = idxs_sorted[:2]
        pts = []
        pts_x = []
        pts_y = []

        for index in idxs_low:
            pts_single_region = pts_region[index]
            for points in pts_single_region:
                pts_x.append(points[0])
                pts_y.append(points[1])

        pts.append(np.array([pts_x, pts_y]).T)

        return pts[0]

    elif len(my) > 1 and (my_sorted[1] - my_sorted[0]) > disk_size:

        idxs_low = idxs_sorted[0]
        pts = pts_region[idxs_low]

        return pts

    if len(my) == 1:
        return pts_region


def get_points(pts_region):
    """ This function takes the contours of an image and returns the points of the
        contours.
    """

    pts = []
    pts_x = []
    pts_y = []

    for region in pts_region:
        for points in region:
            pts_x.append(points[0])
            pts_y.append(points[1])
    pts.append(np.array([pts_x, pts_y]).T)

    return pts[0]


def get_axis_line(pts, axis_name='x', show=False):
    """ This function takes the points of a region and returns the line that
        best fits the points.
    """

    if axis_name == 'y':
        # invert the axis of pts
        pts = np.array([pts[:, 1], pts[:, 0]]).T

    most_common = np.median(pts[:, 1])

    close_pts_to_axis = pts[np.logical_and(pts[:, 1] > most_common - 20, pts[:, 1] < most_common + 20)]

    p = np.polyfit(close_pts_to_axis[:, 0], close_pts_to_axis[:, 1], 1)

    if show:
        plt.figure()
        x = np.linspace(np.min(close_pts_to_axis[:, 0]), np.max(close_pts_to_axis[:, 0]), 100)
        y = p[0] * x + p[1]
        plt.plot(x, y, 'r', linewidth=5)
        plt.scatter(pts[:, 0], pts[:, 1], c='b')

    if axis_name == 'y':
        return 1 / p[0], -p[1] / p[0]
    elif axis_name == 'x':
        return p[0], p[1]


def get_center(m_horizontal, b_horizontal, m_vertical, b_vertical):
    """ This function takes the lines that best fit the horizontal and vertical
        axis and returns the intersection.
    """

    x_center = (b_vertical - b_horizontal) / (m_horizontal - m_vertical)
    y_center = m_horizontal * x_center + b_horizontal

    return x_center, y_center


def cart2pol(x, y, x_center=0, y_center=0):
    x = x - x_center
    y = y - y_center

    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)

    return rho, phi


def pol2cart(rho, phi, x_center=0, y_center=0):
    x = rho * np.cos(phi) + x_center
    y = rho * np.sin(phi) + y_center
    return x, y


def get_quadrant_ellipse(pts, quadrant=1, show=False):
    m_horizontal, b_horizontal = get_axis_line(pts, axis_name='x')
    m_vertical, b_vertical = get_axis_line(pts, axis_name='y')

    x_center, y_center = get_center(m_horizontal, b_horizontal, m_vertical, b_vertical)

    if quadrant == 1:
        x_final = np.max(pts[:, 0])
        y_final = np.min(pts[:, 1])
    elif quadrant == 2:
        x_final = np.min(pts[:, 0])
        y_final = np.min(pts[:, 1])
    elif quadrant == 3:
        x_final = np.min(pts[:, 0])
        y_final = np.max(pts[:, 1])
    elif quadrant == 4:
        x_final = np.max(pts[:, 0])
        y_final = np.max(pts[:, 1])
    else:
        print('Quadrant not valid')
        return

    x_line_horizontal = np.linspace(x_center, x_final, 100)
    y_line_horizontal = m_horizontal * x_line_horizontal + b_horizontal
    xy_line_horizontal = np.array([x_line_horizontal, y_line_horizontal]).T

    a = np.linalg.norm(xy_line_horizontal[0] - xy_line_horizontal[-1])

    y_line_vertical = np.linspace(y_center, y_final, 100)
    x_line_vertical = (y_line_vertical - b_vertical) / m_vertical
    xy_line_vertical = np.array([x_line_vertical, y_line_vertical]).T
    b = np.linalg.norm(xy_line_vertical[0] - xy_line_vertical[-1])

    _, angle_1 = cart2pol(x_line_horizontal[-1], y_line_horizontal[-1], x_center, y_center)
    _, angle_2 = cart2pol(x_line_vertical[-1], y_line_vertical[-1], x_center, y_center)

    theta = np.linspace(0, 2 * np.pi, 1000)

    x_ellipse = a * np.cos(theta) + x_center
    y_ellipse = b * np.sin(theta) + y_center

    if quadrant == 1:
        idx_vertical = np.where(y_ellipse < m_horizontal * x_final + b_horizontal)[0]
        idx_horizontal = np.where(x_ellipse > (y_final - b_vertical) / m_vertical)[0]

    elif quadrant == 2:
        idx_vertical = np.where(y_ellipse < m_horizontal * x_final + b_horizontal)[0]
        idx_horizontal = np.where(x_ellipse < (y_final - b_vertical) / m_vertical)[0]

    elif quadrant == 3:
        idx_vertical = np.where(y_ellipse > m_horizontal * x_final + b_horizontal)[0]
        idx_horizontal = np.where(x_ellipse < (y_final - b_vertical) / m_vertical)[0]

    elif quadrant == 4:
        idx_vertical = np.where(y_ellipse > m_horizontal * x_final + b_horizontal)[0]
        idx_horizontal = np.where(x_ellipse > (y_final - b_vertical) / m_vertical)[0]

    idx = np.intersect1d(idx_horizontal, idx_vertical)
    x_ellipse = x_ellipse[idx]
    y_ellipse = y_ellipse[idx]

    angle = np.arctan(m_horizontal)

    x_ellipse, y_ellipse = rotate_ellipse(x_ellipse, y_ellipse, angle, x_center, y_center)
    xy_ellipse = np.array([x_ellipse, y_ellipse]).T

    if show:
        plt.figure()
        plt.plot(xy_ellipse[:, 0], xy_ellipse[:, 1], 'r')
        plt.plot(xy_line_horizontal[:, 0], xy_line_horizontal[:, 1], 'b')
        plt.plot(xy_line_vertical[:, 0], xy_line_vertical[:, 1], 'g')
        # set limits
        plt.xlim([np.min(pts[:, 0]) - 100, np.max(pts[:, 0]) + 100])
        plt.ylim([np.min(pts[:, 1]) - 100, np.max(pts[:, 1]) + 100])

    return xy_ellipse, xy_line_horizontal, xy_line_vertical


def get_full_ellipse(pts, return_all=False):
    quadrants = [1, 2, 3, 4]
    xy_ellipse = []

    for quadrant in quadrants:
        xy_ellipse_, _, _ = get_quadrant_ellipse(pts, quadrant)
        xy_ellipse.append(xy_ellipse_)

    xy_ellipse = np.concatenate(xy_ellipse)

    m_vertical, b_vertical = get_axis_line(pts, axis_name='y')
    m_horizontal, b_horizontal = get_axis_line(pts, axis_name='x')
    x_center, y_center = get_center(m_horizontal, b_horizontal, m_vertical, b_vertical)

    phi = np.array([cart2pol(x, y, x_center, y_center)[1] for x, y in xy_ellipse])
    xy_ellipse = xy_ellipse[np.argsort(phi)]

    if return_all:
        # create dictionary to store all this
        features = {'m_horizontal': m_horizontal, 'b_horizontal': b_horizontal,
                    'm_vertical': m_vertical, 'b_vertical': b_vertical,
                    'x_center': x_center, 'y_center': y_center}

        return xy_ellipse, features

    else:
        return xy_ellipse


def rotate_point(x, y, angle, x_center=0, y_center=0):
    x = x - x_center
    y = y - y_center

    x_rot = x * np.cos(angle) - y * np.sin(angle)
    y_rot = x * np.sin(angle) + y * np.cos(angle)

    x_rot = x_rot + x_center
    y_rot = y_rot + y_center

    return x_rot, y_rot


def rotate_ellipse(x_ellipse, y_ellipse, angle, x_center=0, y_center=0):
    x_rot = []
    y_rot = []

    for i in range(len(x_ellipse)):
        x, y = rotate_point(x_ellipse[i], y_ellipse[i], angle, x_center, y_center)
        x_rot.append(x)
        y_rot.append(y)

    return np.array(x_rot), np.array(y_rot)

def get_points_enclosed_by_ellipse(img, pts_region=None, threshold=50, show=False, quadrant=1):
    img_to_show = img.copy()

    if pts_region is None:
        pts_region = get_contour(img, threshold=threshold, approx=False, single_region=True)
    xy_ellipse, features = get_full_ellipse(pts_region, return_all=True)

    # mask the image by the points given in pts_ellipse
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, [xy_ellipse.astype(int)], (255, 255, 255))
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    # mask img by mask
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Set the values outside the mask to NaN
    image_masked = img.copy().astype(float)
    image_masked[mask == 0] = np.nan
    # img = cv2.bitwise_and(img, img, mask=mask)
    # img_reverse = cv2.bitwise_and(img, img, mask=~mask)

    if quadrant is not None:

        x_center = features['x_center']
        y_center = features['y_center']

        if quadrant == 1:
            image_masked[:, :x_center.astype(int)] = np.nan
            image_masked[y_center.astype(int):, :] = np.nan

        elif quadrant == 2:
            image_masked[:, x_center.astype(int):] = np.nan
            image_masked[y_center.astype(int):, :] = np.nan

        elif quadrant == 3:
            image_masked[:, x_center.astype(int):] = np.nan
            image_masked[:y_center.astype(int), :] = np.nan

        elif quadrant == 4:
            image_masked[:, :x_center.astype(int)] = np.nan
            image_masked[:y_center.astype(int), :] = np.nan

    values = image_masked[~np.isnan(image_masked)].flatten()

    if show:
        plt.figure()
        plt.imshow(image_masked, cmap='gray')
        plt.imshow(img_to_show, alpha=0.1, cmap='gray')
        plt.plot(xy_ellipse[:, 0], xy_ellipse[:, 1], 'r')

    return values


def get_features_of_ellipse(img, threshold=50):
    pts = get_contour(img, threshold=threshold, approx=False, single_region=True)

    if len(pts) == 0:
        print('No points found')
        return None, None

    xy_ellipse, features = get_full_ellipse(pts, return_all=True)

    bright_values = [get_points_enclosed_by_ellipse(img, pts_region=pts, threshold=threshold, show=False, quadrant=2),
                     get_points_enclosed_by_ellipse(img, pts_region=pts, threshold=threshold, show=False, quadrant=4)]
    dark_values = [get_points_enclosed_by_ellipse(img, pts_region=pts, threshold=threshold, show=False, quadrant=1),
                   get_points_enclosed_by_ellipse(img, pts_region=pts, threshold=threshold, show=False, quadrant=3)]

    bright_median = np.median(np.concatenate(bright_values))
    dark_median = np.median(np.concatenate(dark_values))

    features['bright_median'] = bright_median
    features['dark_median'] = dark_median
    features['area_region'] = cv2.contourArea(np.array(xy_ellipse).astype(int))

    return xy_ellipse, features


def get_plot_features(img, threshold):
    ellipse, features = get_features_of_ellipse(img, threshold=threshold)

    m_horizontal, b_horizontal = features['m_horizontal'], features['b_horizontal']
    m_vertical, b_vertical = features['m_vertical'], features['b_vertical']

    x_center, y_center = features['x_center'], features['y_center']

    min_x, max_x = np.min(ellipse[:, 0]), np.max(ellipse[:, 0])
    min_y, max_y = np.min(ellipse[:, 1]), np.max(ellipse[:, 1])

    x_horizontal = np.linspace(min_x, max_x, 1000)
    y_horizontal = m_horizontal * x_horizontal + b_horizontal

    y_vertical = np.linspace(min_y, max_y, 1000)
    x_vertical = (y_vertical - b_vertical) / m_vertical

    return ellipse, x_center, y_center, x_horizontal, y_horizontal, x_vertical, y_vertical


## Functions for camera acquisition
def open_camera(camera_id=0):
    vc = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
    vc.set(cv2.CAP_PROP_FRAME_WIDTH, 1600)
    vc.set(cv2.CAP_PROP_FRAME_HEIGHT, 1200)

    return vc


def close_camera(vc):
    vc.release()


def get_frame(vc, show=False):
    if vc.isOpened():
        _, frame = vc.read()
        _, frame = vc.read()
        if show:
            cv2.imshow("Frame", frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    else:
        print('Camera is not opened')
        return None
