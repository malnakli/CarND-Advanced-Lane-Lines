import numpy as np
import cv2
import math

# Share Variables
# Choose the number of sliding windows
nwindows = 9
# Set height of windows
window_height = np.int(720 / nwindows)
window_width = 50
margin = 100  # How much to slide left and right for searching
window = np.ones(window_width)


def draw_image(binary_warped, window_centroids):
    # If we found any window centers
    if len(window_centroids) > 0:
        # Points used to draw all the left and right windows
        l_points = np.zeros_like(binary_warped)
        r_points = np.zeros_like(binary_warped)

        # Go through each level and draw the windows
        for level in range(0, len(window_centroids)):
            # Window_mask is a function to draw window areas

            l_mask = window_mask(
                window_width, window_height, binary_warped, window_centroids[level][0], level)
            r_mask = window_mask(
                window_width, window_height, binary_warped, window_centroids[level][1], level)
            # Add graphic points from window mask here to total pixels found
            l_points[(l_points == 255) | ((l_mask == 1))] = 255
            r_points[(r_points == 255) | ((r_mask == 1))] = 255

        # Draw the results
        # add both left and right window pixels together
        template = np.array(r_points + l_points, np.uint8)
        # create a zero color channel
        zero_channel = np.zeros_like(template)
        # make window pixels green
        template = np.array(
            cv2.merge((zero_channel, template, zero_channel)), np.uint8)
        # making the original road pixels 3 color channels
        warpage = np.dstack(
            (binary_warped, binary_warped, binary_warped)) * 255
        # overlay the original road image with window results
        output = cv2.addWeighted(warpage, .7, template, 1, 0.0)

    # If no window centers found, just display original road image
    else:
        output = np.array(
            cv2.merge((binary_warped, binary_warped, binary_warped)), np.uint8)

    return output


def window_mask(width, height, img_ref, center, level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0] - (level + 1) * height):int(img_ref.shape[0] - level * height),
           max(0, int(center - width / 2)): min(int(center + width / 2), img_ref.shape[1])] = 1
    return output


"""
Offset
You can assume the camera is mounted at the center of the car,
such that the lane center is the midpoint at the bottom of the image
between the two lines you've detected.The offset of the lane center from the center of the image
(converted from pixels to meters) is your distance from the center of the lane.
"""


def update_line_centroids(binary_warped, old_line_centroids, tops=.1, line=None):

    if tops > .95:
        line_centroids = []

        if line == 'l':
            pre_center = first_l_centroids(binary_warped)
        else:
            pre_center = first_r_centroids(binary_warped)

        line_centroids.append(pre_center)
    else:
        tops = math.ceil(len(old_line_centroids) * tops)
        line_centroids = old_line_centroids[:-tops]
        pre_center = line_centroids[-1]

    for level in range(len(line_centroids), (int)(binary_warped.shape[0] / window_height)):

        conv_signal = _conv_signal(binary_warped, level)
        # Find the best left centroid by using past left center as a reference
        center = _get_the_next_center(
            binary_warped, conv_signal, pre_center)
        # Find the best right centroid by using past right center as a reference

        # Add what we found for that layer
        line_centroids = np.append(line_centroids, center)

    return line_centroids


def first_l_centroids(binary_warped):
    # get the quarter bottom  image, slicing horizontally
    bottom_image = binary_warped[int(
        binary_warped.shape[0] * (3 / 4)):]
    # get the left image
    l_img = bottom_image[:, : int(binary_warped.shape[1] / 2)]
    l_sum = np.sum(l_img, axis=0)
    # apply a convolution to maximize the number of "hot" pixels in each window
    l_convolve = np.convolve(window, l_sum, 'same')
    return np.argmax(l_convolve)


def first_r_centroids(binary_warped):
    # get the quarter bottom  image, slicing horizontally
    bottom_image = binary_warped[int(
        binary_warped.shape[0] * (3 / 4)):]
    # get the right image
    r_img = bottom_image[:, int(binary_warped.shape[1] / 2):]
    r_sum = np.sum(r_img, axis=0)
    # apply a convolution to maximize the number of "hot" pixels in each window
    r_convolve = np.convolve(window, r_sum, 'same')
    return np.argmax(r_convolve) + int(binary_warped.shape[1] / 2)


def convolve(binary_warped):
    """
    search for window_centroids from scratch
    """
    # Store the (left,right) window centroid positions per level
    window_centroids = []
    # Create our window template that we will use for convolutions

    # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
    # and then np.convolve the vertical image slice with the window template

    left_centroids = update_line_centroids(
        binary_warped, [], tops=1, line='l')
    right_centroids = update_line_centroids(
        binary_warped, [], tops=1, line='r')

    window_centroids = np.array((left_centroids, right_centroids)).T
    return window_centroids


def _get_the_next_center(binary_warped, conv_signal, pre_center):
    # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
    offset = window_width / 2
    min_index = int(max(pre_center + offset - margin, 0))
    max_index = int(
        min(pre_center + offset + margin, binary_warped.shape[1]))
    center = np.argmax(
        conv_signal[min_index:max_index]) + min_index - offset

    return center


def _conv_signal(binary_warped, level):
     # convolve the window into the vertical slice of the image
    image_layer = binary_warped[int(binary_warped.shape[0] - (
        level + 1) * window_height): int(binary_warped.shape[0] - level * window_height), :]
    image_layer_sum = np.sum(image_layer, axis=0)

    return np.convolve(window, image_layer_sum)
