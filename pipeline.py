import pickle
import cv2
import numpy as np
import sliding_windows as sw


def distortion_image(img):
    # Read in the saved camera matrix and distortion coefficients
    # These are the arrays you calculated using cv2.calibrateCamera()
    dist_pickle = pickle.load(open("wide_dist_pickle.p", "rb"))
    mtx = dist_pickle["mtx"]
    dist = dist_pickle["dist"]
    return cv2.undistort(img, mtx, dist, None, mtx)


def abs_sobel_thresh(gray, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Calculate directional gradient
    # Apply threshold
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    else:
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    grad_binary = np.zeros_like(scaled_sobel)
    grad_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return grad_binary


def mag_thresh(gray, sobel_kernel=3, mag_thresh=(0, 255)):
    # Calculate gradient magnitude
    # Apply threshold
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobelxy = np. sqrt(np.square(sobelx) + np.square(sobely))
    scaled_sobel = np.uint8(255 * abs_sobelxy / np.max(abs_sobelxy))
    mag_binary = np.zeros_like(scaled_sobel)
    mag_binary[(scaled_sobel >= mag_thresh[0]) &
               (scaled_sobel <= mag_thresh[1])] = 1
    return mag_binary


def dir_threshold(gray, sobel_kernel=3, thresh=(0, np.pi / 2)):
    # Calculate gradient direction
    # Apply threshold
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    scaled_sobel = np.arctan2(abs_sobely, abs_sobelx)
    dir_binary = np.zeros_like(scaled_sobel)
    dir_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return dir_binary


def combining_thresholds_gradient(img):
    # Choose a Sobel kernel size
    ksize = 3  # Choose a larger odd number to smooth gradient measurements
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply each of the thresholding functions
    gradx = abs_sobel_thresh(
        gray, orient='x', sobel_kernel=7, thresh=(100, 247))
    grady = abs_sobel_thresh(
        gray, orient='y', sobel_kernel=7, thresh=(100, 247))
    mag_binary = mag_thresh(gray, sobel_kernel=5, mag_thresh=(100, 247))
    dir_binary = dir_threshold(gray, sobel_kernel=15, thresh=(0.7, 1.3))

    combined = np.zeros_like(dir_binary)

    combined[((gradx == 1) | (grady == 1)) | (
        (mag_binary == 1) & (dir_binary == 1))] = 1
    return combined


"""
Taken from http://www.bogotobogo.com/python/OpenCV_Python/python_opencv3_Changing_ColorSpaces_RGB_HSV_HLS.php

The big reason is that it separates color information (chroma) from intensity or lighting (luma).
Because value is separated, you can construct a histogram or thresholding rules using only saturation and hue.
This in theory will work regardless of lighting changes in the value channel.
In practice it is just a nice improvement.
Even by singling out only the hue you still have a very meaningful representation of the base color that will likely work much better than RGB.
The end result is a more robust color thresholding over simpler parameters."
"""


def color_s_channel(img, s_thresh=(0, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    s_channel = hls[:, :, 2]
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    return s_binary


def combined_binary_thresholds(img):
    thresholds_gradient = combining_thresholds_gradient(img)
    s_binary = color_s_channel(img, s_thresh=(120, 246))
    combined_binary = np.zeros_like(s_binary)
    combined_binary[(s_binary == 1) | (thresholds_gradient == 1)] = 1
    return combined_binary * 255


def region_of_interest(img):
     # draw a rectangle
    img_size = img.shape[:2]
    vertices = np.array([[
        (img_size[1] * .13, img_size[0] * .96),
        (img_size[1] * .43, img_size[0] * .64),
        (img_size[1] * .58,  img_size[0] * .64),
        (img_size[1] * .92, img_size[0] * .96)
    ]], dtype=np.int32)
    # mask color
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # defining a blank mask to start with
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    masked_image = cv2.bitwise_and(img, mask)

    return masked_image, vertices


def transfrom_street_lane(img):
    img, vertices = region_of_interest(img)
    img_size = img.shape[:2]

    src = np.float32(list(vertices))
    dst = np.float32(

        [[img_size[1] * .1, img_size[0]],
         [0, 0],
         [img_size[1],  0],
         [img_size[1] * .92, img_size[0]]])

    M = cv2.getPerspectiveTransform(src, dst)
    # Compute the inverse perspective transform
    Minv = cv2.getPerspectiveTransform(dst, src)

    warped = cv2.warpPerspective(
        img, M, img_size[::-1], flags=cv2.INTER_LINEAR)
    return warped, Minv


def identify_lane_line(img):
    window_centroids = sw.convolve(img)
    
    # to cover same y-range as image
    ploty = np.linspace(0, img.shape[0], num=len(window_centroids))
    leftx, rightx =  np.array(window_centroids).T

    img = sw.draw_image(img, window_centroids)
    return img, leftx, rightx, ploty


def draw_on_original_image(warped, leftx, rightx, ploty, Minv, image):
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    
    # top to bottom
    leftx = np.flip(leftx, axis=0)
    rightx = np.flip(rightx, axis=0)

    # Recast the x and y points into usable format for cv2.fillPoly()
    left_fit = np.polyfit(ploty, leftx, 2)
    left_fitx = left_fit[0] * ploty**2 + \
        left_fit[1] * ploty + left_fit[2]

    right_fit = np.polyfit(ploty, rightx, 2)
    right_fitx = right_fit[0] * ploty ** 2 + \
        right_fit[1] * ploty + right_fit[2]

    pts_left = np.array([np.transpose(
        np.vstack([left_fitx, ploty]))])

    pts_right = np.array([np.flipud(np.transpose(
        np.vstack([right_fitx, ploty])))])

    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
    newwarp = cv2.warpPerspective(
        color_warp, Minv, (image.shape[1], image.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(image, 1, newwarp, 0.3, 0)
    return result
