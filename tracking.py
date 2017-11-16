
# Tracking class

import pipeline
import numpy as np
import sliding_windows as sw


class Tracking():
    def __init__(self, left_line, right_line):
        self.l_line = left_line
        self.r_line = right_line
        self.ploty = []

    def next_frame(self, frame):
        img = np.copy(frame)
        # undistort the frame
        undistort = pipeline.distortion_image(img)
        # convert frame to black and white to identify lanes
        binary = pipeline.combined_binary_thresholds(undistort)
        # crop region of interset and transfroming it to a bird a view
        warped, Minv = pipeline.transfrom_street_lane(binary)
        # search for lines
        binary_warped_line = self.identify_lane_line(warped)
        result = pipeline.draw_on_original_image(
            warped=warped, ploty=self.ploty, leftx=self.l_line.allx, rightx=self.r_line.allx, Minv=Minv, image=frame)

        return result

    def identify_lane_line(self, img):
        window_centroids = sw.convolve(img)
        if len(window_centroids) > 0:
            self.l_line.allx, self.r_line.allx, self.ploty, self.l_line.radius_of_curvature, \
                self.r_line.radius_of_curvature = sw.radius_of_curvature(
                    img, window_centroids)
            self.check_similar_curvature
        return img

    def check_similar_curvature(self):
        print(self.l_line.radius_of_curvature /
              self.r_line.radius_of_curvature)
