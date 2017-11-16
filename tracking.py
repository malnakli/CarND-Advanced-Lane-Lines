
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

            if self.check_similar_curvature():
                self.save_history()
            else:
                self.adjust_points()

        return sw.draw_image(img, window_centroids)

    def adjust_points(self):
        if(self.l_line.detected and self.r_line.detected):
            left_fit = np.polyfit(self.ploty, self.l_line.allx, 2)
            right_fit = np.polyfit(self.ploty, self.r_line.allx, 2)

            self.l_line.diffs = np.diff(
                [left_fit, self.l_line.current_fit], axis=0)
            self.r_line.diffs = np.diff(
                [right_fit, self.r_line.current_fit], axis=0)
           # print(self.l_line.diffs, self.r_line.diffs)

    def save_history(self):
        self.l_line.detected = True
        self.r_line.detected = True

        self.l_line.recent_xfitted = self.l_line.allx
        self.r_line.recent_xfitted = self.r_line.allx

        self.l_line.current_fit = np.polyfit(
            self.ploty, self.l_line.recent_xfitted, 2)
        self.r_line.current_fit = np.polyfit(
            self.ploty, self.r_line.recent_xfitted, 2)

        self.l_line.bestx = (
            np.mean(self.l_line.recent_xfitted) + self.l_line.bestx) / 2
        self.r_line.bestx = (
            np.mean(self.r_line.recent_xfitted) + self.r_line.bestx) / 2

        self.l_line.best_fit = (
            np.mean(self.l_line.current_fit) + self.l_line.best_fit) / 2
        self.r_line.best_fit = (
            np.mean(self.r_line.current_fit) + self.r_line.best_fit) / 2

    def check_similar_curvature(self):
        # print(self.l_line.radius_of_curvature, 'px',
        #       self.r_line.radius_of_curvature, 'px')

        if self.l_line.radius_of_curvature > self.r_line.radius_of_curvature:
            smaller_v = self.r_line.radius_of_curvature
            bigger_v = self.l_line.radius_of_curvature
        else:
            smaller_v = self.l_line.radius_of_curvature
            bigger_v = self.r_line.radius_of_curvature

        similarity = int((smaller_v / bigger_v) * 100)

        # since we always divided the smaller/bigger then the value should be between 0 and 1
        if similarity in range(50, 100):
            return True

        return False
