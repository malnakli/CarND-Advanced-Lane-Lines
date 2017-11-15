import numpy as np


# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        # average x values of the fitted line over the last n iterations
        self.bestx = None
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        # radius of curvature of the line in meter
        self.radius_of_curvature = None
        # distance in meters of vehicle center from the line
        self.line_base_pos = None
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0, 0, 0], dtype='float')
        # x values for detected line pixels
        self.allx = None
        # y values for detected line pixels
        self.ally = None

    def cal_curvature(self, current_curvature):
        if self.__similar_curvature(current_curvature):
            self.detected = True
            self.radius_of_curvature = current_curvature
            return self.radius_of_curvature
        else:
            return (self.radius_of_curvature + current_curvature / 2.0)

    def __similar_curvature(self, current_curvature):
        offest = 100  # in meter
        if self.detected:
            if current_curvature in range(int(self.radius_of_curvature - offest), int(self.radius_of_curvature + offest)):
                return True

            return False
        else:
            return True


# Tracking class
class Tracking():
    def __init__(self, left_line, right_line):
        self.left_line = left_line
        self.right_line = right_line

    def recent_xfitted(self, left, right):
        self.left_line.recent_xfitted = left
        self.right_line.recent_xfitted = right

    def allxy(self, y, leftx, rightx):
        self.left_line.allx = leftx
        self.left_line.ally = y
        self.right_line.allx = rightx
        self.right_line.ally = y
