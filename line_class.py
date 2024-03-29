import numpy as np


# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = np.array([])
        # average x values of the fitted line over the last n iterations
        self.bestx = 0
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = 0
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
        # current # frame
        self.current_frame_num = 0
        # last frame where line where detected
        self.last_frame_detected = 0
