
# Tracking class

import pipeline
import numpy as np
import sliding_windows as sw
from scipy import stats


class Tracking():

    def __init__(self, left_line, right_line):
        self.l = left_line
        self.r = right_line
        # since y are the same for left and right lines then used the following instead
        self.ploty = []
        # the left and right points will be used to draw lines area on the original image
        self.leftx = None
        self.rightx = None

    def next_frame(self, frame):
        img = np.copy(frame)
        # update frame number
        self.l.current_frame_num += 1
        self.r.current_frame_num += 1

        # undistort the frame
        undistort = pipeline.distortion_image(img)
        # convert frame to black and white to identify lanes
        binary = pipeline.combined_binary_thresholds(undistort)
        # crop region of interset and transfroming it to a bird a view
        warped, Minv = pipeline.transform_street_lane(binary)
        # search for lines
        binary_warped_line = self.identify_lane_line(warped)

        result = pipeline.draw_on_original_image(
            warped=warped, ploty=self.ploty, leftx=self.leftx, rightx=self.rightx, Minv=Minv, image=frame)

        return result

    def identify_lane_line(self, img):
        # init for first frame
        if self.leftx is None:
            window_centroids = sw.convolve(img)
            if len(window_centroids) > 0:
                self._update_l_r(img, window_centroids)
                # calculate radius of curvature
                self._cal_radius_of_curvature(img)

                # 1
                if self._check_similar_curvature():
                    self._save_history()
                    self.leftx = self.l.recent_xfitted
                    self.rightx = self.r.recent_xfitted
                else:
                    self.leftx = self.l.allx
                    self.rightx = self.r.allx

                # draw lines
                return sw.draw_image(img, window_centroids)
            else:
                # return same image, no lines detected
                return img
        else:
            # lines have been detected
            l_tops, r_tops = self._lines_search()

            left_centroids = sw.update_top_line_centroids(
                img, self.leftx, tops=l_tops, line='l')

            right_centroids = sw.update_top_line_centroids(
                img, self.rightx, tops=r_tops, line='r')

            window_centroids = np.array((left_centroids, right_centroids)).T
            if len(window_centroids) > 0:
                self._update_l_r(img, window_centroids)
                # calculate radius of curvature
                self._cal_radius_of_curvature(img)

                if self._check_similar_curvature() and self._check_distance_horizontally() and self._check_lines_are_parallel():
                    self._save_history()
                    self.leftx = self.l.recent_xfitted
                    self.rightx = self.r.recent_xfitted
                else:
                    self._adjust_points()

            return sw.draw_image(img, window_centroids)

    # private
    def _lines_search(self):

        # if the last frame where detected
        l_frame_diff = np.absolute(np.diff((self.l.current_frame_num,
                                            self.l.last_frame_detected), axis=0))

        if self.l.detected:
            l_tops = .2 * l_frame_diff
        else:
            l_tops = .3 * l_frame_diff

        if self.r.detected:
            r_tops = .2 * l_frame_diff
        else:
            r_tops = .3 * l_frame_diff

        return l_tops, r_tops

    def _adjust_points(self):
        DIFF_SUM = .0

        def sum_diffs(ploty, allx, current_fit):
            line_fit = np.polyfit(ploty, allx, 2)
            diffs = np.diff(
                [current_fit, line_fit], axis=0)

            return np.sum(np.absolute(np.divide(diffs, line_fit)))

        # check if the left line was not detected correctly

        # BWT this can be done with calculation of radius of curvature by comparing the diffs between two frames
        diffs = sum_diffs(self.ploty, self.l.allx, self.l.current_fit)
        if diffs > DIFF_SUM:
            self.l.detected = False
            self.leftx = self.l.recent_xfitted
        else:
            self.leftx = self.l.allx

        # check if the right line was detected in the last frame
        if self.r.detected:
            # check if the right line was not detected correctly
            diffs = sum_diffs(self.ploty, self.r.allx, self.r.current_fit)
            if diffs > DIFF_SUM:
                self.r.detected = False
                self.rightx = self.r.recent_xfitted
            else:
                self.rightx = self.r.allx

        """
        when both lines are not detected:
        - the sum of the differences of line_fits between two frames are bigger than 1
        - both lines are not detected in the first frame.
        """
        if not (self.l.detected and self.r.detected):
            self.leftx = self.l.allx
            self.rightx = self.r.allx

    def _check_distance_horizontally(self):
        distance = 836  # in pixel
        midpoint = 636  # in pixel
        left_fitx = self.l.current_fit[0] * \
            self.ploty**2 + self.l.current_fit[1] * \
            self.ploty + self.l.current_fit[2]

        right_fitx = self.r.current_fit[0] * self.ploty**2 + \
            self.r.current_fit[1] * self.ploty + self.r.current_fit[2]

        dist = np.absolute(np.average(left_fitx - right_fitx))
        if int(dist) in range(int(distance - 1), int(distance + 1)):
            print(dist, 'dist')
            return True

        return False

    def _check_lines_are_parallel(self):
        left_fitx = self.l.current_fit[0] * \
            self.ploty**2 + self.l.current_fit[1] * \
            self.ploty + self.l.current_fit[2]

        right_fitx = self.r.current_fit[0] * self.ploty**2 + \
            self.r.current_fit[1] * self.ploty + self.r.current_fit[2]

        left_slop = stats.linregress(left_fitx, self.ploty)[0]
        right_slop = stats.linregress(right_fitx, self.ploty)[0]
        diff = np.absolute(np.diff((left_slop, right_slop), axis=0))

        if diff < 0:
            print(diff, 's')
            return True

        return False

    def _check_similar_curvature(self):

        if self.l.radius_of_curvature > self.r.radius_of_curvature:
            smaller_v = self.r.radius_of_curvature
            bigger_v = self.l.radius_of_curvature
        else:
            smaller_v = self.l.radius_of_curvature
            bigger_v = self.r.radius_of_curvature

        similarity = int((smaller_v / bigger_v) * 100)

        # since we always divided the smaller/bigger then the value should be between 0 and 1
        if similarity in range(99, 100):
            print(self.l.radius_of_curvature, 'lc',
                  self.r.radius_of_curvature, 'rc')
            return True

        return False

    def _cal_radius_of_curvature(self, binary_warped):

        # Define y-value where we want radius of curvature
        # I'll choose the maximum y-value, corresponding to the bottom of the image
        y_eval = binary_warped.shape[0]

        self.l.radius_of_curvature = (
            (1 + (2 * self.l.current_fit[0] * y_eval + self.l.current_fit[1])**2)**1.5) / np.absolute(2 * self.l.current_fit[0])

        self.r.radius_of_curvature = (
            (1 + (2 * self.r.current_fit[0] * y_eval + self.r.current_fit[1])**2)**1.5) / np.absolute(2 * self.r.current_fit[0])

    def _save_history(self):
        self._save_history_l_line()
        self._save_history_r_line()

    def _save_history_l_line(self):
        self.l.detected = True
        self.l.recent_xfitted = self.l.allx
        self.l.bestx = (
            np.mean(self.l.recent_xfitted) + self.l.bestx) / 2
        self.l.best_fit = (
            np.mean(self.l.current_fit) + self.l.best_fit) / 2
        self.l.last_frame_detected = self.l.current_frame_num

    def _save_history_r_line(self):
        self.r.detected = True
        self.r.recent_xfitted = self.r.allx
        self.r.bestx = (
            np.mean(self.r.recent_xfitted) + self.r.bestx) / 2
        self.r.best_fit = (
            np.mean(self.r.current_fit) + self.r.best_fit) / 2
        self.r.last_frame_detected = self.r.current_frame_num

    def _update_l_r(self, binary_warped, window_centroids):
        # to cover same y-range as image
        self.ploty = np.linspace(
            0, binary_warped.shape[0], num=len(window_centroids))
        self.l.allx, self.r.allx = np.array(window_centroids).T

        self.l.current_fit = np.polyfit(
            self.ploty, self.l.allx, 2)

        self.r.current_fit = np.polyfit(
            self.ploty, self.r.allx, 2)
