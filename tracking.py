
# Tracking class
import pipeline
import numpy as np
import sliding_windows as sw
from scipy import stats


SIMILARITY_RADIUS_OF_CURVATURE = 70  # % of how much left and right radius of curvature similar
PARALLEL = 50  # % of parallel of left and right lines
HORIZONTAL_DISTANCE_MATCH = 97  # % of how much distance match actual one
LINE_CHANGED = 35 # % of how much line has changed from previous frame


class Tracking():
    """
    The Tracking class is responsible for keep tracking of detected lines
    Tracking start by calling next_frame() function 
    """

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
        # draw the detected lines on the original image
        line_drawn_on_original_image = pipeline.draw_on_original_image(
            warped=warped, ploty=self.ploty, leftx=self.leftx, rightx=self.rightx, Minv=Minv, image=frame)

        result = self._draw_some_text(line_drawn_on_original_image)
        return result

    def identify_lane_line(self, img):
        # init for first frame
        if self.leftx is None:
            window_centroids = sw.convolve(img)
        else:
            # lines have been detected
            l_tops, r_tops = self._lines_search()

            left_centroids = sw.update_line_centroids(
                img, self.leftx, tops=l_tops, line='l')

            right_centroids = sw.update_line_centroids(
                img, self.rightx, tops=r_tops, line='r')

            window_centroids = np.array((left_centroids, right_centroids)).T

        if len(window_centroids) > 0:
            self._update_l_r(img, window_centroids)
            # calculate radius of curvature
            self._cal_radius_of_curvature(img)

            if self._sanity_check():
                self._save_history()
                self.leftx = self.l.recent_xfitted
                self.rightx = self.r.recent_xfitted
            else:
                self._adjust_points_for_each_line()

        return sw.draw_image(img, window_centroids)

    # private
    def _draw_some_text(self, img):
        left_curverad, right_curverad = self._cal_radius_of_curvature_in_meter(
            img)
        radius_of_curvature = int(np.average((left_curverad, right_curverad)))
        radius_of_curvature_text = "Radius of curvature = " + \
            str(radius_of_curvature) + "(m)"
        # left from center
        vehicle_position_text = "Vehicle is " +\
            str(self._position_of_the_vehicle_with_respect_to_center(
                img)) + " m left of center"

        result = pipeline.draw_text_on_image(
            img, radius_of_curvature_text, location=(320, 40))
        result = pipeline.draw_text_on_image(
            img, vehicle_position_text, location=(320, 80))

        return result

    def _sanity_check(self):
        return self._check_similar_curvature() and self._check_distance_horizontally() and self._check_lines_are_parallel()

    def _lines_search(self):

        # if the last frame where detected
        l_frame_diff = np.absolute(np.diff((self.l.current_frame_num,
                                            self.l.last_frame_detected), axis=0))

        if self.l.detected:
            l_tops = .1 * l_frame_diff
        else:
            l_tops = .25 * l_frame_diff

        if self.r.detected:
            r_tops = .1 * l_frame_diff
        else:
            r_tops = .25 * l_frame_diff

        return l_tops, r_tops

    def _adjust_points_for_each_line(self):
        """
        Identify which lines left or right was not detected properly. 
        by compared to the polynomial of last detected frame
        """
    
        def sum_diffs(ploty, recent_xfitted, current_fit):

            if recent_xfitted.any():
                line_fit = np.polyfit(ploty, recent_xfitted, 2)
                diffs = np.diff(
                    [current_fit, line_fit], axis=0)

                return np.sum(np.absolute(np.divide(diffs, current_fit))) * 100

            return -1

        # Get how mach differences between this frame and last one correctly detected for the left line
        diffs = sum_diffs(self.ploty, self.l.recent_xfitted,
                          self.l.current_fit)
        if diffs > LINE_CHANGED:
            self.l.detected = False
            self.leftx = self.l.recent_xfitted
        else:
            self.leftx = self.l.allx

        # Get how mach differences between this frame and last one correctly detected for the right line
        diffs = sum_diffs(self.ploty, self.r.recent_xfitted,
                          self.r.current_fit)
        if diffs > LINE_CHANGED:
            self.r.detected = False
            self.rightx = self.r.recent_xfitted
        else:
            self.rightx = self.r.allx

    def _check_distance_horizontally(self):
        actual_distance = 836  # in pixel
        left_fitx = self.l.current_fit[0] * \
            self.ploty**2 + self.l.current_fit[1] * \
            self.ploty + self.l.current_fit[2]

        right_fitx = self.r.current_fit[0] * self.ploty**2 + \
            self.r.current_fit[1] * self.ploty + self.r.current_fit[2]

        estimated_distance = np.absolute(np.average(left_fitx - right_fitx))

        distance_match = 100 - self.percentage_difference(actual_distance,estimated_distance)
        if distance_match > HORIZONTAL_DISTANCE_MATCH:
            #print(distance_match,'distance_match')
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
        diff = 100 - self.percentage_difference(left_slop,right_slop)
        
        if diff > PARALLEL:
            #print(diff,'parallel')
            return True

        return False

    def _check_similar_curvature(self):

        similarity = 100 - self.percentage_difference(self.l.radius_of_curvature,self.r.radius_of_curvature)
        # since we always divided the smaller/bigger then the value should be between 0 and 1
        if similarity > SIMILARITY_RADIUS_OF_CURVATURE:
            #print(similarity,'similarity')
            return True

        return False

    # compute the radius of curvature of the fit
    def _cal_radius_of_curvature(self, binary_warped):

        # Define y-value where we want radius of curvature
        # I'll choose the maximum y-value, corresponding to the bottom of the image
        y_eval = binary_warped.shape[0]

        self.l.radius_of_curvature = (
            (1 + (2 * self.l.current_fit[0] * y_eval + self.l.current_fit[1])**2)**1.5) / np.absolute(2 * self.l.current_fit[0])

        self.r.radius_of_curvature = (
            (1 + (2 * self.r.current_fit[0] * y_eval + self.r.current_fit[1])**2)**1.5) / np.absolute(2 * self.r.current_fit[0])

    def _cal_radius_of_curvature_in_meter(self, binary_warped):
        # Define y-value where we want radius of curvature
        # I'll choose the maximum y-value, corresponding to the bottom of the image
        y_eval = binary_warped.shape[0]

        # Define conversions in x and y from pixels space to meters
        # meters per pixel in y dimension
        ym_per_pix = 30 / binary_warped.shape[0]
        # meters per pixel in x dimension
        xm_per_pix = 3.7 / 880  # the lane distance in pixel was calculated manual

        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(
            self.ploty * ym_per_pix, self.leftx * xm_per_pix, 2)
        right_fit_cr = np.polyfit(
            self.ploty * ym_per_pix, self.rightx * xm_per_pix, 2)

        # Calculate the new radii of curvature
        left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix +
                               left_fit_cr[1])**2)**1.5) / np.absolute(2 * left_fit_cr[0])
        right_curverad = (
            (1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2 * right_fit_cr[0])

        return left_curverad, right_curverad

    def _position_of_the_vehicle_with_respect_to_center(self, binary_warped):
        l_distance = self.leftx[0] # get the pixel of left line
        lane_distance_in_pixel = 880  # the lane distance in pixel was calculated manual
        xm_per_pix = 3.7 / lane_distance_in_pixel # get meter per pixel
        midpoint = l_distance + (lane_distance_in_pixel / 2) # midpoint between left and right lines
        center = binary_warped.shape[1] / 2 # image center
        convert_to_meter = (midpoint - center) * xm_per_pix # position in meter
        return format(convert_to_meter, '.5f')

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

    @classmethod
    def percentage_difference(cls,x,y):
        x = np.absolute(x)
        y = np.absolute(y)
        diff =  np.diff((x, y), axis=0)
        percentage = diff / x if x > y else diff / y
        return round(np.absolute(percentage)[0] * 100)