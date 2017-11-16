
# Tracking class

import pipeline
import numpy as np
import sliding_windows as sw


class Tracking():
    def __init__(self, left_line, right_line):
        self.l_line = left_line
        self.r_line = right_line
        self.ploty = []
        self.leftx = None
        self.rightx= None

    def next_frame(self, frame):
        img = np.copy(frame)
        # update frame number
        self.l_line.current_frame_num += 1
        self.r_line.current_frame_num += 1

        # undistort the frame
        undistort = pipeline.distortion_image(img)
        # convert frame to black and white to identify lanes
        binary = pipeline.combined_binary_thresholds(undistort)
        # crop region of interset and transfroming it to a bird a view
        warped, Minv = pipeline.transfrom_street_lane(binary)
        # search for lines
        binary_warped_line = self.identify_lane_line(warped)
        
        result = pipeline.draw_on_original_image(
            warped=warped, ploty=self.ploty, leftx=self.leftx, rightx=self.rightx, Minv=Minv, image=frame)

        return result

    def identify_lane_line(self, img):
        window_centroids = sw.convolve(img)
        self._update_x_y(window_centroids=window_centroids,binary_warped=img)

        if len(window_centroids) > 0:
            self.l_line.radius_of_curvature, self.r_line.radius_of_curvature = sw.radius_of_curvature(
                    img, ploty=self.ploty,leftx=self.l_line.allx,rightx=self.r_line.allx)

            if self.check_similar_curvature():
                self._save_history()
                self.leftx = self.l_line.recent_xfitted
                self.rightx = self.r_line.recent_xfitted
            else:
                self.adjust_points()

        return sw.draw_image(img, window_centroids)
    
    def adjust_points(self):
        DIFF_SUM = 1
        # check if the left line was detected in the last frame
        if self.l_line.detected:
            left_fit = np.polyfit(self.ploty, self.l_line.allx, 2) 
            self.l_line.diffs = np.diff(
                [self.l_line.current_fit, left_fit ], axis=0) 
            
            # check if the left line was not detected correctly 
            # BWT this can be done with calculation of radius of curvature by comparing the diffs between two frames
            diffs = np.sum(np.absolute(np.divide(self.l_line.diffs,left_fit)))
            print(diffs,'l')
            if  diffs > DIFF_SUM:
                self.l_line.detected = False
                self.leftx = self.l_line.recent_xfitted
                print(self.l_line.current_frame_num,'LLCF',self.l_line.last_frame_detected,'LLLF')
            else:
                self.leftx = self.l_line.allx

        # check if the right line was detected in the last frame
        if self.r_line.detected:
            right_fit = np.polyfit(self.ploty, self.r_line.allx, 2) 
            self.r_line.diffs = np.diff(
                [self.r_line.current_fit ,right_fit ], axis=0) 

            # check if the right line was not detected correctly 
            diffs = np.sum(np.absolute(np.divide(self.r_line.diffs,right_fit)))
            print(diffs,'r')
            if diffs > DIFF_SUM:
                self.r_line.detected = False 
                self.rightx = self.r_line.recent_xfitted
                print(self.r_line.current_frame_num,'RLCF',self.r_line.last_frame_detected,'RLLF')
            else:
                self.rightx = self.r_line.allx
        # when both lines are not detected:
        # - the sum of the differences of line_fits between two frames are bigger than 1  
        # - both lines are not detected in the first frame.
        if not (self.l_line.detected and self.r_line.detected):
            self.leftx = self.l_line.allx
            self.rightx = self.r_line.allx

            
            

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
    
    # private
    def _save_history(self):
        self._save_history_l_line()
        self._save_history_r_line()

    def _save_history_l_line(self):
        self.l_line.detected = True
        self.l_line.recent_xfitted = self.l_line.allx
        self.l_line.current_fit = np.polyfit(
            self.ploty, self.l_line.recent_xfitted, 2)
        self.l_line.bestx = (
            np.mean(self.l_line.recent_xfitted) + self.l_line.bestx) / 2
        self.l_line.best_fit = (
            np.mean(self.l_line.current_fit) + self.l_line.best_fit) / 2  
        self.l_line.last_frame_detected = self.l_line.current_frame_num

    def _save_history_r_line(self):
        self.r_line.detected = True
        self.r_line.recent_xfitted = self.r_line.allx
        self.r_line.current_fit = np.polyfit(
            self.ploty, self.r_line.recent_xfitted, 2)
        self.r_line.bestx = (
            np.mean(self.r_line.recent_xfitted) + self.r_line.bestx) / 2
        self.r_line.best_fit = (
            np.mean(self.r_line.current_fit) + self.r_line.best_fit) / 2
        self.r_line.last_frame_detected = self.r_line.current_frame_num

    def _update_x_y(self,binary_warped,window_centroids):
        # to cover same y-range as image
        self.ploty = np.linspace(0, binary_warped.shape[0], num=len(window_centroids))
        levels = [level for level in window_centroids]
        # top to bottom
        self.l_line.allx = np.flip([left for left, right in levels], axis=0)
        self.r_line.allx = np.flip([right for left, right in levels], axis=0)