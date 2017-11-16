
# Tracking class

import pipeline
import numpy as np
import sliding_windows as sw


class Tracking():
    
    def __init__(self, left_line, right_line):
        self.l = left_line
        self.r = right_line
        # since y are the same for left and right lines then used the following instead
        self.ploty = []
        # the left and right points will be used to draw lines area on the original image
        self.leftx = None
        self.rightx= None


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
        warped, Minv = pipeline.transfrom_street_lane(binary)
        # search for lines
        binary_warped_line = self.identify_lane_line(warped)
        
        result = pipeline.draw_on_original_image(
            warped=warped, ploty=self.ploty, leftx=self.leftx, rightx=self.rightx, Minv=Minv, image=frame)

        return result

    def identify_lane_line(self, img):
        
        if self.leftx is not None or self.rightx is not None > 0 :
            l_tops = 0.2 if self.l.detected else 0.8
            r_tops = 0.2 if self.r.detected else 0.8
            self.l.allx =  sw.update_top_line_centroids(img,self.leftx,tops=l_tops)
            self.r.allx =  sw.update_top_line_centroids(img,self.rightx,tops=r_tops)
            window_centroids = np.array((self.l.allx,self.r.allx)).T
        else:
            window_centroids = sw.convolve(img)
            self._update_x_y(img,window_centroids)

        if len(self.l.allx) and len(self.r.allx) > 0:
            # calculate radius of curvature
            self.l.radius_of_curvature, self.r.radius_of_curvature = sw.radius_of_curvature(
                    img, ploty=self.ploty,leftx=self.l.allx,rightx=self.r.allx)

            if self.check_similar_curvature():
                self._save_history()
                self.leftx = self.l.recent_xfitted
                self.rightx = self.r.recent_xfitted
            else:
                self.adjust_points()
        
        else:
            # if a frame never detected any window_centroids then use the old values
            self.l.detected = False 
            self.r.detected = False 

        return sw.draw_image(img, window_centroids)
    
    def adjust_points(self):
        DIFF_SUM = .4
        # check if the left line was detected in the last frame
        if self.l.detected:
            left_fit = np.polyfit(self.ploty, self.l.allx, 2) 
            self.l.diffs = np.diff(
                [self.l.current_fit, left_fit ], axis=0) 
            
            # check if the left line was not detected correctly 
            # BWT this can be done with calculation of radius of curvature by comparing the diffs between two frames
            diffs = np.sum(np.absolute(np.divide(self.l.diffs,left_fit)))
            print(diffs,'l')
            if  diffs > DIFF_SUM:
                self.l.detected = False
                self.leftx = self.l.recent_xfitted
                print(self.l.current_frame_num,'LLCF',self.l.last_frame_detected,'LLLF')
            else:
                self.leftx = self.l.allx

        # check if the right line was detected in the last frame
        if self.r.detected:
            right_fit = np.polyfit(self.ploty, self.r.allx, 2) 
            self.r.diffs = np.diff(
                [self.r.current_fit ,right_fit ], axis=0) 

            # check if the right line was not detected correctly 
            diffs = np.sum(np.absolute(np.divide(self.r.diffs,right_fit)))
            print(diffs,'r')
            if diffs > DIFF_SUM:
                self.r.detected = False 
                self.rightx = self.r.recent_xfitted
                print(self.r.current_frame_num,'RLCF',self.r.last_frame_detected,'RLLF')
            else:
                self.rightx = self.r.allx
        # when both lines are not detected:
        # - the sum of the differences of line_fits between two frames are bigger than 1  
        # - both lines are not detected in the first frame.
        if not (self.l.detected and self.r.detected):
            self.leftx = self.l.allx
            self.rightx = self.r.allx

            
            

    def check_similar_curvature(self):
        # print(self.l.radius_of_curvature, 'px',
        #       self.r.radius_of_curvature, 'px')

        if self.l.radius_of_curvature > self.r.radius_of_curvature:
            smaller_v = self.r.radius_of_curvature
            bigger_v = self.l.radius_of_curvature
        else:
            smaller_v = self.l.radius_of_curvature
            bigger_v = self.r.radius_of_curvature

        similarity = int((smaller_v / bigger_v) * 100)

        # since we always divided the smaller/bigger then the value should be between 0 and 1
        if similarity in range(70, 100):
            print(self.l.radius_of_curvature,'lc' , self.r.radius_of_curvature,'rc')
            return True

        return False
    
    # private
    def _save_history(self):
        self._save_history_l_line()
        self._save_history_r_line()

    def _save_history_l_line(self):
        self.l.detected = True
        self.l.recent_xfitted = self.l.allx
        self.l.current_fit = np.polyfit(
            self.ploty, self.l.recent_xfitted, 2)
        self.l.bestx = (
            np.mean(self.l.recent_xfitted) + self.l.bestx) / 2
        self.l.best_fit = (
            np.mean(self.l.current_fit) + self.l.best_fit) / 2  
        self.l.last_frame_detected = self.l.current_frame_num

    def _save_history_r_line(self):
        self.r.detected = True
        self.r.recent_xfitted = self.r.allx
        self.r.current_fit = np.polyfit(
            self.ploty, self.r.recent_xfitted, 2)
        self.r.bestx = (
            np.mean(self.r.recent_xfitted) + self.r.bestx) / 2
        self.r.best_fit = (
            np.mean(self.r.current_fit) + self.r.best_fit) / 2
        self.r.last_frame_detected = self.r.current_frame_num

    def _update_x_y(self,binary_warped,window_centroids):
        # to cover same y-range as image
        self.ploty = np.linspace(0, binary_warped.shape[0], num=len(window_centroids))
        self.l.allx, self.r.allx  = np.array(window_centroids).T
        