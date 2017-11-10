import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import pickle


def main():
    # prepare object points
    nx = 9
    ny = 6

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)
    objp = np.zeros((ny*nx,3), np.float32)
    objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.
    img_size = (0,0) # to be used for calibrateCamera
    # get images name
    images = glob.glob('camera_cal/calibration*.jpg') 

    # Step through the list and search for chessboard corners
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)

        # update img_size to be user later
        img_size = (img.shape[1], img.shape[0])

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw and display the corners
            # cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
            # plt.imshow(img)
            # plt.show()
        else:
            print('Failed to find corners for: ', fname)

    # Do camera calibration given object points and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)
    # Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
    dist_pickle = {}
    dist_pickle["mtx"] = mtx
    dist_pickle["dist"] = dist
    pickle.dump( dist_pickle, open( "wide_dist_pickle.p", "wb" ) )

    # Step through the images and  test undistortion on them 
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)

        dst = cv2.undistort(img, mtx, dist, None, mtx)
        filepath = "output_images/undistort-calibration"+ str(idx+1) + ".jpg"
        cv2.imwrite(filepath,dst)

        # Visualize undistortion
        # f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
        # ax1.imshow(img)
        # ax1.set_title('Original Image', fontsize=30)
        # ax2.imshow(dst)
        # ax2.set_title('Undistorted Image', fontsize=30)
        # plt.show()


if __name__ == "__main__":
    main()