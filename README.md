# Advanced Lane Finding Project

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/undistort-calibration1.jpg "Undistorted"
[image2]: ./output_images/undistort-straight_lines1.jpg "Road Transformed"
[image3]: ./output_images/thresholds-straight_lines1.jpg "Binary Example"
[image4]: ./output_images/warped-straight_lines1.jpg "Warp Example"
[image5]: ./output_images/line-fit-straight_lines2.jpg "Fit Visual"
[image6]: ./output_images/output-straight_lines1.jpg "Output"
[video1]: ./output_project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

> Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the file called `camera_calibration.py`).  

I start by preparing "object points", which will be the (9, 6, 0) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (9, 6) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (9, 6) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

Basically I loaded mtx and dist from the saved pickle `wide_dist_pickle.p` and used  `cv2.undistort()` function and obtained the result. Code can be found in line #7 in `pipeline.py`.

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines #16 through #101 in `pipeline.py`).  Here's an example of my output for this step.  

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `transform_street_lane()`, which appears in lines 129 through 147 in the file `pipeline.py`. The `transform_street_lane()` function takes as inputs an image (`img`). First I calculate the src by calling `region_of_interest()` function, which appears in lines 104 through 126.  I chose the hardcode the source and destination points in the following manner:

```python

vertices = np.array([[
        (img_size[1] * .13, img_size[0] * .96),
        (img_size[1] * .43, img_size[0] * .64),
        (img_size[1] * .58,  img_size[0] * .64),
        (img_size[1] * .92, img_size[0] * .96)
    ]], dtype=np.int32)

src = np.float32(list(vertices))
dst = np.float32(
    [[img_size[1] * .1, img_size[0]],
    [0, 0],
    [img_size[1],  0],
    [img_size[1] * .92, img_size[0]]])
    
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 166.4, 691.2      | 128, 720        | 
| 550.4, 460.8      | 0, 0      |
| 742.4, 460.8     | 1280, 0      |
| 1177.6, 691.2      | 1177.6, 720        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

I used sliding windows with convolve approach to find the right and left lines. First I obtained the center point of the left and right lines from the bottom quarter of an image. These codes can be found on lines # 102 through # 123 on `sliding_windows.py`.

Next I used these center pixels number as a based to find find the other centers. Look at `convolve()` line # 126 on `sliding_windows.py` , which take warped image as an argument. 
Finally I draw the result on the image, and here is an example:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # 161 through # 192 in my code in `pipeline.py` in the function `draw_on_original_image()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output-project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

