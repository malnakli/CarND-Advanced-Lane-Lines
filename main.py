import numpy as np
import cv2
import pipeline
import glob

def operations_on_frame(frame):
    img = np.copy(frame)
    img = pipeline.distortion_image(img)
    img = pipeline.combining_thresholds_gradient(img)
    return img

def read_video(filename='challenge_video.mp4'):
    cap = cv2.VideoCapture(filename)

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Our operations on the frame come here
        frame = operations_on_frame(frame)
        # Display the resulting frame
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

def read_test_images():
    images = glob.glob('test_images/*.jpg') 

    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        dst = pipeline.distortion_image(img)
        filepath = "output_images/undistort-"+ str(fname.split('/')[-1])
        cv2.imwrite(filepath,dst)

def main():
    read_video()
    #read_test_images()

if __name__ == "__main__":
    main()