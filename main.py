import numpy as np
import cv2
import pipeline
import glob
import argparse
from line_class import Tracking,Line

def operations_on_frame(frame,tracking):
    img = np.copy(frame)
    distort = pipeline.distortion_image(img)
    warped,Minv = pipeline.transfrom_street_lane(distort)
    binary_warped = pipeline.combined_binary_thresholds(warped)
    binary_warped_line = pipeline.identify_lane_line(binary_warped,tracking)
    result = pipeline.draw_on_original_image(warped=binary_warped,tracking=tracking,Minv=Minv,image=frame)
    return result

def read_video(filename='challenge_video.mp4'):
    cap = cv2.VideoCapture(filename)
    # create Tracking object
    tracking = Tracking(Line(),Line())

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if frame is None:
            break

        frame = operations_on_frame(frame,tracking)
        
        # Our operations on the frame come here
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
        # dst = pipeline.distortion_image(img)
        # dst = pipeline.transfrom_street_lane(img)
        img = pipeline.distortion_image(img)
        img = pipeline.transfrom_street_lane(img)
        img = pipeline.combined_binary_thresholds(img)
        dst = pipeline.identify_lane_line(img)
        filepath = "output_images/fit-lines-2-"+ str(fname.split('/')[-1])
        cv2.imwrite(filepath,dst)
        break

def main(args):
    read_video(filename=args.fileinput)
    #read_test_images()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--fileinput',
                            type=str, help='finename of a video file')
    
    main(parser.parse_args())