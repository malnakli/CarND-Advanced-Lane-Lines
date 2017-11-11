import numpy as np
import cv2
import pipeline
import glob
import argparse

def operations_on_frame(frame):
    img = np.copy(frame)
    img = pipeline.distortion_image(img)
    img = pipeline.transfrom_street_lane(img)
    img = pipeline.combined_binary_thresholds(img)
    img = pipeline.identify_lane_line(img)
    return img

def read_video(filename='challenge_video.mp4'):
    cap = cv2.VideoCapture(filename)

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if frame is None:
            break

        frame = operations_on_frame(frame)
        
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
        pipeline.identify_lane_line(img)
        # filepath = "output_images/transform-"+ str(fname.split('/')[-1])
        # cv2.imwrite(filepath,dst)

def main(args):
    read_video(filename=args.fileinput)
    #read_test_images()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--fileinput',
                            type=str, help='finename of a video file')
    
    main(parser.parse_args())