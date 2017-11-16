import numpy as np
import cv2
import pipeline
import glob
import argparse
from line_class import Line
from tracking import Tracking


def read_video(filename='challenge_video.mp4', saved=False):
    cap = cv2.VideoCapture(filename)
    # create Tracking object
    tracking = Tracking(Line(), Line())

    if saved:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Be sure to use lower case
        output = "output-" + filename
        out = cv2.VideoWriter(output, fourcc, 20.0, (1280, 720))

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if frame is None:
            break

        img = tracking.next_frame(frame)

        if saved:
            out.write(img)

        # Our operations on the frame come here
        # Display the resulting frame
        cv2.imshow('img', img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


def read_test_images():
    images = glob.glob('test_images/*.jpg')

    def save(img, name):
        filepath = "output_images/" + name + "-" + str(fname.split('/')[-1])
        cv2.imwrite(filepath, img)

    for idx, fname in enumerate(images):
        frame = cv2.imread(fname)
        img = np.copy(frame)
        undistort = pipeline.distortion_image(img)
        save(undistort, 'undistort')
        thresholds = pipeline.combined_binary_thresholds(undistort)
        save(thresholds, 'thresholds')
        warped, Minv = pipeline.transfrom_street_lane(thresholds)
        save(warped, 'warped')
        line_fit, leftx, rightx, ploty = pipeline.identify_lane_line(warped)
        save(line_fit, 'line-fit')
        output = pipeline.draw_on_original_image(
            warped=warped, leftx=leftx, rightx=rightx, ploty=ploty, Minv=Minv, image=frame)
        save(output, 'output')


def main(args):
    read_video(filename=args.fileinput, saved=args.save_video)
    # read_test_images()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--fileinput',
                        type=str, help='finename of a video file')
    parser.add_argument('-s', '--save_video',
                        type=bool, help='Either to save the output result or not')

    main(parser.parse_args())
