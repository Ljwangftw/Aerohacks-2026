from __future__ import print_function
import cv2 as cv
import argparse
import setup_cameras
import video
'''
What is HSV Color Space?

The HSV color space represents colors in a way that's more aligned with human perception. 
Unlike the RGB color space, which mixes red, green, and blue light, HSV separates hue (the type of color) from saturation (color intensity) 
and value (brightness). This separation helps isolate colors more effectively, especially under different lighting conditions.
hue = 0 : Red
hue = 120 : Green
hue = 240 : Blue

HSV 0 0 0 : Black
HSV 0 0 100% : White
'''
max_value = 255 #max value for saturation
max_value_H = 360 // 2 #the type of color, ranging from 0° to 179° in OpenCV (as opposed to 0° to 360°)
low_H = 0
low_S = 0
low_V = 0
high_H = max_value_H
high_S = max_value
high_V = max_value
window_capture_name = 'Video Capture'
window_detection_name = 'Object Detection'
low_H_name = 'Low H'
low_S_name = 'Low S'
low_V_name = 'Low V'
high_H_name = 'High H'
high_S_name = 'High S'
high_V_name = 'High V'


def on_low_H_thresh_trackbar(val):
    global low_H
    global high_H
    low_H = val
    low_H = min(high_H - 1, low_H)
    cv.setTrackbarPos(low_H_name, window_detection_name, low_H)


def on_high_H_thresh_trackbar(val):
    global low_H
    global high_H
    high_H = val
    high_H = max(high_H, low_H + 1)
    cv.setTrackbarPos(high_H_name, window_detection_name, high_H)


def on_low_S_thresh_trackbar(val):
    global low_S
    global high_S
    low_S = val
    low_S = min(high_S - 1, low_S)
    cv.setTrackbarPos(low_S_name, window_detection_name, low_S)


def on_high_S_thresh_trackbar(val):
    global low_S
    global high_S
    high_S = val
    high_S = max(high_S, low_S + 1)
    cv.setTrackbarPos(high_S_name, window_detection_name, high_S)


def on_low_V_thresh_trackbar(val):
    global low_V
    global high_V
    low_V = val
    low_V = min(high_V - 1, low_V)
    cv.setTrackbarPos(low_V_name, window_detection_name, low_V)


def on_high_V_thresh_trackbar(val):
    global low_V
    global high_V
    high_V = val
    high_V = max(high_V, low_V + 1)
    cv.setTrackbarPos(high_V_name, window_detection_name, high_V)


def run_threshold_tester(video_src):

#parser = argparse.ArgumentParser(description='Code for Thresholding Operations using inRange tutorial.')
#parser.add_argument('--camera', help='Camera divide number.', default=0, type=int)
#args = parser.parse_args()

    cap = video.create_capture(video_src, fallback = None)

    # self.camera1 = 'camera1'

    # _ret1, self.frame1 = self.cam_feed1.read()

    # cv.namedWindow(self.camera1)
    # cv.setMouseCallback(self.camera1, self.onmouse)

    cv.namedWindow(window_capture_name)
    cv.namedWindow(window_detection_name)

    cv.createTrackbar(low_H_name, window_detection_name, low_H, max_value_H, on_low_H_thresh_trackbar)
    cv.createTrackbar(high_H_name, window_detection_name, high_H, max_value_H, on_high_H_thresh_trackbar)
    cv.createTrackbar(low_S_name, window_detection_name, low_S, max_value, on_low_S_thresh_trackbar)
    cv.createTrackbar(high_S_name, window_detection_name, high_S, max_value, on_high_S_thresh_trackbar)
    cv.createTrackbar(low_V_name, window_detection_name, low_V, max_value, on_low_V_thresh_trackbar)
    cv.createTrackbar(high_V_name, window_detection_name, high_V, max_value, on_high_V_thresh_trackbar)

    while True:

        ret, frame = cap.read()
        if frame is None:
            break

        frame_HSV = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        #HSV = Hue saturation value


        frame_threshold = cv.inRange(frame_HSV, (low_H, low_S, low_V), (high_H, high_S, high_V))
        cv.imshow(window_capture_name, frame)
        cv.imshow(window_detection_name, frame_threshold)


        key = cv.waitKey(30)
        if key == ord('q') or key == 27:
            break
            #low_H:0, low_S:51, low_V :180
            #high_H:0, high_S:255, high_V:255

#blue light filter
            #low_H:100, low_S:54, low_V :0
            #high_H:140, high_S:255, high_V:255
# red light filter
    # low_H:0, low_S:148, low_V :135
    # high_H:10, high_S:255, high_V:255

def run_threshold(video_src, colour_name, low_hsv , high_hsv):
    #low_hsv = (low_H, low_S, low_V)
    #high_hsv = (high_H, high_S, high_V)
    cap = video.create_capture(video_src, fallback = None)
    cv.namedWindow(window_capture_name+ " " + colour_name.upper())
    cv.namedWindow(window_detection_name)


    while True:

        ret, frame = cap.read()
        if frame is None:
            break

        frame_HSV = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        # HSV = Hue saturation value
        #red
        #frame_threshold = cv.inRange(frame_HSV, (0, 148, 135), (10, 255, 255))
        frame_threshold = cv.inRange(frame_HSV, low_hsv, high_hsv)

        cv.imshow(window_capture_name, frame)
        cv.imshow(window_detection_name, frame_threshold)
        key = cv.waitKey(30)
        if key == ord('q') or key == 27:
            break
            # low_H:0, low_S:51, low_V :180
            # high_H:0, high_S:255, high_V:255
