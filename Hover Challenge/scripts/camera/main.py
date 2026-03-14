
'''
Camshift tracker from https://github.com/opencv/opencv/blob/3.4/samples/python/camshift.py
================

This is a demo that shows mean-shift based tracking
You select a color objects sit tracks it.
This reads from video camera (0 by default, or the camera number the user enters)

[1] http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.14.7673

Usage:
------
    camshift.py [<video source>]

    To initialize tracking, select the object with mouse

Keys:
-----
    ESC   - exit
    b     - toggle back-projected probability visualization
'''

# Python 2/3 compatibility
from __future__ import print_function
import sys
import os
PY3 = sys.version_info[0] == 3

if PY3:
    xrange = range

import numpy as np
import cv2 as cv
import setup_cameras

# local module
import video
from video import presets
class Camera(object):
    def __init__(self, video_src):
        self.name = 'camera1'
        self.cam_feed = video.create_capture(video_src, fallback=None)



class App(object):
    def __init__(self, video_src1, video_src2):

        self.cam_feed1 = video.create_capture(video_src1, fallback = None)
        self.cam_feed2 = video.create_capture(video_src2, fallback = None)
        self.camera1 = 'camera1'
        self.camera2 = 'camera2'
        _ret1, self.frame1 = self.cam_feed1.read()
        _ret2, self.frame2 = self.cam_feed2.read()
        cv.namedWindow(self.camera1)
        cv.setMouseCallback(self.camera1, self.onmouse)

        cv.namedWindow(self.camera2)
        cv.setMouseCallback(self.camera2, self.onmouse)

        self.selection = None
        self.drag_start = None
        self.show_backproj = False
        self.track_window = None

    def onmouse(self, event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            self.drag_start = (x, y)
            self.track_window = None
        if self.drag_start:
            xmin = min(x, self.drag_start[0])
            ymin = min(y, self.drag_start[1])
            xmax = max(x, self.drag_start[0])
            ymax = max(y, self.drag_start[1])
            self.selection = (xmin, ymin, xmax, ymax)
        if event == cv.EVENT_LBUTTONUP:
            self.drag_start = None
            self.track_window = (xmin, ymin, xmax - xmin, ymax - ymin)

    def show_hist(self):
        bin_count = self.hist.shape[0]
        bin_w = 24
        img = np.zeros((256, bin_count*bin_w, 3), np.uint8)
        for i in xrange(bin_count):
            h = int(self.hist[i])
            cv.rectangle(img, (i*bin_w+2, 255), ((i+1)*bin_w-2, 255-h), (int(180.0*i/bin_count), 255, 255), -1)
        img = cv.cvtColor(img, cv.COLOR_HSV2BGR)
        cv.imshow('hist', img)

    def run(self):
        while True:
            _ret1, self.frame1 = self.cam_feed1.read()
            _ret2, self.frame2 = self.cam_feed2.read()
            vis1 = self.frame1.copy()
            if(self.frame2 is None):
                print("")
            vis2 = self.frame2.copy()
            hsv1 = cv.cvtColor(self.frame1, cv.COLOR_BGR2HSV)
            hsv2 = cv.cvtColor(self.frame2, cv.COLOR_BGR2HSV)
            mask1 = cv.inRange(hsv1, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
            mask2 = cv.inRange(hsv2, np.array((0., 60., 32.)), np.array((180., 255., 255.)))

            if self.selection:
                x0, y0, x1, y1 = self.selection
                hsv_roi = hsv1[y0:y1, x0:x1]
                mask_roi = mask1[y0:y1, x0:x1]
                hist = cv.calcHist( [hsv_roi], [0], mask_roi, [16], [0, 180] )
                cv.normalize(hist, hist, 0, 255, cv.NORM_MINMAX)
                self.hist = hist.reshape(-1)
                self.show_hist()

                vis_roi = vis1[y0:y1, x0:x1]
                cv.bitwise_not(vis_roi, vis_roi)
                vis1[mask1 == 0] = 0

            if self.track_window and self.track_window[2] > 0 and self.track_window[3] > 0:
                self.selection = None
                prob = cv.calcBackProject([hsv1], [0], self.hist, [0, 180], 1)
                prob &= mask1
                term_crit = ( cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1 )
                track_box, self.track_window = cv.CamShift(prob, self.track_window, term_crit)

                if self.show_backproj:
                    vis1[:] = prob[...,np.newaxis]
                try:
                    cv.ellipse(vis1, track_box, (0, 0, 255), 2)
                except:
                    print(track_box)

            cv.imshow(self.camera1, vis1)
            cv.imshow(self.camera2, vis2)

            ch = cv.waitKey(5)
            if ch == 27:
                break
            if ch == ord('b'):
                self.show_backproj = not self.show_backproj
        cv.destroyAllWindows()


if __name__ == '__main__':
    print(__doc__)
    import sys
    cameras = setup_cameras.get_available_cameras()
    print(cameras)
    #vid_path = "C:\\Users\\Manhands\\Documents\\Comp Sci\\McGill Aerohacks\\"
    vid_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))+"\\data\\"
    print( "video dir: " +vid_path)
    vid_1 = "drone video long.mp4"
    vid_2 = "drone video short.mp4"
    try:
        video_src1 = sys.argv[1]
    except:
        video_src1 = cameras[0]
        #video_src1 = vid_path + vid_1
    try:
        video_src2 = sys.argv[2]
    except:
        video_src2 = cameras[1]
        #video_src2 = vid_path + vid_2

    App(video_src1, video_src2).run()
