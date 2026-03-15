'''
https://github.com/jlengrand/image_processing/blob/master/LedDetector/data/output/2200.jpg
Created on 25 mai 2012

@author: jlengrand
'''
# Python 2/3 compatibility
from __future__ import print_function

import sys
PY3 = sys.version_info[0] == 3

if PY3:
    xrange = range
import os
import numpy as np
import cv2 as cv

# local module
import video
import setup_cameras

class Led_Detect(object):
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


        #frame processing

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

    def to_gray(self, frame):
        """
        Converts the input in grey levels
        Returns a one channel image
        """
        return cv.cvtColor(frame, cv.COLOR_BGR2GRAY)


    # def grey_histogram(self, img, nBins=64):
    #     """
    #     Returns a one dimension histogram for the given image
    #     The image is expected to have one channel, 8 bits depth
    #     nBins can be defined between 1 and 255
    #     """
    #     hist_size = [nBins]
    #     h_ranges = [0, 256]
    #     #hist = cv.CreateHist(hist_size, cv.HIST_ARRAY, [[0, 255]], 1)
    #     hist = cv.calcHist(images = [img], mask= None, channels=[0],  histSize=hist_size, ranges=h_ranges)
    #     cv.normalize(hist, hist, 0, 255, cv.NORM_MINMAX)
    #
    #     return hist

    def extract_bright(self, grey_frame, histogram=False):
        """
        Extracts brightest part of the image.
        Expected to be the LEDs (provided that there is a dark background)
        Returns a Thresholded image
        histgram defines if we use the hist calculation to find the best margin
        """
        ## Searches for image maximum (brightest pixel)
        # We expect the LEDs to be brighter than the rest of the image
        [minVal, maxVal, minLoc, maxLoc] = cv.minMaxLoc(grey_frame)
        print("Brightest pixel val is %d" % maxVal)


        # We retrieve only the brightest part of the image
        # Here is use a fixed margin (80%), but you can use hist to enhance this one
        margin = 0.9
        thresh = int(maxVal * margin)  # in pix value to be extracted


        # if histogram:
        #     ## Histogram may be used to wisely define the margin
        #     # We expect a huge spike corresponding to the mean of the background
        #     # and another smaller spike of bright values (the LEDs)
        #     hist = self.grey_histogram(grey_frame, nBins=64)
        #     [hminValue, hmaxValue, hminIdx, hmaxIdx] = cv.GetMinMaxHistValue(hist)
        #     margin = 0  # statistics to be calculated using hist data
        # else:
        #     margin = 0.8

        print("Threshold is defined as %d" % (thresh))

        #ALGO CHOICE, use built in histogram?
        # In the first case, global thresholding with a value of 127 is applied.  ##HERE using margin 0.8 * max
            #ret1, th1 = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
        # In the second case, Otsu's thresholding is applied directly.
            #ret2, th2 = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        # In the third case, the image is first filtered with a 5x5 gaussian kernel to remove the noise, then Otsu thresholding is applied.
            #blur = cv.GaussianBlur(img, (5, 5), 0)
            #ret3, th3 = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        # See how noise filtering improves the result.
        #https://docs.opencv.org/3.4/d7/d4d/tutorial_py_thresholding.html
        cv.imshow('grey frame', grey_frame)
        #using global threshold
        ret,thresh_img1 = cv.threshold(grey_frame, thresh, 255, cv.THRESH_BINARY)
        cv.imshow('thresh image - global thresholding', thresh_img1)
        # #using otsu only
        # ret,thresh_img2 = cv.threshold(grey_frame, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
        # cv.imshow('thresh image - Otsu Threshold applied directly', thresh_img2)
        # #using otsu and blur
        # blur = cv.GaussianBlur(grey_frame, (5, 5), 0)
        # ret3, thresh_img = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        # cv.imshow('thresh image - Otsu Threshold with blur', thresh_img)
        return thresh_img1

    def find_leds(self, thresh_img):
        """
        Given a binary image showing the brightest pixels in an image,
        returns a result image, displaying found leds in a rectangle
        """
        """
        contours = cv.FindContours(thresh_img,
                                   cv.CreateMemStorage(),
                                   mode=cv.CV_RETR_EXTERNAL,
                                   method=cv.CV_CHAIN_APPROX_NONE,
                                   offset=(0, 0))
        """
        #im2, contours, hierarchy = cv.findContours(thresh_img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        contours, hierarchy = cv.findContours(thresh_img, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

        regions = []
        #while contours:
        for points in contours:
            p = np.squeeze(points,axis=1).tolist()
            pts = [pts for pts in p]
            x,y = zip(*pts)
            x, y = zip(*pts)
            min_x, min_y = min(x), min(y)
            width, height = max(x) - min_x + 1, max(y) - min_y + 1
            regions.append((min_x, min_y, width, height))
            #pts =[pt[0] for pt in contours]
            #print(pts)

            #contours = contours.h_next()

            #out_img =  cv.CreateImage(cv.GetSize(grey_img), 8, 3)
            out_img = np.zeros(thresh_img.shape, np.int16)
            #out_img = np.zeros((grey_img.shape[0], grey_img.shape[1]), np.int16)


        for x, y, width, height in regions:
            pt1 = x, y
            pt2 = x + width, y + height
            color = (0, 0, 255, 0)
            thickness = 2
            #cv.Rectangle(out_img, pt1, pt2, color, 2)
            #cv2.rectangle(out_img, start_point, end_point, color, thickness)
            cv.rectangle(out_img, pt1, pt2, color, thickness)
        return out_img, regions

    def leds_positions(self, regions):
        """
        Function using the regions in input to calculate the position of found leds
        """
        centers = []
        for x, y, width, height in regions:
            centers.append([x + (width / 2), y + (height / 2)])

        return centers
    def process(self, cam_feed):
        _ret1, frame = cam_feed.read()
        '''
        vis1 = self.frame1.copy()
        vis2 = self.frame2.copy()
        hsv1 = cv.cvtColor(self.frame1, cv.COLOR_BGR2HSV)
        hsv2 = cv.cvtColor(self.frame2, cv.COLOR_BGR2HSV)
        mask1 = cv.inRange(hsv1, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
        mask2 = cv.inRange(hsv2, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
        video_file = self.
        '''
        ####
        # Starts image processing here
        ####
        # Turns to one channel image
        grey_img = self.to_gray(frame)
        #display_img(grey_img, 1000)

        # Detect brightest point in image :
        thresh_img = self.extract_bright(grey_img)
        #display_img(thresh_img, delay=1000)

        # We want to extract the elements left, and count their number
        led_img, regions = self.find_leds(thresh_img)
        #display_img(led_img, delay=1000)

        cv.imshow('frame', led_img)

        centers = self.leds_positions(regions)

        print
        "Total number of Leds found : %d !" % (len(centers))
        print
        "###"
        print
        "Led positions :"
        for c in centers:
            print
            "x : %d; y : %d" % (c[0], c[1])
        print
        "###"

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
            self.process(self.cam_feed1)

            ch = cv.waitKey(5)
            if ch == 27:
                break
            if ch == ord('b'):
                self.show_backproj = not self.show_backproj
        cv.destroyAllWindows()

'''
# ---- Useful functions ----

def init_video(video_file):
    """
    Given the name of the video, prepares the stream and checks that everything works as attended
    """
    capture = cv.CaptureFromFile(video_file)

    nFrames = int(cv.GetCaptureProperty(capture, cv.CV_CAP_PROP_FRAME_COUNT))
    fps = cv.GetCaptureProperty(capture, cv.CV_CAP_PROP_FPS)
    if fps != 0:
        waitPerFrameInMillisec = int(1 / fps * 1000 / 1)

        print
        'Num. Frames = ', nFrames
        print
        'Frame Rate = ', fps, ' frames per sec'

        print
        '----'

        return capture
    else:
        return None


def display_img(img):
    """
    One liner that displays the given image on screen
    """
    cv.NamedWindow("Vid", cv.CV_WINDOW_AUTOSIZE)
    cv.ShowImage("Vid", img)


def display_video(my_video, frame_inc=100, delay=100):
    """
    Displays frames of the video in a dumb way.
    Used to see if everything is working fine
    my_video = cvCapture object
    frame_inc = Nmber of increments between each frame displayed
    delay = time delay between each image 
    """
    cpt = 0
    img = cv.QueryFrame(my_video)

    if img != None:
        cv.NamedWindow("Vid", cv.CV_WINDOW_AUTOSIZE)
    else:
        return None

    nFrames = int(cv.GetCaptureProperty(my_video, cv.CV_CAP_PROP_FRAME_COUNT))
    while cpt < nFrames:
        for ii in range(frame_inc):
            img = cv.QueryFrame(my_video)
            cpt + 1

        cv.ShowImage("Vid", img)
        cv.WaitKey(delay)


def grab_images(video_file, frame_inc=100, delay=100):
    """
    Walks through the entire video and save image for each increment
    """
    my_video = init_video(video_file)
    if my_video != None:
        # Display the video and save evry increment frames
        cpt = 0
        img = cv.QueryFrame(my_video)

        if img != None:
            cv.NamedWindow("Vid", cv.CV_WINDOW_AUTOSIZE)
        else:
            return None

        nFrames = int(cv.GetCaptureProperty(my_video, cv.CV_CAP_PROP_FRAME_COUNT))
        while cpt < nFrames:
            for ii in range(frame_inc):
                img = cv.QueryFrame(my_video)
                cpt += 1

            cv.ShowImage("Vid", img)
            out_name = "data/output/" + str(cpt) + ".jpg"
            cv.SaveImage(out_name, img)
            print
            out_name, str(nFrames)
            cv.WaitKey(delay)
    else:
        return None


def to_gray(img):
    """
    Converts the input in grey levels
    Returns a one channel image
    """

    grey_img = cv.CreateImage(cv.GetSize(img), img.depth, 1)
    cv.CvtColor(img, grey_img, cv.CV_RGB2GRAY)

    return grey_img


def grey_histogram(img, nBins=64):
    """
    Returns a one dimension histogram for the given image
    The image is expected to have one channel, 8 bits depth
    nBins can be defined between 1 and 255 
    """
    hist_size = [nBins]
    h_ranges = [0, 255]
    hist = cv.CreateHist(hist_size, cv.CV_HIST_ARRAY, [[0, 255]], 1)
    cv.CalcHist([img], hist)

    return hist


def extract_bright(grey_img, histogram=False):
    """
    Extracts brightest part of the image.
    Expected to be the LEDs (provided that there is a dark background)
    Returns a Thresholded image
    histgram defines if we use the hist calculation to find the best margin
    """
    ## Searches for image maximum (brightest pixel)
    # We expect the LEDs to be brighter than the rest of the image
    [minVal, maxVal, minLoc, maxLoc] = cv.MinMaxLoc(grey_img)
    print
    "Brightest pixel val is %d" % (maxVal)

    # We retrieve only the brightest part of the image
    # Here is use a fixed margin (80%), but you can use hist to enhance this one    
    if 0:
        ## Histogram may be used to wisely define the margin
        # We expect a huge spike corresponding to the mean of the background
        # and another smaller spike of bright values (the LEDs)
        hist = grey_histogram(img, nBins=64)
        [hminValue, hmaxValue, hminIdx, hmaxIdx] = cv.GetMinMaxHistValue(hist)
        margin = 0  # statistics to be calculated using hist data
    else:
        margin = 0.8

    thresh = int(maxVal * margin)  # in pix value to be extracted
    print
    "Threshold is defined as %d" % (thresh)

    thresh_img = cv.CreateImage(cv.GetSize(img), img.depth, 1)
    cv.Threshold(grey_img, thresh_img, thresh, 255, cv.CV_THRESH_BINARY)

    return thresh_img


def find_leds(thresh_img):
    """
    Given a binary image showing the brightest pixels in an image, 
    returns a result image, displaying found leds in a rectangle
    """
    contours = cv.FindContours(thresh_img,
                               cv.CreateMemStorage(),
                               mode=cv.CV_RETR_EXTERNAL,
                               method=cv.CV_CHAIN_APPROX_NONE,
                               offset=(0, 0))

    regions = []
    while contours:
        pts = [pt for pt in contours]
        x, y = zip(*pts)
        min_x, min_y = min(x), min(y)
        width, height = max(x) - min_x + 1, max(y) - min_y + 1
        regions.append((min_x, min_y, width, height))
        contours = contours.h_next()

        out_img = cv.CreateImage(cv.GetSize(grey_img), 8, 3)
    for x, y, width, height in regions:
        pt1 = x, y
        pt2 = x + width, y + height
        color = (0, 0, 255, 0)
        cv.Rectangle(out_img, pt1, pt2, color, 2)

    return out_img, regions


def leds_positions(regions):
    """
    Function using the regions in input to calculate the position of found leds
    """
    centers = []
    for x, y, width, height in regions:
        centers.append([x + (width / 2), y + (height / 2)])

    return centers
'''

if __name__ == '__main__':

    print(__doc__)
    import sys

    cameras = setup_cameras.get_available_cameras()
    print(cameras)
    # vid_path = "C:\\Users\\Manhands\\Documents\\Comp Sci\\McGill Aerohacks\\"
    vid_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__))) + "\\data\\"
    print("video dir: " + vid_path)
    # vid_1 = "drone video long.mp4"
    # vid_2 = "drone video short.mp4"
    vid_1 = "drone cam 1.mp4"
    vid_2 = "drone cam 2.mp4"

    try:
        video_src1 = sys.argv[1]
    except:
        # video_src1 = cameras[0]
        video_src1 = vid_path + vid_1
    try:
        video_src2 = sys.argv[2]
    except:
        # video_src2 = cameras[0]
        video_src2 = vid_path + vid_2

    Led_Detect(video_src1, video_src2).run()


