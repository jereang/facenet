# -*- coding: utf-8 -*-

""" 检测视频帧是否有运动目标
"""

import cv2
from datetime import datetime
import time

class MotionDetector():
    """ 运动检测器
    """

    def __init__(self, threshold=1, doRecord=True, showWindows=True):
        self.writer = None
        self.font = None
        self.doRecord = doRecord
        self.show = showWindows
        self.frame = None

        self.capture = cv2.VideoCapture(0)
        _, self.frame = self.capture.read()
        if doRecord:
            self.initRecorder()

        self.gray_frame = None
        self.absdiff_frame = None
        self.previous_frame = None
        self.average_frame = None

        self.surface = self.capture.get(3) * self.capture.get(4)
        self.currentsurface = 0
        self.currentcontours = None
        self.threshold = threshold
        self.isRecording = False
        self.trigger_time = 0 # Hold timestamp of the last detection

        if showWindows:
            cv2.namedWindow("Image")
            #cv2.createTrackbar("Detection threshold: ", "Image", self.threshold, 100, self.onChange)

    def initRecorder(self): # create the recorder
        pass
        #codec = cv.CV_FOURCC('M', 'J', 'P', "G")
        #self.writer=cv.CreateVideoWriter(datetime.now().strftime("%b-%d_%H_%M_%S")+".wmv", codec, 5, cv.GetSize(self.frame), 1)
        #FPS set to 5 because it seems to be the fps of my cam but should be adjusted to your needs
        #self.font = cv.InitFont(cv.CV_FONT_HERSHEY_SIMPLEX, 1, 1, 0, 2, 8) # create a font

    def run(self):
        started = time.time()
        while True:
            _, currentframe = self.capture.read()
            instant = time.time() # get timestamp of the frame

            self.processImage(currentframe) # process the iamge

            if not self.isRecording:
                if self.somethingHasMoved():
                    self.trigger_time = instant # update the trigger_time
                    if instant > started + 10: # wait 5 second after the webcam start for luminosity adjusting etc..
                        print("Something is moving !")
                        if self.doRecord:
                            self.isRecording = True
                cv2.drawContours(currentframe, self.currentcontours, -1, (0, 0, 255), 1)
            else:
                if instant >= self.trigger_time + 10:
                    print("Stop recording")
                    self.isRecording = False
                else:
                    pass
                    #cv2.putText(currentframe, datetime.now().strftime("%b %d, %H:%M:%S"), (25,30),self.font, 0)
                    #cv.WriteFrame(self.writer, currentframe)

            if self.show:
                cv2.imshow("Image", currentframe)

            if cv2.waitKey(10) & 0xFF == ord('q'): # break if user enters 'q'
                break

    def processImage(self, curframe):
        self.average_frame = cv2.blur(curframe, (21, 21))
        if self.previous_frame is None:
            self.previous_frame = self.average_frame

        self.absdiff_frame = cv2.absdiff(self.average_frame, self.previous_frame)
        self.previous_frame = self.average_frame

        self.gray_frame = cv2.cvtColor(self.absdiff_frame, cv2.COLOR_BGR2GRAY)
        _, self.gray_frame = cv2.threshold(self.gray_frame, 50, 255, cv2.THRESH_BINARY)

        self.gray_frame = cv2.dilate(self.gray_frame, None, 15) # to get object blobs
        self.gray_frame = cv2.erode(self.gray_frame, None, 10)
        cv2.imshow('gray', self.gray_frame)

    def somethingHasMoved(self):
        """ 检测运动
        """
        ## find contours
        _, contours, _ = cv2.findContours(self.gray_frame, cv2.RETR_EXTERNAL,
                                          cv2.CHAIN_APPROX_SIMPLE)

        self.currentcontours = contours

        for contour in contours:
            self.currentsurface += cv2.contourArea(contour)

        # calc the average of contour area on the total size
        avg = (self.currentsurface * 100) / self.surface

        print(avg, self.currentsurface, self.surface)
        self.currentsurface = 0

        if avg >= self.threshold:
            return True
        else:
            return False

if __name__ == '__main__':
    detector = MotionDetector(doRecord=False)
    detector.run()