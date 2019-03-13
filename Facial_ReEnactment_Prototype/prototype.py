import cv2
import numpy as np
import dlib
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import time
import dlib

def dlib_rect_to_bb(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return (x, y, w, h)


sourceName = 'source.mp4'
targetName = 'target.mp4'
predictor_data = 'shape_predictor_68_face_landmarks.dat'

source = cv2.VideoCapture(sourceName)
target = cv2.VideoCapture(targetName)

#sourceVc = cv2.VideoCapture.open(sourceName)
#targetVc = cv2.VideoCapture.open(targetName)

sourceSuccess = True
targetSuccess = True
frameTime = 1/24
print(frameTime)

face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor(predictor_data)

fig = plt.figure()
srcWindow = fig.add_subplot(121)
trgWindow = fig.add_subplot(122)
#srcWindow = plt.subplot(121)
#trgWindow = plt.subplot(122)
sourceSuccess, sourceFrame = source.read()
targetSuccess, targetFrame = target.read()
sourceFrame = cv2.cvtColor(sourceFrame, cv2.COLOR_BGR2RGB)
targetFrame = cv2.cvtColor(targetFrame, cv2.COLOR_BGR2RGB)
srcPlot = srcWindow.imshow(sourceFrame)
trgPlot = trgWindow.imshow(targetFrame)
plt.ion()



#TODO - Separate processing stack in own file/class/module w/e

while targetSuccess and sourceSuccess:
    #get frames
    sourceSuccess, sourceFrame = source.read()
    targetSuccess, targetFrame = target.read()
    #convert from BGR to RGB
    sourceFrame = cv2.cvtColor(sourceFrame, cv2.COLOR_BGR2RGB)
    targetFrame = cv2.cvtColor(targetFrame, cv2.COLOR_BGR2RGB)
    #convert to gray for detection

    srcGray = cv2.cvtColor(sourceFrame, cv2.COLOR_RGB2GRAY)
    trgGray = cv2.cvtColor(sourceFrame, cv2.COLOR_RGB2GRAY)

    srcBB = face_detector(srcGray, 1)
    trgBB = face_detector(trgGray, 1)

    #for now assume one face only
    srcBB = srcBB[0]
    trgBB = trgBB[0]

    srcLandmarks = landmark_predictor(srcGray, srcBB)
    trgLandmarks = landmark_predictor(trgGray, trgBB)

    srcPlot.set_data(sourceFrame)
    trgPlot.set_data(targetFrame)

    x,y,w,h = dlib_rect_to_bb(srcBB)
    srcWindow.add_patch(matplotlib.patches.Rectangle((x,y), w, h, fill=False))
    for i in range(0,68):
        srcWindow.add_patch(matplotlib.patches.Circle((srcLandmarks.part(i).x, srcLandmarks.part(i).y), 5))
    x,y,w,h = dlib_rect_to_bb(trgBB)
    trgWindow.add_patch(matplotlib.patches.Rectangle((x,y),w,h, fill= False))
    for i in range(0,68):
        trgWindow.add_patch(matplotlib.patches.Circle((trgLandmarks.part(i).x, trgLandmarks.part(i).y), 5))

    plt.pause(0.1)

source.release()
target.release()