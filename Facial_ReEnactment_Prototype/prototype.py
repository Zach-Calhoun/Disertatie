import cv2
import numpy as np
import dlib
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import time
import dlib
from profiling import PerformanceTimer
from transforms import dlib_rect_to_bb, landmarks_to_points
from processing import get_scaled_rgb_frame, get_face_bb_landmarks, triangulate_landmarks

profiler = PerformanceTimer()

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

scalingFactor = 0.25
sourceSuccess, sourceFrame = get_scaled_rgb_frame(source, scalingFactor)
targetSuccess, targetFrame = get_scaled_rgb_frame(target, scalingFactor)

#scalingFactor = 0.12
srcPlot = srcWindow.imshow(sourceFrame)
trgPlot = trgWindow.imshow(targetFrame)
plt.ion()
plt.show()


frameCount = 1

while targetSuccess and sourceSuccess:
 
    print("Processing frame {} ...".format(frameCount))
    frameCount = frameCount + 1
    profiler.tick("Process Frame")
    #get frames
    sourceSuccess, sourceFrame = get_scaled_rgb_frame(source, scalingFactor)
    targetSuccess, targetFrame = get_scaled_rgb_frame(target, scalingFactor)

    #convert to gray for detection
    profiler.tick("Get Face BB and landmarks")

    #this is very slow, google says face detection is slow compared to landmark extraction
    #TODO extract face region more rarely, maybe even half the rate of landmark prediction
    srcBB, srcLandmarks = get_face_bb_landmarks(sourceFrame, face_detector, landmark_predictor)
    trgBB, trgLandmarks = get_face_bb_landmarks(targetFrame, face_detector, landmark_predictor)
    profiler.tock()
    if srcBB is None or trgBB is None:
        continue
    
    #draw frames, bounding box and landmarks
    srcPlot.set_data(sourceFrame)
    trgPlot.set_data(targetFrame)

    x,y,w,h = dlib_rect_to_bb(srcBB)
    srcWindow.add_patch(matplotlib.patches.Rectangle((x,y), w, h, fill=False))
    #triangulate
    im_h,im_w = sourceFrame.shape[:2]
    

    #TODO , do not triangulate eachtime, use frame deltas, otherwise triangulation will be unstable
    profiler.tick("Triangulat Landmarks")
    srcTriangles = triangulate_landmarks(srcLandmarks, im_h, im_w)
    trgTriangles = triangulate_landmarks(trgLandmarks, im_h, im_w)
    profiler.tock()

    landmarks = zip(srcLandmarks, trgLandmarks)
    
    #combining sequences to reduce loop overhead

    for srcPoint, trgPoint in landmarks:
        srcWindow.add_patch(matplotlib.patches.Circle(srcPoint, 1))
        trgWindow.add_patch(matplotlib.patches.Circle(trgPoint, 1))

    for srcTri, trgTri in zip(srcTriangles, trgTriangles):
        srcWindow.add_patch(matplotlib.patches.Polygon(srcTri, True, fill=False))
        trgWindow.add_patch(matplotlib.patches.Polygon(trgTri, True, fill=False))
   

    #convert in a neutral scale invariant space

    #move stuff arround
   

    profiler.tock()
    #clear patches
    plt.pause(frameTime)
    for p in reversed(srcWindow.patches):
        p.remove()
    for p in reversed(trgWindow.patches):
        p.remove()

source.release()
target.release()



