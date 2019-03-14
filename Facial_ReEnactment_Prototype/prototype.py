import cv2
import numpy as np
import dlib
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import time
import dlib
from profiling import PerformanceTimer
from transforms import dlib_rect_to_bb, landmarks_to_points, verts_to_indices, landmark_indices_to_triangles
from processing import get_scaled_rgb_frame, get_face_bb_landmarks, triangulate_landmarks, get_transforms
from visualisation import view_landmarks, debug_fill_triangles


DEBUG_TRIANGLES = False

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
srcWindow = fig.add_subplot(131)
trgWindow = fig.add_subplot(132)
resWindow = fig.add_subplot(133)

scalingFactor = 0.25
sourceSuccess, sourceFrame = get_scaled_rgb_frame(source, scalingFactor)
targetSuccess, targetFrame = get_scaled_rgb_frame(target, scalingFactor)

#scalingFactor = 0.12
srcPlot = srcWindow.imshow(sourceFrame)
trgPlot = trgWindow.imshow(targetFrame)
resPlot = resWindow.imshow(targetFrame)
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
    

    #instead of triangulating both, triangulate only source, then use landmark indices ( which should be fixed points ) 
    #to obtain the second triangulation to obtain matching triangle areas
    profiler.tick("Triangulate Landmarks")
    srcTriangles = triangulate_landmarks(srcLandmarks, im_h, im_w)
    srcLandmarkTrianglesIndices, srcTriangles = verts_to_indices(srcTriangles, srcLandmarks)
    trgTriangles = landmark_indices_to_triangles(trgLandmarks, srcLandmarkTrianglesIndices)

    profiler.tock()

    
    
    view_landmarks(srcLandmarks, trgLandmarks, srcWindow, trgWindow)
    if(DEBUG_TRIANGLES):
        debug_fill_triangles(srcTriangles, trgTriangles, srcWindow, trgWindow, sourceFrame, targetFrame)
    
    srcPlot.set_data(sourceFrame)
    trgPlot.set_data(targetFrame)


    transforms = get_transforms(srcTriangles, trgTriangles)
    #srcLocalTris = []
    #trgLocalTris = []
    resFrame = np.copy(targetFrame)
    for i, transform in enumerate(transforms):
        #skip while debugging triangles
        if(DEBUG_TRIANGLES):
            continue
        srcTri = srcTriangles[i]
        trgTri = trgTriangles[i]
        srcLocalTris = []
        trgLocalTris = []
        M, srcBB, trgBB = transform
        if(trgBB[2] > im_h or trgBB[3] > im_w or srcBB[2] > im_h or srcBB[3] > im_w):
            continue

        #keep in mind the target has become our data source
        #and we need to wrap it with source as reference for dimensions 

        for j in range(0,3):
            srcLocalTriangle = (srcTri[j][0] - srcBB[0], srcTri[j][1] - srcBB[1])
            trgLocalTriangle = (trgTri[j][0] - trgBB[0], trgTri[j][1] - trgBB[1])
            srcLocalTris.append(srcLocalTriangle)
            trgLocalTris.append(trgLocalTriangle)
        #our source are pixels from the target image
        srcCrop = targetFrame[trgBB[1] : trgBB[1] + trgBB[3], trgBB[0] : trgBB[0] + trgBB[2]]
        #target is the result image
        #dimensions will corespond to the original expresion source
        trgCrop = cv2.warpAffine(srcCrop, M, (srcBB[2], srcBB[3]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        mask = np.zeros((srcBB[3], srcBB[2], 3), dtype = np.float32)
        cv2.fillConvexPoly(mask, np.int32(trgLocalTris),(1.0, 1.0, 1.0), 16, 0)
        trgCrop = trgCrop * mask
        resFrame[trgBB[1]:trgBB[1]+trgBB[3], trgBB[0]:trgBB[0]+trgBB[2]] = resFrame[trgBB[1]:trgBB[1]+trgBB[3], trgBB[0]:trgBB[0]+trgBB[2]] * ((1.0,1.0,1.0) - mask)
        resFrame[trgBB[1]:trgBB[1]+trgBB[3], trgBB[0]:trgBB[0]+trgBB[2]] = resFrame[trgBB[1]:trgBB[1]+trgBB[3], trgBB[0]:trgBB[0]+trgBB[2]] + trgCrop
        
    #convert in a neutral scale invariant space
    resPlot.set_data(resFrame)
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



