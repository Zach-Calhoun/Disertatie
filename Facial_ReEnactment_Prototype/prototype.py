import cv2
import numpy as np
import dlib
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import time
import dlib
import argparse

from profiling import PerformanceTimer
from transforms import dlib_rect_to_bb, landmarks_to_points, verts_to_indices, landmark_indices_to_triangles, apply_transform
from processing import get_scaled_rgb_frame, get_face_bb_landmarks, triangulate_landmarks, get_transforms, get_face_coordinates_system
from visualisation import view_landmarks, debug_fill_triangles
from utils import *

parser = argparse.ArgumentParser(description='Facial re-enactment prototype from source to target clip')
parser.add_argument("--source", metavar="/path/to/source.mp4", help="path to source video clip")
parser.add_argument("--target", metavar="/path/to/target.mp4", help="path to target video clip")
parser.add_argument("--output", required=False, metavar="output.mp4", help="output file path, optional")
parser.add_argument("--debugtriangles", required=False, )
args = parser.parse_args()

assert args.source, "Argument --source must be provided"
assert args.target, "Argument --target must be provided"



DEBUG_TRIANGLES = args.debugtriangles
generated_colors = False
generated_triangulation = False
colors = []
    
profiler = PerformanceTimer()

sourceName = args.source
targetName = args.target
outputName = args.output

predictor_data = 'shape_predictor_68_face_landmarks.dat'

source = cv2.VideoCapture(sourceName)
target = cv2.VideoCapture(targetName)

#sourceVc = cv2.VideoCapture.open(sourceName)
#targetVc = cv2.VideoCapture.open(targetName)

sourceSuccess = True
targetSuccess = True
frameTime = 1/24

face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor(predictor_data)

fig = plt.figure()

#source frame
srcWindow = fig.add_subplot(231)
#target frame
trgWindow = fig.add_subplot(232)
#result frame
resWindow = fig.add_subplot(233)

#source local face space
src_face_space_window = fig.add_subplot(234)
#target local face space
trg_face_space_window = fig.add_subplot(235)

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
srcLandmarkTrianglesIndices = []
while targetSuccess and sourceSuccess:
 
    print("Processing frame {} ...".format(frameCount))
    frameCount = frameCount + 1
    profiler.tick("Process Frame")
    #get frames
    sourceSuccess, sourceFrame = get_scaled_rgb_frame(source, scalingFactor)
    targetSuccess, targetFrame = get_scaled_rgb_frame(target, scalingFactor)

    #convert to gray for detection
    profiler.tick("Get Face BB and landmarks")
    im_h,im_w = sourceFrame.shape[:2]
    
    #this is very slow, google says face detection is slow compared to landmark extraction
    #TODO extract face region more rarely, maybe even half the rate of landmark prediction
    srcBB, srcLandmarks = get_face_bb_landmarks(sourceFrame, face_detector, landmark_predictor)
    trgBB, trgLandmarks = get_face_bb_landmarks(targetFrame, face_detector, landmark_predictor)
    
    # sourceAxes = np.float32([(srcLandmarks[FACE_AXIS_TOP_INDEX][0], srcLandmarks[FACE_AXIS_TOP_INDEX][1]), 
    #                         (srcLandmarks[FACE_AXIS_RIGHT_INDEX][0],srcLandmarks[FACE_AXIS_RIGHT_INDEX][1]),  
                            
    #                         (srcLandmarks[FACE_AXIS_LEFT_INDEX][0],srcLandmarks[FACE_AXIS_LEFT_INDEX][1])
    #                         ])
    # targetAxes = np.float32([(trgLandmarks[FACE_AXIS_TOP_INDEX][0], trgLandmarks[FACE_AXIS_TOP_INDEX][1]), 
    #                         (trgLandmarks[FACE_AXIS_RIGHT_INDEX][0],trgLandmarks[FACE_AXIS_RIGHT_INDEX][1]),  
                         
    #                         (trgLandmarks[FACE_AXIS_LEFT_INDEX][0],trgLandmarks[FACE_AXIS_LEFT_INDEX][1])
    #                         ])

    # target_to_source_T = cv2.getAffineTransform(targetAxes, sourceAxes)
    # source_to_target_T = cv2.getAffineTransform(sourceAxes, targetAxes)

    #for now just transform everything in source space, to see difference
    #targetFrame =  cv2.warpAffine(targetFrame, target_to_source_T, (im_w, im_h), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    #TODO go from target to source, perform calculations, but go back to target space when getting pixel data
    #trgLandmarks = apply_transform(trgLandmarks, target_to_source_T)

    profiler.tock()
    if srcBB is None or trgBB is None:
        continue
    
    #draw frames, bounding box and landmarks
    srcPlot.set_data(sourceFrame)
    trgPlot.set_data(targetFrame)

    x,y,w,h = dlib_rect_to_bb(srcBB)
    srcWindow.add_patch(matplotlib.patches.Rectangle((x,y), w, h, fill=False))
    #triangulate
   

    #instead of triangulating both, triangulate only source, then use landmark indices ( which should be fixed points ) 
    #to obtain the second triangulation to obtain matching triangle areas
    #PS : now triangulating just once 
    profiler.tick("Triangulate Landmarks")
    #TODO triangulate more than once, every few frames to adapt to chaning structure
    srcTriangles = None
    if not generated_triangulation:
        srcTriangles = triangulate_landmarks(srcLandmarks, im_h, im_w)
        srcLandmarkTrianglesIndices, srcTriangles = verts_to_indices(srcTriangles, srcLandmarks)
        #generated_triangulation = True
    else:
        srcTriangles = landmark_indices_to_triangles(srcLandmarks, srcLandmarkTrianglesIndices)
    
   
    trgTriangles = landmark_indices_to_triangles(trgLandmarks, srcLandmarkTrianglesIndices)

    source_face_center, source_x_scale, source_y_scale, source_rot, source_local_landmarks = get_face_coordinates_system(srcLandmarks,src_face_space_window )
    target_face_center, target_x_scale, target_y_scale, target_rot, target_local_landmarks = get_face_coordinates_system(trgLandmarks,trg_face_space_window )

    if DEBUG_TRIANGLES and generated_colors == False:
        generated_colors = True
        for i in range(0, len(trgTriangles)):
            colors.append(np.uint8(np.random.uniform(0, 255, 3)))

        
    profiler.tock()

    
    
    view_landmarks(srcLandmarks, trgLandmarks, srcWindow, trgWindow)
    if(DEBUG_TRIANGLES):
        debug_fill_triangles(srcTriangles, trgTriangles, srcWindow, trgWindow, sourceFrame, targetFrame, colors=None)
    
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
        M, srcBB, localSrcTriangle, trgBB, localTrgTriangle = transform
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
        resFrame[srcBB[1]:srcBB[1]+srcBB[3], srcBB[0]:srcBB[0]+srcBB[2]] = resFrame[srcBB[1]:srcBB[1]+srcBB[3], srcBB[0]:srcBB[0]+srcBB[2]] * ((1.0,1.0,1.0) - mask)
        resFrame[srcBB[1]:srcBB[1]+srcBB[3], srcBB[0]:srcBB[0]+srcBB[2]] = resFrame[srcBB[1]:srcBB[1]+srcBB[3], srcBB[0]:srcBB[0]+srcBB[2]] + trgCrop
        # resFrame[trgBB[1]:trgBB[1]+trgBB[3], trgBB[0]:trgBB[0]+trgBB[2]] = resFrame[trgBB[1]:trgBB[1]+trgBB[3], trgBB[0]:trgBB[0]+trgBB[2]] * ((1.0,1.0,1.0) - mask)
        # resFrame[trgBB[1]:trgBB[1]+trgBB[3], trgBB[0]:trgBB[0]+trgBB[2]] = resFrame[trgBB[1]:trgBB[1]+trgBB[3], trgBB[0]:trgBB[0]+trgBB[2]] + trgCrop
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



