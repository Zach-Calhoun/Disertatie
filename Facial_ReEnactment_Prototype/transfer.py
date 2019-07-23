#transfer.py

import cv2
import numpy as np
import dlib
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt

from expression_prototypes import get_expression_prototypes, get_matching_expression_prototypes
from profiling import PerformanceTimer
from transforms import dlib_rect_to_bb, landmarks_to_points, verts_to_indices, landmark_indices_to_triangles, apply_transform
from processing import get_scaled_rgb_frame, get_face_bb_landmarks, triangulate_landmarks, get_transforms, get_face_coordinates_system, landmarks_to_image_space
from visualisation import view_landmarks, debug_fill_triangles
from utils import *

generated_colors = False
generated_triangulation = False
srcLandmarkTrianglesIndices = []
predictor_data = 'shape_predictor_68_face_landmarks.dat'
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor(predictor_data)
colors = []


def transfer(sourceFrame, targetFrame, srcWindow, trgWindow,srcPlot, trgPlot, DEBUG_TRIANGLES=False, profiler=PerformanceTimer()):

    #convert to gray for detection
    profiler.tick("Get Face BB and landmarks")
    im_h,im_w = sourceFrame.shape[:2]
    trg_h, trg_w = targetFrame.shape[:2]
    
    #this is very slow, google says face detection is slow compared to landmark extraction
    #TODO extract face region more rarely, maybe even half the rate of landmark prediction
    srcBB, srcLandmarks = get_face_bb_landmarks(sourceFrame, face_detector, landmark_predictor)
    trgBB, trgLandmarks = get_face_bb_landmarks(targetFrame, face_detector, landmark_predictor)
    
    profiler.tock()
    if srcBB is None or trgBB is None:
        print("No face in frame!")
        return
    
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

    #todo account for face perspective rotation
    #obtain landmark data in face space
    source_face_center, source_x_scale, source_y_scale, source_rot, source_local_landmarks = get_face_coordinates_system(srcLandmarks)
    target_face_center, target_x_scale, target_y_scale, target_rot, target_local_landmarks = get_face_coordinates_system(trgLandmarks)


    transformed_source_landmarks = landmarks_to_image_space(source_local_landmarks, target_rot, target_face_center, target_x_scale, target_y_scale, trgWindow)
    #do not move arround nose or face edges
    source_to_target_landmarks = trgLandmarks[0:17] + transformed_source_landmarks[17:27] + trgLandmarks[27:36] + transformed_source_landmarks[36:]
    #source_to_target_landmarks = trgLandmarks[0:17] + list(map(lambda x : np.add(x[0],x[1]),zip(trgLandmarks[17:],src_landmark_deltas[17:])))
    #generate local triangle data in face space using previous triangulation and new local landmarks
    # source_local_triangles = landmark_indices_to_triangles(source_local_landmarks, srcLandmarkTrianglesIndices)
    source_to_target_triangles = landmark_indices_to_triangles(source_to_target_landmarks, srcLandmarkTrianglesIndices)
    
    #move source local triangles to target local space and calculate from there
    if DEBUG_TRIANGLES and generated_colors == False:
        generated_colors = True
        for i in range(0, len(trgTriangles)):
            colors.append(np.uint8(np.random.uniform(0, 255, 3)))

        
    profiler.tock()

    #TODO
    #transform from source to prototype
    #transform from transformed prototype  to target
    
    view_landmarks(srcLandmarks, trgLandmarks, srcWindow, trgWindow)
    if(DEBUG_TRIANGLES):
        debug_fill_triangles(srcTriangles, trgTriangles, srcWindow, trgWindow, sourceFrame, targetFrame, colors=None)
    
    srcPlot.set_data(sourceFrame)
    trgPlot.set_data(targetFrame)

    #TODO ignore edge landmarks when transforming, as to not affect target shape face, just inside features
    #TODO keep a reference point for each triangle so that at least one point remains static
    #todo refactor this
    transforms = get_transforms(source_to_target_triangles, trgTriangles)
    #srcLocalTris = []
    #trgLocalTris = []
    resFrame = None
    # if args.blank:
    #     resFrame = np.zeros((trg_h, trg_w,3), dtype=np.uint8)
    # else:
    #     resFrame = np.copy(targetFrame)
    resFrame = np.zeros((trg_h, trg_w,3), dtype=np.uint8)

    for i, transform in enumerate(transforms):
        #skip while debugging triangles
        if(DEBUG_TRIANGLES):
            continue
        srcTri = source_to_target_triangles[i]
        trgTri = trgTriangles[i]
        srcLocalTris = []
        trgLocalTris = []
        M, srcBB, localSrcTriangle, trgBB, localTrgTriangle = transform
        # if(trgBB[2] > im_h or trgBB[3] > im_w or srcBB[2] > im_h or srcBB[3] > im_w):
        #     continue

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
        cv2.fillConvexPoly(mask, np.int32(srcLocalTris),(1.0, 1.0, 1.0), 16, 0)
        trgCrop = trgCrop * mask
        resFrame[srcBB[1]:srcBB[1]+srcBB[3], srcBB[0]:srcBB[0]+srcBB[2]] = resFrame[srcBB[1]:srcBB[1]+srcBB[3], srcBB[0]:srcBB[0]+srcBB[2]] * ((1.0,1.0,1.0) - mask)
        resFrame[srcBB[1]:srcBB[1]+srcBB[3], srcBB[0]:srcBB[0]+srcBB[2]] = resFrame[srcBB[1]:srcBB[1]+srcBB[3], srcBB[0]:srcBB[0]+srcBB[2]] + mask * trgCrop
        #resFrame[srcBB[1]:srcBB[1]+srcBB[3], srcBB[0]:srcBB[0]+srcBB[2]] =  resFrame[srcBB[1]:srcBB[1]+srcBB[3], srcBB[0]:srcBB[0]+srcBB[2]] + mask * trgCrop
        
        cloneMask = np.zeros(targetFrame.shape, targetFrame.dtype)
        #cloneMask = cloneMask.fill(255)
        cloneContour = cv2.convexHull(np.int32(transformed_source_landmarks))
        #mskBB = cv2.boundingRect(cloneContour)
        #cloneMask[mskBB[1]:mskBB[1]+mskBB[3], mskBB[0]:mskBB[0]+mskBB[2]] = (255,255,255)
        #targetFrame = cv2.fillConvexPoly(targetFrame, cloneContour, (0,0,0))
        cloneMask = cv2.fillConvexPoly(cloneMask, cloneContour, (255,255,255))
        cloneMask = cv2.erode(cloneMask, np.ones((5,5)))
        #finalFrame = cv2.seamlessClone(resFrame, np.copy(targetFrame), cloneMask ,(int(trg_w/2),int(trg_h/2)),cv2.MIXED_CLONE)
        finalFrame = cv2.seamlessClone(resFrame, np.copy(targetFrame), cloneMask ,(int(target_face_center[0]),int(target_face_center[1])),cv2.NORMAL_CLONE)
        #resFrame[srcBB[1]:srcBB[1]+srcBB[3], srcBB[0]:srcBB[0]+srcBB[2]] =  resFrame[srcBB[1]:srcBB[1]+srcBB[3], srcBB[0]:srcBB[0]+srcBB[2]] + mask * trgCrop
        
        if args.slow:
            resPlot.set_data(finalFrame)
            plt.pause(frameTime)
        # resFrame[trgBB[1]:trgBB[1]+trgBB[3], trgBB[0]:trgBB[0]+trgBB[2]] = resFrame[trgBB[1]:trgBB[1]+trgBB[3], trgBB[0]:trgBB[0]+trgBB[2]] * ((1.0,1.0,1.0) - mask)
        # resFrame[trgBB[1]:trgBB[1]+trgBB[3], trgBB[0]:trgBB[0]+trgBB[2]] = resFrame[trgBB[1]:trgBB[1]+trgBB[3], trgBB[0]:trgBB[0]+trgBB[2]] + trgCrop
    #convert in a neutral scale invariant space
    resPlot.set_data(finalFrame)
    #move stuff arround
    if output:
        output.write(np.uint8(cv2.cvtColor(finalFrame, cv2.COLOR_RGB2BGR)))

    profiler.tock()
    #clear patches
    plt.pause(frameTime)
    for p in reversed(srcWindow.patches):
        p.remove()
    for p in reversed(trgWindow.patches):
        p.remove()