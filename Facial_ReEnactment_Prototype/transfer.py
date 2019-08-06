#transfer_expression.py

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
from headpose import headpose_estimate

generated_colors = False
generated_triangulation = False
srcLandmarkTrianglesIndices = []
predictor_data = 'shape_predictor_68_face_landmarks.dat'
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor(predictor_data)
colors = []
frameTime = 1/24


fig = plt.figure()
src_m_win = fig.add_subplot(131)
trg_m_win = fig.add_subplot(132)
trg_m2s_win = fig.add_subplot(133)

def transfer_expression(sourceFrame, targetFrame, trgWindow, profiler=PerformanceTimer()):

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
        return None, None, None, None
    

    source_translation, source_rotation, source_t_matrix = headpose_estimate(sourceFrame, srcLandmarks)
    target_translation, target_rotation, target_t_matrix = headpose_estimate(targetFrame, trgLandmarks)
    # assumption can be made that applying the inverse of source_t_matrix to the source landmarks 
    # followed by the target_t_matrix should solve the overlaying of target landmarks to source landmarks
    # the edge case of extreme rotations is not treated
    # care should be taken as landmarks are currently 2D, but assuming 0 depth axis should wokr

    #x,y,w,h = dlib_rect_to_bb(srcBB)

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

    source_to_target_triangles = landmark_indices_to_triangles(source_to_target_landmarks, srcLandmarkTrianglesIndices)
    
    #move source local triangles to target local space and calculate from there
        
    profiler.tock()
    
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
        
    #clone inner mouth form source to target
    source_inner_mouth_landmarks = srcLandmarks[INNER_MOUTH_BEGIN:INNER_MOUTH_END+1]
    target_inner_mouth_landmarks = source_to_target_landmarks[INNER_MOUTH_BEGIN:INNER_MOUTH_END+1]

    src_mouth_center = np.average(source_inner_mouth_landmarks, axis=0)
    trg_mouth_center = np.average(target_inner_mouth_landmarks, axis=0)
    src_mouth_bb = cv2.boundingRect(np.array(source_inner_mouth_landmarks, dtype=np.int32))
    trg_mouth_bb = cv2.boundingRect(np.array(target_inner_mouth_landmarks, dtype=np.int32))
    local_inner_target_mouth = target_inner_mouth_landmarks - np.array((trg_mouth_bb[0], trg_mouth_bb[1]))
    local_inner_target_mouth = list(map(lambda  x : tuple(x) ,local_inner_target_mouth))
    local_inner_source_mouth = source_inner_mouth_landmarks - np.array((src_mouth_bb[0], src_mouth_bb[1]))
    local_inner_source_mouth = list(map(lambda  x : tuple(x) ,local_inner_source_mouth))
    src_inner_mouth_rect = sourceFrame[src_mouth_bb[1]:src_mouth_bb[1]+src_mouth_bb[3], src_mouth_bb[0]:src_mouth_bb[0]+src_mouth_bb[2]]
    #transform source mouth to match size of target transformed mouth - to account for different scaling as naive cloning has scaling issues
    target_inner_mouth_temp_area = np.zeros((trg_mouth_bb[3],trg_mouth_bb[2],3),dtype=np.uint8)
    source_inner_mouth_to_target_size = transfer_poly(target_inner_mouth_temp_area, src_inner_mouth_rect, local_inner_target_mouth, local_inner_source_mouth)

    #src_m_win.imshow(src_inner_mouth_rect)
    #trg_m_win.imshow(source_inner_mouth_to_target_size)

    inner_mouth_mask = np.zeros(source_inner_mouth_to_target_size.shape, dtype=np.uint8)

    inner_mouth_mask = cv2.fillConvexPoly(inner_mouth_mask, np.array(local_inner_target_mouth,dtype=np.int32), (255,255,255))
    #finalFrame = cv2.fillConvexPoly(finalFrame, np.array(target_inner_mouth_landmarks, dtype=np.int32), (64,64,64))
    #innerMouthMask[:,:] = (255,255,255)
    #clear inner mouth for transfer
    resFrame = cv2.fillConvexPoly(resFrame, np.array(target_inner_mouth_landmarks, dtype=np.int32), (0,0,0))
    resFrame[trg_mouth_bb[1]:trg_mouth_bb[1]+trg_mouth_bb[3], trg_mouth_bb[0]:trg_mouth_bb[0]+trg_mouth_bb[2]] = resFrame[trg_mouth_bb[1]:trg_mouth_bb[1]+trg_mouth_bb[3], trg_mouth_bb[0]:trg_mouth_bb[0]+trg_mouth_bb[2]] + source_inner_mouth_to_target_size * inner_mouth_mask
    
    #blend mouth from source to target
    source_mouth_landmarks = srcLandmarks[MOUTH_BEGIN:MOUTH_END+1]
    target_mouth_landmarks = source_to_target_landmarks[MOUTH_BEGIN:MOUTH_END+1]

    src_mouth_center = np.average(source_mouth_landmarks, axis=0)
    trg_mouth_center = np.average(target_mouth_landmarks, axis=0)
    src_mouth_bb = cv2.boundingRect(np.array(source_mouth_landmarks, dtype=np.int32))
    trg_mouth_bb = cv2.boundingRect(np.array(target_mouth_landmarks, dtype=np.int32))
    local_target_mouth = target_mouth_landmarks - np.array((trg_mouth_bb[0], trg_mouth_bb[1]))
    local_target_mouth = list(map(lambda  x : tuple(x) ,local_target_mouth))
    local_source_mouth = source_mouth_landmarks - np.array((src_mouth_bb[0], src_mouth_bb[1]))
    local_source_mouth = list(map(lambda  x : tuple(x) ,local_source_mouth))
    src_mouth_rect = sourceFrame[src_mouth_bb[1]:src_mouth_bb[1]+src_mouth_bb[3], src_mouth_bb[0]:src_mouth_bb[0]+src_mouth_bb[2]]
    #transform source mouth to match size of target transformed mouth - to account for different scaling as naive cloning has scaling issues
    target_mouth_temp_area = np.zeros((trg_mouth_bb[3],trg_mouth_bb[2],3),dtype=np.uint8)
    source_mouth_to_target_size = transfer_poly(target_mouth_temp_area, src_mouth_rect, local_target_mouth, local_source_mouth)

   
    src_m_win.imshow(src_mouth_rect)
    trg_m_win.imshow(source_mouth_to_target_size)

    mouth_mask = np.zeros(source_mouth_to_target_size.shape, dtype=np.uint8)

    mouth_mask = cv2.fillConvexPoly(mouth_mask, np.array(local_target_mouth,dtype=np.int32), (255,255,255))
    #finalFrame = cv2.fillConvexPoly(finalFrame, np.array(target_inner_mouth_landmarks, dtype=np.int32), (64,64,64))
    #innerMouthMask[:,:] = (255,255,255)
    try:
        #THERE IS AN ISSUE HERE WHEN THE SCALING DOESN'T MATCH, SO FIRST SRC needs to be mapped to TARGET
        res_frame_with_mouth = cv2.seamlessClone(source_mouth_to_target_size, resFrame, mouth_mask, (int(trg_mouth_center[0]),int(trg_mouth_center[1])), cv2.NORMAL_CLONE)
    except Exception as e:
        print("EXCEPTION ON FRAME")
        print(e)
        return None, None, None, None
    
    cloneMask = np.zeros(targetFrame.shape, targetFrame.dtype)
    #cloneMask = cloneMask.fill(255)
    cloneContour = cv2.convexHull(np.int32(transformed_source_landmarks))
    #mskBB = cv2.boundingRect(cloneContour)
    #cloneMask[mskBB[1]:mskBB[1]+mskBB[3], mskBB[0]:mskBB[0]+mskBB[2]] = (255,255,255)
    #targetFrame = cv2.fillConvexPoly(targetFrame, cloneContour, (0,0,0))
    cloneMask = cv2.fillConvexPoly(cloneMask, cloneContour, (255,255,255))
    cloneMask = cv2.erode(cloneMask, np.ones((5,5)))
    #finalFrame = cv2.seamlessClone(resFrame, np.copy(targetFrame), cloneMask ,(int(trg_w/2),int(trg_h/2)),cv2.MIXED_CLONE)
    
    #finalFrame = resFrame
    finalFrame = cv2.seamlessClone(res_frame_with_mouth, np.copy(targetFrame), cloneMask ,(int(target_face_center[0]),int(target_face_center[1])),cv2.NORMAL_CLONE)





    profiler.tock()
    return finalFrame, source_to_target_landmarks, srcLandmarkTrianglesIndices, srcLandmarks
        
def transfer_poly(source, target, sourcePoints, targetPoints, triangle_indices=None):
    height, width = source.shape[:2]
    if(triangle_indices is None):
        target_triangles = triangulate_landmarks(targetPoints, height, width)
        triangle_indices, target_triangles = verts_to_indices(target_triangles, targetPoints)

    source_triangles = landmark_indices_to_triangles(sourcePoints, triangle_indices)
    
    transforms = get_transforms(source_triangles, target_triangles)
    #transforms = get_transforms(target_triangles, source_triangles)

    resFrame = None
    resFrame = np.zeros((height, width,3), dtype=np.uint8)

    for i, transform in enumerate(transforms):
        #skip while debugging triangles
        srcTri = source_triangles[i]
        trgTri = target_triangles[i]
        srcLocalTris = []
        trgLocalTris = []
        transformationMatrix , srcBB, _, trgBB, _ = transform

        for j in range(0,3):
            srcLocalTriangle = (srcTri[j][0] - srcBB[0], srcTri[j][1] - srcBB[1])
            trgLocalTriangle = (trgTri[j][0] - trgBB[0], trgTri[j][1] - trgBB[1])
            srcLocalTris.append(srcLocalTriangle)
            trgLocalTris.append(trgLocalTriangle)

        srcCrop = target[trgBB[1] : trgBB[1] + trgBB[3], trgBB[0] : trgBB[0] + trgBB[2]]
        #suddenly for some reeason, everynow and then the BB is out of bounds so we need to limit it
        max_crop = resFrame[srcBB[1]:srcBB[1]+srcBB[3], srcBB[0]:srcBB[0]+srcBB[2]]
        trgCrop = cv2.warpAffine(srcCrop, transformationMatrix, (max_crop.shape[1], max_crop.shape[0]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
     
     
        mask = np.zeros(max_crop.shape, dtype = np.float32)
        cv2.fillConvexPoly(mask, np.int32(srcLocalTris),(1.0, 1.0, 1.0), 16, 0)
        trgCrop = trgCrop * mask
        resFrame[srcBB[1]:srcBB[1]+srcBB[3], srcBB[0]:srcBB[0]+srcBB[2]] = max_crop * ((1.0,1.0,1.0) - mask)
        resFrame[srcBB[1]:srcBB[1]+srcBB[3], srcBB[0]:srcBB[0]+srcBB[2]] = max_crop + mask * trgCrop
        
        #cloneMask = np.zeros(target.shape, target.dtype)

        #cloneContour = cv2.convexHull(np.int32(targetPoints))
        #cloneMask = cv2.fillConvexPoly(cloneMask, cloneContour, (255,255,255))
        #cloneMask = cv2.erode(cloneMask, np.ones((5,5)))
        #finalFrame = cv2.seamlessClone(resFrame, np.copy(targetFrame), cloneMask ,(int(trg_w/2),int(trg_h/2)),cv2.MIXED_CLONE)
        #finalFrame = cv2.seamlessClone(resFrame, np.copy(targetFrame), cloneMask ,(int(target_face_center[0]),int(target_face_center[1])),cv2.NORMAL_CLONE)
    return resFrame


def transfer_face(sourceFrame, targetFrame, sourceLandmarks, targetLandmarks, triangle_landmark_indices):

    im_h,im_w = sourceFrame.shape[:2]
    trg_h, trg_w = targetFrame.shape[:2]
    
    #this is very slow, google says face detection is slow compared to landmark extraction
    #TODO extract face region more rarely, maybe even half the rate of landmark prediction
    srcLandmarks = sourceLandmarks
    trgLandmarks = targetLandmarks
    
    if sourceLandmarks is None or targetLandmarks is None:
        print("No face in frame!")
        return


    source_face_center, source_x_scale, source_y_scale, source_rot, source_local_landmarks = get_face_coordinates_system(srcLandmarks)
    target_face_center, target_x_scale, target_y_scale, target_rot, target_local_landmarks = get_face_coordinates_system(trgLandmarks)


    transformed_source_landmarks = landmarks_to_image_space(source_local_landmarks, target_rot, target_face_center, target_x_scale, target_y_scale)
    #do not move arround nose or face edges
    source_to_target_landmarks = trgLandmarks[0:17] + transformed_source_landmarks[17:27] + trgLandmarks[27:36] + transformed_source_landmarks[36:]


    srcTriangles =  landmark_indices_to_triangles(sourceLandmarks, triangle_landmark_indices)
    trgTriangles =  landmark_indices_to_triangles(targetLandmarks, triangle_landmark_indices)
    source_to_target_triangles = landmark_indices_to_triangles(source_to_target_landmarks, triangle_landmark_indices)
    
    #move source local triangles to target local space and calculate from there  
    transforms = get_transforms(srcTriangles, trgTriangles)

    resFrame = None

    resFrame = np.zeros((trg_h, trg_w,3), dtype=np.uint8)

    for i, transform in enumerate(transforms):
        #skip while debugging triangles
        srcTri = source_to_target_triangles[i]
        trgTri = trgTriangles[i]
        srcLocalTris = []
        trgLocalTris = []
        M, srcBB, localSrcTriangle, trgBB, localTrgTriangle = transform


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

    return finalFrame

    
def transfer_triangle():
    pass
    