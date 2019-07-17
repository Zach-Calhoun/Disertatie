#processings

import cv2
import dlib
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from transforms import landmarks_to_points, triangles_to_verts, apply_transform
from utils import *
def get_scaled_rgb_frame(source : cv2.VideoCapture, scale : float, ):
    """Captures RGB frame from openCV frame and scales it down for performance reasons, or up, depends on scale param"""
    success, frame = source.read()
    if not success:
        return (False, None)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, None, fx=scale, fy=scale)
    return (success, frame)

def get_face_bb_landmarks(frame, face_detector, landmark_predictor):
    """Retrieves face bounding box and landmarks"""
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    bounding_box = face_detector(frame, 1)
    #for now assume one face only
    if len(bounding_box) < 1:
        return (None, None)
    bounding_box = bounding_box[0]
    landmarks = landmark_predictor(frame_gray, bounding_box)
    landmarks = landmarks_to_points(landmarks)
    #skip edges
    #landmarks = landmarks[16:]
    return (bounding_box, landmarks)


def triangulate_landmarks(landmarks, height, width):
    """Calculated Delaunay Triangulation of given points"""
    subdiv = cv2.Subdiv2D((0,0,height, width))
    for pts in landmarks:
        cv_pt = (pts[1],pts[0])
        subdiv.insert(cv_pt)
    
    triangles = subdiv.getTriangleList()
    triangles = triangles_to_verts(triangles)
    return triangles

def get_transforms(sourceTriangles, targetTriangles):
    """Calculates affine transformations between each pair of supplied triangles
    returns (transformation_matrix, source_triangle_bounding_box, source_triangle_points, target_triangle_boudning_box, target_triangle_points)
    """
    transforms = []
    for srcTri, trgTri in zip(sourceTriangles, targetTriangles):
        srcPts = np.float32([[srcTri[0,:], srcTri[1,:], srcTri[2,:]]])
        srcBB = cv2.boundingRect(srcPts)
        localSrcPts = []
        for i in range(0,3):
            localPt = (srcTri[i][0] - srcBB[0] , srcTri[i][1] - srcBB[1])
            localSrcPts.append(localPt)

        trgPts = np.float32([[trgTri[0,:], trgTri[1,:], trgTri[2,:]]])
        trgBB = cv2.boundingRect(trgPts)

        if(np.any(np.array(srcBB) < 0) or np.any(np.array(trgBB) < 0)):
            print('WHOOPS')

        localTrgPts = []
        for i in range(0,3):
            localPt = (trgTri[i][0] - trgBB[0] , trgTri[i][1] - trgBB[1])
            localTrgPts.append(localPt)
        M = cv2.getAffineTransform(np.float32(localTrgPts), np.float32(localSrcPts))
        transforms.append((M, srcBB, localSrcPts, trgBB, localTrgPts))
    return transforms

def get_face_coordinates_system(landmarks, preview_window = None):
    """returns face center, xscale, yscale, rotationmatrix and landmarks in that coordinate system"""
    leftAcc = np.array([0,0])
    for i in LEFT_EYE_INDICES:
        leftAcc += landmarks[i]
    
    rightAcc = np.array([0,0])
    for i in RIGHT_EYE_INDICES:
        rightAcc += landmarks[i]
    #calculate eyes cog
    leftEyePos = leftAcc / EYE_INDICES_COUNT
    rightEyePos = rightAcc / EYE_INDICES_COUNT

    #align eyes
    dx = rightEyePos[0] - leftEyePos[0]
    dy = rightEyePos[1] - leftEyePos[1]

    #TODO why subtract 180?
    angle = np.degrees(np.arctan2(dy,dx)) - 180
    eyesC = (rightEyePos + leftEyePos)/2
    #calculate center based on two main axes
    rotM = cv2.getRotationMatrix2D((eyesC[0],eyesC[1]), angle, 1)
    rotatedLandmarks =  apply_transform(landmarks, rotM)

    face_top = np.array(rotatedLandmarks[FACE_AXIS_TOP_INDEX])
    face_bottom = np.array(rotatedLandmarks[FACE_AXIS_BOTTOM_INDEX])
    face_left = np.array(rotatedLandmarks[FACE_AXIS_LEFT_INDEX])
    face_right = np.array(rotatedLandmarks[FACE_AXIS_RIGHT_INDEX])
    vertical_axis_c = (face_top + face_bottom) / 2
    horizontal_axis_c = (face_left + face_right) / 2
    #consider face center is Y of vertical axis and X of horizontal
    #TODO make use intersection?
    vertical_axis_len = np.linalg.norm(face_top - face_bottom)
    horizontal_axis_len = np.linalg.norm(face_left - face_right)
    face_center = np.array((horizontal_axis_c[0], vertical_axis_c[1]))
    #calculate landmark positions based on center
    localLandmarks = []
    for rl in rotatedLandmarks:
        localLandmarks.append((np.array(rl) - face_center) / (horizontal_axis_len,vertical_axis_len))
    
    

    #scale everything so that the two main axes go between [-1,1]
    #* eyebrows will go above 1 in this case


    if preview_window:
        preview_window.clear()
        #multiply by 100 because scatter points are to large for -1,1 scale
        preview_window.scatter(np.array(localLandmarks)[:,0]*100,-np.array(localLandmarks)[:,1]*100)
        preview_window.add_patch(matplotlib.patches.Circle((0,0), 1))


    #return (face_center, horizontal_axis_len, vertical_axis_len, rotM, localLandmarks)
    #this is a kludge, suposed to return matrix, but sice it's not invertible, angle will do
    return (face_center, horizontal_axis_len, vertical_axis_len, angle, localLandmarks)
   
#TODO implement
def triangles_to_image_space(triangles, rotM, faceCenter, scaleX, scaleY):
    """STUB moves a set of triangles from face local space to target face's image space for later wrapping STUB"""
    invRot = np.linalg.inv(rotM)
    return None

def landmarks_to_image_space(landmarks, rotM, faceCenter, scaleX, scaleY, preview_window = None):
    """moves landmarks from local face space to target face's image space for later triangulation and wrapping"""
    #opencv 2d roattion matrix can not be inverted, not without some extra operations, better just use reverse angle for now
    #invRot = np.linalg.inv(rotM)
    invRot = cv2.getRotationMatrix2D((0,0), -rotM, 1)
    inv_rot_landmarks = apply_transform(landmarks, invRot)
    target_image_landmarks = []
    for local_landmark in inv_rot_landmarks:
        #rescale landmarks up
        target_landmark = np.array(local_landmark) * (scaleX, scaleY)
        #calculate landmark position relative to target face center
        target_landmark = target_landmark + faceCenter
        #TODO see this approach's effect on eye positions
        target_image_landmarks.append(target_landmark)
        if preview_window:
            preview_window.add_patch(matplotlib.patches.Circle(target_landmark, 2, color='#FF0000'))
    
    
    return target_image_landmarks

        
    