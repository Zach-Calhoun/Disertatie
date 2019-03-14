#processings

import cv2
import dlib
import numpy as np
from transforms import landmarks_to_points, triangles_to_verts

def get_scaled_rgb_frame(source : cv2.VideoCapture, scale : float, ):
    success, frame = source.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, None, fx=scale, fy=scale)
    return (success, frame)

def get_face_bb_landmarks(frame, face_detector, landmark_predictor):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    bounding_box = face_detector(frame, 1)
    #for now assume one face only
    if len(bounding_box) < 1:
        return (None, None)
    bounding_box = bounding_box[0]
    landmarks = landmark_predictor(frame_gray, bounding_box)
    landmarks = landmarks_to_points(landmarks)
    return (bounding_box, landmarks)


def triangulate_landmarks(landmarks, height, width):
    subdiv = cv2.Subdiv2D((0,0,height, width))
    for pts in landmarks:
        cv_pt = (pts[1],pts[0])
        subdiv.insert(cv_pt)
    
    triangles = subdiv.getTriangleList()
    triangles = triangles_to_verts(triangles)
    return triangles

def get_transforms(sourceTriangles, targetTriangles):
    transforms = []
    for srcTri, trgTri in zip(sourceTriangles, targetTriangles):
        srcPts = np.float32([[srcTri[0,:], srcTri[1,:], srcTri[2,:]]])
        srcBB = cv2.boundingRect(srcPts)
        trgPts = np.float32([[trgTri[0,:], trgTri[1,:], trgTri[2,:]]])
        trgBB = cv2.boundingRect(trgPts)
        M = cv2.getAffineTransform(srcPts, trgPts)
        transforms.append((M, srcBB, trgBB))
    return transforms

