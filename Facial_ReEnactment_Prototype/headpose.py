# head pose estimation
import cv2
import numpy as np
import utils
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
from transforms import apply_transform 

# https://www.learnopencv.com/head-pose-estimation-using-opencv-and-dlib/

aproximation_3d_face = np.array([
    (0, 0, 0),
    (0, -330, -65),
    (255, 170, -135),
    (-255, 170, -135),
    (150, -150, -125),
    (-150, -150, -125),
], dtype=np.float64)
# ignores lens distortion
#distance_coefficients = np.zeros((4, 1))
distance_coefficients = None
fig = plt.figure()
src_to_target_plot = fig.add_subplot(111)

def headpose_estimate(frame, landmarks, output_plot=None):
    '''Estimates the headpose in a given frame based on dlib landmarks, and returns the translation vector, the rotation vector,
    and for convenience the transformation matrix, which then can be easily inverted
    !!!assumes landmarks are in image space!!!'''
    # assume focal_length value from image width
    focal_length = frame.shape[1]
    center = (frame.shape[1] / 2, frame.shape[0] / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]], dtype=np.float64
    )

    key_landmarks = np.array([
        landmarks[utils.NOSE_TIP],
        landmarks[utils.CHIN],
        landmarks[utils.LEFT_EYE_LEFT],
        landmarks[utils.RIGHT_EYE_RIGHT],
        landmarks[utils.MOUTH_LEFT],
        landmarks[utils.MOUTH_RIGHT]
    ], dtype=np.float64)
    #key_landmarks = key_landmarks - center
    
    
   
    # interestingly enough, in the tutorial the flag had a differnt form, not sure of typoe or version change
    success, rotation_vector, translation_vector = cv2.solvePnP(aproximation_3d_face, key_landmarks, camera_matrix, distance_coefficients, flags=cv2.SOLVEPNP_ITERATIVE)
    if not success:
        raise("Failure to estimate headpose")

    # get transformation matrix for convenience, we're working in 3D now , but should be a huge issue
    rodrigues_matrix, _ = cv2.Rodrigues(rotation_vector)
    # if close to -1 instead of 1, it means it's reversed for some reason

    print(np.linalg.det(rodrigues_matrix))
    rot_matrix = np.zeros((4,4))
    rot_matrix[:3,:3] = rodrigues_matrix
    rot_matrix[3,3] = 1

    T_matrix = np.array([
        [rot_matrix[0,0], rot_matrix[0,1], rot_matrix[0,2], translation_vector[0]],
        [rot_matrix[1,0], rot_matrix[1,1], rot_matrix[1,2], translation_vector[1]],
        [rot_matrix[2,0], rot_matrix[2,1], rot_matrix[2,2], translation_vector[2]],
        [0, 0, 0, 1]
    ])

    inv_R = rodrigues_matrix.transpose()
    inv_T = np.matmul(-1 * inv_R, translation_vector)

    inv_T = np.array([
        [inv_R[0,0], inv_R[0,1], inv_R[0,2], inv_T[0]],
        [inv_R[1,0], inv_R[1,1], inv_R[1,2], inv_T[1]],
        [inv_R[2,0], inv_R[2,1], inv_R[2,2], inv_T[2]],
        [0, 0, 0, 1]
    ])

    hom_cam = np.array([
        [camera_matrix[0,0],camera_matrix[0,1],camera_matrix[0,2],0],
        [camera_matrix[1,0],camera_matrix[1,1],camera_matrix[1,2],0],
        [camera_matrix[2,0],camera_matrix[2,1],camera_matrix[2,2],0],
        [0,0,0,0]
    ])
    P_matrix = np.matmul(hom_cam,T_matrix)
    inv_P_rot = np.linalg.inv(P_matrix[0:3,0:3])
    inv_P_t = np.matmul(-1 * inv_P_rot, P_matrix[0:3,3].transpose())
    inv_P = np.array([
        [inv_P_rot[0,0], inv_P_rot[0,1], inv_P_rot[0,2], inv_P_t[0]],
        [inv_P_rot[1,0], inv_P_rot[1,1], inv_P_rot[1,2], inv_P_t[1]],
        [inv_P_rot[2,0], inv_P_rot[2,1], inv_P_rot[2,2], inv_P_t[2]],
        [0, 0, 0, 1]
    ])

    return translation_vector, rotation_vector, camera_matrix, T_matrix, inv_P

def image_to_world(image_points, camera_matrix, transformation_matrix):
    #invers projection
    #solve z coordinate by projecting ideal model and matching on the Z plane, should be possible
    #invert transformaiton_matrix
    #should be back in model space now
    height = camera_matrix[1,2] * 2
    width = camera_matrix[0,2] * 2
    #normalize coordinates
    scaled_image_points = np.array([(x/width, y/height, 1) for x,y in image_points])
    inverse_camera = np.array([
        [camera_matrix[1,1], 0, -camera_matrix[0,2] * camera_matrix[1,1]],
        [0, camera_matrix[0,0], -camera_matrix[1,2] * camera_matrix[0,0]],
        [0, 0, camera_matrix[0,0] * camera_matrix[1,1]]
    ])
    

def match_to_model_face(source_landmarks):
    '''Assumes the source is as well aligned as possible, or maybe passed through the pose correction algortihm, at least'''
    """returns face center, xscale, yscale, rotationmatrix and landmarks in that coordinate system"""
    leftAcc = np.array([0,0],dtype=np.float64)
    for i in utils.LEFT_EYE_INDICES:
        leftAcc += source_landmarks[i]
    
    rightAcc = np.array([0,0], dtype=np.float64)
    for i in utils.RIGHT_EYE_INDICES:
        rightAcc += source_landmarks[i]
    #calculate eyes cog
    leftEyePos = leftAcc / utils.EYE_INDICES_COUNT
    rightEyePos = rightAcc / utils.EYE_INDICES_COUNT

    #align eyes
    dx = rightEyePos[0] - leftEyePos[0]
    dy = rightEyePos[1] - leftEyePos[1]

    #TODO why subtract 180?
    angle = np.degrees(np.arctan2(dy,dx)) - 180
    eyesC = (rightEyePos + leftEyePos)/2
    #calculate center based on two main axes
    rotM = cv2.getRotationMatrix2D((eyesC[0],eyesC[1]), angle, 1)
    rotatedLandmarks =  apply_transform(source_landmarks, rotM)

    face_top = np.array(rotatedLandmarks[utils.FACE_AXIS_TOP_INDEX])
    face_bottom = np.array(rotatedLandmarks[utils.FACE_AXIS_BOTTOM_INDEX])
    face_left = np.array(rotatedLandmarks[utils.FACE_AXIS_LEFT_INDEX])
    face_right = np.array(rotatedLandmarks[utils.FACE_AXIS_RIGHT_INDEX])
    vertical_axis_c = (face_top + face_bottom) / 2
    horizontal_axis_c = (face_left + face_right) / 2
    #consider face center is Y of vertical axis and X of horizontal
    #TODO make use intersection?
    vertical_axis_len = np.linalg.norm(face_top - face_bottom)
    horizontal_axis_len = np.linalg.norm(face_left - face_right)

    face_center = np.array((rotatedLandmarks[utils.NOSE_TIP][0], rotatedLandmarks[utils.NOSE_TIP][1]))

    #face_center = np.average(rotatedLandmarks, axis=0)
    #calculate landmark positions based on center
    localLandmarks = []

    for rl in rotatedLandmarks:
        localLandmarks.append((np.array(rl) - face_center))

    key_landmarks = np.array([
        localLandmarks[utils.NOSE_TIP],
        localLandmarks[utils.CHIN],
        localLandmarks[utils.LEFT_EYE_LEFT],
        localLandmarks[utils.RIGHT_EYE_RIGHT],
        localLandmarks[utils.MOUTH_LEFT],
        localLandmarks[utils.MOUTH_RIGHT]
    ], dtype=np.float64)

    #now transform the points so that the key ones match the ideal model
    transform, _ = cv2.findHomography(key_landmarks, aproximation_3d_face[:,:2])
    #transform = cv2.getAffineTransform(key_landmarks, aproximation_3d_face[:,:2])
    localLandmarks = apply_transform(localLandmarks, transform)
    ideal_3d_landmarks = np.array([(x,y,-160) for x,y in localLandmarks])

    #likely need to do this for whole landmarks if this is to work at all
    ideal_3d_landmarks[utils.NOSE_TIP][2] = 0
    ideal_3d_landmarks[utils.CHIN][2] = -65
    ideal_3d_landmarks[utils.LEFT_EYE_LEFT][2] - 135
    ideal_3d_landmarks[utils.RIGHT_EYE_RIGHT][2] - 135
    ideal_3d_landmarks[utils.MOUTH_LEFT][2] - 125
    ideal_3d_landmarks[utils.MOUTH_RIGHT][2] - 125

    return ideal_3d_landmarks

def project_back(landmarks_3d, rotation_vector, translation_vector, camera_matrix):
    #why does this work but not the other, 
    #landmarks_3d_t = np.array([landmarks_3d[0],landmarks_3d[1]])
    # I have no clue why I need to recast this as an array as it is already an array
    # but it won't work otherwise
    # perhaps need more sleep
    # final output resuls is kind of messed up...
    target_img_center = (camera_matrix[0][2], camera_matrix[1][2])
    target_local_landmarks, jacobian = cv2.projectPoints(np.array(landmarks_3d), rotation_vector, translation_vector, camera_matrix, distance_coefficients)
    #squeeze to remove reduntant nesting cause by hack fix for project_back
    target_local_landmarks = np.squeeze(target_local_landmarks)
    #target_local_landmarks = target_local_landmarks + target_img_center
    target_local_landmarks = list(map(tuple,target_local_landmarks))

    for landmark in target_local_landmarks:
        src_to_target_plot.add_patch(matplotlib.patches.Circle(landmark, 2))        
    #plt.pause(1)

    return target_local_landmarks, jacobian