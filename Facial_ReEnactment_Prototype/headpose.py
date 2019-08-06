# head pose estimation
import cv2
import numpy as np
import utils

# https://www.learnopencv.com/head-pose-estimation-using-opencv-and-dlib/

aproximation_3d_face = np.array([
    (0, 0, 0),
    (0, -330, -65),
    (255, 170, -135),
    (255, 170, -135),
    (150, -150, -125),
    (150, -150, -125),
], dtype=np.float64)


def headpose_estimate(frame, landmarks, output_plot=None):
    '''Estimates the headpose in a given frame based on dlib landmarks, and returns the translation vector, the rotation vector,
    and for convenience the transformation matrix, which then can be easily inverted'''
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
    # ignores length distortion
    distance_coefficients = np.zeros((4, 1))
    # interestingly enough, in the tutorial the flag had a differnt form, not sure of typoe or version change
    success, rotation_vector, translation_vector = cv2.solvePnP(
        aproximation_3d_face, key_landmarks, camera_matrix, distance_coefficients, flags=cv2.SOLVEPNP_ITERATIVE)
    if not success:
        raise("Failure to estimate headpose")

    # get transformation matrix for convenience, we're working in 3D now , but should be a huge issue
    rot_matrix = cv2.Rodrigues(rotation_vector)
    T_matrix = np.array(
        [1, 0, 0, translation_vector[0]],
        [0, 1, 0, translation_vector[1]],
        [0, 0, 1, translation_vector[2]],
        [0, 0, 0, 1]
    )

    transformation_matrix = T_matrix * rot_matrix 
    print(translation_vector)
    print(rotation_vector)
    return translation_vector, rotation_vector, transformation_matrix
