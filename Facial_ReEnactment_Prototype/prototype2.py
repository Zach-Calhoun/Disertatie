import cv2
import numpy as np
import dlib
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import time
import dlib
import argparse
import os

from expression_prototypes import get_expression_prototypes, get_matching_expression_prototypes
from profiling import PerformanceTimer
from transforms import dlib_rect_to_bb, landmarks_to_points, verts_to_indices, landmark_indices_to_triangles, apply_transform
from processing import get_scaled_rgb_frame, get_face_bb_landmarks, triangulate_landmarks, get_transforms, get_face_coordinates_system, landmarks_to_image_space
from visualisation import view_landmarks, debug_fill_triangles
from transfer import transfer_expression, transfer_face, transfer_poly
from utils import *


parser = argparse.ArgumentParser(description='Facial re-enactment prototype from source to target clip')
parser.add_argument("--source", metavar="/path/to/source.mp4", help="path to source video clip")
parser.add_argument("--target", metavar="/path/to/target.mp4", help="path to target video clip")
parser.add_argument("--output", required=False, metavar="output.mp4", help="output file path, optional")
parser.add_argument("--debugtriangles", required=False, action='store_true', help="specify this to see tracking triangles")
parser.add_argument("--blank", required=False, action='store_true', help="include this to make the resulting face on top of a blank canvas, for easier inspection")
parser.add_argument("--slow", required=False, action='store_true', help="include this to see the face reconstruction in a slow rate for debugging purposes")
parser.add_argument("--srcscale", type=float, required=False, metavar=1, help="Specify scaling of source frames ( use smaller values to improve performance )", default=0.25)
parser.add_argument("--trgscale", type=float, required=False, metavar=1, help="Specify scaling of target frames ( use smaller values to improve performance )", default=0.25)
parser.add_argument("--use3D", required=False, action='store_true', help="Specify to use 3D transfer method instead of 2D")

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
output = None

sourceSuccess = True
targetSuccess = True
frameTime = 1/24

face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor(predictor_data)

fig = plt.figure()

#some offline processing before rendering


#source frame
srcWindow = fig.add_subplot(161)
#target frame
trgWindow = fig.add_subplot(162)
#target prototype
protoWindow = fig.add_subplot(163)
#transformed prototype
transformedProtoWindow = fig.add_subplot(164)

#transformed target frame
transformedTargetWindow = fig.add_subplot(165)
#final result frame
protoToTargetWindow = fig.add_subplot(166)


#get landmark faces in source video using some sort of disimilarity measure

#average sum euclidean distance?
#some others - compare performance?

#performance measure - self re-enacntment contour comparison makes most sense 

srcScalingFactor = args.srcscale
trgScalingFactor = args.trgscale

sourceSuccess, sourceFrame = get_scaled_rgb_frame(source, srcScalingFactor)
targetSuccess, targetFrame = get_scaled_rgb_frame(target, trgScalingFactor)
im_h,im_w = sourceFrame.shape[:2]
trg_h, trg_w = targetFrame.shape[:2]

srcPlot = srcWindow.imshow(sourceFrame)
trgPlot = trgWindow.imshow(targetFrame)
transformedTargetPlot = transformedTargetWindow.imshow(targetFrame)
protoPlot = protoWindow.imshow(targetFrame)
transformedProtoPlot = transformedProtoWindow.imshow(targetFrame)
finalPlot = protoToTargetWindow.imshow(targetFrame)

plt.ion()
plt.show()


target_prototype_matching_faces = []


source_prototype_faces, source_frame_prototype_index = get_expression_prototypes(source, srcScalingFactor, profiler)

#go through target video and identify key face frames that  matched source video prototypes
#use similarity measure to also point which reference face should be matched durring time measures

target_prototype_faces_frames_landmarks = get_matching_expression_prototypes(target, trgScalingFactor, source_prototype_faces, profiler)



frameCount = 0
srcLandmarkTrianglesIndices = []

#go 

#reopen clips to reset capture
source = cv2.VideoCapture(sourceName)
target = cv2.VideoCapture(targetName)
sourceFPS = source.get(cv2.CAP_PROP_FPS)

base_output_folder = 'debug\\{}'.format(outputName)
if not os.path.exists(base_output_folder):
    os.makedirs(base_output_folder, exist_ok=True)

base_output_path = base_output_folder+'\\{}.jpeg'


while targetSuccess and sourceSuccess:
    use3D = args.use3D
    print("Processing frame {} ...".format(frameCount))
    frameCount = frameCount + 1

    profiler.tick("Process Frame")
    #get frames
    sourceSuccess, sourceFrame = get_scaled_rgb_frame(source, srcScalingFactor)
    targetSuccess, targetFrame = get_scaled_rgb_frame(target, trgScalingFactor)
    if not sourceSuccess or not targetSuccess:
        break

    try:
        target_prototype,target_prototype_landmarks = target_prototype_faces_frames_landmarks[source_frame_prototype_index[frameCount-1]]
    except:
        target_prototype,target_prototype_landmarks = target_prototype_faces_frames_landmarks[source_frame_prototype_index[len(source_frame_prototype_index)-1]]

    srcPlot.set_data(sourceFrame)
    trgPlot.set_data(targetFrame)
    protoPlot.set_data(target_prototype)

   

    sourceToTarget, newTargetLandmarks, triangle_landmark_indices, source_frame_landmarks, target_landmarks = transfer_expression(sourceFrame, targetFrame, trgWindow, srcWindow, profiler, use3D)


    sourceToProto, newProtoLandmarks, _ , _, _ = transfer_expression(sourceFrame, target_prototype, protoWindow, srcWindow, profiler, use3D)

    
    #sourceToProto, newProtoLandmarks, _ , _= transfer_expression(sourceFrame, targetFrame, protoWindow, srcWindow, profiler)
    if(sourceToTarget is None or sourceToProto is None):
        continue

    transformedTargetPlot.set_data(sourceToTarget)
    transformedProtoPlot.set_data(sourceToProto)
    protoToSource = transfer_poly(sourceToTarget, sourceToProto, newTargetLandmarks, newProtoLandmarks)
    #protoToSource = transfer_poly(sourceToProto,sourceToTarget, newProtoLandmarks, newTargetLandmarks)

    cloneMask = np.zeros(protoToSource.shape, protoToSource.dtype)
    cloneContour = cv2.convexHull(np.int32(target_landmarks))
    cloneMask = cv2.fillConvexPoly(cloneMask, cloneContour, (255,255,255))
    cloneMask = cv2.erode(cloneMask, np.ones((5,5)))
    #finalFrame = cv2.seamlessClone(protoToSource, np.copy(targetFrame), cloneMask ,(int(trg_w/2),int(trg_h/2)),cv2.NORMAL_CLONE)
    target_face_center, _, _, _, target_local_landmarks  = get_face_coordinates_system(target_landmarks);
    finalFrame = cv2.seamlessClone(protoToSource, np.copy(targetFrame), cloneMask ,(int(target_face_center[0]),int(target_face_center[1]-10)),cv2.NORMAL_CLONE)

   # finalFrame = protoToSource;
    #protoToSource = transfer_face(sourceToProto, sourceToTarget, newProtoLandmarks, newTargetLandmarks, triangle_landmark_indices)
    #protoToSource = transfer_face(sourceFrame,  source_frame_landmarks ,sourceToProto, sourceToTarget, triangle_landmark_indices)
    if args.slow:
        plt.pause(frameTime)
        
    finalPlot.set_data(finalFrame)

    src_out_path = base_output_path.format('{}_src'.format(frameCount))
    cv2.imwrite(src_out_path, cv2.cvtColor(sourceFrame, cv2.COLOR_RGB2BGR))
    trg_out_path = base_output_path.format('{}_trg'.format(frameCount))
    cv2.imwrite(trg_out_path, cv2.cvtColor(targetFrame, cv2.COLOR_RGB2BGR))
    transformed_target_out_path = base_output_path.format('{}_src2trg'.format(frameCount))
    cv2.imwrite(transformed_target_out_path, cv2.cvtColor(sourceToTarget, cv2.COLOR_RGB2BGR))
    proto_out_path = base_output_path.format('{}_proto'.format(frameCount))
    cv2.imwrite(proto_out_path, cv2.cvtColor(target_prototype, cv2.COLOR_RGB2BGR))
    transformed_proto_out_path = base_output_path.format('{}_src2proto'.format(frameCount))
    cv2.imwrite(transformed_proto_out_path, cv2.cvtColor(sourceToProto, cv2.COLOR_RGB2BGR))
    final_out_path = base_output_path.format('{}_result'.format(frameCount))
    cv2.imwrite(final_out_path, cv2.cvtColor(finalFrame, cv2.COLOR_RGB2BGR))

    if args.output and output is None:
        output = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc('m', 'j', 'p', 'g'), sourceFPS, (finalFrame.shape[1], finalFrame.shape[0]))
    if output:
        output.write(np.uint8(cv2.cvtColor(finalFrame, cv2.COLOR_RGB2BGR)))
    print("Done one full frame process")
    profiler.tock()

    plt.pause(frameTime)
    for p in reversed(srcWindow.patches):
        p.remove()
    for p in reversed(trgWindow.patches):
        p.remove()
    for p in reversed(protoWindow.patches):
        p.remove()
    for p in reversed(transformedProtoWindow.patches):
        p.remove()

print('Transfer complete')
output.release()
source.release()
target.release()



