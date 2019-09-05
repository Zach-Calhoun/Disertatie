import dlib
predictor_data = 'shape_predictor_68_face_landmarks.dat'
from processing import get_scaled_rgb_frame, get_face_bb_landmarks, get_face_coordinates_system
from measures import sum_squared_euclideean_distances as ssed, sum_euclideean_distances as sed
import cv2;

DISIMILARITY_TRESHHOLD = 0.2
#MAX_PROTO = 9999999999999999
MAX_PROTO = 999
#TODO think about paralizing parts of these

measure = sed;

def get_expression_prototypes(source, srcScalingFactor,profiler=None):

    source_prototype_faces = []
    source_frame_prototype_index = []

    face_detector = dlib.get_frontal_face_detector()
    landmark_predictor = dlib.shape_predictor(predictor_data)
    #go through source video and identify key face frames that should be matched in target video if possible
    #use similarity measure to also point which reference face should be matched durring time measures
    #thing video compression
    #[F1,F1,F1,F1...F3,F3,F3,F3,....F1,F1,F1,F1....F2,F2,F2,F2 etc]
    sourceSuccess, sourceFrame = get_scaled_rgb_frame(source, srcScalingFactor)
    frameCount = 0
    while sourceSuccess:
        if(len(source_prototype_faces) >= MAX_PROTO):
            break
        print("Processing source frame {} ...".format(frameCount))
        frameCount = frameCount + 1
        profiler.tick("Process Frame")
        #get frames
        sourceSuccess, sourceFrame = get_scaled_rgb_frame(source, srcScalingFactor)
        if not sourceSuccess:
            break

        #convert to gray for detection
        profiler.tick("Get Face BB and landmarks")
        #im_h,im_w = sourceFrame.shape[:2]

        srcBB, srcLandmarks = get_face_bb_landmarks(sourceFrame, face_detector, landmark_predictor)
        if(srcBB is None):
            print('Failed to find face in frame, skipping')
            continue
        source_face_center, source_x_scale, source_y_scale, source_rot, source_local_landmarks = get_face_coordinates_system(srcLandmarks)

        if(frameCount == 1):
            source_prototype_faces += [source_local_landmarks]
            source_frame_prototype_index += [0]
            src_proto_path = 'debug\\source_proto_{}.jpeg'.format(frameCount)
            cv2.imwrite(src_proto_path, cv2.cvtColor(sourceFrame, cv2.COLOR_RGB2BGR))
            continue

        #if disimilarity measure is > treshhold, we asume it's the same reference face
        #if different, we store it as a separate landmark frame

        disimilarity_min = 99999999999999999999999999999999999
        disimilarity_min_index = 0
        for p_index,prototype in enumerate(source_prototype_faces):
            disimilarity = measure(prototype, source_local_landmarks)
            if(disimilarity < disimilarity_min):
                disimilarity_min = disimilarity
                disimilarity_min_index = p_index
            print('Calculated disimilarity between frame {} and prototype {}'.format(frameCount,p_index))
            print(disimilarity_min)
        if(disimilarity_min >  DISIMILARITY_TRESHHOLD):
            print('Found new prototype')
            source_prototype_faces += [source_local_landmarks] 
            source_frame_prototype_index += [disimilarity_min_index]
            src_proto_path = 'debug\\source_proto_{}.jpeg'.format(frameCount)
            cv2.imwrite(src_proto_path, cv2.cvtColor(sourceFrame, cv2.COLOR_RGB2BGR))
        else:
            print('Not unique enough, closest prototype is {}'.format(disimilarity_min_index))
            source_frame_prototype_index += [disimilarity_min_index]

        #print unique prototypes and frame corespondence

    print("Finished processing source video, found {} unique prototypes".format(len(source_prototype_faces)))
    for prototype_index in source_frame_prototype_index:
        print("#{}#".format(prototype_index), end='')    

    return (source_prototype_faces, source_frame_prototype_index)



def get_matching_expression_prototypes(target, trgScalingFactor,source_prototype_faces, profiler=None):

    #TODO should these be stored as touples?
    target_prototype_faces_frames_landmarks = {}
    minimum_disimilarity = {}

    face_detector = dlib.get_frontal_face_detector()
    landmark_predictor = dlib.shape_predictor(predictor_data)
    #go through source video and identify key face frames that should be matched in target video if possible
    #use similarity measure to also point which reference face should be matched durring time measures
    #thing video compression
    #[F1,F1,F1,F1...F3,F3,F3,F3,....F1,F1,F1,F1....F2,F2,F2,F2 etc]
    targetSuccess, targetFrame = get_scaled_rgb_frame(target, trgScalingFactor)
    frameCount = 0
    while targetSuccess:
        if(len(target_prototype_faces_frames_landmarks) >= MAX_PROTO):
            break
        print("Processing target frame {} ...".format(frameCount))
        frameCount = frameCount + 1
        profiler.tick("Process Frame")
        #get frames
        targetSuccess, targetFrame = get_scaled_rgb_frame(target, trgScalingFactor)
        if not targetSuccess:
            break

        #convert to gray for detection
        profiler.tick("Get Face BB and landmarks")
        #im_h,im_w = targetFrame.shape[:2]

        trgBB, trgLandmarks = get_face_bb_landmarks(targetFrame, face_detector, landmark_predictor)
        if(trgBB is None):
            print('Failed to find face in frame, skipping')
            continue
        target_face_center, target_x_scale, target_y_scale, target_rot, target_local_landmarks = get_face_coordinates_system(trgLandmarks)


        #go through target faces and for each prototype face in source try and find the matches for each in each frame
        #while also storing the similarity value, if next frames have better similarity we replace the old matching
        #frame per prototype

        for p_index,prototype in enumerate(source_prototype_faces):
            disimilarity = measure(prototype, target_local_landmarks)
            print('Calculated disimilarity between frame {} and prototype {}'.format(frameCount,p_index))
            #if we started storing matches
            if p_index in minimum_disimilarity:
                if(disimilarity < minimum_disimilarity[p_index]):
                    print('Found better match for prototype {} at frame {}'.format(p_index, frameCount))
                    minimum_disimilarity[p_index] = disimilarity
                    target_prototype_faces_frames_landmarks[p_index] = (targetFrame, trgLandmarks)
            #else take whatever
            else:
                print('Take whatever')
                minimum_disimilarity[p_index] = disimilarity
                target_prototype_faces_frames_landmarks[p_index] = (targetFrame, trgLandmarks)

    print("Finished processing target video,")
    for i in range(0, len(target_prototype_faces_frames_landmarks)):
        frame = target_prototype_faces_frames_landmarks[i][0]
        src_proto_path = 'debug\\target_proto_{}.jpeg'.format(i)
        cv2.imwrite(src_proto_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        
    return target_prototype_faces_frames_landmarks