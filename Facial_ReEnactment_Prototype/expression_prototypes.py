import dlib
predictor_data = 'shape_predictor_68_face_landmarks.dat'
from processing import get_scaled_rgb_frame, get_face_bb_landmarks, get_face_coordinates_system
from measures import sum_squared_euclideean_distances as ssed

def get_expression_prototypes(source, srcScalingFactor,profiler=None):
    DISIMILARITY_TRESHHOLD = 0.1
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
        print("Processing source frame {} ...".format(frameCount))
        frameCount = frameCount + 1
        profiler.tick("Process Frame")
        #get frames
        sourceSuccess, sourceFrame = get_scaled_rgb_frame(source, srcScalingFactor)
        if not sourceSuccess:
            break

        #convert to gray for detection
        profiler.tick("Get Face BB and landmarks")
        im_h,im_w = sourceFrame.shape[:2]

        srcBB, srcLandmarks = get_face_bb_landmarks(sourceFrame, face_detector, landmark_predictor)
        if(srcBB is None):
            print('Failed to find face in frame, skipping')
            continue
        source_face_center, source_x_scale, source_y_scale, source_rot, source_local_landmarks = get_face_coordinates_system(srcLandmarks)

        if(frameCount == 1):
            source_prototype_faces += [source_local_landmarks]
            source_frame_prototype_index += [0]
            continue

        #if similarity measure is > treshhold, we asume it's the same reference face
        #if different, we store it as a separate landmark frame

        disimilarity_min = 99999999999999999999999999999999999
        disimilarity_min_index = 0
        for p_index,prototype in enumerate(source_prototype_faces):
            disimilarity = ssed(prototype, source_local_landmarks)
            if(disimilarity < disimilarity_min):
                disimilarity_min = disimilarity
                disimilarity_min_index = p_index
            print('Calculated disimilarity between frame {} and prototype {}'.format(frameCount,p_index))
            print(disimilarity_min)
            if(disimilarity_min >  DISIMILARITY_TRESHHOLD):
                print('Found new prototype')
                source_prototype_faces += [source_local_landmarks]
                source_frame_prototype_index += [p_index]
                break
        else:
            print('Not unique enough, closest prototype is {}'.format(disimilarity_min_index))
            source_frame_prototype_index += [disimilarity_min_index]

        #print unique prototypes and frame corespondence

    print("Finished processing source video, found {} unique prototypes".format(len(source_prototype_faces)))
    for prototype_index in source_frame_prototype_index:
        print("#{}#".format(prototype_index), end='')    

    return (source_prototype_faces, source_frame_prototype_index)



def get_matching_expression_prototypes(target, trgScalingFactor,source_prototype_faces, profiler=None):
    DISIMILARITY_TRESHHOLD = 0.1
    #TODO should these be stored as touples?
    target_prototype_faces = []
    target_prototype_faces_frames = []
    maximum_similarity = []

    face_detector = dlib.get_frontal_face_detector()
    landmark_predictor = dlib.shape_predictor(predictor_data)
    #go through source video and identify key face frames that should be matched in target video if possible
    #use similarity measure to also point which reference face should be matched durring time measures
    #thing video compression
    #[F1,F1,F1,F1...F3,F3,F3,F3,....F1,F1,F1,F1....F2,F2,F2,F2 etc]
    targetSuccess, targetFrame = get_scaled_rgb_frame(target, trgScalingFactor)
    frameCount = 0
    while targetSuccess:
        print("Processing source frame {} ...".format(frameCount))
        frameCount = frameCount + 1
        profiler.tick("Process Frame")
        #get frames
        targetSuccess, targetFrame = get_scaled_rgb_frame(target, trgScalingFactor)
        if not targetSuccess:
            break

        #convert to gray for detection
        profiler.tick("Get Face BB and landmarks")
        im_h,im_w = targetFrame.shape[:2]

        srcBB, srcLandmarks = get_face_bb_landmarks(targetFrame, face_detector, landmark_predictor)
        if(srcBB is None):
            print('Failed to find face in frame, skipping')
            continue
        source_face_center, source_x_scale, source_y_scale, source_rot, source_local_landmarks = get_face_coordinates_system(srcLandmarks)


        #if similarity measure is > treshhold, we asume it's matches the prototype face and store it as a match
        #while also storing the similarity value, if next frames have better similarity we replace the old matching
        #frame

        disimilarity_min = 99999999999999999999999999999999999
        disimilarity_min_index = 0
        for p_index,prototype in enumerate(source_prototype_faces):
            disimilarity = ssed(prototype, source_local_landmarks)
            if(disimilarity < disimilarity_min):
                disimilarity_min = disimilarity
                disimilarity_min_index = p_index
            print('Calculated disimilarity between frame {} and prototype {}'.format(frameCount,p_index))
            print(disimilarity_min)
            if(disimilarity_min >  DISIMILARITY_TRESHHOLD):
                print('Found new prototype')
                source_prototype_faces += [source_local_landmarks]
                source_frame_prototype_index += [p_index]
                break
        else:
            print('Not unique enough, closest prototype is {}'.format(disimilarity_min_index))
            source_frame_prototype_index += [disimilarity_min_index]

        #print unique prototypes and frame corespondence

    print("Finished processing source video, found {} unique prototypes".format(len(source_prototype_faces)))
    for prototype_index in source_frame_prototype_index:
        print("#{}#".format(prototype_index), end='')    
        
    return (source_prototype_faces, source_frame_prototype_index)