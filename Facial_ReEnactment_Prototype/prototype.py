import cv2
import numpy as np
import dlib
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import time
import dlib

def dlib_rect_to_bb(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return (x, y, w, h)


sourceName = 'source.mp4'
targetName = 'target.mp4'
predictor_data = 'shape_predictor_68_face_landmarks.dat'

source = cv2.VideoCapture(sourceName)
target = cv2.VideoCapture(targetName)

#sourceVc = cv2.VideoCapture.open(sourceName)
#targetVc = cv2.VideoCapture.open(targetName)

sourceSuccess = True
targetSuccess = True
frameTime = 1/24
print(frameTime)

face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor(predictor_data)

fig = plt.figure()
srcWindow = fig.add_subplot(121)
trgWindow = fig.add_subplot(122)
#srcWindow = plt.subplot(121)
#trgWindow = plt.subplot(122)
sourceSuccess, sourceFrame = source.read()
targetSuccess, targetFrame = target.read()
sourceFrame = cv2.cvtColor(sourceFrame, cv2.COLOR_BGR2RGB)
targetFrame = cv2.cvtColor(targetFrame, cv2.COLOR_BGR2RGB)
#scalingFactor = 0.12
scalingFactor = 0.25
sourceFrame = cv2.resize(sourceFrame, None, fx=scalingFactor, fy=scalingFactor)
targetFrame = cv2.resize(targetFrame, None, fx=scalingFactor, fy=scalingFactor)
srcPlot = srcWindow.imshow(sourceFrame)
trgPlot = trgWindow.imshow(targetFrame)
plt.ion()
plt.show()


#TODO - Separate processing stack in own file/class/module w/e
frameCount = 1

while targetSuccess and sourceSuccess:
 
    print("Processing frame {} ...".format(frameCount))
    frameCount = frameCount + 1
    tick = time.time()
    #get frames
    sourceSuccess, sourceFrame = source.read()
    targetSuccess, targetFrame = target.read()
    sourceFrame = cv2.resize(sourceFrame, None, fx=scalingFactor, fy=scalingFactor)
    targetFrame = cv2.resize(targetFrame, None, fx=scalingFactor, fy=scalingFactor)
    #convert from BGR to RGB
    sourceFrame = cv2.cvtColor(sourceFrame, cv2.COLOR_BGR2RGB)
    targetFrame = cv2.cvtColor(targetFrame, cv2.COLOR_BGR2RGB)
    #convert to gray for detection

    srcGray = cv2.cvtColor(sourceFrame, cv2.COLOR_RGB2GRAY)
    trgGray = cv2.cvtColor(targetFrame, cv2.COLOR_RGB2GRAY)

    srcBB = face_detector(srcGray, 1)
    trgBB = face_detector(trgGray, 1)

    #for now assume one face only
    if len(srcBB) < 1 or len(trgBB)< 1:
        continue
    srcBB = srcBB[0]
    trgBB = trgBB[0]

    #obtain landmarks
    srcLandmarks = landmark_predictor(srcGray, srcBB)
    trgLandmarks = landmark_predictor(trgGray, trgBB)

    #draw frames, bounding box and landmarks
    srcPlot.set_data(sourceFrame)
    trgPlot.set_data(targetFrame)

    x,y,w,h = dlib_rect_to_bb(srcBB)
    srcWindow.add_patch(matplotlib.patches.Rectangle((x,y), w, h, fill=False))
    #triangulate
    im_h,im_w = sourceFrame.shape[:2]
    subdiv = cv2.Subdiv2D((0,0,im_h+2,im_w+2))
    print("Processing frame {} {}".format(im_h, im_w))
    for i in range(0,68):
        pt_x = srcLandmarks.part(i).x
        pt_y = srcLandmarks.part(i).y
        pt_y = np.clip(pt_y, 0, im_h)
        pt_x = np.clip(pt_x, 0, im_w)

        pts = (pt_x,pt_y)
        #axes are reveres for openCV wich caused out of bounds issues
        cv_pts = (pt_y, pt_x)
        
        print("Point {} : {} {} ".format(i, pts[0],pts[1]))
        srcWindow.add_patch(matplotlib.patches.Circle(pts, 1))
        subdiv.insert((cv_pts))
    triangles = subdiv.getTriangleList()
    for i in range(0, len(triangles)):
        pt1 = (triangles[i][1], triangles[i][0])
        pt2 = (triangles[i][3], triangles[i][2])
        pt3 = (triangles[i][5], triangles[i][4])
        verts = np.zeros((3,2))
        verts[0,:] = pt1
        verts[1,:] = pt2
        verts[2,:] = pt3
        srcWindow.add_patch(matplotlib.patches.Polygon(verts, True, fill=False))

    #convert in a neutral scale invariant space

    #move stuff arround
   
    
    
    
    
    
    
    
   
    x,y,w,h = dlib_rect_to_bb(trgBB)
    trgWindow.add_patch(matplotlib.patches.Rectangle((x,y),w,h, fill= False))
    for i in range(0,68):
        trgWindow.add_patch(matplotlib.patches.Circle((trgLandmarks.part(i).x, trgLandmarks.part(i).y), 1))


   


    tock = time.time()
    print("Done, elasped time: {}".format(tock-tick))
    #clear patches
    plt.pause(frameTime)
    for p in reversed(srcWindow.patches):
        p.remove()
    for p in reversed(trgWindow.patches):
        p.remove()

source.release()
target.release()



