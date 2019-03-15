#visualisation utils
import matplotlib
import matplotlib.pyplot as plt
import cv2
import numpy as np

def view_landmarks(srcLandmarks, trgLandmarks, srcWindow, trgWindow):
    #combining sequences to reduce loop overhead
    landmarks = zip(srcLandmarks, trgLandmarks)
    for srcPoint, trgPoint in landmarks:
        srcWindow.add_patch(matplotlib.patches.Circle(srcPoint, 1))
        trgWindow.add_patch(matplotlib.patches.Circle(trgPoint, 1))



def debug_fill_triangles(srcTriangles, trgTriangles, srcWindow, trgWindow, srcFrame, trgFrame, colors=None):
    color_index = -1
    for srcTri, trgTri in zip(srcTriangles, trgTriangles):
        color_index = color_index + 1
        if colors == None:
            color = np.uint8(np.random.uniform(0, 255, 3))
        else:
            color = colors[color_index]
        c = tuple(map(int, color))
        alpha = 0.2
        srcPts = np.float32([[srcTri[0,:], srcTri[1,:], srcTri[2,:]]])
        srcBB = cv2.boundingRect(srcPts)    
        triPts = []   
        for j in range(0,3):
            srcLocalTriangle = (srcTri[j][0] - srcBB[0], srcTri[j][1] - srcBB[1])
            triPts.append(srcLocalTriangle)
        mask = np.zeros((srcBB[3], srcBB[2], 3), dtype = np.float32)
        tri = np.zeros((srcBB[3], srcBB[2], 3), dtype = np.uint8)
        cv2.fillConvexPoly(mask, np.int32(triPts),(0.2, 0.2, 0.2), 16, 0)
        cv2.fillConvexPoly(tri, np.int32(triPts),c, 16, 0)
        triangle = tri * (.2,0.2,0.2)
        srcFrame[srcBB[1]:srcBB[1]+srcBB[3], srcBB[0]:srcBB[0]+srcBB[2]] =  srcFrame[srcBB[1]:srcBB[1]+srcBB[3], srcBB[0]:srcBB[0]+srcBB[2]] *  ((1.0,1.0,1.0) - mask)
        srcFrame[srcBB[1]:srcBB[1]+srcBB[3], srcBB[0]:srcBB[0]+srcBB[2]] =  srcFrame[srcBB[1]:srcBB[1]+srcBB[3], srcBB[0]:srcBB[0]+srcBB[2]] + triangle
        srcWindow.add_patch(matplotlib.patches.Polygon(srcTri, True, fill=False))
        #cv2.fillConvexPoly(srcFrame, np.int32(srcTri),c, 16, 0)
        
        trgPts = np.float32([[trgTri[0,:], trgTri[1,:], trgTri[2,:]]])
        trgBB = cv2.boundingRect(trgPts)   
        triPts = []   
        for j in range(0,3):
            trgLocalTriangle = (trgTri[j][0] - trgBB[0], trgTri[j][1] - trgBB[1])
            triPts.append(trgLocalTriangle)    
        mask = np.zeros((trgBB[3], trgBB[2], 3), dtype = np.float32)
        tri = np.zeros((trgBB[3], trgBB[2], 3), dtype = np.float32)
        cv2.fillConvexPoly(mask, np.int32(triPts),(alpha, alpha, alpha), 16, 0)
        cv2.fillConvexPoly(tri, np.int32(triPts),c, 16, 0)
        trgFrame[trgBB[1]:trgBB[1]+trgBB[3], trgBB[0]:trgBB[0]+trgBB[2]] = trgFrame[trgBB[1]:trgBB[1]+trgBB[3], trgBB[0]:trgBB[0]+trgBB[2]] *  ((1.0,1.0,1.0) - mask) + ((alpha,alpha,alpha) * tri)
        trgWindow.add_patch(matplotlib.patches.Polygon(trgTri, True, fill=False))
        #cv2.fillConvexPoly(trgFrame, np.int32(trgTri),c, 16, 0)