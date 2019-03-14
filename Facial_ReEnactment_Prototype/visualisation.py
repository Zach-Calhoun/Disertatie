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

def debug_fill_triangles(srcTriangles, trgTriangles, srcWindow, trgWindow, srcFrame, trgFrame):
    for srcTri, trgTri in zip(srcTriangles, trgTriangles):
        color = np.uint8(np.random.uniform(0, 255, 3))
        c = tuple(map(int, color))
        srcWindow.add_patch(matplotlib.patches.Polygon(srcTri, True, fill=False))
        cv2.fillConvexPoly(srcFrame, np.int32(srcTri),c, 16, 0)
        trgWindow.add_patch(matplotlib.patches.Polygon(trgTri, True, fill=False))
        cv2.fillConvexPoly(trgFrame, np.int32(trgTri),c, 16, 0)