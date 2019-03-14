#data transforms
import numpy as np

def dlib_rect_to_bb(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return (x, y, w, h)

def landmarks_to_points(landmarks):
    points = []
    for i in range(0, 68):
        pt_x = landmarks.part(i).x
        pt_y = landmarks.part(i).y
        points.append((pt_x, pt_y))
    return points

#triangles come as list of points
def triangles_to_verts(triangles):
    triangle_list = []
    for i in range(0, len(triangles)):
        verts = np.zeros((3,2))
        verts[0,:] = (triangles[i][1], triangles[i][0])
        verts[1,:] = (triangles[i][3], triangles[i][2])
        verts[2,:] = (triangles[i][5], triangles[i][4])
        triangle_list.append(verts)
    return triangle_list
