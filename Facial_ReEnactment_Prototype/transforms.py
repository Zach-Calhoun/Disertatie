#data transforms
import numpy as np

def dlib_rect_to_bb(rect):
    """Converts a dlib rectangle to a top position width and height tuple"""
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return (x, y, w, h)

def landmarks_to_points(landmarks):
    """Converts dlib landmarks to coordinates tuples"""
    points = []
    for i in range(0, 68):
        pt_x = landmarks.part(i).x
        pt_y = landmarks.part(i).y
        points.append((pt_x, pt_y))
    return points

#triangles come as list of points
def triangles_to_verts(triangles):
    """converts openCV subdiv2D triangulation to a np.array representation of triangles
    format that is usable by matplotlib polygon drawing"""
    triangle_list = []
    for i in range(0, len(triangles)):
        verts = np.zeros((3,2),dtype=np.int32)
        verts[0,:] = (triangles[i][1], triangles[i][0])
        verts[1,:] = (triangles[i][3], triangles[i][2])
        verts[2,:] = (triangles[i][5], triangles[i][4])
        triangle_list.append(verts)
    return triangle_list


def verts_to_indices(vertices, landmark_list):
    """converts triangulation coordinates to indexes of landmakrs
    so that we can rebuild the triangles for the second face
    without running a second delaunay triangulation
    will also filter out triangles that do not match to landmarks 
    meaning it will drop contour triangles and only keep the faces one
    otherwise we'd still have mismatching indices issues
    """
    indices = []
    validTriangles = []
    landmarks = list.copy(landmark_list)
    for triangle_verts in vertices:
        v1 = triangle_verts[0,:]
        v2 = triangle_verts[1,:]
        v3 = triangle_verts[2,:]
        try:
            #TODO see if this can be otpimised by prior sorting and using a binary search
            v1pos = landmarks.index((v1[0],v1[1]))
            v2pos = landmarks.index((v2[0],v2[1]))
            v3pos = landmarks.index((v3[0],v3[1]))
        except Exception as e:
            #print(e)
            continue
        indices.append((v1pos,v2pos,v3pos))
        validTriangles.append(triangle_verts)
    return indices, validTriangles

def landmark_indices_to_triangles(landmarks, indices):
    """converts a set of landmarks and triangle indices to an output similar to
    triangles_to_verts"""
    triangle_list = []
    for v1index,v2index,v3index in indices:
        verts = np.zeros((3,2))
        verts[0,:] = (landmarks[v1index][0], landmarks[v1index][1])
        verts[1,:] = (landmarks[v2index][0], landmarks[v2index][1])
        verts[2,:] = (landmarks[v3index][0], landmarks[v3index][1])
        triangle_list.append(verts)
    
    return triangle_list

def apply_transform(landmarks, matrix):
    new_landmarks = []
    for landmark in landmarks:
        #tmp = np.array([[landmark[0]],[landmark[1]],[1]])
        tmp = np.array([landmark[0],landmark[1],1])
        res = np.dot(matrix, tmp)
        new_lm = (res[0],res[1])
        new_landmarks.append(new_lm)
    return new_landmarks
    