#measures.py
import math

#TODO add other measure
#hausdroff?


def sum_squared_euclideean_distances(landmarks1, landmarks2):
    """For best results, apply to points in local face space, to avoid
    differences caused by rotations / translation,
    avoids sqrt for performance reasons"""
    total = 0
    for pair1,pair2 in zip(landmarks1,landmarks2):
        d1 = pair1[0] - pair2[0]
        d2 = pair1[1] - pair2[1]
        d = d1*d1 + d2*d2
        total = total + d
    return total

def sum_euclideean_distances(landmarks1, landmarks2):
    """Like sum_squared_euclidean_distances put applies square root,
    so it's a true euclidean distance"""
    total = 0
    for pair1,pair2 in zip(landmarks1,landmarks2):
        d1 = pair1[0] - pair2[0]
        d2 = pair1[1] - pair2[1]
        d = math.sqrt(d1*d1 + d2*d2)
        total = total + d
    return total



    

