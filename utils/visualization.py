import cv2
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree

def draw_group_silhouettes(frame,
                           points,
                           component_label):
    """
    Draw silhouettes and MST for a group of points.

    Args:
        frame (numpy.ndarray): Frame to draw on.
        points (numpy.ndarray): Points in the cluster.
        component_label (int): Label for the cluster.
    """
    # Generate a unique color for the cluster
    color = tuple(map(int, np.random.randint(0, 256, size=3)))

    # Draw points
    for point in points:
        cv2.circle(frame, tuple(point), 3, color, -1)

    # Draw MST if there are enough points
    if len(points) > 1:
        dist_matrix = squareform(pdist(points))  # Compute pairwise distances
        mst = minimum_spanning_tree(dist_matrix).toarray().astype(float)  # Compute MST
        for i in range(mst.shape[0]):
            for j in range(i + 1, mst.shape[1]):
                if mst[i, j] > 0:  # Draw edges of MST
                    cv2.line(frame, tuple(points[i]), tuple(points[j]), color, 2)