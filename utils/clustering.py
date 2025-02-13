import numpy as np
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from utils.visualization import draw_group_silhouettes
def group_and_draw_silhouettes(boxes,
                               max_distance,
                               frame,
                               calculate_contours=False,
                               previous_labels=None,
                               previous_points=None):
    """
    Group detected objects and draw silhouettes.

    Args:
        boxes (list): List of bounding boxes [x1, y1, x2, y2].
        max_distance (float): Maximum distance for clustering.
        frame (numpy.ndarray): Frame to draw silhouettes on.
        calculate_contours (bool): Whether to calculate new contours.
        previous_labels (list): Labels from the previous frame.
        previous_points (list): Points from the previous frame.

    Returns:
        component_labels (list): Labels for each cluster.
        group_points (list): Points for each cluster.
    """
    if len(boxes) == 0:
        return [], []

    # Compute centers of bounding boxes
    centers = np.array([[((x1 + x2) / 2), ((y1 + y2) / 2)] for x1, y1, x2, y2 in boxes])

    # Compute pairwise distances between centers
    distances = cdist(centers, centers)

    # Create adjacency matrix (1 if distance < max_distance, else 0)
    adjacency_matrix = (distances < max_distance).astype(int)

    # Find connected components (clusters)
    graph = csr_matrix(adjacency_matrix)
    n_components, labels = connected_components(csgraph=graph, directed=False)

    # Group points and draw silhouettes
    component_labels = []
    group_points = []
    for component_label in range(n_components):
        group_indices = np.where(labels == component_label)[0]
        if len(group_indices) > 3:  # Only consider clusters with at least 3 objects
            component_labels.append(component_label)
            points = np.array([centers[idx] for idx in group_indices], dtype=np.int32)
            group_points.append(points)
            draw_group_silhouettes(frame, points, component_label)

    return component_labels, group_points