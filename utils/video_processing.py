import cv2

def apply_frame_clahe(frame, gridsize=(3, 3)):
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to a frame.

    Args:
        frame (numpy.ndarray): Input frame.
        gridsize (tuple): Grid size for CLAHE.

    Returns:
        numpy.ndarray: Frame with CLAHE applied.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=4, tileGridSize=gridsize)
    gray = clahe.apply(gray)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)