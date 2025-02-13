import argparse
import cv2
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm
from utils.video_processing import apply_frame_clahe
from utils.clustering import group_and_draw_silhouettes

def process_video_with_yolo_tracking(model_path,
                                     input_video_path,
                                     output_video_path,
                                     apply_clahe=False,
                                     group_distance=1.6):
    """
    Process a video using YOLOv8 for object detection and dynamic clustering.

    Args:
        model_path (str): Path to the YOLOv8 model.
        input_video_path (str): Path to the input video file.
        output_video_path (str): Path to save the output video.
        apply_clahe (bool): Whether to apply CLAHE for contrast enhancement.
        group_distance (float): Maximum distance for clustering (multiplied by average object width).
    """
    # Load YOLOv8 model
    model = YOLO(model_path)
    model.fuse()  # Fuse model for faster inference

    # Open video file
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Error opening video file.")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    # Initialize variables for tracking and clustering
    frame_counter = 0
    previous_labels = None
    previous_points = None

    # Process video frames
    with tqdm(total=frame_count,
              desc="Processing video",
              unit="frame") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            original_frame = frame.copy()

            # Apply CLAHE for contrast enhancement (if enabled)
            if apply_clahe:
                frame = apply_frame_clahe(frame)

            # Run YOLOv8 tracking
            results = model.track(frame,
                                  imgsz=1088,
                                  iou=0.25,
                                  conf=0.35,
                                  classes=[0, 1],
                                  max_det=700,
                                  persist=True,
                                  verbose=False,
                                  half=True,
                                  tracker='botsort.yaml')

            # Extract bounding boxes for pedestrians (class 0)
            class0_boxes = []
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        class_id = int(box.cls[0])
                        if class_id == 0:  # Pedestrians
                            class0_boxes.append([x1, y1, x2, y2])

            # Cluster objects every 10 frames (to reduce computational load)
            calculate_contours = (frame_counter % 10 == 0)
            previous_labels, previous_points = group_and_draw_silhouettes(
                class0_boxes,
                max_distance=np.mean([x2 - x1 for x1, y1, x2, y2 in class0_boxes]) * group_distance,
                frame=original_frame,
                calculate_contours=calculate_contours,
                previous_labels=previous_labels,
                previous_points=previous_points
            )

            # Write processed frame to output video
            out.write(original_frame)
            frame_counter += 1
            pbar.update(1)

    # Release resources
    cap.release()
    out.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-Time Dynamic Clustering with YOLOv8")
    parser.add_argument("--input-video", type=str, required=True, help="Path to the input video file.")
    parser.add_argument("--output-video", type=str, required=True, help="Path to save the output video.")
    parser.add_argument("--apply-clahe", action="store_true", help="Apply CLAHE for contrast enhancement.")
    parser.add_argument("--group-distance", type=float, default=1.6, help="Maximum distance for clustering.")
    args = parser.parse_args()

    process_video_with_yolo_tracking(
        model_path="no_aug_no_freeze_0_3_2p.pt",
        input_video_path=args.input_video,
        output_video_path=args.output_video,
        apply_clahe=args.apply_clahe,
        group_distance=args.group_distance
    )