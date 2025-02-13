# Real-Time Dynamic Clustering with YOLOv8 P2, CSR Matrix, and Minimum Spanning Tree

This repository implements real-time dynamic clustering of detected objects using YOLOv8 p2,
CSR matrices, and Minimum Spanning Trees (MST). It is designed for applications like crowd analysis, traffic monitoring, and scene understanding.

![demo_gif](https://github.com/chiflmas/dynamic_clustering/blob/main/dynamic_clustering_slow.gif)

## Features
- Real-time object detection using YOLOv8.
- Dynamic clustering based on spatial proximity.
- Visualization of clusters using Minimum Spanning Trees (MST).
- Efficient computation using CSR matrices.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/chiflmas/dynamic_clustering.git
   cd dynamic_clustering
   ```
2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ``` 
## Usage
```bash
python main.py --input-video path/to/input.mp4 --output-video path/to/output.mp4
```

## Parameters
- input-video: Path to the input video file.
- output-video: Path to save the output video.
- apply-clahe: (Optional) Apply CLAHE for contrast enhancement.
- group-distance: (Optional) Maximum distance for clustering (default: 1.6).

## Example
```bash
python main.py --input-video input.mp4 --output-video output.mp4 --apply-clahe --group-distance 1.7
```

## License
This project is licensed under the MIT License. See LICENSE for details.