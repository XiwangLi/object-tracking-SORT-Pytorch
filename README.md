# Object Tracking using SORT in Pytorch

## Yolo for object detection in Videos

The basic logic for object detection in videos is:

1. extract the frames from video using OpenCV:

```python
cap = cv2.VideoCapture(videopath)
 _, frame = cap.read()
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
```
2. apply yolo detection on the extracted frame

```python
pilimg = Image.fromarray(frame)
detections = detect_image(pilimg)
```
## Object tracking in Videos using SORT
[SORT (Simple Online and Realtime Tracking)](https://arxiv.org/pdf/1602.00763.pdf) combines object detection and [Kalman filter](https://en.wikipedia.org/wiki/Kalman_filter) for object tracking. 

## Run this code:
1. install dependencies
```bash
pip install -r requirements.txt
```
2. downoad yolo weights
```bash
wget https://pjreddie.com/media/files/yolov3.weights -O config/yolov3.weights
```
3. upload a `.mp4` video to `./videos` and run [object_tracking_Sort.ipynb](./object_tracking_Sort.ipynb)

You can also ran [object_tracker.py](./object_tracker.py) to save the video with annotations


## References:

1. YOLOv3: https://pjreddie.com/darknet/yolo/
1. YOLO paper: https://pjreddie.com/media/files/papers/YOLOv3.pdf
1. SORT paper: https://arxiv.org/pdf/1602.00763.pdf
1. Alex Bewley's SORT implementation: https://github.com/abewley/sort