# How to Use YOLOv11 for Lane Detection

YOLO (You Only Look Once) is a state-of-the-art, real-time object detection model. For lanes, we don't just want bounding boxes; we want the exact pixels of the lane. Therefore, we use **YOLOv11 Instance Segmentation (YOLOv11-seg)**.

## 1. Understanding the Data Format
To train YOLO, you need a dataset folder with `images/` and `labels/`.
Each image needs a corresponding `.txt` label file. For segmentation, the text file looks like this:
`[class_id] [x1] [y1] [x2] [y2] ... [xn] [yn]`
* `class_id`: 0 for lane line.
* `x, y`: Normalized coordinates (0.0 to 1.0) of the polygon outlining the lane.

## 2. Setting up the Config (data.yaml)
You must create a `data.yaml` file to tell YOLO where your data is:
```yaml
train: ../data/processed/train/images
val: ../data/processed/valid/images

nc: 1
names: ['lane']
```

## 3. Training the Model
Using the Ultralytics Python library, training is very simple:
```python
from ultralytics import YOLO

# Load a pre-trained segmentation model
model = YOLO('yolo11n-seg.pt')

# Train the model on your dataset
results = model.train(data='data.yaml', epochs=50, imgsz=640)
```

## 4. Inference (Using the Model)
Once trained, you can run the model on a new video frame to get the lane masks:
```python
results = model('test_image.jpg')
for result in results:
    masks = result.masks  # This contains the pixel data of the detected lanes!
    # You can now pass these masks to your Spline Fitting algorithm
```

## 5. Why YOLO?
YOLO is robust against lighting changes, shadows, and missing lane markers because it learns the *context* of a road, unlike classic math-based computer vision.
