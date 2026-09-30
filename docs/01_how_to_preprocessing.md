# How to Preprocess Data for Lane Detection

Preprocessing is the act of transforming raw data (dashcam videos) into clean, standardized inputs that an AI model or computer vision algorithm can easily understand.

## 1. Video to Frames Extraction
Machine learning models process static images, not video files.
* **The Process:** Use OpenCV (`cv2.VideoCapture`) to read a video file and save individual frames as `.jpg` or `.png` files.
* **Frame Rate Sampling:** Do not save all 30 frames per second (FPS). Save 1 or 2 frames per second to reduce data redundancy.

## 2. Region of Interest (ROI) Cropping
Lanes only appear in the bottom half of a dashcam video.
* **The Process:** Crop the image to remove the sky, hood of the car, and distant horizon.
* **Why?** It stops the model from learning useless features (like clouds) and reduces the image size, making the model run faster.

## 3. Resizing
Neural networks require fixed input sizes (e.g., 640x640 pixels).
* **The Process:** Use `cv2.resize()` to scale your cropped frames down. Keep the aspect ratio in mind; stretching the image too much can distort the lanes.

## 4. Color Space Conversion & Filtering (Classic CV)
If you are using classic methods (like Hough Transform), color is critical.
* **Grayscale:** Converts the image to black and white to simplify edge detection.
* **HSV/HSL Filtering:** Lanes are usually white or yellow. Converting the image to HSV (Hue, Saturation, Value) makes it much easier to filter out everything except white and yellow pixels, regardless of lighting conditions.

## 5. Normalization (For Deep Learning)
Neural networks prefer small numbers (between 0 and 1).
* **The Process:** Divide all pixel values (which are 0-255) by 255.0. This standardizes the data and helps the neural network learn faster and more stably.
