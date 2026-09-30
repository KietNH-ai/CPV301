# How to Use Hough Transform

The **Hough Transform** is the most famous classic computer vision algorithm for finding straight lines in an image. Before AI, this was how all lane detection systems worked.

## 1. The Pipeline
Because Hough Transform uses pure math (not AI), the data must be heavily preprocessed first.

1. **Grayscale:** Convert image to black and white.
2. **Gaussian Blur:** Blur the image to remove tiny noise and gravel textures.
3. **Canny Edge Detection:** Find sharp gradients (changes in pixel color). This leaves you with an image that looks like a black canvas with white outlines.
4. **ROI Mask:** Crop out the sky and other lanes so you only look at the road directly in front of you.
5. **Hough Transform:** Apply the math.

## 2. How it Works (The Math)
A line in coordinate space `(x, y)` can be represented as a single point in Hough space `(m, b)` where $y = mx + b$. 
The algorithm checks every single edge pixel from step 3. If many edge pixels form a straight line, their curves in Hough space will all intersect at a single point. The algorithm looks for these intersections to declare: "I found a line!"

## 3. Python Implementation
```python
import cv2
import numpy as np

# Assuming 'edges' is your image after Canny Edge Detection
lines = cv2.HoughLinesP(
    edges,
    rho=1,                # Distance resolution in pixels
    theta=np.pi/180,      # Angle resolution in radians
    threshold=50,         # Min number of intersections to be a line
    minLineLength=100,    # Minimum length of a line
    maxLineGap=50         # Max gap between dashed lines to connect them
)
```

## 4. Pros and Cons
**Pros:**
* Extremely fast. Doesn't require a GPU.
* No training data or labeling required.

**Cons:**
* Only detects **straight lines**. It fails on curved roads.
* Highly sensitive to shadows, glare, and changing weather conditions.
