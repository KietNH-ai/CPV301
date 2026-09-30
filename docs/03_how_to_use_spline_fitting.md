# How to Use Spline Fitting

When an AI (like YOLO) or a classic algorithm detects a lane, it often outputs a noisy, jagged, or disconnected set of points. **Spline Fitting** is the mathematical process of drawing a smooth, continuous curve through these points.

## 1. What is a Spline?
A spline is a piecewise polynomial curve. Instead of forcing one rigid straight line (like a ruler) through your points, a spline bends to fit the curve of the road smoothly.

## 2. The Workflow

### Step A: Extract Points
From your YOLO mask or Hough edge detection, you get a list of pixel coordinates `(x, y)` where the lane is present.

### Step B: Filter and Sort
Lanes go from the bottom of the image to the top. Sort your `(x, y)` points based on the `y` coordinate (from bottom to top) so the math knows in which direction the curve flows.

### Step C: Fit the Spline
In Python, we use `scipy.interpolate` to fit the curve. 
```python
import numpy as np
from scipy.interpolate import splprep, splev

# y_pts and x_pts are your lane coordinates
# Fit a B-spline to the data (k=3 means cubic spline)
tck, u = splprep([y_pts, x_pts], s=2, k=3)

# Generate new, perfectly smooth points along the curve
u_new = np.linspace(u.min(), u.max(), 100)
y_smooth, x_smooth = splev(u_new, tck)
```
*Note: `s` is a smoothing factor. A higher `s` ignores noisy outliers.*

### Step D: Draw the Spline
Using OpenCV, you connect the newly generated `(x_smooth, y_smooth)` points with a thick colored line and overlay it on your original image.

## 3. Why Splines are Better
1. **Curves:** They handle winding roads effortlessly.
2. **Missing Data:** If a dashed lane line has gaps, the spline naturally bridges the gap smoothly.
3. **Outlier Rejection:** They can be tuned to ignore random points that YOLO falsely detected as a lane.
