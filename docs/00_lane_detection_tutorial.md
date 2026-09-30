# Lane Detection System: Concept & Data Cleaning Tutorial

Welcome to the Lane Detection Project! Since the goal here is to **learn and study**, it is essential to understand *why* we take certain steps before writing the code. 

This tutorial covers the theory behind our pipeline, focusing heavily on data cleaning and how YOLOv11 and Spline Fitting work together.

## 1. The Big Picture: How the System Works

A lane detection system isn't just one algorithm; it's a pipeline of distinct steps. 

1. **Preprocessing:** We clean the images and prepare them for the AI.
2. **YOLOv11 (The AI):** It looks at the image and highlights "where the lanes are" (segmentation mask).
3. **Extraction:** We take that highlighted mask and convert it into X, Y coordinates (math data).
4. **Spline Fitting (The Math):** We use a mathematical curve (spline) to connect those X, Y points smoothly, representing the curved lane line.

## 2. Data Cleaning & Preparation

In Machine Learning, there is a golden rule: **"Garbage In, Garbage Out."**

### A. Frame Extraction & Deduplication
Videos are just 30 images (frames) per second. If a car is stopped at a red light, 300 frames will look exactly the same. We write a script to sample frames (e.g., 1 frame every second) and remove duplicates.

### B. Handling Environmental Edge Cases
Dashcam footage includes rain, glare, nighttime, and dirty windshields. We must keep a balanced amount of these edge cases in our dataset so the AI learns what a lane looks like in bad weather.

### C. Cropping the Region of Interest (ROI)
The top 40% of a dashcam video is usually the sky, trees, and buildings. Lanes don't exist in the sky. By cropping out the sky, we remove useless data, speed up training, and prevent the AI from mistaking a powerline for a lane.

### D. Labeling (The Ground Truth)
To train YOLO, humans must draw shapes over the lanes in the training images. YOLO requires a specific `.txt` format where coordinates are normalized (between 0.0 and 1.0).

## 3. How YOLOv11 Fits In

YOLO (You Only Look Once) is incredibly fast. We will likely use **YOLOv11-Seg (Instance Segmentation)**. 
YOLO classifies every single pixel in the image as "Lane" or "Not Lane", reducing a complex color image down to just a set of points.

## 4. How Spline Fitting Works

Once YOLO outputs its prediction, we have a raw list of pixels (X, Y coordinates). Why not just draw a straight line through them? Because roads curve!
A **Cubic Spline** or **B-Spline** is great for lane detection. It smooths out noise and creates a mathematical curve ($y = ax^3 + bx^2 + cx + d$) that we can draw back onto the original frame.
