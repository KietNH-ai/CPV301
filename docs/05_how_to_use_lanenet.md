# How to Use LaneNet

**LaneNet** is a deep learning architecture specifically designed for end-to-end lane detection. Unlike YOLO (which is a general object detector), LaneNet was built from the ground up to solve the unique challenges of lane lines.

## 1. The Core Concept
Lane detection has a problem: How many lanes are on a highway? Sometimes 2, sometimes 6. LaneNet handles this unknown number of lanes by splitting the neural network into two branches:

### Branch 1: Binary Segmentation
This branch acts as a filter. It classifies every pixel as either "Lane" or "Background". It doesn't care *which* lane it is, just that it is a lane marker.

### Branch 2: Instance Embedding
This is the magic of LaneNet. It assigns an "embedding" (a mathematical vector) to every lane pixel. 
* Pixels that belong to the *same* lane line are given similar embeddings.
* Pixels belonging to *different* lane lines are forced to have very different embeddings.

## 2. Post-Processing (Clustering)
Once the neural network outputs the data from both branches:
1. We take all the pixels identified as "Lane" (from Branch 1).
2. We look at their embeddings (from Branch 2) and use a clustering algorithm (like DBSCAN) to group them together.
3. Group A becomes the Left Lane, Group B becomes the Right Lane, etc.

## 3. H-Net (Perspective Transformation)
LaneNet often includes a secondary network called H-Net. H-Net learns the optimal perspective transformation matrix. 
Instead of manually tuning a Bird's-Eye-View (Inverse Perspective Mapping) matrix, H-Net automatically projects the curved lanes into a flat, straight plane, fits a polynomial curve, and projects it back.

## 4. When to Use LaneNet
* **Use it when:** You want a highly specialized, academic-grade lane detector that handles lane changes, varying numbers of lanes, and complex curves elegantly.
* **Don't use it when:** You need a quick, easy-to-deploy solution. LaneNet requires a complex, custom training pipeline (like the TuSimple dataset) and is harder to set up than an off-the-shelf YOLO model.
