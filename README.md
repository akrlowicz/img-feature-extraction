# img-feature-extraction
Feature extraction from images: SIFT method using Bag of Visual Words for feature extraction from images, Sobel filter for edge detection

# Bag of Visual Words
In `sift_bovw.py` file there is SIFT() class that applies Scale-invariant feature transform to images and calculates descriptors of keypoints. With KMeans clustering is performed to obtain histograms.

Usage example:

```
sift = SIFT(n_classes)
sift.apply_sift(X_train)

sift_train = sift.get_features(X_train)
sift_val = sift.get_features(X_val)
sift_test = sift.get_features(X_test)
```

# Edge detection
Edge detection can be performed with variety of kernels. In current file edge detection is performed with Sobel filter.
The Sobel Operator detects edges that are marked by sudden changes in pixel intensity.
These are the kernels used for Sobel Edge Detection:
- X-Direction Kernel:
<img src="https://latex.codecogs.com/svg.image?\bg_black&space;&space;\begin{bmatrix}&space;-1&space;&&space;0&space;&&space;&plus;1&space;\\&space;-2&space;&&space;0&space;&&space;&plus;2&space;\\&space;-1&space;&&space;0&space;&&space;&plus;1&space;\end{bmatrix}" title="\bg_black \begin{bmatrix} -1 & 0 & +1 \\ -2 & 0 & +2 \\ -1 & 0 & +1 \end{bmatrix}" />

- Y-Direction Kernel:
<img src="https://latex.codecogs.com/png.image?\dpi{110}&space;&space;\begin{bmatrix}&space;&plus;1&space;&&space;&plus;2&space;&&space;&plus;1&space;\\&space;0&space;&&space;0&space;&&space;0&space;\\&space;-1&space;&&space;-2&space;&&space;-1&space;\end{bmatrix}" title=" \begin{bmatrix} +1 & +2 & +1 \\ 0 & 0 & 0 \\ -1 & -2 & -1 \end{bmatrix}" />

The final approximation of  the gradient magnitude, G can be computed as:
<img src="https://latex.codecogs.com/svg.image?\bg_black&space;\begin{align*}\begin{equation*}&space;G&space;=&space;\sqrt{G_x^2&space;*&space;G_y^2}&space;\end{equation*}\end{align*}" title="\bg_black \begin{align*}\begin{equation*} G = \sqrt{G_x^2 * G_y^2} \end{equation*}\end{align*}" />

And the orientation of the gradient can then be approximated as:

<img src="https://latex.codecogs.com/png.image?\dpi{110}&space;\begin{equation}&space;\Theta&space;=&space;arctan(G_y&space;/&space;G_x)&space;\end{equation}" title="\begin{equation} \Theta = arctan(G_y / G_x) \end{equation}" />

Usage example:

```
sobel_train = apply_sobel(X_train) 
```
