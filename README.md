# virtual-drawing-pad
This project was to understand the concepts of **thresholding**, **contours** and various **smoothing functions** performed on images.
The drawing pad works with any red colour objects and points will be plotted at the centroid of that object. The drawing pad
has been implemented in two ways:

### 1. Centroid of contour
1. threshold the HSV (for adapting to different lighting conditions) converted image to find the red portions of image.
2. detect the contours in the image and consider the one with largest area to be required contour.
3. plot line from centroid of previous contour to the centroid of current contour.

### 2. Centre of white-mass
1. threshold the HSV (for adapting to different lighting conditions) converted image to find the red portions of image.
2. take mean of all the x-coordinates and the y-coordinates that are white in the thresholded image. This will give us the mean
position at which we have to plot the point. 
3. plot line from previous such point to current point.



To use the program, press s to toggle drawing(initially off) and press e to toggle erasing(initially off). To save image and 
quit, press q.

This [blogpost](https://towardsdatascience.com/tutorial-webcam-paint-opencv-dbe356ab5d6c) really helped with the Centroid-of-contour method.
