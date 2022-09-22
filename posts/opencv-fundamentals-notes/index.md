---
categories:
- opencv
- python
- numpy
- notes
date: 2022-1-16
description: My notes from Jason Dsouza's introductory video on OpenCV for Python.
hide: false
layout: post
search_exclude: false
title: Notes on OpenCV Fundamentals
toc: false

---

* [Overview](#overview)
* [What is OpenCV](#what-is-opencv)
* [Installation](#installation)
* [Reading Images and Video](#reading-images-and-video)
* [Resizing and Rescaling Frames](#resizing-and-rescaling-frames)
* [Drawing Shapes and Text](#drawing-shapes-and-text)
* [Essential Functions](#essential-functions)
* [Image Transformations](#image-transformations)
* [Contour Detection](#contour-detection)
* [Color Spaces](#color-spaces)
* [Color Channels](#color-channels)
* [Blur](#blur)
* [Bitwise Operators](#bitwise-operators)
* [Masking](#masking)
* [Computing Histograms](#computing-histograms)
* [Thresholding Images](#thresholding-images)
* [Edge Detection](#edge-detection)
* [Face Detection](#face-detection)



## Overview

Here are some notes some notes I took while watching Jason Dsouza's [video](https://www.youtube.com/watch?v=oXlwWbU8l2o) providing an introduction to OpenCV for Python.



## What is OpenCV
* [Website](https://opencv.org/)
* An open source computer vision and machine learning library
- Built to provide a common infrastructure for computer vision applications and to accelerate the use of machine perception in commercial products.
- Has more than 2500 optimized algorithms including both classic and state-of-the-art algorithms
- Natively written in C++
- Has C++, Python, Java and MATLAB interfaces
- Supports Windows, Linux, Android, and Mac OS
- Primarily targeted towards real-time applications
- Applications
    - detect and recognize faces
    - identify objects
    - classify human actions in videos
    - track camera movements
    - track moving objects
    - extract 3D models of objects
    - produce 3D point clouds from stereo cameras
    - stitch images together to produce a high resolution image of an entire scene
    - find similar images from an image database
    - remove red eyes from images taken using flash
    - follow eye movements
    - recognize scenery and establish markers to overlay it with augmented reality




## Installation

### Python

- [Website](https://www.python.org/downloads/)
- Use `3.7` or later
- Open Installer
    - Check the box to add Python to path before clicking Install Now
- Verify python installation and version in Command Prompt
    - `python --version`

### **[OpenCV](https://github.com/opencv/opencv-python)**

- Pre-built OpenCV packages for Python
- Default install is CPU-only
- `pip install opencv-contrib-python`
    - includes everything in the main module as well as modules contributed by the community
- `python -m pip install --user opencv-contrib-python`
    - might need to uninstall existing opencv package
        - `pip uninstall opencv-python`



## Streamlit

- [Streamlit - The fastest way to build and share data apps](https://streamlit.io/)
- Turns data scripts into shareable web apps
- `pip install streamlit`
- Test Installation: `streamlit hello`
- Run apps: `streamlit run main.py`
- Format text using [Markdown](https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet)





## Reading Images and Video

- OpenCV does not have a built-in way for displaying images larger than the screen so that the whole image is visible

```python
import cv2 as cv

# Read in an image
img = cv.imread('flower.jpg')

# Display image in new window
# Input 1: Name of window
# Input 2: Variable
cv.imshow('Flower', img)

# Wait for key to be pressed before
# closing the above window
cv.waitKey(0)
```

```python
import cv2 as cv

# Capture live webcam
# input is the index for the available
# capture devices
# capture = cv.VideoCapture(0)

# Read in a video file
capture = cv.VideoCapture('video_sample.mp4')

# read video frame by frame
while True:
    isTrue, frame = capture.read()
    cv.imshow('Video Feed', frame)

    # Stop playing video for after 20 seconds
    # or if the 'd' key is pressed
    if cv.waitKey(20) and 0xFF == ord('d'):
        break

capture.release()
cv.destroyAllWindows()
```

## Resizing and Rescaling Frames

```python
import cv2 as cv

def rescale_frame(frame, scale=0.75):
    # Calculate the new dimensions
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)

    # resize the frame to the new dimensions
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

# Read in an image
img = cv.imread('flower.jpg')
frame_resized = rescale_frame(img)

# Display image in new window
# Input 1: Name of window
# Input 2: Variable
cv.imshow('Flower', img)

# Wait for key to be pressed before
# closing the above window
cv.waitKey(0)
```

```python
import cv2 as cv

def rescale_frame(frame, scale=0.75):
    # Calculate the new dimensions
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)

    # resize the frame to the new dimensions
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

# Capture live webcam
# input is the index for the available
# capture devices
# capture = cv.VideoCapture(0)

# Read in a video file
capture = cv.VideoCapture('video_sample.mp4')

# read video frame by frame
while True:
    isTrue, frame = capture.read()
    cv.imshow('Video Feed', rescale_frame(frame))

    # Stop playing video for after 20 seconds
    # or if the 'd' key is pressed
    if cv.waitKey(20) and 0xFF == ord('d'):
        break

capture.release()
cv.destroyAllWindows()
```

[VideoCapture Properties](https://docs.opencv.org/3.4/d4/d15/group__videoio__flags__base.html#gaeb8dd9c89c10a5c63c139bf7c4f5704d)

- `3`: width
- `4`: height

```python
import cv2 as cv

# Change properties for live video capture
def changeRes(width, height):
    # Update the width property for the video capture
    capture.set(3, width)
    # Update the height propert for the video capture
    capture.set(4, height)

# Capture live webcam
# input is the index for the available
# capture devices
capture = cv.VideoCapture(0)

changeRes(960, 540)

# read video frame by frame
while True:
    isTrue, frame = capture.read()
    cv.imshow("Video Feed", frame)

    # Stop playing video if it has been
    # playing for at least 20 milliseconds
    # and the 'd' key is pressed
    if cv.waitKey(20) and 0xFF == ord("d"):
        break

capture.release()
cv.destroyAllWindows()
```



## Drawing Shapes and Text

[Fonts Documentation](https://docs.opencv.org/3.1.0/d0/de1/group__core.html#ga0f9314ea6e35f99bb23f29567fc16e11)

**Streamlit Demo:** [https://share.streamlit.io/cj-mills/opencv-notes/main/streamlit-demo-draw-shapes.py](https://share.streamlit.io/cj-mills/opencv-notes/main/streamlit-demo-draw-shapes.py)

```python
import streamlit as st
import cv2 as cv

st.title("Drawing Shapes and Text")
st.header("Input Image")
img_bgr = cv.imread("images/flower.jpg")
st.image(cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB), caption="Input")


st.header("Shape Selection")
shape_option = st.selectbox(
    "Select a shape", ("Rectangle", "Circle", "Line", "Text"), index=0
)

st.subheader("Shape Color")
color_picker_col1, color_picker_col2 = st.columns([1, 1])
shape_color = color_picker_col1.color_picker("Pick A Color", "#00f900")
color_picker_col2.write(f"Hex Color Code: {shape_color}")
color_r = int(shape_color[1:3], 16)
color_g = int(shape_color[3:5], 16)
color_b = int(shape_color[5:7], 16)
color_picker_col2.write(f"RGB Color Code: ({color_r},{color_g},{color_b})")


col1, col2 = st.columns([1, 1])
col1.subheader("Origin Coorindates")
origin_x = col1.slider("X:", min_value=0, max_value=img_bgr.shape[1], value=20)
origin_y = col1.slider("Y:", min_value=0, max_value=img_bgr.shape[0], value=20)

if shape_option == "Rectangle":
    col2.subheader("Dimensions")
    width = col2.slider("Width:", min_value=0, max_value=img_bgr.shape[1], value=40)
    height = col2.slider("Height:", min_value=0, max_value=img_bgr.shape[0], value=40)
    thickness = st.slider("Thickness", min_value=-1, max_value=20, value=2)
    cv.rectangle(
        # Image to draw on
        img_bgr,
        # Starting coords
        (origin_x, origin_y),
        # ending coords
        (origin_x + width, origin_y + height),
        # Color (in BGR)
        (color_b, color_g, color_r),
        # Line thickness (-1 to fill the shape)
        thickness=thickness,
    )
if shape_option == "Circle":
    col2.subheader("Dimensions")
    radius = col2.slider(
        "Radius:", min_value=0, max_value=(min(img_bgr.shape[:2]) // 2), value=40
    )
    thickness = st.slider("Thickness", min_value=-1, max_value=20, value=2)
    cv.circle(
        # Image to draw on
        img_bgr,
        # Starting coords
        (origin_x, origin_y),
        # Radius
        radius,
        # Color (in BGR)
        (color_b, color_g, color_r),
        # Line thickness (-1 to fill the shape)
        thickness=thickness,
    )
if shape_option == "Line":
    col2.subheader("Ending Coordinates")
    end_x = col2.slider("X:", min_value=0, max_value=img_bgr.shape[1], value=40)
    end_y = col2.slider("Y:", min_value=0, max_value=img_bgr.shape[0], value=40)
    thickness = st.slider("Thickness", min_value=1, max_value=40, value=2)
    cv.line(
        # Image to draw on
        img_bgr,
        # Starting coords
        (origin_x, origin_y),
        # Ending coords
        (end_x, end_y),
        # Color (in BGR)
        (color_b, color_g, color_r),
        # Line thickness
        thickness=thickness,
    )
if shape_option == "Text":
    col2.subheader("Input")
    text = col2.text_input("Enter some text:", "Hello World")
    font_size = col2.slider("Font Size:", min_value=0.0, max_value=10.0, value=1.0)
    cv.putText(
        # Image to draw on
        img_bgr,
        # Text
        text,
        # Starting coords
        (origin_x, origin_y),
        # Font type
        cv.FONT_HERSHEY_TRIPLEX,
        # Font size
        font_size,
        # Color (in BGR)
        (color_b, color_g, color_r),
    )
st.image(cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB), caption="Output")


```



## Essential Functions

**Streamlit Demo:** [https://share.streamlit.io/cj-mills/opencv-notes/main/streamlit-demo-essential-functions.py](https://share.streamlit.io/cj-mills/opencv-notes/main/streamlit-demo-essential-functions.py)

```python
import cv2 as cv

# Read in an image
img = cv.imread("flower.jpg")

# Convert image to grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow("Grayscale", gray)

# Blur
# Image to blur
# Kernel size for the blur filter (larger kernel -> more blur)
# Border pixels are reflected by default
blur = cv.GaussianBlur(img, (9, 9), cv.BORDER_DEFAULT)
cv.imshow("Blur", blur)

# Edge Cascade
canny = cv.Canny(img, 125, 175)
cv.imshow("Canny", canny)

# Dilating the image
dilated = cv.dilate(canny, (7, 7), iterations=3)
cv.imshow("Dilated", dilated)

# Eroding
eroded = cv.erode(dilated, (7, 7), iterations=3)
cv.imshow("Eroded", eroded)

# Resize
resize = cv.resize(
    img, (img.shape[1] * 3, img.shape[0] * 3), interpolation=cv.INTER_CUBIC
)
cv.imshow("Resize", resize)

# Cropping
cropped = img[150:351, 150:351]
cv.imshow("Cropped", cropped)

# Wait for key to be pressed before
# closing the above window
cv.waitKey(0)
```



## Image Transformations

**Streamlit Demo:** [https://share.streamlit.io/cj-mills/opencv-notes/main/streamlit-demo-transformations.py](https://share.streamlit.io/cj-mills/opencv-notes/main/streamlit-demo-transformations.py)

```python
import streamlit as st
import cv2 as cv
import numpy as np

st.title("Image Transformations")

st.header("Input Image")
img_bgr = cv.imread("images/flower.jpg")
st.image(cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB), caption="Input")

st.header("Translation")
def translate(img, x, y):
    transMat = np.float32([[1, 0, x], [0, 1, y]])
    dimensions = (img.shape[1], img.shape[0])
    return cv.warpAffine(img, transMat, dimensions)

translate_horizontal = st.slider(
    "Horizontal Translation",
    min_value=-img_bgr.shape[1],
    max_value=img_bgr.shape[1],
    value=img_bgr.shape[1] // 2,
    step=1,
)
translate_vertical = st.slider(
    "Vertical Translation",
    min_value=-img_bgr.shape[0],
    max_value=img_bgr.shape[0],
    value=img_bgr.shape[0] // 2,
    step=1,
)
translated = translate(img_bgr, translate_horizontal, translate_vertical)
st.image(cv.cvtColor(translated, cv.COLOR_BGR2RGB), caption="Translated")



st.header("Rotation")
def rotate(img, angle, rotationPoint=None):
    (height, width) = img.shape[:2]

    if rotationPoint is None:
        rotationPoint = (width // 2, height // 2)
    rotMat = cv.getRotationMatrix2D(rotationPoint, angle, scale=1.0)
    dimensions = (width, height)
    return cv.warpAffine(img, rotMat, dimensions)

rotation_angle = st.slider(
    "Rotation Angle", min_value=0, max_value=360, value=45, step=1,
)
rotated = rotate(img_bgr, rotation_angle)
st.image(cv.cvtColor(rotated, cv.COLOR_BGR2RGB), caption="Rotated")


st.header("Flip")
flip_code = st.radio("Flip Code", (1, 0, -1))
flip = cv.flip(img_bgr, flip_code)
st.image(cv.cvtColor(flip, cv.COLOR_BGR2RGB), caption="Flipped")

```



## Contour Detection

**Streamlit Demo:** [https://share.streamlit.io/cj-mills/opencv-notes/main/streamlit-demo-contour-detection.py](https://share.streamlit.io/cj-mills/opencv-notes/main/streamlit-demo-contour-detection.py)

```python
import streamlit as st
import cv2 as cv
import numpy as np

st.title("Contour Detection")

st.header("Input Image")
img_bgr = cv.imread("images/flower.jpg")
st.image(cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB), caption="Input")

blank = np.zeros(img_bgr.shape, dtype="uint8")

st.header("Grayscale")
gray = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)
st.image(gray, caption="Grayscale")


st.header("Edge Cascade")
threshold_1 = st.slider("Threshold 1", min_value=1, max_value=500, value=125, step=1)
threshold_2 = st.slider("Threshold 2", min_value=1, max_value=500, value=175, step=1)
canny = cv.Canny(img_bgr, threshold_1, threshold_2)
st.image(cv.cvtColor(canny, cv.COLOR_BGR2RGB), caption="Canny Edge Cascade")


st.header("Find Contours")
# Find Countours
# Looks at the edges found in an image
# Returns a list of coordinates for all contours found in an image
# and a hierarchical representation of the contours
# CHAIN_APPROX_NONE: does nothing
contours, hierarchies = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
st.write(f"Found {len(contours)} contours")

st.header("Threshold")
# Pixels with intensity below 125 are set to black
# Pixels with intensity above 125 are set to white
threshold = st.slider("Threshold", min_value=0, max_value=255, value=125, step=1)
ret, thresh = cv.threshold(gray, threshold, 255, cv.THRESH_BINARY)
st.image(cv.cvtColor(thresh, cv.COLOR_BGR2RGB), caption="Threshold")

st.header("Find Contours #2")
# Looks at the edges found in an image
# Returns a list of coordinates for all contours found in an image
# and a hierarchical representation of the contours
# CHAIN_APPROX_NONE: does nothing
contours, hierarchies = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
st.write(f"Found {len(contours)} contours")

st.header("Visualize contours")
cv.drawContours(blank, contours, -1, (0, 0, 255), thickness=1)
st.image(cv.cvtColor(blank, cv.COLOR_BGR2RGB), caption="Contours")

```





## Color Spaces

**Streamlit Demo:** [https://share.streamlit.io/cj-mills/opencv-notes/main/streamlit-demo-color-spaces.py](https://share.streamlit.io/cj-mills/opencv-notes/main/streamlit-demo-color-spaces.py)

```python
import streamlit as st
import cv2 as cv
import numpy as np

st.title("Color Spaces")

st.header("RGB")
img_bgr = cv.imread("images/flower.jpg")
st.image(cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB), caption="Input")

st.header("BGR")
st.image(img_bgr, "BGR")

st.header("Grayscale")
gray = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)
st.image(gray, caption="Grayscale")

st.header("BGR to HSV (Hue Saturation and Value)")
hsv = cv.cvtColor(img_bgr, cv.COLOR_BGR2HSV)
st.image(hsv, "HSV")

st.header("BGR to LAB")
lab = cv.cvtColor(img_bgr, cv.COLOR_BGR2LAB)
st.image(lab, "LAB")

st.header("Grayscale to HSV")
gray_bgr = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
bgr_hsv = cv.cvtColor(gray_bgr, cv.COLOR_BGR2HSV)
st.image(bgr_hsv, "Gray to HSV")
```



## Color Channels

**Streamlit Demo:** [https://share.streamlit.io/cj-mills/opencv-notes/main/streamlit-demo-color-channels.py](https://share.streamlit.io/cj-mills/opencv-notes/main/streamlit-demo-color-channels.py)

```python
import streamlit as st
import cv2 as cv
import numpy as np

st.title("Color Spaces")

st.header("RGB")
img_bgr = cv.imread("images/flower.jpg")
st.image(cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB), caption="Input")

blank = np.zeros(img_bgr.shape[:2], dtype="uint8")

b, g, r = cv.split(img_bgr)

st.header("Visualize Colors")
blue = cv.merge([b, blank, blank])
st.image(cv.cvtColor(blue, cv.COLOR_BGR2RGB), caption="Visualize Blue")
green = cv.merge([blank, g, blank])
st.image(cv.cvtColor(green, cv.COLOR_BGR2RGB), caption="Visualize Green")
red = cv.merge([blank, blank, r])
st.image(cv.cvtColor(red, cv.COLOR_BGR2RGB), caption="Visualize Red")

st.header("Visualize Color Intensitry")
st.image(b, "Blue Intensity")
st.image(g, "Green Intensity")
st.image(r, "Red Intensity")

st.write(f"Image Shape: {img_bgr.shape}")
st.write(f"Blue Shape: {b.shape}")
st.write(f"Green Shape: {g.shape}")
st.write(f"Red Shape: {r.shape}")

merged = cv.merge([b, g, r])
st.image(cv.cvtColor(merged, cv.COLOR_BGR2RGB), "Merged")

```



## Blur

- typically add blur to images that have a lot of noise

**Streamlit Demo:** [https://share.streamlit.io/cj-mills/opencv-notes/main/streamlit-demo-blur.py](https://share.streamlit.io/cj-mills/opencv-notes/main/streamlit-demo-blur.py)

```python
import streamlit as st
import cv2 as cv

st.title("Blur")

st.header("Input Image")
img_bgr = cv.imread("images/flower.jpg")
st.image(cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB), caption="Input")

st.header("Average Blur")
avg_kernel_size = st.slider(
    "Average Kernel Size", min_value=1, max_value=15, value=7, step=2
)
average = cv.blur(img_bgr, (avg_kernel_size, avg_kernel_size))
st.image(cv.cvtColor(average, cv.COLOR_BGR2RGB), "Average Blur")

st.header("Gaussian Blur")
# Gives weight to each pixel in the kernal window
# Basically a weighted average blur?
gaussian_kernel_size = st.slider(
    "Gaussian Kernel Size", min_value=1, max_value=15, value=7, step=2
)
gaussian = cv.GaussianBlur(
    img_bgr, (gaussian_kernel_size, gaussian_kernel_size), sigmaX=0
)
st.image(cv.cvtColor(gaussian, cv.COLOR_BGR2RGB), "Gaussian Blur")

st.header("Median Blur")
# tends to be more effective for reducing noise
# not meant for high kernel sizes
median_kernel_size = st.slider(
    "Median Kernel Size", min_value=1, max_value=35, value=7, step=2
)
median = cv.medianBlur(img_bgr, median_kernel_size)
st.image(cv.cvtColor(median, cv.COLOR_BGR2RGB), "Median Blur")

st.header("Bilateral Blur")
# most effective
bilateral_diameter = st.slider(
    "Bilateral Diameter", min_value=1, max_value=30, value=10, step=1
)
bilateral_sigma_color = st.slider(
    "Bilateral Sigma Color", min_value=1, max_value=100, value=35, step=1
)
bilateral_sigma_space = st.slider(
    "Bilateral Sigma Space", min_value=1, max_value=100, value=25, step=1
)
bilateral = cv.bilateralFilter(
    img_bgr,
    bilateral_diameter,
    sigmaColor=bilateral_sigma_color,
    sigmaSpace=bilateral_sigma_space,
)
st.image(cv.cvtColor(bilateral, cv.COLOR_BGR2RGB), "Bilateral Blur")

```



## Bitwise Operators

**Streamlit Demo:** [https://share.streamlit.io/cj-mills/opencv-notes/main/streamlit-demo-bitwise.py](https://share.streamlit.io/cj-mills/opencv-notes/main/streamlit-demo-bitwise.py)

```python
import cv2 as cv
import numpy as np

blank = np.zeros((400, 400), dtype="uint8")

rectangle = cv.rectangle(blank.copy(), (30, 30), (370, 370), 255, -1)
circle = cv.circle(blank.copy(), (200, 200), 200, 255, -1)

cv.imshow("Rectangle", rectangle)
cv.imshow("Circle", circle)

# AND Operator
bitwise_and = cv.bitwise_and(rectangle, circle)
cv.imshow("AND", bitwise_and)

# OR Operator
bitwise_or = cv.bitwise_or(rectangle, circle)
cv.imshow("OR", bitwise_or)

# XOR Operator
bitwise_xor = cv.bitwise_xor(rectangle, circle)
cv.imshow("XOR", bitwise_xor)

# NOT Operator
bitwise_not_rect = cv.bitwise_not(rectangle)
cv.imshow("NOT Rectangle", bitwise_not_rect)

# NOT Operator
bitwise_not_circle = cv.bitwise_not(circle)
cv.imshow("NOT Circle", bitwise_not_circle)

# Wait for key to be pressed before
# closing the above windows
cv.waitKey(0)
```



## Masking

**Streamlit Demo:** [https://share.streamlit.io/cj-mills/opencv-notes/main/streamlit-demo-masking.py](https://share.streamlit.io/cj-mills/opencv-notes/main/streamlit-demo-masking.py)

```python
import streamlit as st
import numpy as np
import cv2 as cv


st.title("Masking")

st.header("Input Image")
img_bgr = cv.imread("images/flower.jpg")
st.image(cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB), caption="Input")


blank = np.zeros(img_bgr.shape[:2], dtype="uint8")

rect_col1, rect_col2 = st.columns([1, 1])
rect_col1.subheader("Rectangle Origin")
rect_origin_x = rect_col1.slider(
    "Rect X:", min_value=0, max_value=img_bgr.shape[1], value=30
)
rect_origin_y = rect_col1.slider(
    "Rect Y:", min_value=0, max_value=img_bgr.shape[0], value=30
)

rect_col2.subheader("Rectangle Dimensions")
rect_width = rect_col2.slider(
    "Rect Width:", min_value=0, max_value=img_bgr.shape[1], value=img_bgr.shape[0] // 2
)
rect_height = rect_col2.slider(
    "Rect Height:", min_value=0, max_value=img_bgr.shape[0], value=img_bgr.shape[1] // 2
)
rectangle_mask = cv.rectangle(
    blank.copy(),
    (rect_origin_x, rect_origin_y),
    (rect_origin_x + rect_width, rect_origin_y + rect_height),
    255,
    -1,
)
st.image(cv.cvtColor(rectangle_mask, cv.COLOR_BGR2RGB), "Rectangle Mask")


circle_col1, circle_col2 = st.columns([1, 1])
circle_col1.subheader("Circle Origin")
circle_origin_x = circle_col1.slider(
    "Circle X:", min_value=0, max_value=img_bgr.shape[1], value=img_bgr.shape[1] // 2
)
circle_origin_y = circle_col1.slider(
    "Circle Y:", min_value=0, max_value=img_bgr.shape[0], value=img_bgr.shape[0] // 2
)

circle_col2.subheader("Circle Dimensions")
radius = circle_col2.slider(
    "Radius:", min_value=0, max_value=img_bgr.shape[0], value=100
)
circle_mask = cv.circle(
    blank.copy(), (circle_origin_x, circle_origin_y), radius, 100, -1
)
st.image(cv.cvtColor(circle_mask, cv.COLOR_BGR2RGB), "Circle Mask")

dual_mask = rectangle_mask + circle_mask
st.image(cv.cvtColor(dual_mask, cv.COLOR_BGR2RGB), "Dual Mask")

# AND Operator
bitwise_and = cv.bitwise_and(img_bgr, img_bgr, mask=dual_mask)
st.image(cv.cvtColor(bitwise_and, cv.COLOR_BGR2RGB), "AND")

# OR Operator
bitwise_or = cv.bitwise_or(img_bgr, img_bgr, mask=dual_mask)
st.image(cv.cvtColor(bitwise_or, cv.COLOR_BGR2RGB), "OR")

# XOR Operator
bitwise_xor = cv.bitwise_xor(img_bgr, img_bgr, mask=dual_mask)
st.image(cv.cvtColor(bitwise_xor, cv.COLOR_BGR2RGB), "XOR")

# NOT Operator
bitwise_not = cv.bitwise_not(img_bgr, mask=dual_mask)
st.image(cv.cvtColor(bitwise_not, cv.COLOR_BGR2RGB), "NOT Dual")

```

## Computing Histograms

- histograms allow you to visualize the distribution of pixel intensity

**Streamlit Demo:** [https://share.streamlit.io/cj-mills/opencv-notes/main/streamlit-demo-histograms.py](https://share.streamlit.io/cj-mills/opencv-notes/main/streamlit-demo-histograms.py)

```python
import streamlit as st
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


st.title("Computing Histograms")

st.header("Input Image")
img_bgr = cv.imread("images/flower_2.jpg")
st.image(cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB), caption="Input")


st.header("Grayscale")
gray = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)
st.image(gray, caption="Grayscale")

gray_hist = cv.calcHist(
    [gray], channels=[0], mask=None, histSize=[256], ranges=[0, 256]
)

fig, ax = plt.subplots()
plt.title("Grayscale Histogram")
plt.xlabel("Bins")
plt.ylabel("# of pixels")
plt.plot(gray_hist)
plt.xlim([0, 256])
ax.hist(gray_hist, bins=20)
st.pyplot(fig)


st.header("Color")
colors = ("b", "g", "r")

fig, ax = plt.subplots()
plt.title("Color Histogram")
plt.xlabel("Bins")
plt.ylabel("# of pixels")
plt.xlim([0, 256])
for i, col in enumerate(colors):
    hist = cv.calcHist(
        [img_bgr], channels=[i], mask=None, histSize=[256], ranges=[0, 256]
    )
    plt.plot(hist, color=col)
st.pyplot(fig)

```



## Thresholding Images

**Streamlit Demo:** [https://share.streamlit.io/cj-mills/opencv-notes/main/streamlit-demo-thresholding.py](https://share.streamlit.io/cj-mills/opencv-notes/main/streamlit-demo-thresholding.py)

```python
import streamlit as st
import numpy as np
import cv2 as cv


st.title("Thresholding Images")

st.header("Input Image")
img_bgr = cv.imread("images/flower_2.jpg")
st.image(cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB), caption="Input")

st.header("Simple Thresholding")
simple_thresh = st.slider(
    "Simple Threshold", min_value=0, max_value=255, value=100, step=1
)
threshold, thresh = cv.threshold(
    img_bgr, thresh=simple_thresh, maxval=255, type=cv.THRESH_BINARY
)
st.image(cv.cvtColor(thresh, cv.COLOR_BGR2RGB), "Simple Threshold")

st.header("Inverse Thresholding")
inv_thresh = st.slider(
    "Inverse Threshold", min_value=0, max_value=255, value=100, step=1
)
threshold, thresh_inv = cv.threshold(
    img_bgr, thresh=inv_thresh, maxval=255, type=cv.THRESH_BINARY_INV
)
st.image(cv.cvtColor(thresh_inv, cv.COLOR_BGR2RGB), "Threshold Inverse")


st.header("Grayscale")
gray = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)
st.image(gray, caption="Grayscale")

st.header("Adaptive Threshold")
blocksize = st.slider("Blocksize", min_value=3, max_value=33, value=11, step=2)
adaptive_thresh = cv.adaptiveThreshold(
    gray,
    maxValue=255,
    adaptiveMethod=cv.ADAPTIVE_THRESH_MEAN_C,
    thresholdType=cv.THRESH_BINARY,
    blockSize=blocksize,
    C=3,
)
st.image(adaptive_thresh, "Adaptive Thresholding")

```



## Edge Detection

**Streamlit Demo:** [https://share.streamlit.io/cj-mills/opencv-notes/main/streamlit-demo-edge-detection.py](https://share.streamlit.io/cj-mills/opencv-notes/main/streamlit-demo-edge-detection.py)

```python
import streamlit as st
import numpy as np
import cv2 as cv


st.title("Edge Detection")

st.header("Input Image")
img_bgr = cv.imread("images/flower_2.jpg")
st.image(cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB), caption="Input")

st.header("Grayscale")
gray = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)
st.image(gray, caption="Grayscale")

st.header("Laplacian")
lap = cv.Laplacian(src=gray, ddepth=cv.CV_64F)
lap = np.uint8(np.absolute(lap))
st.image(lap, "Laplacian")

st.header("Sobel")
sobelx = cv.Sobel(src=gray, ddepth=cv.CV_64F, dx=1, dy=0)
sobely = cv.Sobel(src=gray, ddepth=cv.CV_64F, dx=0, dy=1)

st.image(sobelx, "Sobel X", clamp=True)
st.image(sobely, "Sobel Y", clamp=True)
st.image(cv.bitwise_or(sobelx, sobely), "Sobel Combined")

st.header("Canny")
threshold_1 = st.slider("Threshold 1", min_value=1, max_value=500, value=125, step=1)
threshold_2 = st.slider("Threshold 2", min_value=1, max_value=500, value=175, step=1)
canny = cv.Canny(img_bgr, threshold_1, threshold_2)
st.image(cv.cvtColor(canny, cv.COLOR_BGR2RGB), caption="Canny")

```






## Face Detection

### Face Detection with Haar Cascades

- [OpenCV GitHub Haar Cascades](https://github.com/opencv/opencv/tree/4.x/data/haarcascades)
- [Documentation](https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html)
- Lower `minNeighbors` value increases false detections, while higher increases missed detections
- [dlib](http://dlib.net/) would be more appropriate for production

```python
import cv2 as cv

img = cv.imread("faces-1.jpg")
cv.imshow("Face", img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow("Gray", gray)

haar_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

print(f"Number of faces found: {len(faces_rect)}")
for (x,y,w,h) in faces_rect:
    cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), thickness=2)

cv.imshow('Detected Faces', img)

cv.waitKey(0)
```

### Face Detection with OpenCV’s Built-in Recognizer

- [Dataset](https://www.kaggle.com/dansbecker/5-celebrity-faces-dataset)
- Not the best detector

```python
import numpy as np
import cv2 as cv
import os

DIR = "./faces/val"
haar_cascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")
people = []

for i in os.listdir(DIR):
    people.append(i)

print(f"Face Classes: {people}")

features = np.load("features.npy", allow_pickle=True)
labels = np.load("labels.npy")

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read("face_trained.yml")

img = cv.imread(
    f"{DIR}/ben_afflek/httpafilesbiographycomimageuploadcfillcssrgbdprgfacehqwMTENDgMDUODczNDcNTcjpg.jpg"
)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow("Person", gray)

# Detect
faces_rect = haar_cascade.detectMultiScale(gray, 1.1, 4)

for (x, y, w, h) in faces_rect:
    faces_roi = gray[y : y + h, x : x + h]

    label, confidence = face_recognizer.predict(faces_roi)
    print(f"{label} with a confidence of {confidence}")

    cv.putText(img, str(people[label]), (10,30), cv.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), thickness=2)
    cv.rectangle(img, (x,y), (x+w, y+h), (0,255,0), thickness=2)
    cv.imshow('Detected Face', img)

cv.waitKey(0)
```








**References:**

* [OpenCV Course - Full Tutorial with Python](https://www.youtube.com/watch?v=oXlwWbU8l2o)

* [Streamlit Digital Book](https://surendraredd.github.io/Books/mlapps.html)
