---
title: "Real-Time Object Tracking with YOLOX and ByteTrack"
date: 2023-10-27
image: /images/empty.gif
hide: false
search_exclude: false
categories: [onnx, object-detection, object-tracking, yolox, byte-track, tutorial]
description: "Learn how to track objects across video frames with YOLOX and ByteTrack."

twitter-card:
  creator: "@cdotjdotmills"
  site: "@cdotjdotmills"
  image: /images/default-preview-image-black.png
open-graph:
  image: /images/default-preview-image-black.png
---

::: {.callout-tip}
## This post is part of the following series:
* [**Training YOLOX Models for Real-Time Object Detection in PyTorch**](/series/tutorials/pytorch-train-object-detector-yolox-series.html)
:::


* [Introduction](#introduction)
* [Getting Started with the Code](#getting-started-with-the-code)
* [Setting Up Your Python Environment](#setting-up-your-python-environment)
* [Importing the Required Dependencies](#importing-the-required-dependencies)
* [Setting Up the Project](#setting-up-the-project)
* [Loading the Checkpoint Data](#loading-the-checkpoint-data)
* [Defining Utility Functions](#defining-utility-functions)
* [Tracking Objects in Videos](#tracking-objects-in-videos)
* [Conclusion](#conclusion)


## Introduction

Welcome back to this series on real-time object detection with YOLOX! Previously, we [fine-tuned a YOLOX](../) model in PyTorch to detect hand signs and [exported it to ONNX](../onnx-export). This tutorial combines our YOLOX model with the [ByteTrack](https://arxiv.org/abs/2110.06864) object tracker to track objects continuously across video frames.

Tracking objects over time unlocks a wide range of potential applications. With our hand-sign detector, we could implement gesture-based controls to control devices and create interactive gaming and multimedia experiences. Beyond our specific model, object tracking has applications in everything from sports analysis to wildlife monitoring.

By the end of this tutorial, you will understand how to combine a YOLOX object detection model with ByteTrack, enabling you to effectively track hand signs or other objects across consecutive video frames.



::: {.callout-important title="This post assumes the reader has completed the previous tutorial linked below:"}
* [Exporting YOLOX Models from PyTorch to ONNX](../onnx-export/)
:::





## Getting Started with the Code

As with the previous tutorial, the code is available as a Jupyter Notebook.

| Jupyter Notebook                                             | Google Colab                                                 |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [GitHub Repository](https://github.com/cj-mills/pytorch-yolox-object-detection-tutorial-code/blob/main/notebooks/pytorch-yolox-object-tracking-onnx-byte-track.ipynb) | [Open In Colab](https://colab.research.google.com/github/cj-mills/pytorch-yolox-object-detection-tutorial-code/blob/main/notebooks/pytorch-yolox-object-tracking-onnx-byte-track-colab.ipynb) |





## Setting Up Your Python Environment

We need to add a couple of new libraries to our [Python environment](../#setting-up-your-python-environment). We will use [OpenCV](https://opencv.org/) to read and write video files. I also made a package with a standalone implementation of ByteTrack. Make sure to install `onnx` and `onnxruntime` if you did not follow the previous tutorial.

::: {.callout-note title="Package Descriptions" collapse="true"}

| Package          | Description                                                  |
| ---------------- | ------------------------------------------------------------ |
| `onnx`           | This package provides a Python API for working with ONNX models. ([link](https://pypi.org/project/onnx/)) |
| `onnxruntime`    | ONNX Runtime is a runtime accelerator for machine learning models. ([link](https://onnxruntime.ai/)) |
| `opencv-python`  | Wrapper package for [OpenCV](https://opencv.org/) python bindings. ([link]()) |
| `cjm-byte-track` | A standalone Python implementation of the [ByteTrack](https://arxiv.org/abs/2110.06864) multi-object tracker based on the official implementation. ([link](https://github.com/cj-mills/cjm-byte-track)) |



:::

Run the following command to install these additional libraries:

```bash
# Install packages
pip install onnx onnxruntime opencv-python cjm_byte_track
```





## Importing the Required Dependencies

With our environment updated, we can dive into the code. First, we will import the necessary Python dependencies into our Jupyter Notebook.


```python
# Import Python Standard Library dependencies
from dataclasses import dataclass
import json
from pathlib import Path
import random
import time
from typing import List

# Import ByteTrack package
from cjm_byte_track.core import BYTETracker
from cjm_byte_track.matching import match_detections_with_tracks

# Import utility functions
from cjm_psl_utils.core import download_file
from cjm_pil_utils.core import resize_img

# Import OpenCV
import cv2

# Import numpy
import numpy as np

# Import the pandas package
import pandas as pd

# Import PIL for image manipulation
from PIL import Image, ImageDraw, ImageFont

# Import ONNX dependencies
import onnx # Import the onnx module
import onnxruntime as ort # Import the ONNX Runtime

# Import tqdm for progress bar
from tqdm.auto import tqdm
```



## Setting Up the Project

In this section, we will set the folder locations for our project and the directory with the ONNX model and JSON colormap file. We should also ensure we have a font file for annotating images.

### Set the Directory Paths


```python
# The name for the project
project_name = f"pytorch-yolox-object-detector"

# The path for the project folder
project_dir = Path(f"./{project_name}/")

# Create the project directory if it does not already exist
project_dir.mkdir(parents=True, exist_ok=True)

# The path to the checkpoint folder
checkpoint_dir = Path(project_dir/f"2023-08-17_16-14-43")
# checkpoint_dir = Path(project_dir/f"pretrained-coco")

pd.Series({
    "Project Directory:": project_dir,
    "Checkpoint Directory:": checkpoint_dir,
}).to_frame().style.hide(axis='columns')
```

<div style="overflow-x:auto; max-height:500px">
<table id="T_125b0">
  <thead>
  </thead>
  <tbody>
    <tr>
      <th id="T_125b0_level0_row0" class="row_heading level0 row0" >Project Directory:</th>
      <td id="T_125b0_row0_col0" class="data row0 col0" >pytorch-yolox-object-detector</td>
    </tr>
    <tr>
      <th id="T_125b0_level0_row1" class="row_heading level0 row1" >Checkpoint Directory:</th>
      <td id="T_125b0_row1_col0" class="data row1 col0" >pytorch-yolox-object-detector/2023-08-17_16-14-43</td>
    </tr>
  </tbody>
</table>
</div>



::: {.callout-tip title="I made an ONNX model  available on Hugging Face Hub with a colormap file in the repository linked below:"}
* [cj-mills/yolox-hagrid-onnx](https://huggingface.co/cj-mills/yolox-hagrid-onnx/tree/main)
:::


::: {.callout-tip title="Those following along on Google Colab can drag the contents of their checkpoint folder into Colab's file browser. "}
:::


### Download a Font File


```python
# Set the name of the font file
font_file = 'KFOlCnqEu92Fr1MmEU9vAw.ttf'

# Download the font file
download_file(f"https://fonts.gstatic.com/s/roboto/v30/{font_file}", "./")
```





## Loading the Checkpoint Data

Now, we can load the colormap and set the max stride value and input dimension slice.

### Load the Colormap


```python
# The colormap path
colormap_path = list(checkpoint_dir.glob('*colormap.json'))[0]

# Load the JSON colormap data
with open(colormap_path, 'r') as file:
        colormap_json = json.load(file)

# Convert the JSON data to a dictionary        
colormap_dict = {item['label']: item['color'] for item in colormap_json['items']}

# Extract the class names from the colormap
class_names = list(colormap_dict.keys())

# Make a copy of the colormap in integer format
int_colors = [tuple(int(c*255) for c in color) for color in colormap_dict.values()]
```



### Set the Preprocessing and Post-Processing Parameters


```python
max_stride = 32
input_dim_slice = slice(2, 4, None)
```



## Defining Utility Functions

Next, we will define some utility functions for preparing the input data and processing the model output.

### Define a Function to Prepare Images for Inference

OpenCV uses the BGR (Blue, Green, Red) color format for images, so we must change the current video frame to RGB before performing the [standard preprocessing steps](../#preparing-input-data) for the YOLOX model.


```python
def prepare_image_for_inference(frame:np.ndarray, target_sz:int, max_stride:int):

    """
    Prepares an image for inference by performing a series of preprocessing steps.
    
    Steps:
    1. Converts a BGR image to RGB.
    2. Resizes the image to a target size without cropping, considering a given divisor.
    3. Calculates input dimensions as multiples of the max stride.
    4. Calculates offsets based on the resized image dimensions and input dimensions.
    5. Computes the scale between the original and resized image.
    6. Crops the resized image based on calculated input dimensions.
    
    Parameters:
    - frame (numpy.ndarray): The input image in BGR format.
    - target_sz (int): The target minimum size for resizing the image.
    - max_stride (int): The maximum stride to be considered for calculating input dimensions.
    
    Returns:
    tuple: 
    - rgb_img (PIL.Image): The converted RGB image.
    - input_dims (list of int): Dimensions of the image that are multiples of max_stride.
    - offsets (numpy.ndarray): Offsets from the resized image dimensions to the input dimensions.
    - min_img_scale (float): Scale factor between the original and resized image.
    - input_img (PIL.Image): Cropped image based on the calculated input dimensions.
    """

    # Convert the BGR image to RGB
    rgb_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    # Resize image without cropping to multiple of the max stride
    resized_img = resize_img(rgb_img, target_sz=target_sz, divisor=1)
    
    # Calculating the input dimensions that multiples of the max stride
    input_dims = [dim - dim % max_stride for dim in resized_img.size]
    # Calculate the offsets from the resized image dimensions to the input dimensions
    offsets = (np.array(resized_img.size) - input_dims) / 2
    # Calculate the scale between the source image and the resized image
    min_img_scale = min(rgb_img.size) / min(resized_img.size)
    
    # Crop the resized image to the input dimensions
    input_img = resized_img.crop(box=[*offsets, *resized_img.size - offsets])
    
    return rgb_img, input_dims, offsets, min_img_scale, input_img
```

### Define Functions to Process YOLOX Output

We can use the same [utility functions](../onnx-export/#define-utility-functions) defined in the previous tutorial on exporting the model to ONNX.

#### Define a function to generate the output grids


```python
def generate_output_grids_np(height, width, strides=[8,16,32]):
    """
    Generate a numpy array containing grid coordinates and strides for a given height and width.

    Args:
        height (int): The height of the image.
        width (int): The width of the image.

    Returns:
        np.ndarray: A numpy array containing grid coordinates and strides.
    """

    all_coordinates = []

    for stride in strides:
        # Calculate the grid height and width
        grid_height = height // stride
        grid_width = width // stride

        # Generate grid coordinates
        g1, g0 = np.meshgrid(np.arange(grid_height), np.arange(grid_width), indexing='ij')

        # Create an array of strides
        s = np.full((grid_height, grid_width), stride)

        # Stack the coordinates along with the stride
        coordinates = np.stack((g0.flatten(), g1.flatten(), s.flatten()), axis=-1)

        # Append to the list
        all_coordinates.append(coordinates)

    # Concatenate all arrays in the list along the first dimension
    output_grids = np.concatenate(all_coordinates, axis=0)

    return output_grids
```

#### Define a function to calculate bounding boxes and probabilities


```python
def calculate_boxes_and_probs(model_output:np.ndarray, output_grids:np.ndarray) -> np.ndarray:
    """
    Calculate the bounding boxes and their probabilities.

    Parameters:
    model_output (numpy.ndarray): The output of the model.
    output_grids (numpy.ndarray): The output grids.

    Returns:
    numpy.ndarray: The array containing the bounding box coordinates, class labels, and maximum probabilities.
    """
    # Calculate the bounding box coordinates
    box_centroids = (model_output[..., :2] + output_grids[..., :2]) * output_grids[..., 2:]
    box_sizes = np.exp(model_output[..., 2:4]) * output_grids[..., 2:]

    x0, y0 = [t.squeeze(axis=2) for t in np.split(box_centroids - box_sizes / 2, 2, axis=2)]
    w, h = [t.squeeze(axis=2) for t in np.split(box_sizes, 2, axis=2)]

    # Calculate the probabilities for each class
    box_objectness = model_output[..., 4]
    box_cls_scores = model_output[..., 5:]
    box_probs = np.expand_dims(box_objectness, -1) * box_cls_scores

    # Get the maximum probability and corresponding class for each proposal
    max_probs = np.max(box_probs, axis=-1)
    labels = np.argmax(box_probs, axis=-1)

    return np.array([x0, y0, w, h, labels, max_probs]).transpose((1, 2, 0))
```

#### Define a function to extract object proposals from the raw model output


```python
def process_outputs(outputs:np.ndarray, input_dims:tuple, bbox_conf_thresh:float):

    """
    Process the model outputs to generate bounding box proposals filtered by confidence threshold.
    
    Parameters:
    - outputs (numpy.ndarray): The raw output from the model, which will be processed to calculate boxes and probabilities.
    - input_dims (tuple of int): Dimensions (height, width) of the input image to the model.
    - bbox_conf_thresh (float): Threshold for the bounding box confidence/probability. Bounding boxes with a confidence
                                score below this threshold will be discarded.
    
    Returns:
    - numpy.array: An array of proposals where each proposal is an array containing bounding box coordinates
                   and its associated probability, sorted in descending order by probability.
    """

    # Process the model output
    outputs = calculate_boxes_and_probs(outputs, generate_output_grids_np(*input_dims))
    # Filter the proposals based on the confidence threshold
    max_probs = outputs[:, :, -1]
    mask = max_probs > bbox_conf_thresh
    proposals = outputs[mask]
    # Sort the proposals by probability in descending order
    proposals = proposals[proposals[..., -1].argsort()][::-1]
    return proposals
```

#### Define a function to calculate the intersection-over-union


```python
def calc_iou(proposals:np.ndarray) -> np.ndarray:
    """
    Calculates the Intersection over Union (IoU) for all pairs of bounding boxes (x,y,w,h) in 'proposals'.

    The IoU is a measure of overlap between two bounding boxes. It is calculated as the area of
    intersection divided by the area of union of the two boxes.

    Parameters:
    proposals (2D np.array): A NumPy array of bounding boxes, where each box is an array [x, y, width, height].

    Returns:
    iou (2D np.array): The IoU matrix where each element i,j represents the IoU of boxes i and j.
    """

    # Calculate coordinates for the intersection rectangles
    x1 = np.maximum(proposals[:, 0], proposals[:, 0][:, None])
    y1 = np.maximum(proposals[:, 1], proposals[:, 1][:, None])
    x2 = np.minimum(proposals[:, 0] + proposals[:, 2], (proposals[:, 0] + proposals[:, 2])[:, None])
    y2 = np.minimum(proposals[:, 1] + proposals[:, 3], (proposals[:, 1] + proposals[:, 3])[:, None])
    
    # Calculate intersection areas
    intersections = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)

    # Calculate union areas
    areas = proposals[:, 2] * proposals[:, 3]
    unions = areas[:, None] + areas - intersections

    # Calculate IoUs
    iou = intersections / unions

    # Return the iou matrix
    return iou
```

#### Define a function to filter bounding box proposals using Non-Maximum Suppression


```python
def nms_sorted_boxes(iou:np.ndarray, iou_thresh:float=0.45) -> np.ndarray:
    """
    Applies non-maximum suppression (NMS) to sorted bounding boxes.

    It suppresses boxes that have high overlap (as defined by the IoU threshold) with a box that 
    has a higher score.

    Parameters:
    iou (np.ndarray): An IoU matrix where each element i,j represents the IoU of boxes i and j.
    iou_thresh (float): The IoU threshold for suppression. Boxes with IoU > iou_thresh are suppressed.

    Returns:
    keep (np.ndarray): The indices of the boxes to keep after applying NMS.
    """

    # Create a boolean mask to keep track of boxes
    mask = np.ones(iou.shape[0], dtype=bool)

    # Apply non-max suppression
    for i in range(iou.shape[0]):
        if mask[i]:
            # Suppress boxes with higher index and IoU > threshold
            mask[(iou[i] > iou_thresh) & (np.arange(iou.shape[0]) > i)] = False

    # Return the indices of the boxes to keep
    return np.arange(iou.shape[0])[mask]
```

### Define a Function to Annotate Images with Bounding Boxes

Likewise, we can use the same [function for annotating images](../onnx-export/#define-a-function-to-annotate-an-image-with-bounding-boxes) with bounding boxes with PIL.


```python
def draw_bboxes_pil(image, boxes, labels, colors, font, width=2, font_size=18, probs=None):
    """
    Annotates an image with bounding boxes, labels, and optional probability scores.

    Parameters:
    - image (PIL.Image): The input image on which annotations will be drawn.
    - boxes (list of tuples): A list of bounding box coordinates where each tuple is (x, y, w, h).
    - labels (list of str): A list of labels corresponding to each bounding box.
    - colors (list of str): A list of colors for each bounding box and its corresponding label.
    - font (str): Path to the font file to be used for displaying the labels.
    - width (int, optional): Width of the bounding box lines. Defaults to 2.
    - font_size (int, optional): Size of the font for the labels. Defaults to 18.
    - probs (list of float, optional): A list of probability scores corresponding to each label. Defaults to None.

    Returns:
    - annotated_image (PIL.Image): The image annotated with bounding boxes, labels, and optional probability scores.
    """
    
    # Define a reference diagonal
    REFERENCE_DIAGONAL = 1000
    
    # Scale the font size using the hypotenuse of the image
    font_size = int(font_size * (np.hypot(*image.size) / REFERENCE_DIAGONAL))
    
    # Add probability scores to labels if provided
    if probs is not None:
        labels = [f"{label}: {prob*100:.2f}%" for label, prob in zip(labels, probs)]

    # Create an ImageDraw object for drawing on the image
    draw = ImageDraw.Draw(image)

    # Load the font file (outside the loop)
    fnt = ImageFont.truetype(font, font_size)
    
    # Compute the mean color value for each color
    mean_colors = [np.mean(np.array(color)) for color in colors]

    # Loop through the bounding boxes, labels, and colors
    for box, label, color, mean_color in zip(boxes, labels, colors, mean_colors):
        # Get the bounding box coordinates
        x, y, w, h = box

        # Draw the bounding box on the image
        draw.rectangle([x, y, x+w, y+h], outline=color, width=width)
        
        # Get the size of the label text box
        label_w, label_h = draw.textbbox(xy=(0,0), text=label, font=fnt)[2:]
        
        # Draw the label rectangle on the image
        draw.rectangle([x, y-label_h, x+label_w, y], outline=color, fill=color)

        # Draw the label text on the image
        font_color = 'black' if mean_color > 127.5 else 'white'
        draw.multiline_text((x, y-label_h), label, font=fnt, fill=font_color)
        
    return image
```

That takes care of the required utility functions. In the next section, we will use our ONNX model with ByteTrack to track objects in a video.



## Tracking Objects in Videos

We will first initialize an inference session with our ONNX model.

### Create an Inference Session


```python
# Get a filename for the ONNX model
onnx_file_path = list(checkpoint_dir.glob('*.onnx'))[0]
```


```python
# Load the model and create an InferenceSession
providers = [
    'CPUExecutionProvider',
    # "CUDAExecutionProvider",
]
sess_options = ort.SessionOptions()
session = ort.InferenceSession(onnx_file_path, sess_options=sess_options, providers=providers)
```



### Select a Test Video

Next, we need a video to test the object tracking performance. We can use this one from [Pexels](https://www.pexels.com/video/a-woman-giving-a-thumbs-up-10373924/), a free stock photo & video site.


```python
# Specify the directory where videos are or will be stored.
video_dir = "./videos/"

# Name of the test video to be used.
test_video_name = "pexels-rodnae-productions-10373924.mp4"

# Construct the full path for the video using the directory and video name.
video_path = f"{video_dir}{test_video_name}"

# Define the URL for the test video stored on Huggingface's server.
test_video_url = f"https://huggingface.co/datasets/cj-mills/pexels-object-tracking-test-videos/resolve/main/{test_video_name}"

# Download the video file from the specified URL to the local video directory.
download_file(test_video_url, video_dir, False)

# Display the video using the Video function (assuming an appropriate library/module is imported).
Video(video_path)
```

![](./video/pexels-rodnae-productions-10373924.mp4){fig-align="center"}



### Initialize a `VideoCapture` Object

Now that we have a test video, we can use OpenCV's [`VideoCapture`](https://docs.opencv.org/4.x/d8/dfe/classcv_1_1VideoCapture.html) class to iterate through it and access relevant metadata.


```python
# Open the video file located at 'video_path' using OpenCV
video_capture = cv2.VideoCapture(video_path)

# Retrieve the frame width of the video
frame_width = int(video_capture.get(3))
# Retrieve the frame height of the video
frame_height = int(video_capture.get(4))
# Retrieve the frames per second (FPS) of the video
frame_fps = int(video_capture.get(5))
# Retrieve the total number of frames in the video
frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

# Create a pandas Series containing video metadata and convert it to a DataFrame
pd.Series({
    "Frame Width:": frame_width,
    "Frame Height:": frame_height,
    "Frame FPS:": frame_fps,
    "Frames:": frames
}).to_frame().style.hide(axis='columns')
```

<div style="overflow-x:auto; max-height:500px">
<table id="T_b9b06">
  <thead>
  </thead>
  <tbody>
    <tr>
      <th id="T_b9b06_level0_row0" class="row_heading level0 row0" >Frame Width:</th>
      <td id="T_b9b06_row0_col0" class="data row0 col0" >720</td>
    </tr>
    <tr>
      <th id="T_b9b06_level0_row1" class="row_heading level0 row1" >Frame Height:</th>
      <td id="T_b9b06_row1_col0" class="data row1 col0" >1280</td>
    </tr>
    <tr>
      <th id="T_b9b06_level0_row2" class="row_heading level0 row2" >Frame FPS:</th>
      <td id="T_b9b06_row2_col0" class="data row2 col0" >29</td>
    </tr>
    <tr>
      <th id="T_b9b06_level0_row3" class="row_heading level0 row3" >Frames:</th>
      <td id="T_b9b06_row3_col0" class="data row3 col0" >226</td>
    </tr>
  </tbody>
</table>
</div>



### Initialize a `VideoWriter` Object

We will use OpenCV's [`VideoWriter`](https://docs.opencv.org/3.4/dd/d9e/classcv_1_1VideoWriter.html) class to save the annotated version of our test video.


```python
# Construct the output video path 
video_out_path = f"{(video_dir)}{Path(video_path).stem}-byte-track.mp4"

# Initialize a VideoWriter object for video writing.
# 1. video_out_path: Specifies the name of the output video file.
# 2. cv2.VideoWriter_fourcc(*'mp4v'): Specifies the codec for the output video. 'mp4v' is used for .mp4 format.
# 3. frame_fps: Specifies the frames per second for the output video.
# 4. (frame_width, frame_height): Specifies the width and height of the frames in the output video.
video_writer = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc(*'mp4v'), frame_fps, (frame_width, frame_height))
```



### Define Inference Parameters


```python
test_sz = 288
bbox_conf_thresh = 0.1
iou_thresh = 0.45
```



### Detect, Track, and Annotate Objects in Video Frames

In this section, we'll iterate through each frame of our test video, detect objects, track those objects across video frames, and then annotate the video frames with the corresponding bounding boxes and tracking IDs.

We start by initializing a ByteTracker object. Then, we can iterate over the video frames using our VideoCapture object. For each frame, we pass it through the `prepare_image_for_inference` function to apply the preprocessing steps. We then convert the image to a NumPy array and scale the values to the range `[0,1]`. 

Next, we pass the input to our YOLOX model to get the raw prediction data. After that, we can process the raw output to extract the bounding box predictions. We then pass the bounding box predictions to the ByteTracker object so it can update its current object tracks. 

Once we match the updated track data with the current bounding box predictions, we can annotate the current frame with bounding boxes and associated track IDs.


```python
# Initialize a ByteTracker object
tracker = BYTETracker(track_thresh=0.25, track_buffer=30, match_thresh=0.8, frame_rate=frame_fps)

with tqdm(total=frames, desc="Processing frames") as pbar:
    # Iterate through each frame in the video
    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if ret:
            
            start_time = time.perf_counter()
        
            # Prepare the input image for inference
            rgb_img, input_dims, offsets, min_img_scale, input_img = prepare_image_for_inference(frame, test_sz, max_stride)
            
            # Convert the input image to NumPy format for the model
            input_tensor_np = np.array(input_img, dtype=np.float32).transpose((2, 0, 1))[None]/255
                            
            # Run inference using the ONNX session
            outputs = session.run(None, {"input": input_tensor_np})[0]
        
            # Process the model output to get object proposals
            proposals = process_outputs(outputs, input_tensor_np.shape[input_dim_slice], bbox_conf_thresh)
            
            # Apply non-max suppression to filter overlapping proposals
            proposal_indices = nms_sorted_boxes(calc_iou(proposals[:, :-2]), iou_thresh)
            proposals = proposals[proposal_indices]
            
            # Extract bounding boxes, labels, and probabilities from proposals
            bbox_list = (proposals[:,:4]+[*offsets, 0, 0])*min_img_scale
            label_list = [class_names[int(idx)] for idx in proposals[:,4]]
            probs_list = proposals[:,5]
    
            # Initialize track IDs for detected objects
            track_ids = [-1]*len(bbox_list)
    
            # Convert bounding boxes to top-left bottom-right (tlbr) format
            tlbr_boxes = bbox_list.copy()
            tlbr_boxes[:, 2:4] += tlbr_boxes[:, :2]
    
            # Update tracker with detections
            tracks = tracker.update(
                output_results=np.concatenate([tlbr_boxes, probs_list[:, np.newaxis]], axis=1),
                img_info=rgb_img.size,
                img_size=rgb_img.size)
    
            if len(tlbr_boxes) > 0 and len(tracks) > 0:
                # Match detections with tracks
                track_ids = match_detections_with_tracks(tlbr_boxes=tlbr_boxes, track_ids=track_ids, tracks=tracks)
        
                # Filter object detections based on tracking results
                bbox_list, label_list, probs_list, track_ids = zip(*[(bbox, label, prob, track_id) 
                                                                    for bbox, label, prob, track_id 
                                                                    in zip(bbox_list, label_list, probs_list, track_ids) if track_id != -1])
                
                if len(bbox_list) > 0:
                    # Annotate the current frame with bounding boxes and tracking IDs
                    annotated_img = draw_bboxes_pil(
                        image=rgb_img, 
                        boxes=bbox_list, 
                        labels=[f"{track_id}-{label}" for track_id, label in zip(track_ids, label_list)],
                        probs=probs_list,
                        colors=[int_colors[class_names.index(i)] for i in label_list],  
                        font=font_file,
                    )
                    annotated_frame = cv2.cvtColor(np.array(annotated_img), cv2.COLOR_RGB2BGR)
            else:
                # If no detections, use the original frame
                annotated_frame = frame
            
            video_writer.write(annotated_frame)
            pbar.update(1)
        else:
            break
video_capture.release()
video_writer.release()
```



Finally, we can check the annotated video to see how the object tracker performed.

![](./video/pexels-rodnae-productions-10373924-byte-track.mp4){fig-align="center"}



The ByteTracker had no issue tracking the two hands throughout the video, as the track IDs remained the same for each hand.



::: {.callout-caution}
## Google Colab Users
1. Don't forget to download the the annotated video from the Colab Environment's file browser. ([tutorial link](https://christianjmills.com/posts/google-colab-getting-started-tutorial/#working-with-data)) 
:::








## Conclusion

Congratulations on reaching the end of this tutorial on object tracking with YOLOX and ByteTrack! With this knowledge, we have unlocked a new realm of potential applications for our YOLOX model.

Combining YOLOX's robust detection capabilities with ByteTrack's tracking efficiency gives you a powerful toolset to work on myriad projects, from video analysis to immersive augmented reality experiences.

As a follow-up project, consider integrating our hand sign detector with ByteTrack in an application for gesture-based controls or training a new YOLOX model for other domains. The potential applications of this powerful combination are vast, limited only by your imagination.





{{< include /_tutorial-cta.qmd >}}




{{< include /_about-author-cta.qmd >}}
