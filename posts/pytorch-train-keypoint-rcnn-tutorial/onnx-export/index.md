---
title: "Exporting Keypoint R-CNN Models from PyTorch to ONNX"
date: 2024-01-30
date-modified: last-modified
image: /images/empty.gif
hide: false
search_exclude: false
categories: [pytorch, onnx, keypoint-rcnn, keypoint-estimation, tutorial]
description: "Learn how to export Keypoint R-CNN models from PyTorch to ONNX and perform inference using ONNX Runtime."

twitter-card:
  creator: "@cdotjdotmills"
  site: "@cdotjdotmills"
  image: /images/default-preview-image-black.png
open-graph:
  image: /images/default-preview-image-black.png
---

::: {.callout-tip}
## This post is part of the following series:
* [**Training Keypoint R-CNN Models with PyTorch**](/series/tutorials/pytorch-train-keypoint-rcnn-series.html)
:::


* [Introduction](#introduction)
* [Getting Started with the Code](#getting-started-with-the-code)
* [Setting Up Your Python Environment](#setting-up-your-python-environment)
* [Importing the Required Dependencies](#importing-the-required-dependencies)
* [Setting Up the Project](#setting-up-the-project)
* [Loading the Checkpoint Data](#loading-the-checkpoint-data)
* [Exporting the Model to ONNX](#exporting-the-model-to-onnx)
* [Performing Inference with ONNX Runtime](#performing-inference-with-onnx-runtime)
* [Conclusion](#conclusion)


## Introduction

Welcome back to this series on training Keypoint R-CNN models with PyTorch. Previously, we demonstrated how to fine-tune a Keypoint R-CNN model by training it to identify the locations of human noses and faces. This tutorial builds on that by showing how to export the model to [ONNX](https://onnx.ai/) and perform inference using [ONNX Runtime](https://onnxruntime.ai/docs/). 

ONNX (Open Neural Network Exchange) is an open format to represent machine learning models and make them portable across various platforms. ONNX Runtime is a cross-platform inference accelerator that provides interfaces to hardware-specific libraries. By exporting our model to ONNX, we can deploy it to multiple devices and leverage hardware acceleration  for faster inference. The Keypoint R-CNN model is computationally intensive, so any improvements to inference speed are welcome.

Additionally, we'll implement the functionality to annotate images with key points without relying on PyTorch as a dependency. By the end of this tutorial, you will have an ONNX version of our Keypoint R-CNN model that you can deploy to servers and edge devices using ONNX Runtime.



::: {.callout-important title="This post assumes the reader has completed the previous tutorial linked below:"}
* [Training Keypoint R-CNN Models with PyTorch](../)
:::





## Getting Started with the Code

As with the previous tutorial, the code is available as a Jupyter Notebook.

| Jupyter Notebook                                             | Google Colab                                                 |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [GitHub Repository](https://github.com/cj-mills/pytorch-keypoint-rcnn-tutorial-code/blob/main/notebooks/pytorch-keypoint-r-cnn-onnx-export.ipynb) | [Open In Colab](https://colab.research.google.com/github/cj-mills/pytorch-keypoint-rcnn-tutorial-code/blob/main/notebooks/pytorch-keypoint-r-cnn-onnx-export-colab.ipynb) |





## Setting Up Your Python Environment

We'll need to add a few new libraries to our [Python environment](../#setting-up-your-python-environment) for working with ONNX models.

::: {.callout-note title="Package Descriptions" collapse="true"}



| Package           | Description                                                  |
| ----------------- | ------------------------------------------------------------ |
| `onnx`            | This package provides a Python API for working with ONNX models. ([link](https://pypi.org/project/onnx/)) |
| `onnxruntime`     | ONNX Runtime is a runtime accelerator for machine learning models. ([link](https://onnxruntime.ai/)) |
| `onnx-simplifier` | This package helps simplify ONNX models. ([link](https://pypi.org/project/onnx-simplifier/)) |



:::

Run the following command to install these additional libraries:

```bash
# Install ONNX packages
pip install onnx onnxruntime onnx-simplifier
```



## Importing the Required Dependencies

With our environment updated, we can dive into the code. First, we will  import the necessary Python dependencies into our Jupyter Notebook.


```python
# Import Python Standard Library dependencies
import json
from pathlib import Path
import random

# Import utility functions
from cjm_psl_utils.core import download_file
from cjm_pil_utils.core import resize_img

# Import numpy
import numpy as np

# Import the pandas package
import pandas as pd

# Import PIL for image manipulation
from PIL import Image, ImageDraw, ImageFont

# Import PyTorch dependencies
import torch

# Import Keypoint R-CNN
from torchvision.models.detection import keypointrcnn_resnet50_fpn
from torchvision.models.detection.keypoint_rcnn import KeypointRCNNPredictor

# Import ONNX dependencies
import onnx # Import the onnx module
from onnxsim import simplify # Import the method to simplify ONNX models
import onnxruntime as ort # Import the ONNX Runtime
```



## Setting Up the Project

In this section, we’ll set the folder locations for our project and  training session with the PyTorch checkpoint. Let’s also ensure we have a font file for annotating images.

### Set the Directory Paths


```python
# The name for the project
project_name = f"pytorch-keypoint-r-cnn"

# The path for the project folder
project_dir = Path(f"./{project_name}/")

# Create the project directory if it does not already exist
project_dir.mkdir(parents=True, exist_ok=True)

# The path to the checkpoint folder
checkpoint_dir = Path(project_dir/f"2024-01-30_10-44-52")

pd.Series({
    "Project Directory:": project_dir,
    "Checkpoint Directory:": checkpoint_dir,
}).to_frame().style.hide(axis='columns')
```

<div style="overflow-x:auto; max-height:500px">
<table id="T_d7624">
  <thead>
  </thead>
  <tbody>
    <tr>
      <th id="T_d7624_level0_row0" class="row_heading level0 row0" >Project Directory:</th>
      <td id="T_d7624_row0_col0" class="data row0 col0" >pytorch-keypoint-r-cnn</td>
    </tr>
    <tr>
      <th id="T_d7624_level0_row1" class="row_heading level0 row1" >Checkpoint Directory:</th>
      <td id="T_d7624_row1_col0" class="data row1 col0" >pytorch-keypoint-r-cnn/2024-01-30_10-44-52</td>
    </tr>
  </tbody>
</table>
</div>



::: {.callout-tip title="I made a model checkpoint available on Hugging Face Hub in the repository linked below:"}
* [cj-mills/keypoint-rcnn-eyes-noses-pytorch](https://huggingface.co/cj-mills/keypoint-rcnn-eyes-noses-pytorch/tree/main)
:::

::: {.callout-tip title="Those following along on Google Colab can drag the contents of their checkpoint folder into Colab's file browser. Keep in mind the model checkpoint has a large file size. "}
:::



### Download a Font File


```python
# Set the name of the font file
font_file = 'KFOlCnqEu92Fr1MmEU9vAw.ttf'

# Download the font file
download_file(f"https://fonts.gstatic.com/s/roboto/v30/{font_file}", "./")
```



## Loading the Checkpoint Data

Now, we can load the colormap used during training and initialize a Keypoint R-CNN model with the saved checkpoint.

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

### Load the Model Checkpoint


```python
# The model checkpoint path
checkpoint_path = list(checkpoint_dir.glob('*.pth'))[0]

# Load the model checkpoint onto the CPU
model_checkpoint = torch.load(checkpoint_path, map_location='cpu')
```

### Load the Trained Keypoint R-CNN Model


```python
# Load a pre-trained model
model = keypointrcnn_resnet50_fpn(weights='DEFAULT')

# Replace the classifier head with the number of keypoints
in_features = model.roi_heads.keypoint_predictor.kps_score_lowres.in_channels
model.roi_heads.keypoint_predictor = KeypointRCNNPredictor(in_channels=in_features, num_keypoints=len(class_names))

# Initialize the model with the checkpoint parameters and buffers
model.load_state_dict(model_checkpoint)
```

```text
<All keys matched successfully>
```





## Exporting the Model to ONNX

Before exporting the model, let’s ensure the model is in evaluation mode.


```python
model.eval();
```

### Prepare the Input Tensor

We need a sample input tensor for the export process.


```python
input_tensor = torch.randn(1, 3, 256, 256)
```

### Export the Model to ONNX

We can export the model using PyTorch’s [`torch.onnx.export()`](https://pytorch.org/docs/stable/onnx.html#torch.onnx.export) function. This function performs a single pass through the model and records all operations to generate a [TorchScript graph](https://pytorch.org/docs/stable/jit.html). It then exports this graph to ONNX by decomposing each graph node  (which contains a PyTorch operator) into a series of ONNX operators.

If we want the ONNX model to support different input sizes, we must set the width and height input axes as dynamic.


```python
# Set a filename for the ONNX model
onnx_file_path = f"{checkpoint_dir}/{colormap_path.stem.removesuffix('-colormap')}-{checkpoint_path.stem}.onnx"

# Export the PyTorch model to ONNX format
torch.onnx.export(model.cpu(),
                  input_tensor.cpu(),
                  onnx_file_path,
                  export_params=True,
                  do_constant_folding=False,
                  input_names = ['input'],
                  output_names = ['boxes', 'labels', 'scores', 'keypoints', 'keypoints_scores'],
                  dynamic_axes={'input': {2 : 'height', 3 : 'width'}}
                 )
```



::: {.callout-note}
The export function may return some `UserWarning` messages when we export the model. We can ignore these warnings as the exported model functions as expected.
:::



### Simplify the ONNX Model

The ONNX models generated by PyTorch are not always the most concise. We can use the [`onnx-simplifier`](https://pypi.org/project/onnx-simplifier/) package to tidy up the exported model.


```python
# Load the ONNX model from the onnx_file_name
onnx_model = onnx.load(onnx_file_path)

# Simplify the model
model_simp, check = simplify(onnx_model)

# Save the simplified model to the onnx_file_name
onnx.save(model_simp, onnx_file_path)
```



## Performing Inference with ONNX Runtime

Now that we have our ONNX model, it’s time to test it with ONNX Runtime.

### Create an Inference Session

We interact with models in ONNX Runtime through an [`InferenceSession`](https://onnxruntime.ai/docs/api/python/api_summary.html#load-and-run-a-model) object. Here we can specify which Execution Providers to use for inference and other configuration information. [Execution Providers](https://onnxruntime.ai/docs/execution-providers/) are the interfaces for hardware-specific inference engines like [TensorRT](https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html) for NVIDIA and [OpenVINO](https://onnxruntime.ai/docs/execution-providers/OpenVINO-ExecutionProvider.html) for Intel. By default, the `InferenceSession` uses the generic `CPUExecutionProvider`.


```python
# Load the model and create an InferenceSession
session = ort.InferenceSession(onnx_file_path, providers=['CPUExecutionProvider'])
```



### Define Annotation Function

Next, we need to annotate images with key points. PIL includes functionality to draw circles on images.


```python
def draw_keypoints_pil(image, keypoints, labels, colors, radius:int=5):

    """
    Annotates an image with keypoints, each marked by a circle and associated with specific labels and colors.

    This function draws circles on the provided image at given keypoint coordinates. Each keypoint is associated 
    with a label and a color. The radius of the circles can be adjusted.

    Parameters:
    image (PIL.Image): The input image on which annotations will be drawn.
    keypoints (list of tuples): A list of (x, y) tuples representing the coordinates of each keypoint.
    labels (list of str): A list of labels corresponding to each keypoint.
    colors (list of tuples): A list of RGB tuples for each keypoint, defining the color of the circle to be drawn.
    radius (int, optional): The radius of the circles to be drawn for each keypoint. Defaults to 5.

    Returns:
    annotated_image (PIL.Image): The image annotated with keypoints, each represented as a colored circle.
    """
        
    # Create a copy of the image
    annotated_image = image.copy()

    # Create an ImageDraw object for drawing on the image
    draw = ImageDraw.Draw(annotated_image)

    # Loop through the bounding boxes and labels in the 'annotation' DataFrame
    for i in range(len(labels)):
        # Get the key point coordinates
        x, y = keypoints[i]

        # Draw a circle
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=colors[i])
        
    return annotated_image
```



### Select a Test Image

We can download an image from one of my HuggingFace repositories to verify the exported model performs as expected.


```python
test_img_name = "pexels-2769554-man-doing-rock-and-roll-sign.jpg"
test_img_url = f"https://huggingface.co/datasets/cj-mills/pexel-hand-gesture-test-images/resolve/main/{test_img_name}"

download_file(test_img_url, './', False)

test_img = Image.open(test_img_name)
display(test_img)

pd.Series({
    "Test Image Size:": test_img.size, 
}).to_frame().style.hide(axis='columns')
```


![](./images/output_32_1.png){fig-align="center"}


<div style="overflow-x:auto; max-height:500px">
<table id="T_daf38">
  <thead>
  </thead>
  <tbody>
    <tr>
      <th id="T_daf38_level0_row0" class="row_heading level0 row0" >Test Image Size:</th>
      <td id="T_daf38_row0_col0" class="data row0 col0" >(640, 960)</td>
    </tr>
  </tbody>
</table>
</div>





### Prepare the Test Image


```python
# Set test image size
test_sz = 512

## Resize the test image
input_img = resize_img(test_img, target_sz=test_sz, divisor=1)

# Calculate the scale between the source image and the resized image
min_img_scale = min(test_img.size) / min(input_img.size)

display(input_img)

# Print the prediction data as a Pandas DataFrame for easy formatting
pd.Series({
    "Source Image Size:": test_img.size,
    "Input Dims:": input_img.size,
    "Min Image Scale:": min_img_scale,
    "Input Image Size:": input_img.size
}).to_frame().style.hide(axis='columns')
```
![](./images/output_34_0.png){fig-align="center"}



<div style="overflow-x:auto; max-height:500px">
<table id="T_54014">
  <thead>
  </thead>
  <tbody>
    <tr>
      <th id="T_54014_level0_row0" class="row_heading level0 row0" >Source Image Size:</th>
      <td id="T_54014_row0_col0" class="data row0 col0" >(640, 960)</td>
    </tr>
    <tr>
      <th id="T_54014_level0_row1" class="row_heading level0 row1" >Input Dims:</th>
      <td id="T_54014_row1_col0" class="data row1 col0" >(512, 768)</td>
    </tr>
    <tr>
      <th id="T_54014_level0_row2" class="row_heading level0 row2" >Min Image Scale:</th>
      <td id="T_54014_row2_col0" class="data row2 col0" >1.250000</td>
    </tr>
    <tr>
      <th id="T_54014_level0_row3" class="row_heading level0 row3" >Input Image Size:</th>
      <td id="T_54014_row3_col0" class="data row3 col0" >(512, 768)</td>
    </tr>
  </tbody>
</table>
</div>




### Prepare the Input Tensor

When we convert the PIL input image to a NumPy array, we need to reorder the array values to channels-first format, scale the values from `[0,255]` to `[0,1]`, and add a batch dimension.


```python
# Convert the input image to NumPy format
input_tensor_np = np.array(input_img, dtype=np.float32).transpose((2, 0, 1))[None]/255
```

### Compute the Predictions

Now, we can finally perform inference with our ONNX model.


```python
# Run inference
model_output = session.run(None, {"input": input_tensor_np})

# Set the confidence threshold
conf_threshold = 0.8

# Filter the output based on the confidence threshold
scores_mask = model_output[2] > conf_threshold

# Extract and scale the predicted keypoints
predicted_keypoints = (model_output[3][scores_mask])[:,:,:-1].reshape(-1,2)*min_img_scale
predicted_keypoints

labels=class_names*sum(scores_mask).item()

draw_keypoints_pil(test_img, 
                predicted_keypoints, 
                labels=labels,
                colors=[int_colors[i] for i in [class_names.index(label) for label in labels]],
               )
```
![](./images/output_38_0.png){fig-align="center"}



The model appears to work as intended, even on this new image.



::: {.callout-caution}
## Google Colab Users
1. Don't forget to download the ONNX model from the Colab Environment's file browser. ([tutorial link](https://christianjmills.com/posts/google-colab-getting-started-tutorial/#working-with-data)) 
:::










## Conclusion

Congratulations on reaching the end of this tutorial! We previously trained a Keypoint R-CNN model in PyTorch, and now we've exported that model to ONNX. With this, we can streamline our deployment process and leverage platform-specific hardware optimizations through ONNX Runtime.

As you move forward, consider exploring more about ONNX and its ecosystem. Check out the available [Execution Providers](https://onnxruntime.ai/docs/execution-providers/) that provide flexible interfaces to different hardware acceleration libraries.





{{< include /_tutorial-cta.qmd >}}




{{< include /_about-author-cta.qmd >}}
