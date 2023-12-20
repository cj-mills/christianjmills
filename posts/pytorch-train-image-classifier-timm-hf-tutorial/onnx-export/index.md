---
title: "Exporting timm Image Classifiers from PyTorch to ONNX"
date: 2023-8-23
image: /images/empty.gif
hide: false
search_exclude: false
categories: [pytorch, onnx, image-classification, tutorial]
description: "Learn how to export timm image classification models from PyTorch to ONNX and perform inference using ONNX Runtime."

twitter-card:
  creator: "@cdotjdotmills"
  site: "@cdotjdotmills"
  image: ../social-media/cover.jpg
open-graph:
  image: ../social-media/cover.jpg
---

::: {.callout-tip}
## This post is part of the following series:
* [**Fine-Tuning Image Classifiers with PyTorch and the timm library for Beginners**](/series/tutorials/pytorch-train-image-classifier-series.html)
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

Welcome back to this series on fine-tuning image classifiers with PyTorch and the timm library! [Previously](../), we demonstrated how to fine-tune a ResNet18-D model from the timm library in PyTorch by creating a hand gesture classifier. This tutorial builds on that by showing how to export the model to [ONNX](https://onnx.ai/) and perform inference using [ONNX Runtime](https://onnxruntime.ai/docs/).

ONNX (Open Neural Network Exchange) is an open format to represent machine learning models and make them portable across various platforms. ONNX Runtime is a cross-platform inference accelerator that provides interfaces to hardware-specific libraries. By exporting our model to ONNX, we can deploy it to multiple devices and leverage hardware acceleration for faster inference.

Additionally, we'll wrap the PyTorch model with the required preprocessing and post-processing steps to include them in the ONNX model. By the end of this tutorial, you will have an ONNX version of our ResNet18-D model that you can deploy to servers and edge devices using ONNX Runtime.



::: {.callout-important title="This post assumes the reader has completed the previous tutorial linked below:"}
* [Fine-Tuning Image Classifiers with PyTorch and the timm library for Beginners](../)
:::





## Getting Started with the Code

As with the previous tutorial, the code is available as a Jupyter Notebook.

| Jupyter Notebook                                             | Google Colab                                                 |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [GitHub Repository](https://github.com/cj-mills/pytorch-timm-gesture-recognition-tutorial-code/blob/main/notebooks/pytorch-timm-image-classifier-onnx-export.ipynb) | [Open In Colab](https://colab.research.google.com/github/cj-mills/pytorch-timm-gesture-recognition-tutorial-code/blob/main/notebooks/pytorch-timm-image-classifier-onnx-export-colab.ipynb) |





## Setting Up Your Python Environment

We'll need to add a few new libraries to our [Python environment](../#setting-up-your-python-environment) for working with ONNX models.

| Package           | Description                                                  |
| ----------------- | ------------------------------------------------------------ |
| `onnx`            | This package provides a Python API for working with ONNX models. ([link](https://pypi.org/project/onnx/)) |
| `onnxruntime`     | ONNX Runtime is a runtime accelerator for machine learning models. ([link](https://onnxruntime.ai/)) |
| `onnx-simplifier` | This package helps simplify ONNX models. ([link](https://pypi.org/project/onnx-simplifier/)) |


Run the following command to install these additional libraries:

```bash
# Install ONNX packages
pip install onnx onnxruntime onnx-simplifier
```





## Importing the Required Dependencies

With our environment updated, we can dive into the code. First, we will import the necessary Python dependencies into our Jupyter Notebook.


```python
# Import Python Standard Library dependencies
import json
from pathlib import Path
import random

# Import utility functions
from cjm_psl_utils.core import download_file, get_source_code
from cjm_pil_utils.core import resize_img

# Import numpy
import numpy as np

# Import the pandas package
import pandas as pd

# Import PIL for image manipulation
from PIL import Image

# Import timm library
import timm

# Import PyTorch dependencies
import torch
from torch import nn

# Import ONNX dependencies
import onnx # Import the onnx module
from onnxsim import simplify # Import the method to simplify ONNX models
import onnxruntime as ort # Import the ONNX Runtime
```



## Setting Up the Project

In this section, we'll set the folder locations for our project and training session with the PyTorch checkpoint.

### Set the Directory Paths


```python
# The name for the project
project_name = f"pytorch-timm-image-classifier"

# The path for the project folder
project_dir = Path(f"./{project_name}/")

# Create the project directory if it does not already exist
project_dir.mkdir(parents=True, exist_ok=True)

# The path to the checkpoint folder
checkpoint_dir = Path(project_dir/f"2023-08-12_15-21-16")

pd.Series({
    "Project Directory:": project_dir,
    "Checkpoint Directory:": checkpoint_dir,
}).to_frame().style.hide(axis='columns')
```



<div style="overflow-x:auto; max-height:500px">
<table id="T_2c96e">
  <thead>
  </thead>
  <tbody>
    <tr>
      <th id="T_2c96e_level0_row0" class="row_heading level0 row0" >Project Directory:</th>
      <td id="T_2c96e_row0_col0" class="data row0 col0" >pytorch-timm-image-classifier</td>
    </tr>
    <tr>
      <th id="T_2c96e_level0_row1" class="row_heading level0 row1" >Checkpoint Directory:</th>
      <td id="T_2c96e_row1_col0" class="data row1 col0" >pytorch-timm-image-classifier/2023-08-12_15-21-16</td>
    </tr>
  </tbody>
</table>
</div>




::: {.callout-tip title="Those following along on Google Colab can drag the contents of their checkpoint folder into Colab's file browser. "}
:::





## Loading the Checkpoint Data

Now, we can load the class labels and initialize a ResNet18-D model with the saved checkpoint.

### Load the Class Labels


```python
# The class labels path
class_labels_path = list(checkpoint_dir.glob('*classes.json'))[0]

# Load the JSON class labels data
with open(class_labels_path, 'r') as file:
        class_labels_json = json.load(file)

# Get the list of classes
class_names = class_labels_json['classes']

# Print the list of classes
pd.DataFrame(class_names)
```



<div style="overflow-x:auto; max-height:500px">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>call</td>
    </tr>
    <tr>
      <th>1</th>
      <td>dislike</td>
    </tr>
    <tr>
      <th>2</th>
      <td>fist</td>
    </tr>
    <tr>
      <th>3</th>
      <td>four</td>
    </tr>
    <tr>
      <th>4</th>
      <td>like</td>
    </tr>
    <tr>
      <th>5</th>
      <td>mute</td>
    </tr>
    <tr>
      <th>6</th>
      <td>no_gesture</td>
    </tr>
    <tr>
      <th>7</th>
      <td>ok</td>
    </tr>
    <tr>
      <th>8</th>
      <td>one</td>
    </tr>
    <tr>
      <th>9</th>
      <td>palm</td>
    </tr>
    <tr>
      <th>10</th>
      <td>peace</td>
    </tr>
    <tr>
      <th>11</th>
      <td>peace_inverted</td>
    </tr>
    <tr>
      <th>12</th>
      <td>rock</td>
    </tr>
    <tr>
      <th>13</th>
      <td>stop</td>
    </tr>
    <tr>
      <th>14</th>
      <td>stop_inverted</td>
    </tr>
    <tr>
      <th>15</th>
      <td>three</td>
    </tr>
    <tr>
      <th>16</th>
      <td>three2</td>
    </tr>
    <tr>
      <th>17</th>
      <td>two_up</td>
    </tr>
    <tr>
      <th>18</th>
      <td>two_up_inverted</td>
    </tr>
  </tbody>
</table>
</div>




### Load the Model Checkpoint


```python
# The model checkpoint path
checkpoint_path = list(checkpoint_dir.glob('*.pth'))[0]

# Load the model checkpoint onto the CPU
model_checkpoint = torch.load(checkpoint_path, map_location='cpu')
```





### Load the Finetuned Model


```python
# Specify the model configuration
model_type = checkpoint_path.stem

# Create a model with the number of output classes equal to the number of class names
model = timm.create_model(model_type, num_classes=len(class_names))

# Initialize the model with the checkpoint parameters and buffers
model.load_state_dict(model_checkpoint)
```

```text
<All keys matched successfully>
```





### Get the Normalization Stats

```python
# Import the resnet module
from timm.models import resnet

# Get the default configuration of the chosen model
model_cfg = resnet.default_cfgs[model_type].default.to_dict()

# Retrieve normalization statistics (mean and std) specific to the pretrained model
mean, std = model_cfg['mean'], model_cfg['std']
norm_stats = (mean, std)
norm_stats
```
```text
((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
```








## Exporting the Model to ONNX

Before exporting the model, we will wrap it with the preprocessing and post-processing steps. These steps will be included in the ONNX model, reducing the code we need to write when deploying the model to other platforms.



### Prepare the Model for Inference

Whenever we make predictions with the model, we must normalize the input data and pass the model output through a [Softmax](https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html) function. We can define a wrapper class that automatically performs these steps.



Additionally, we can include options to scale pixel data from the range `[0,255]` to `[0,1]` and set the model to expect input tensors in channels-last format. These settings can be helpful when deploying to platforms where tensor operations are less convenient.



#### Define model export wrapper


```python
class InferenceWrapper(nn.Module):
    def __init__(self, model, normalize_mean, normalize_std, scale_inp=False, channels_last=False):
        super().__init__()
        self.model = model
        self.register_buffer("normalize_mean", normalize_mean)
        self.register_buffer("normalize_std", normalize_std)
        self.scale_inp = scale_inp
        self.channels_last = channels_last
        self.softmax = nn.Softmax(dim=1)

    def preprocess_input(self, x):
        if self.scale_inp:
            x = x / 255.0

        if self.channels_last:
            x = x.permute(0, 3, 1, 2)

        x = (x - self.normalize_mean) / self.normalize_std
        return x

    def forward(self, x):
        x = self.preprocess_input(x)
        x = self.model(x)
        x = self.softmax(x)
        return x
```

#### Wrap model with preprocessing and post-processing steps



::: {.panel-tabset}
## Default Settings

``` {.python}
# Define the normalization mean and standard deviation
mean_tensor = torch.tensor(norm_stats[0]).view(1, 3, 1, 1)
std_tensor = torch.tensor(norm_stats[1]).view(1, 3, 1, 1)

# Set the model to evaluation mode
model.eval();

# Wrap the model with preprocessing and post-processing steps
wrapped_model = InferenceWrapper(model, 
                                 mean_tensor, 
                                 std_tensor, 
                                 scale_inp=False, # Scale input values from the rang [0,255] to [0,1]
                                 channels_last=False, # Have the model expect input in channels-last format
                                )
```


## Inverse

``` {.python}
# Define the normalization mean and standard deviation
mean_tensor = torch.tensor(norm_stats[0]).view(1, 3, 1, 1)
std_tensor = torch.tensor(norm_stats[1]).view(1, 3, 1, 1)

# Set the model to evaluation mode
model.eval();

# Wrap the model with preprocessing and post-processing steps
wrapped_model = InferenceWrapper(model, 
                                 mean_tensor, 
                                 std_tensor, 
                                 scale_inp=True, # Scale input values from the rang [0,255] to [0,1]
                                 channels_last=True, # Have the model expect input in channels-last format
                                )
```


:::



::: {.callout-note title="Settings for Unity's Barracuda Inference Library:"}
```python
scale_inp=False
channels_last=False
```
:::

### Prepare the Input Tensor

We need a sample input tensor for the export process.

::: {.panel-tabset}
## Default

``` {.python}
input_tensor = torch.randn(1, 3, 256, 256)
```

## Channels-Last

``` {.python}
input_tensor = torch.randn(1, 256, 256, 3)
```
:::



### Export the Model to ONNX

We can export the model using the [`torch.onnx.export()`](https://pytorch.org/docs/stable/onnx.html#torch.onnx.export) function. This function performs a single pass through the model and records all operations to generate a [TorchScript graph](https://pytorch.org/docs/stable/jit.html). It then exports this graph to ONNX by decomposing each graph node (which contains a PyTorch operator) into a series of ONNX operators.

If we want the ONNX model to support different input sizes, we must set the width and height input axes as dynamic. However, attempting to do this with the ResNet18-D model causes the export function to throw an error. Therefore, we'll need to modify the exported ONNX model directly.


::: {.panel-tabset}
## Default
```python
# Set a filename for the ONNX model
onnx_file_path = f"{checkpoint_dir}/{class_labels_path.stem.removesuffix('-classes')}-{model_type}.onnx"

# Export the PyTorch model to ONNX format
torch.onnx.export(wrapped_model.cpu(),
                  input_tensor.cpu(),
                  onnx_file_path,
                  export_params=True,
                  do_constant_folding=False,
                  input_names = ['input'],
                  output_names = ['output'],
                  # dynamic_axes={'input': {2 : 'height', 3 : 'width'}}
                 )
```
```text
============= Diagnostic Run torch.onnx.export version 2.0.1+cu118 =============
verbose: False, log level: Level.ERROR
======================= 0 NONE 0 NOTE 0 WARNING 0 ERROR ========================
```

## Channels-Last
```python
# Set a filename for the ONNX model
onnx_file_path = f"{checkpoint_dir}/{class_labels_path.stem.removesuffix('-classes')}-{model_type}.onnx"

# Export the PyTorch model to ONNX format
torch.onnx.export(wrapped_model.cpu(),
                  input_tensor.cpu(),
                  onnx_file_path,
                  export_params=True,
                  do_constant_folding=False,
                  input_names = ['input'],
                  output_names = ['output'],
                  # dynamic_axes={'input': {1 : 'height', 2 : 'width'}}
                 )
```
```text
============= Diagnostic Run torch.onnx.export version 2.0.1+cu118 =============
verbose: False, log level: Level.ERROR
======================= 0 NONE 0 NOTE 0 WARNING 0 ERROR ========================
```

:::





### Enable Dynamic Input Dimensions

We can enable dynamic input dimensions for the ONNX model by changing the appropriate values in the input shape to `-1`.

::: {.panel-tabset}
## Default
```python
# Load the ONNX model from the onnx_file_name
onnx_model = onnx.load(onnx_file_path)

# Get the first input node for the ONNX model
input_tensor = onnx_model.graph.input[0]

# Set the height and width dimensions to dynamic values (using -1)
height_dim = input_tensor.type.tensor_type.shape.dim[2]
height_dim.dim_value = -1

width_dim = input_tensor.type.tensor_type.shape.dim[3]
width_dim.dim_value = -1
```

## Channels-Last
```python
# Load the ONNX model from the onnx_file_name
onnx_model = onnx.load(onnx_file_path)

# Assume the first input needs its dimensions modified
input_tensor = onnx_model.graph.input[0]

# Set the height and width dimensions to dynamic values (using -1)
height_dim = input_tensor.type.tensor_type.shape.dim[1]
height_dim.dim_value = -1

width_dim = input_tensor.type.tensor_type.shape.dim[2]
width_dim.dim_value = -1
```

:::



We then save the updated ONNX model back to disk for later. Before we do so, we'll perform one more modification.



### Simplify the ONNX Model
The ONNX models generated by PyTorch are not always the most concise. We can use the [`onnx-simplifier`](https://pypi.org/project/onnx-simplifier/) package to tidy up the exported model.

```python
# Simplify the model
model_simp, check = simplify(onnx_model)

# Save the simplified model to the onnx_file_name
onnx.save(model_simp, onnx_file_path)
```






## Performing Inference with ONNX Runtime

Now that we have our ONNX model, it's time to test it with ONNX Runtime.

### Create an Inference Session

We interact with models in ONNX Runtime through an [`InferenceSession`](https://onnxruntime.ai/docs/api/python/api_summary.html#load-and-run-a-model) object. Here, we can specify which Execution Providers to use for inference and other configuration information. [Execution Providers](https://onnxruntime.ai/docs/execution-providers/) are the interfaces for hardware-specific inference engines like [TensorRT](https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html) for NVIDIA and [OpenVINO](https://onnxruntime.ai/docs/execution-providers/OpenVINO-ExecutionProvider.html) for Intel. By default, the `InferenceSession` uses the generic `CPUExecutionProvider`.


```python
# Load the model and create an InferenceSession
session = ort.InferenceSession(onnx_file_path, , providers=['CPUExecutionProvider'])
```



With our inference session initialized, we can select an image to test our ONNX model.



### Select a Test Image

Let's use the same test image and input size from the [previous tutorial](../#testing-the-model-on-new-data) to compare the results with the PyTorch model.


```python
test_img_name = "pexels-2769554-man-doing-rock-and-roll-sign.jpg"
test_img_url = f"https://huggingface.co/datasets/cj-mills/pexel-hand-gesture-test-images/resolve/main/{test_img_name}"

download_file(test_img_url, './', False)

test_img = Image.open(test_img_name)
display(test_img)

target_cls = "rock"

pd.Series({
    "Test Image Size:": test_img.size, 
    "Target Class:": target_cls
}).to_frame().style.hide(axis='columns')
```




![](./images/output_41_1.png){fig-align="center"}


<div style="overflow-x:auto; max-height:500px">
<table id="T_47ce0">
  <thead>
  </thead>
  <tbody>
    <tr>
      <th id="T_47ce0_level0_row0" class="row_heading level0 row0" >Test Image Size:</th>
      <td id="T_47ce0_row0_col0" class="data row0 col0" >(640, 960)</td>
    </tr>
    <tr>
      <th id="T_47ce0_level0_row1" class="row_heading level0 row1" >Target Class:</th>
      <td id="T_47ce0_row1_col0" class="data row1 col0" >rock</td>
    </tr>
  </tbody>
</table>
</div>





### Prepare the Test Image


```python
# Set test image size
test_sz = 288

# Resize image without cropping to multiple of the max stride
input_img = resize_img(test_img.copy(), target_sz=test_sz)

display(input_img)

pd.Series({
    "Input Image Size:": input_img.size
}).to_frame().style.hide(axis='columns')
```



![](./images/output_43_0.png){fig-align="center"}

<div style="overflow-x:auto; max-height:500px">
<table id="T_3ba23">
  <thead>
  </thead>
  <tbody>
    <tr>
      <th id="T_3ba23_level0_row0" class="row_heading level0 row0" >Input Image Size:</th>
      <td id="T_3ba23_row0_col0" class="data row0 col0" >(288, 416)</td>
    </tr>
  </tbody>
</table>
</div>



### Prepare the Input Tensor

When we convert the PIL input image to a NumPy array, we need to reorder the array values to channels-first format, scale the values from `[0,255]` to `[0,1]`, and add a batch dimension. When we enable the `scale_inp` and `channels_last` options, we only need to add a batch dimension.


::: {.panel-tabset}
## Default

``` {.python}
# Convert the existing input image to NumPy format
input_tensor_np = np.array(input_img, dtype=np.float32).transpose((2, 0, 1))[None]/255
```

## Scaled & Channels-Last

``` {.python}
# Convert the existing input image to NumPy format
input_tensor_np = np.array(input_img, dtype=np.float32)[None]
```

:::

### Compute the Predictions

Now, we can finally perform inference with our ONNX model.

```python
# Run inference
outputs = session.run(None, {"input": input_tensor_np})[0]

# Get the highest confidence score
confidence_score = outputs.max()

# Get the class index with the highest confidence score and convert it to the class name
pred_class = class_names[outputs.argmax()]

# Display the image
display(test_img)

# Store the prediction data in a Pandas Series for easy formatting
pd.Series({
    "Input Size:": input_img.size,
    "Target Class:": target_cls,
    "Predicted Class:": pred_class,
    "Confidence Score:": f"{confidence_score*100:.2f}%"
}).to_frame().style.hide(axis='columns')
```


![](./images/output_47_0.png){fig-align="center"}

<div style="overflow-x:auto; max-height:500px">
<table id="T_effd1">
  <thead>
  </thead>
  <tbody>
    <tr>
      <th id="T_effd1_level0_row0" class="row_heading level0 row0" >Input Size:</th>
      <td id="T_effd1_row0_col0" class="data row0 col0" >(288, 416)</td>
    </tr>
    <tr>
      <th id="T_effd1_level0_row1" class="row_heading level0 row1" >Target Class:</th>
      <td id="T_effd1_row1_col0" class="data row1 col0" >rock</td>
    </tr>
    <tr>
      <th id="T_effd1_level0_row2" class="row_heading level0 row2" >Predicted Class:</th>
      <td id="T_effd1_row2_col0" class="data row2 col0" >rock</td>
    </tr>
    <tr>
      <th id="T_effd1_level0_row3" class="row_heading level0 row3" >Confidence Score:</th>
      <td id="T_effd1_row3_col0" class="data row3 col0" >79.93%</td>
    </tr>
  </tbody>
</table>
</div>


The model predictions should be virtually identical to the PyTorch model, but the confidence scores can sometimes vary slightly.


::: {.callout-caution}
## Google Colab Users
1. Don't forget to download the ONNX model from the Colab Environment's file browser. ([tutorial link](https://christianjmills.com/posts/google-colab-getting-started-tutorial/#working-with-data)) 
:::









## Conclusion

Congratulations on reaching the end of this tutorial! We previously fine-tuned a model from the timm library in PyTorch for hand gesture classification and now exported that model to ONNX. With this, we can streamline our deployment process and leverage platform-specific hardware optimizations through ONNX Runtime.

As you move forward, consider exploring more about ONNX and its ecosystem. Check out the available [Execution Providers](https://onnxruntime.ai/docs/execution-providers/) that provide flexible interfaces to different hardware acceleration libraries.

If you found this guide helpful, consider sharing it with others.
