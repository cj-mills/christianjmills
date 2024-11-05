---
title: "Exporting YOLOX Models from PyTorch to TensorFlow.js"
date: 2023-9-28
image: /images/empty.gif
hide: false
search_exclude: false
categories: [pytorch, tensorflow, tensorflow-js, yolox, tutorial]
description: "Learn how to export YOLOX models from PyTorch to TensorFlow.js to leverage efficient object detection in web applications."

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
* [Converting the Model to TensorFlow](#converting-the-model-to-tensorflow)
* [Exporting the Model to TensorFlow.js](#exporting-the-model-to-tensorflow.js)
* [Conclusion](#conclusion)


## Introduction

Welcome back to this series on training YOLOX models for real-time applications! [Previously](../), we demonstrated how to fine-tune a YOLOX model in PyTorch by creating a hand gesture detector. This tutorial builds on that by showing how to export the model to [TensorFlow.js](https://www.tensorflow.org/js). 

TensorFlow.js is an open-source hardware-accelerated JavaScript library for training and deploying machine learning models in web browsers. Converting our YOLOX model to TensorFlow.js allows us to run and integrate it directly into web applications without server-side processing.

Check out a live demo using the YOLOX model in a Unity WebGL application at the link below:

- [Unity TensorFlow.js Inference YOLOX Demo](https://cj-mills.github.io/unity-tfjs-inference-yolox-demo/)



We'll first use a tool called [nobuco](https://github.com/AlexanderLutsenko/nobuco) to translate the PyTorch model to a [TensorFlow Keras](https://www.tensorflow.org/guide/keras) model. We can then use the official TensorFlow.js conversion tool to export the Keras model to a TensorFlow.js Graph model. 

By the end of this tutorial, you will have a TensorFlow.js version of our YOLOX model that you can deploy to web applications and have it run locally in web browsers.



::: {.callout-important title="This post assumes the reader has completed the previous tutorial linked below:"}

* [Training YOLOX Models for Real-Time Object Detection in Pytorch](../)
:::





## Getting Started with the Code

As with the previous tutorial, the code is available as a Jupyter Notebook.

| Jupyter Notebook                                             | Google Colab                                                 |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [GitHub Repository](https://github.com/cj-mills/pytorch-yolox-object-detection-tutorial-code/blob/main/notebooks/pytorch-yolox-object-detector-nobuco-tfjs-export.ipynb) | [Open In Colab](https://colab.research.google.com/github/cj-mills/pytorch-yolox-object-detection-tutorial-code/blob/main/notebooks/pytorch-yolox-object-detector-nobuco-tfjs-export-colab.ipynb) |





## Setting Up Your Python Environment

We'll need to add a couple of new packages to our [Python environment](../#setting-up-your-python-environment).

::: {.callout-note title="Package Descriptions" collapse="true"}

| Package        | Description                                                  |
| -------------- | ------------------------------------------------------------ |
| `nobuco`       | A tool that helps you translate pytorch models into TensorFlow graphs. ([link](https://github.com/AlexanderLutsenko/nobuco)) |
| `tensorflowjs` | A pip package that contains libraries and tools for TensorFlow.js ([link](https://pypi.org/project/tensorflowjs/)) |

:::

Run the following command to install these additional libraries:

```bash
# Install additional packages
pip install nobuco tensorflowjs
```



::: {.callout-warning title="Update 09/26/2024"}

You will need to downgrade the following TensorFlow-related packages for nobuco:


```bash
pip install "keras<3.0.0" "tensorflow<2.16" "tensorflow-decision-forests<1.10.0"
```
:::





## Importing the Required Dependencies

With our environment updated, we can dive into the code. First, we will import the necessary Python dependencies into our Jupyter Notebook.


```python
# Import Python Standard Library dependencies
import json
from pathlib import Path

# Import YOLOX package
from cjm_yolox_pytorch.model import build_model
from cjm_yolox_pytorch.inference import YOLOXInferenceWrapper

# Import the pandas package
import pandas as pd

# Import PyTorch dependencies
import torch

# Import Nobuco dependencies
from nobuco import pytorch_to_keras, ChannelOrder

# Import TensorFlow.js dependencies
from tensorflowjs import converters, quantization
```



## Setting Up the Project

In this section, we'll set the folder locations for our project and training session with the PyTorch checkpoint.

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

pd.Series({
    "Project Directory:": project_dir,
    "Checkpoint Directory:": checkpoint_dir,
}).to_frame().style.hide(axis='columns')
```



<div style="overflow-x:auto; max-height:500px">
<table id="T_3c82a">
  <thead>
  </thead>
  <tbody>
    <tr>
      <th id="T_3c82a_level0_row0" class="row_heading level0 row0" >Project Directory:</th>
      <td id="T_3c82a_row0_col0" class="data row0 col0" >pytorch-yolox-object-detector</td>
    </tr>
    <tr>
      <th id="T_3c82a_level0_row1" class="row_heading level0 row1" >Checkpoint Directory:</th>
      <td id="T_3c82a_row1_col0" class="data row1 col0" >pytorch-yolox-object-detector/2023-08-17_16-14-43</td>
    </tr>
  </tbody>
</table>
</div>



::: {.callout-tip title="I made some model checkpoints available on Hugging Face Hub in the repository linked below:"}
* [cj-mills/yolox-hagrid-pytorch](https://huggingface.co/cj-mills/yolox-hagrid-pytorch/tree/main)
:::


::: {.callout-tip title="Those following along on Google Colab can drag the contents of their checkpoint folder into Colab's file browser. "}
:::



## Loading the Checkpoint Data

Now, we can load the colormap and normalization stats used during training and initialize a YOLOX model with the saved checkpoint.

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

### Load the Normalization Statistics


```python
# The normalization stats path
norm_stats_path = checkpoint_dir/'norm_stats.json'

# Read the normalization stats from the JSON file
with open(norm_stats_path, "r") as f:
    norm_stats_dict = json.load(f)

# Convert the dictionary to a tuple
norm_stats = (norm_stats_dict["mean"], norm_stats_dict["std_dev"])

# Print the mean and standard deviation
pd.DataFrame(norm_stats)
```


<div style="overflow-x:auto; max-height:500px">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.5</td>
      <td>0.5</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
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





### Load the Trained YOLOX Model


```python
# Select the YOLOX model configuration
model_type = checkpoint_path.stem

# Create a YOLOX model with the number of output classes equal to the number of class names
model = build_model(model_type, len(class_names))

# Initialize the model with the checkpoint parameters and buffers
model.load_state_dict(model_checkpoint)
```

```text
<All keys matched successfully>
```



## Converting the Model to TensorFlow

Before exporting the model, we'll wrap it with the preprocessing steps as we did [previously](../#preparing-the-model-for-inference). These steps will be included in the TensorFlow.js model, reducing the code we need to write when deploying the model.

### Prepare the Model for Inference

The [`YOLOXInferenceWrapper`](https://cj-mills.github.io/cjm-yolox-pytorch/inference.html#yoloxinferencewrapper) class has some optional settings we did not explore in the previous tutorial. The `scale_inp` parameter will scale pixel data from the range `[0,255]` to `[0,1]`, and `channels_last` sets the model to expect input tensors in channels-last format. 

Image data in JavaScript tends to be in the range `[0,255]`, so we'll want to enable the `scale_inp` setting. The nobuco conversion tool automatically sets the model to the channels-last format for TensorFlow.

Additionally, we can turn off the post-processing steps to compute the predicted bounding box information and probability scores. We'll need to do this when converting the model to TensorFlow using the nobuco tool as it throws an error with them enabled.



``` {.python}
# Convert the normalization stats to tensors
mean_tensor = torch.tensor(norm_stats[0]).view(1, 3, 1, 1)
std_tensor = torch.tensor(norm_stats[1]).view(1, 3, 1, 1)

# Set the model to evaluation mode
model.eval();

# Wrap the model with preprocessing and post-processing steps
wrapped_model = YOLOXInferenceWrapper(model, 
                                      mean_tensor, 
                                      std_tensor, 
                                      scale_inp=True, # Scale input values from the rang [0,255] to [0,1]
                                      channels_last=False, # Have the model expect input in channels-last format
                                      run_box_and_prob_calculation=False # Enable or disable post-processing steps
                                     )
```






### Prepare the Input Tensor

We need a sample input tensor for the conversion process.

``` {.python}
input_tensor = torch.randn(1, 3, 224, 224)
```

::: {.callout-warning}

The exported TensorFlow.js model will lock to this input resolution, so pick dimensions suitable for your intended use case.

:::



### Convert the PyTorch Model to Keras 

We use the `pytorch_to_keras` function included with nobuco to convert the YOLOX model from PyTorch to a [Keras](https://www.tensorflow.org/guide/keras) model. While we can stick with the default channel order for the model input, we need to maintain the output channel order from the original PyTorch model.


```python
keras_model = pytorch_to_keras(
    wrapped_model, 
    args=[input_tensor],
    outputs_channel_order=ChannelOrder.PYTORCH, 
)
```





### Save the Keras Model in SavedModel format

Next, we save the Keras model in TensorFlow's SavedModel format, the recommended format for exporting to TensorFlow.js.


```python
# Set the folder path for the SavedModel files
savedmodel_dir = Path(f"{checkpoint_dir}/{colormap_path.stem.removesuffix('-colormap')}-{model_type}-tf")
# Save the TensorFlow model to disk
keras_model.save(savedmodel_dir, save_format="tf")
```







## Exporting the Model to TensorFlow.js

With our TensorFlow model saved to disk, we can use the TensorFlow.js conversion tool to export it to a TensorFlow.js Graph model. 

Since the model will run locally in the browser, it must first download to the user's device. The larger the model, the longer users must wait for it to download. 

Fortunately, the TensorFlow.js conversion tool lets us quantize the model weights (i.e., convert them from 32-bit floating-point precision to 8-bit integers), significantly reducing their file size.


```python
# Set the path for TensorFlow.js model files
tfjs_model_dir = f"{savedmodel_dir}js-uint8"

# Convert the TensorFlow SavedModel to a TensorFlow.js Graph model
converters.convert_tf_saved_model(saved_model_dir=str(savedmodel_dir), 
                                  output_dir=tfjs_model_dir, 
                                  quantization_dtype_map={quantization.QUANTIZATION_DTYPE_UINT8:True}
                                 )
```










::: {.callout-caution}
## Google Colab Users
1. Don't forget to download the archive file containing the TensorFlow.js model files from the Colab Environment's file browser. ([tutorial link](https://christianjmills.com/posts/google-colab-getting-started-tutorial/#working-with-data)) 
:::










## Conclusion

Congratulations on reaching the end of this tutorial! We previously trained a YOLOX model in PyTorch for hand gesture detection, and now we've exported that model to TensorFlow.js. With it, we can deploy our model to the web and run it locally in users' browsers.



{{< include /_tutorial-cta.qmd >}}




{{< include /_about-author-cta.qmd >}}
