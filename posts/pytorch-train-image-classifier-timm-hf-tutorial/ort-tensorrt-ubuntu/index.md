---
title: "Quantizing timm Image Classifiers with ONNX Runtime and TensorRT in Ubuntu"
date: 2024-4-7
image: /images/empty.gif
hide: false
search_exclude: false
categories: [onnx, cuda, tensorrt, image classification, tutorial]
description: "Learn how to quantize timm image classification models with ONNX Runtime and TensorRT for int8 inference."

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
* [Quantization Process](#quantization-process)
* [Getting Started with the Code](#getting-started-with-the-code)
* [Setting Up Your Python Environment](#setting-up-your-python-environment)
* [Importing the Required Dependencies](#importing-the-required-dependencies)
* [Setting Up the Project](#setting-up-the-project)
* [Loading the Checkpoint Data](#loading-the-checkpoint-data)
* [Loading the Dataset](#loading-the-dataset)
* [Collecting Calibration Data](#collecting-calibration-data)
* [Performing Inference with TensorRT](#performing-inference-with-tensorrt)
* [Conclusion](#conclusion)



## Introduction

Welcome back to this series on image classification with the timm library. Previously, we [fine-tuned a ResNet 18-D](../) model in PyTorch to classify hand signs and [exported it to ONNX](../onnx-export). This tutorial covers quantizing our ONNX model and performing int8 inference using ONNX Runtime and TensorRT.

Quantization aims to make inference more computationally and memory efficient using a lower precision data type (e.g., 8-bit integer (int8)) for the model weights and activations. Modern devices increasingly have specialized hardware for running models at these lower precisions for improved performance.

ONNX Runtime includes tools to assist with quantizing our model from its original float32 precision to int8. ONNX Runtime's execution providers also make it easier to leverage the hardware-specific inference libraries used to run models on the specialized hardware. In this tutorial, we will use the TensorRT Execution Provider to perform int8-precision inference.

TensorRT is a high-performance inference library for NVIDIA hardware. For our purposes it allows us to run our image classification model at 16-bit and 8-bit precision, while leveraging the specialized tensor cores in modern NVIDIA devices.

::: {.callout-important title="This post assumes the reader has completed the previous tutorial linked below:"}
* [Exporting timm Image Classifiers from PyTorch to ONNX](../onnx-export)
:::



::: {.callout-important title="TensorRT Hardware Requirements:"}

TensorRT requires NVIDIA hardware with CUDA [Compute Capability](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities) 7.0 or higher (e.g., RTX 20-series or newer). Check the Compute Capability tables at the link below for your Nvidia hardware:

* [GPU Computer Capability Tables](https://developer.nvidia.com/cuda-gpus#compute)



 You can follow along using the free GPU-tier of Google Colab if you do not have any supported hardware.

:::





## Quantization Process

Quantizing our model involves converting the original 32-bit floating point values to 8-bit integers. float32 precision allows for a significantly greater range of possible values versus int8. To find the best way to map the float32 values to int8, we must compute the range of float32 values in the model. 

The float32 values for the model weights are static, while the activation values depend on the input fed to the model. We can calculate a suitable range of activation values by feeding sample inputs through the model and recording the activations. TensorRT can then use this information when quantizing the model. We will use a subset of images from the [original training dataset](../#loading-and-exploring-the-dataset) to generate this calibration data.





## Getting Started with the Code

As with the previous tutorial, the code is available as a Jupyter Notebook.

| Jupyter Notebook                                             | Google Colab                                                 |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [GitHub Repository](https://github.com/cj-mills/pytorch-timm-gesture-recognition-tutorial-code/blob/main/notebooks/timm-image-classifier-ort-tensorrt-int8-calibration-inference.ipynb) | [Open In Colab](https://colab.research.google.com/github/cj-mills/pytorch-timm-gesture-recognition-tutorial-code/blob/main/notebooks/timm-image-classifier-ort-tensorrt-int8-calibration-inference-colab.ipynb) |







## Setting Up Your Python Environment

First, we must add a few new libraries to our [Python environment](../onnx-export/#setting-up-your-python-environment). 



### Install CUDA Package

Both ONNX Runtime and TensorRT require CUDA for use with NVIDIA GPUs. The most recent CUDA version supported by ONNX Runtime is `12.2`.

Run the following command to install CUDA in our Python environment with [Conda/Mamba](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#conda-installation).

::: {.panel-tabset}

## Conda

```bash
conda install cuda -c nvidia/label/cuda-12.2.0 -y
```

## Mamba

```bash
mamba install cuda -c nvidia/label/cuda-12.2.0 -y
```

:::

### Install ONNX Runtime and TensorRT

The only additional libraries we need are ONNX Runtime with GPU support and TensorRT, assuming the packages used in the previous two tutorials are already in the Python environment.


::: {.callout-note title="Package Descriptions" collapse="true"}

| Package           | Description                                                  |
| ----------------- | ------------------------------------------------------------ |
| `onnxruntime-gpu` | ONNX Runtime is a runtime accelerator for machine learning models. ([link](https://pypi.org/project/onnxruntime-gpu/)) |
| `tensorrt`        | A high performance deep learning inference library for Nvidia devices. ([link](https://pypi.org/project/tensorrt/)) |

:::

Run the following commands to install the libraries:

```bash
# Install TensorRT packages
pip install 'tensorrt==10.0.1' --extra-index-url https://pypi.nvidia.com

# Install ONNX Runtime for CUDA 12
pip install -U 'onnxruntime-gpu==1.18.0' --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/
```



With our environment updated, we can dive into the code. 



## Importing the Required Dependencies

First, we will import the necessary Python dependencies into our Jupyter Notebook.


```python
# Import Python Standard Library dependencies
import json
import os
from pathlib import Path
import random

# Import utility functions
from cjm_psl_utils.core import download_file, file_extract
from cjm_pil_utils.core import resize_img, get_img_files

# Import numpy
import numpy as np

# Import the pandas package
import pandas as pd

# Do not truncate the contents of cells and display all rows and columns
pd.set_option('max_colwidth', None, 'display.max_rows', None, 'display.max_columns', None)

# Import PIL for image manipulation
from PIL import Image

# Import ONNX dependencies
import onnxruntime as ort # Import the ONNX Runtime
from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference
from onnxruntime.quantization import CalibrationDataReader, CalibrationMethod, create_calibrator, write_calibration_table

# Import tensorrt_libs
import tensorrt_libs
```



::: {.callout-important}

Make sure to import the `tensorrt_libs` module that is part of the `tensorrt` pip package. Otherwise, you will need to update the `LD_LIBRARY_PATH` environment variable with the path to the TensorRT library files.

:::





## Setting Up the Project

Next, we will set the folder locations for our project, the calibration dataset, and the directory with the ONNX model and JSON class labels file.

### Setting the Directory Paths

Readers following the tutorial on their local machine should select locations with read and write access to store the archived and extracted dataset. For a cloud service like Google Colab, you can set it to the  current directory.


```python
# The name for the project
project_name = f"pytorch-timm-image-classifier"

# The path for the project folder
project_dir = Path(f"./{project_name}/")

# Create the project directory if it does not already exist
project_dir.mkdir(parents=True, exist_ok=True)

# Define path to store datasets
dataset_dir = Path("/mnt/980_1TB_2/Datasets/")
# Create the dataset directory if it does not exist
dataset_dir.mkdir(parents=True, exist_ok=True)

# Define path to store archive files
archive_dir = dataset_dir/'../Archive'
# Create the archive directory if it does not exist
archive_dir.mkdir(parents=True, exist_ok=True)

# The path to the checkpoint folder
checkpoint_dir = Path(project_dir/f"2024-02-02_15-41-23")

pd.Series({
    "Project Directory:": project_dir, 
    "Dataset Directory:": dataset_dir, 
    "Archive Directory:": archive_dir,
    "Checkpoint Directory:": checkpoint_dir,
}).to_frame().style.hide(axis='columns')
```

<div style="overflow-x:auto; max-height:500px">
<table id="T_98a20">
  <thead>
  </thead>
  <tbody>
    <tr>
      <th id="T_98a20_level0_row0" class="row_heading level0 row0" >Project Directory:</th>
      <td id="T_98a20_row0_col0" class="data row0 col0" >pytorch-timm-image-classifier</td>
    </tr>
    <tr>
      <th id="T_98a20_level0_row1" class="row_heading level0 row1" >Dataset Directory:</th>
      <td id="T_98a20_row1_col0" class="data row1 col0" >/mnt/980_1TB_2/Datasets</td>
    </tr>
    <tr>
      <th id="T_98a20_level0_row2" class="row_heading level0 row2" >Archive Directory:</th>
      <td id="T_98a20_row2_col0" class="data row2 col0" >/mnt/980_1TB_2/Datasets/../Archive</td>
    </tr>
    <tr>
      <th id="T_98a20_level0_row3" class="row_heading level0 row3" >Checkpoint Directory:</th>
      <td id="T_98a20_row3_col0" class="data row3 col0" >pytorch-timm-image-classifier/2024-02-02_15-41-23</td>
    </tr>
  </tbody>
</table>
</table>
</div>



::: {.callout-tip title="Those following along on Google Colab can drag the contents of their checkpoint folder into Colab's file browser. "}
:::



## Loading the Checkpoint Data

Now, we can load the class labels, set the path for the ONNX model.

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



### Set Model Checkpoint Information


```python
# The onnx model path
onnx_file_path = list(checkpoint_dir.glob('*.onnx'))[0]
```



## Loading the Dataset

Now that we set up the project, we can download our dataset and select a subset to use for calibration.

### Setting the Dataset Path

We first need to construct the name for the Hugging Face Hub dataset and define where to download and extract the dataset.


```python
# Set the name of the dataset
dataset_name = 'hagrid-classification-512p-no-gesture-150k-zip'

# Construct the HuggingFace Hub dataset name by combining the username and dataset name
hf_dataset = f'cj-mills/{dataset_name}'

# Create the path to the zip file that contains the dataset
archive_path = Path(f'{archive_dir}/{dataset_name.removesuffix("-zip")}.zip')

# Create the path to the directory where the dataset will be extracted
dataset_path = Path(f'{dataset_dir}/{dataset_name.removesuffix("-zip")}')

# Creating a Series with the dataset name and paths and converting it to a DataFrame for display
pd.Series({
    "HuggingFace Dataset:": hf_dataset, 
    "Archive Path:": archive_path, 
    "Dataset Path:": dataset_path
}).to_frame().style.hide(axis='columns')
```

<div style="overflow-x:auto; max-height:500px">
<table id="T_04900">
  <thead>
  </thead>
  <tbody>
    <tr>
      <th id="T_04900_level0_row0" class="row_heading level0 row0" >HuggingFace Dataset:</th>
      <td id="T_04900_row0_col0" class="data row0 col0" >cj-mills/hagrid-classification-512p-no-gesture-150k-zip</td>
    </tr>
    <tr>
      <th id="T_04900_level0_row1" class="row_heading level0 row1" >Archive Path:</th>
      <td id="T_04900_row1_col0" class="data row1 col0" >/mnt/980_1TB_2/Datasets/../Archive/hagrid-classification-512p-no-gesture-150k.zip</td>
    </tr>
    <tr>
      <th id="T_04900_level0_row2" class="row_heading level0 row2" >Dataset Path:</th>
      <td id="T_04900_row2_col0" class="data row2 col0" >/mnt/980_1TB_2/Datasets/hagrid-classification-512p-no-gesture-150k</td>
    </tr>
  </tbody>
</table>
</div>



### Downloading the Dataset

We can now download the dataset archive file and extract the dataset. We can delete the archive afterward to save space.


```python
# Construct the HuggingFace Hub dataset URL
dataset_url = f"https://huggingface.co/datasets/{hf_dataset}/resolve/main/{dataset_name.removesuffix('-zip')}.zip"
print(f"HuggingFace Dataset URL: {dataset_url}")

# Set whether to delete the archive file after extracting the dataset
delete_archive = True

# Download the dataset if not present
if dataset_path.is_dir():
    print("Dataset folder already exists")
else:
    print("Downloading dataset...")
    download_file(dataset_url, archive_dir)    
    
    print("Extracting dataset...")
    file_extract(fname=archive_path, dest=dataset_dir)
    
    # Delete the archive if specified
    if delete_archive: archive_path.unlink()
```



### Get Image File Paths

Once downloaded, we can get the paths to the images in the dataset.


```python
# Get a list of all JPG image files in the dataset
img_file_paths = list(dataset_path.glob("./**/*.jpeg"))

# Print the number of image files
print(f"Number of Images: {len(img_file_paths)}")

# Display the first five entries from the dictionary using a Pandas DataFrame
pd.DataFrame(img_file_paths).head()
```

```text
Number of Images: 153735
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
      <td>/mnt/980_1TB_2/Datasets/hagrid-classification-512p-no-gesture-150k/call/3ffbf0a0-1837-42cd-8f13-33977a2b47aa.jpeg</td>
    </tr>
    <tr>
      <th>1</th>
      <td>/mnt/980_1TB_2/Datasets/hagrid-classification-512p-no-gesture-150k/call/7f4d415e-f570-42c3-aa5a-7c907d2d461e.jpeg</td>
    </tr>
    <tr>
      <th>2</th>
      <td>/mnt/980_1TB_2/Datasets/hagrid-classification-512p-no-gesture-150k/call/0003d6d1-3489-4f57-ab7a-44744dba93fd.jpeg</td>
    </tr>
    <tr>
      <th>3</th>
      <td>/mnt/980_1TB_2/Datasets/hagrid-classification-512p-no-gesture-150k/call/00084dfa-60a2-4c8e-9bd9-25658382b8b7.jpeg</td>
    </tr>
    <tr>
      <th>4</th>
      <td>/mnt/980_1TB_2/Datasets/hagrid-classification-512p-no-gesture-150k/call/0010543c-be59-49e7-8f6d-fbea8f5fdc6b.jpeg</td>
    </tr>
  </tbody>
</table>
</div>




### Select Sample Images

Using every image in the dataset for the calibration process would be unnecessary and time-consuming, so we'll select a random subset.


```python
random.seed(1234) # Set random seed for consistency 
sample_percentage = 0.05
random.shuffle(img_file_paths)
sample_img_paths = random.sample(img_file_paths, int(len(img_file_paths)*sample_percentage))
```



::: {.callout-tip title='Other Datasets:'}

Try to have at least `200` samples for the calibration set if adapting this tutorial to another dataset. 

:::



## Collecting Calibration Data

With the dataset samples selected, we can feed them through the model and collect the calibration data.

### Implement a CalibrationDataReader

First, we will implement a [`CalibrationDataReader`](https://github.com/microsoft/onnxruntime/blob/07788e082ef2c78c3f4e72f49e7e7c3db6f09cb0/onnxruntime/python/tools/quantization/calibrate.py#L30) class to load and prepare samples to feed through the model.


```python
class CalibrationDataReaderCV(CalibrationDataReader):
    """
    A subclass of CalibrationDataReader specifically designed for handling
    image data for calibration in computer vision tasks. This reader loads,
    preprocesses, and provides images for model calibration.
    """
    
    def __init__(self, img_file_paths, target_sz, input_name='input'):
        """
        Initializes a new instance of the CalibrationDataReaderCV class.
        
        Args:
            img_file_paths (list): A list of image file paths.
            target_sz (tuple): The target size (width, height) to resize images to.
            input_name (str, optional): The name of the input node in the ONNX model. Default is 'input'.
        """
        super().__init__()  # Initialize the base class
        
        # Initialization of instance variables
        self._img_file_paths = img_file_paths
        self.input_name = input_name
        self.enum = iter(img_file_paths)  # Create an iterator over the image paths
        self.target_sz = target_sz
        
    def get_next(self):
        """
        Retrieves, processes, and returns the next image in the sequence as a NumPy array suitable for model input.
        
        Returns:
            dict: A dictionary with a single key-value pair where the key is `input_name` and the value is the
                  preprocessed image as a NumPy array, or None if there are no more images.
        """
        
        img_path = next(self.enum, None)  # Get the next image path
        if not img_path:
            return None  # If there are no more paths, return None

        # Load the image from the filepath and convert to RGB
        image = Image.open(img_path).convert('RGB')

        # Resize the image to the target size
        input_img = resize_img(image, target_sz=self.target_sz, divisor=1)
        
        # Convert the image to a NumPy array, normalize, and add a batch dimension
        input_tensor_np = np.array(input_img, dtype=np.float32).transpose((2, 0, 1))[None] / 255

        # Return the image in a dictionary under the specified input name
        return {self.input_name: input_tensor_np}
```



::: {.callout-warning title='Preprocessing Steps:'}

This `CalibrationDataReader` class does not normalize the input as our ONNX model performs that step internally. Be sure to include any required input normalization if adapting this tutorial to another model that does not include it internally.

:::



### Specify a Cache Folder

Next, we will create a folder to store the collected calibration data and any cache files generated by TensorRT.


```python
trt_cache_dir = checkpoint_dir/'trt_engine_cache'
trt_cache_dir.mkdir(parents=True, exist_ok=True)
trt_cache_dir
```


```text
PosixPath('pytorch-timm-image-classifier/2024-02-02_15-41-23/trt_engine_cache')
```



### Collect Calibration Data

Now, we can create a calibrator object and an instance of our custom `CalibrationDataReader` object to collect the activation values and compute the range of values. The calibrator object creates a temporary ONNX model for the calibration process that we can delete afterward.

After feeding the data samples through the model, we will save the generated calibration file for TensorRT to use later. 


```python
%%time

target_sz = 288

# Save path for temporary ONNX model used during calibration process
augmented_model_path = onnx_file_path.parent/f"{onnx_file_path.stem}-augmented.onnx"

try:
    # Create a calibrator object for the ONNX model.
    calibrator = create_calibrator(
        model=onnx_file_path, 
        op_types_to_calibrate=None, 
        augmented_model_path=augmented_model_path, 
        calibrate_method=CalibrationMethod.MinMax
    )

    # Set the execution providers for the calibrator.
    calibrator.set_execution_providers(["CUDAExecutionProvider", "CPUExecutionProvider"])

    # Initialize the custom CalibrationDataReader object
    calibration_data_reader = CalibrationDataReaderCV(img_file_paths=sample_img_paths, 
                                                      target_sz=target_sz, 
                                                      input_name=calibrator.model.graph.input[0].name)

    # Collect calibration data using the specified data reader.
    calibrator.collect_data(data_reader=calibration_data_reader)

    # Initialize an empty dictionary to hold the new compute range values.
    new_compute_range = {}

    # Compute data and update the compute range for each key in the calibrator's data.
    for k, v in calibrator.compute_data().data.items():
        # Extract the min and max values from the range_value.
        v1, v2 = v.range_value
        # Convert the min and max values to float and store them in the new_compute_range dictionary.
        new_compute_range[k] = (float(v1.item()), float(v2.item()))
        
    # Write the computed calibration table to the specified directory.
    write_calibration_table(new_compute_range, dir=str(trt_cache_dir))
    
except Exception as e:
    # Catch any exceptions that occur during the calibration process.
    print("An error occurred:", e)

finally:
    # Remove temporary ONNX file created during the calibration process
    if augmented_model_path.exists():
        augmented_model_path.unlink()
```

```text
CPU times: user 48.1 s, sys: 5.88 s, total: 53.9 s
Wall time: 1min 4s
```



### Inspect TensorRT Cache Folder

Looking in the cache folder, we should see three new files.


```python
# Print the content of the module folder as a Pandas DataFrame
pd.DataFrame([path.name for path in trt_cache_dir.iterdir()])
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
      <td>calibration.cache</td>
    </tr>
    <tr>
      <th>1</th>
      <td>calibration.flatbuffers</td>
    </tr>
    <tr>
      <th>2</th>
      <td>calibration.json</td>
    </tr>
  </tbody>
</table>
</div>


That takes care of the calibration process. In the next section, we will create an ONNX Runtime inference session and perform inference with TensorRT.



## Performing Inference with TensorRT

To have TensorRT quantize the model for int8 inference, we need to specify the path to the cache folder and the calibration table file name and enable int8 precision when initializing the inference session.

### Create an Inference Session


```python
ort.get_available_providers()
```


```text
['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
```


```python
providers = [
    ('TensorrtExecutionProvider', {
        'device_id': 0, # The device ID
        'trt_max_workspace_size': 4e9, # Maximum workspace size for TensorRT engine (1e9 ≈ 1GB)
        'trt_engine_cache_enable': True, # Enable TensorRT engine caching
        'trt_engine_cache_path': str(trt_cache_dir), # Path for TensorRT engine, profile files, and int8 calibration table
        'trt_int8_enable': True, # Enable int8 mode in TensorRT
        'trt_int8_calibration_table_name': 'calibration.flatbuffers', # int8 calibration table file for non-QDQ models in int8 mode
    })
]

sess_opt = ort.SessionOptions()

# Load the model and create an InferenceSession
session = ort.InferenceSession(onnx_file_path, sess_options=sess_opt, providers=providers)
```



::: {.callout-note title='TensorRT Warning Messages:'}

You might see warning messages like the example below when creating the inference session with TensorRT. These are normal, and you can safely ignore them.

```text
2024-03-28 13:07:04.725964281 [W:onnxruntime:Default, tensorrt_execution_provider.h:83 log] [2024-03-28 20:07:04 WARNING] onnx2trt_utils.cpp:374: Your ONNX model has been generated with INT64 weights, while TensorRT does not natively support INT64. Attempting to cast down to INT32.
2024-03-28 13:07:04.725986806 [W:onnxruntime:Default, tensorrt_execution_provider.h:83 log] [2024-03-28 20:07:04 WARNING] onnx2trt_utils.cpp:400: One or more weights outside the range of INT32 was clamped
2024-03-28 13:07:04.738993049 [W:onnxruntime:Default, tensorrt_execution_provider.h:83 log] [2024-03-28 20:07:04 WARNING] onnx2trt_utils.cpp:374: Your ONNX model has been generated with INT64 weights, while TensorRT does not natively support INT64. Attempting to cast down to INT32.
2024-03-28 13:07:04.739015348 [W:onnxruntime:Default, tensorrt_execution_provider.h:83 log] [2024-03-28 20:07:04 WARNING] onnx2trt_utils.cpp:400: One or more weights outside the range of INT32 was clamped
```

:::



### Select a Test Image

We can use the same test image and input size from the [previous tutorial](../onnx-export/#select-a-test-image).


```python
test_img_name = 'pexels-elina-volkova-16191659.jpg'
test_img_url = f"https://huggingface.co/datasets/cj-mills/pexel-hand-gesture-test-images/resolve/main/{test_img_name}"

download_file(test_img_url, './', False)

test_img = Image.open(test_img_name)
display(test_img)

pd.Series({
    "Test Image Size:": test_img.size, 
}).to_frame().style.hide(axis='columns')
```




![](./images/output_41_1.png){fig-align="center"}

<div style="overflow-x:auto; max-height:500px">
<table id="T_cba4b">
  <thead>
  </thead>
  <tbody>
    <tr>
      <th id="T_cba4b_level0_row0" class="row_heading level0 row0" >Test Image Size:</th>
      <td id="T_cba4b_row0_col0" class="data row0 col0" >(637, 960)</td>
    </tr>
  </tbody>
</table>
</div>




### Prepare the Test Image


```python
# Set the input image size
test_sz = 288

# Resize image without cropping
input_img = resize_img(test_img, target_sz=test_sz)

display(input_img)

pd.Series({
    "Input Image Size:": input_img.size
}).to_frame().style.hide(axis='columns')
```



![](./images/output_43_0.png){fig-align="center"}



<div style="overflow-x:auto; max-height:500px">
<table id="T_796f0">
  <thead>
  </thead>
  <tbody>
    <tr>
      <th id="T_796f0_level0_row0" class="row_heading level0 row0" >Input Image Size:</th>
      <td id="T_796f0_row0_col0" class="data row0 col0" >(288, 416)</td>
    </tr>
  </tbody>
</table>
</div>




### Prepare the Input Tensor


```python
# Convert the existing input image to NumPy format
input_tensor_np = np.array(input_img, dtype=np.float32).transpose((2, 0, 1))[None]/255
```

### Build TensorRT Engine

TensorRT will build an optimized and quantized representation of our model called an engine when we first pass input to the inference session. It will save a copy of this engine object to the cache folder we specified earlier. The build process can take a bit, so caching the engine will save time for future use. 


```python
%%time
# Perform a single inference run to build the TensorRT engine for the current input dimensions
session.run(None, {"input": input_tensor_np});
```


    CPU times: user 25.4 s, sys: 1.88 s, total: 27.3 s
    Wall time: 35 s



::: {.callout-note}

TensorRT needs to build separate engine files for different input dimensions.

:::



### Inspect TensorRT Cache Folder

If we look in the cache folder again, we can see a new `.engine` file and a new `.profile` file.


```python
# Print the content of the module folder as a Pandas DataFrame
pd.DataFrame([path.name for path in trt_cache_dir.iterdir()])
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
      <td>calibration.cache</td>
    </tr>
    <tr>
      <th>1</th>
      <td>calibration.flatbuffers</td>
    </tr>
    <tr>
      <th>2</th>
      <td>calibration.json</td>
    </tr>
    <tr>
      <th>3</th>
      <td>TensorrtExecutionProvider_TRTKernel_graph_main_graph_9370993215447387188_0_0_int8_sm89.engine</td>
    </tr>
    <tr>
      <th>4</th>
      <td>TensorrtExecutionProvider_TRTKernel_graph_main_graph_9370993215447387188_0_0_int8_sm89.profile</td>
    </tr>
  </tbody>
</table>
</div>



### Benchmark Quantized Model

With the TensorRT engine built, we can benchmark our quantized model to gauge the raw inference speeds.


```python
%%timeit
session.run(None, {"input": input_tensor_np})
```

```text
376 µs ± 3.92 µs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)
```

In my testing for this model, TensoRT int8 inference tends to be about 3x faster than the CUDA execution provider with the original float32 model.



Of course, it does not matter how much faster the quantized model is if there is a significant drop in accuracy, so let's verify the prediction results.

### Compute the Predictions


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
    "Predicted Class:": pred_class,
    "Confidence Score:": f"{confidence_score*100:.2f}%"
}).to_frame().style.hide(axis='columns')
```



![](./images/output_53_0.png){fig-align="center"}



<div style="overflow-x:auto; max-height:500px">
<table id="T_46d6d">
  <thead>
  </thead>
  <tbody>
    <tr>
      <th id="T_46d6d_level0_row0" class="row_heading level0 row0" >Input Size:</th>
      <td id="T_46d6d_row0_col0" class="data row0 col0" >(288, 416)</td>
    </tr>
    <tr>
      <th id="T_46d6d_level0_row1" class="row_heading level0 row1" >Predicted Class:</th>
      <td id="T_46d6d_row1_col0" class="data row1 col0" >mute</td>
    </tr>
    <tr>
      <th id="T_46d6d_level0_row2" class="row_heading level0 row2" >Confidence Score:</th>
      <td id="T_46d6d_row2_col0" class="data row2 col0" >100.00%</td>
    </tr>
  </tbody>
</table>
</div>



---



The probability scores will likely differ slightly from the full-precision ONNX model, but the predicted class should be the same.



::: {.callout-caution}
## Google Colab Users
Don't forget to download the content of the `trt_engine_cache` folder from the Colab Environment's file browser. ([tutorial link](https://christianjmills.com/posts/google-colab-getting-started-tutorial/#working-with-data)) 
:::










## Conclusion

Congratulations on reaching the end of this tutorial. We previously trained an image classification model in PyTorch for hand gesture recognition, and now we've quantized that model for optimized inference on NVIDIA hardware. Our model is now smaller, faster, and better suited for real-time applications and edge devices like the Jetson Orin Nano.







{{< include /_tutorial-cta.qmd >}}
