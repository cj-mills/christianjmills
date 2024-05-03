---
title: "Working with COCO Bounding Box Annotations in Torchvision"
date: 2024-01-21
image: /images/empty.gif
hide: false
search_exclude: false
categories: [pytorch, image-annotation, object-detection, tutorial]
description: "Learn how to work with COCO bounding box annotations in torchvision for object detection tasks."

twitter-card:
  creator: "@cdotjdotmills"
  site: "@cdotjdotmills"
  image: ../social-media/cover.jpg
open-graph:
  image: ../social-media/cover.jpg
---



::: {.callout-tip}
## This post is part of the following series:
* [**Torchvision Annotation Tutorials**](/series/tutorials/torchvision-annotation-tutorials-series.html)
:::



* [Introduction](#introduction)
* [Getting Started with the Code](#getting-started-with-the-code)
* [Setting Up Your Python Environment](#setting-up-your-python-environment)
* [Importing the Required Dependencies](#importing-the-required-dependencies)
* [Loading and Exploring the Dataset](#loading-and-exploring-the-dataset)
* [Preparing the Data](#preparing-the-data)
* [Conclusion](#conclusion)



## Introduction

Welcome to this hands-on guide for working with [COCO-formatted](https://cocodataset.org/#format-data) bounding box annotations in [torchvision](https://pytorch.org/vision/stable/index.html). Bounding box annotations specify rectangular frames around objects in images to identify and locate them for training object detection models.

![](./images/bounding-box-hero-img.png){fig-align="center"}

The tutorial walks through setting up a Python environment, loading the raw annotations into a [Pandas DataFrame](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html), annotating and augmenting images using torchvision's [Transforms V2 API](https://pytorch.org/vision/stable/auto_examples/transforms/plot_transforms_getting_started.html#sphx-glr-auto-examples-transforms-plot-transforms-getting-started-py), and creating a custom [Dataset](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html) class to feed samples to a model.

This guide is suitable for beginners and experienced practitioners, providing the code, explanations, and resources needed to understand and implement each step. By the end, you will have a solid foundation for working with COCO bounding box annotations in torchvision for object detection tasks.



## Getting Started with the Code

The tutorial code is available as a [Jupyter Notebook](https://jupyter.org/), which you can run locally or in a cloud-based environment like [Google Colab](https://colab.research.google.com/). I have dedicated tutorials for those new to these platforms or who need guidance setting up:

::: {.callout-tip title="Setup Guides" collapse="true"}

* [**Getting Started with Google Colab**](/posts/google-colab-getting-started-tutorial/)

* [**Setting Up a Local Python Environment with Mamba for Machine Learning Projects on Windows**](/posts/mamba-getting-started-tutorial-windows/)



:::

::: {.callout-tip title="Tutorial Code" collapse="true"}

| Jupyter Notebook: | [GitHub Repository](https://github.com/cj-mills/torchvision-annotation-tutorials/blob/main/notebooks/coco/torchvision-coco-bounding-box-annotations.ipynb) | [Open In Colab](https://colab.research.google.com/github/cj-mills/torchvision-annotation-tutorials/blob/main/notebooks/coco/torchvision-coco-bounding-box-annotations.ipynb) |
| ----------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
|                   |                                                              |                                                              |



:::



## Setting Up Your Python Environment

Before diving into the code, we'll cover the steps to create a local Python environment and install the necessary dependencies.



### Creating a Python Environment

First, we'll create a Python environment using [Conda](https://docs.conda.io/en/latest/)/[Mamba](https://mamba.readthedocs.io/en/latest/). Open a terminal with Conda/Mamba installed and run the following commands:



::: {.panel-tabset}
## Conda

``` {.bash}
# Create a new Python 3.10 environment
conda create --name pytorch-env python=3.10 -y
# Activate the environment
conda activate pytorch-env
```

## Mamba

``` {.bash}
# Create a new Python 3.10 environment
mamba create --name pytorch-env python=3.10 -y
# Activate the environment
mamba activate pytorch-env
```

:::





### Installing PyTorch

Next, we'll install [PyTorch](https://pytorch.org/). Run the appropriate command for your hardware and operating system.

::: {.panel-tabset}
## Linux/Windows (CUDA)

``` {.bash}
# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Mac

``` {.bash}
# MPS (Metal Performance Shaders) acceleration is available on MacOS 12.3+
pip install torch torchvision torchaudio
```

## Linux (CPU)

``` {.bash}
# Install PyTorch for CPU only
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

## Windows (CPU)

``` {.bash}
# Install PyTorch for CPU only
pip install torch torchvision torchaudio
```

:::



### Installing Additional Libraries

We also need to install some additional libraries for our project.

::: {.callout-note title="Package Descriptions" collapse="true"}

| Package       | Description                                                  |
| ------------- | ------------------------------------------------------------ |
| `jupyter`     | An  open-source web application that allows you to create and share  documents that contain live code, equations, visualizations, and  narrative text. ([link](https://jupyter.org/)) |
| `matplotlib`  | This package provides a comprehensive collection of visualization tools to  create high-quality plots, charts, and graphs for data exploration and  presentation. ([link](https://matplotlib.org/)) |
| `pandas`      | This package provides fast, powerful, and flexible data analysis and manipulation tools. ([link](https://pandas.pydata.org/)) |
| `pillow`      | The Python Imaging Library adds image processing capabilities. ([link](https://pillow.readthedocs.io/en/stable/)) |
| `tqdm`        | A Python library that provides fast, extensible progress bars for loops and other iterable objects in Python. ([link](https://tqdm.github.io/)) |
| `distinctipy` | A lightweight python package providing functions to generate colours that are visually distinct from one another. ([link](https://distinctipy.readthedocs.io/en/latest/)) |



:::

Run the following commands to install these additional libraries:

```bash
# Install additional dependencies
pip install distinctipy jupyter matplotlib pandas pillow tqdm
```





### Installing Utility Packages

We will also install some utility packages I made, which provide shortcuts for routine tasks.

::: {.callout-note title="Package Descriptions" collapse="true"}

| Package                | Description                                                  |
| ---------------------- | ------------------------------------------------------------ |
| `cjm_pil_utils`        | Some PIL utility functions I frequently use. ([link](https://cj-mills.github.io/cjm-pil-utils/)) |
| `cjm_psl_utils`        | Some utility functions using the Python Standard Library. ([link](https://cj-mills.github.io/cjm-psl-utils/)) |
| `cjm_pytorch_utils`    | Some utility functions for working with PyTorch. ([link](https://cj-mills.github.io/cjm-pytorch-utils/)) |
| `cjm_torchvision_tfms` | Some custom Torchvision tranforms. ([link](https://cj-mills.github.io/cjm-torchvision-tfms/)) |



:::

Run the following commands to install the utility packages:

```python
# Install additional utility packages
pip install cjm_pil_utils cjm_psl_utils cjm_pytorch_utils cjm_torchvision_tfms
```



With our environment set up, we can open our Jupyter Notebook and dive into the code. 



## Importing the Required Dependencies

First, we will import the necessary Python packages into our Jupyter Notebook.


```python
# Import Python Standard Library dependencies
from functools import partial
from pathlib import Path

# Import utility functions
from cjm_pil_utils.core import get_img_files
from cjm_psl_utils.core import download_file, file_extract
from cjm_pytorch_utils.core import tensor_to_pil
from cjm_torchvision_tfms.core import ResizeMax, PadSquare, CustomRandomIoUCrop

# Import the distinctipy module
from distinctipy import distinctipy

# Import matplotlib for creating plots
import matplotlib.pyplot as plt

# Import numpy
import numpy as np

# Import the pandas package
import pandas as pd

# Do not truncate the contents of cells and display all rows and columns
pd.set_option('max_colwidth', None, 'display.max_rows', None, 'display.max_columns', None)

# Import PIL for image manipulation
from PIL import Image

# Import PyTorch dependencies
import torch
from torch.utils.data import Dataset, DataLoader

# Import torchvision dependencies
import torchvision
torchvision.disable_beta_transforms_warning()
from torchvision.tv_tensors import BoundingBoxes
from torchvision.utils import draw_bounding_boxes
import torchvision.transforms.v2  as transforms

# Import tqdm for progress bar
from tqdm.auto import tqdm
```

Torchvision provides dedicated [`torch.Tensor`](https://pytorch.org/docs/stable/tensors.html) subclasses for different annotation types called [`TVTensors`](https://pytorch.org/vision/stable/tv_tensors.html). Torchvision's V2 transforms use these subclasses to update the annotations based on the applied image augmentations. The TVTensor class for bounding box annotations is called [`BoundingBoxes`](https://pytorch.org/vision/stable/generated/torchvision.tv_tensors.BoundingBoxes.html). Torchvision also includes a [`draw_bounding_boxes`](https://pytorch.org/vision/stable/generated/torchvision.utils.draw_bounding_boxes.html) function to annotate images.



## Loading and Exploring the Dataset

After importing the dependencies, we can start working with our data. I annotated a toy dataset with bounding boxes for this tutorial using images from the free stock photo site [Pexels](https://www.pexels.com/). The dataset is available on [HuggingFace Hub](https://huggingface.co/) at the link below:

- **Dataset Repository:** [coco-bounding-box-toy-dataset](https://huggingface.co/datasets/cj-mills/coco-bounding-box-toy-dataset/tree/main)



### Setting the Directory Paths

We first need to specify a place to store our dataset and a location to download the zip file containing it. The following code creates the folders in the current directory (`./`). Update the path if that is not suitable for you.


```python
# Define path to store datasets
dataset_dir = Path("./Datasets/")
# Create the dataset directory if it does not exist
dataset_dir.mkdir(parents=True, exist_ok=True)

# Define path to store archive files
archive_dir = dataset_dir/'../Archive'
# Create the archive directory if it does not exist
archive_dir.mkdir(parents=True, exist_ok=True)

# Creating a Series with the paths and converting it to a DataFrame for display
pd.Series({
    "Dataset Directory:": dataset_dir, 
    "Archive Directory:": archive_dir
}).to_frame().style.hide(axis='columns')
```

<div style="overflow-x:auto; max-height:500px">
<table id="T_5502a">
  <thead>
  </thead>
  <tbody>
    <tr>
      <th id="T_5502a_level0_row0" class="row_heading level0 row0" >Dataset Directory:</th>
      <td id="T_5502a_row0_col0" class="data row0 col0" >Datasets</td>
    </tr>
    <tr>
      <th id="T_5502a_level0_row1" class="row_heading level0 row1" >Archive Directory:</th>
      <td id="T_5502a_row1_col0" class="data row1 col0" >Datasets/../Archive</td>
    </tr>
  </tbody>
</table>
</div>




### Setting the Dataset Path

Next, we construct the name for the Hugging Face Hub dataset and set where to download and extract the dataset.


```python
# Set the name of the dataset
dataset_name = 'coco-bounding-box-toy-dataset'

# Construct the HuggingFace Hub dataset name by combining the username and dataset name
hf_dataset = f'cj-mills/{dataset_name}'

# Create the path to the zip file that contains the dataset
archive_path = Path(f'{archive_dir}/{dataset_name}.zip')

# Create the path to the directory where the dataset will be extracted
dataset_path = Path(f'{dataset_dir}/{dataset_name}')

# Creating a Series with the dataset name and paths and converting it to a DataFrame for display
pd.Series({
    "HuggingFace Dataset:": hf_dataset, 
    "Archive Path:": archive_path, 
    "Dataset Path:": dataset_path
}).to_frame().style.hide(axis='columns')
```

<div style="overflow-x:auto; max-height:500px">
<table id="T_6f643">
  <thead>
  </thead>
  <tbody>
    <tr>
      <th id="T_6f643_level0_row0" class="row_heading level0 row0" >HuggingFace Dataset:</th>
      <td id="T_6f643_row0_col0" class="data row0 col0" >cj-mills/coco-bounding-box-toy-dataset</td>
    </tr>
    <tr>
      <th id="T_6f643_level0_row1" class="row_heading level0 row1" >Archive Path:</th>
      <td id="T_6f643_row1_col0" class="data row1 col0" >Datasets/../Archive/coco-bounding-box-toy-dataset.zip</td>
    </tr>
    <tr>
      <th id="T_6f643_level0_row2" class="row_heading level0 row2" >Dataset Path:</th>
      <td id="T_6f643_row2_col0" class="data row2 col0" >Datasets/coco-bounding-box-toy-dataset</td>
    </tr>
  </tbody>
</table>
</div>




### Downloading the Dataset

We can now download the archive file and extract the dataset using the [`download_file`](https://cj-mills.github.io/cjm-psl-utils/core.html#download_file) and [`file_extract`](https://cj-mills.github.io/cjm-psl-utils/core.html#file_extract) functions from the `cjm_psl_utils` package. We can delete the archive afterward to save space.


```python
# Construct the HuggingFace Hub dataset URL
dataset_url = f"https://huggingface.co/datasets/{hf_dataset}/resolve/main/{dataset_name}.zip"
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



### Getting the Image and Annotation Folders

The dataset has two folders containing the sample images and annotations. The image folder organizes all samples together. The annotations are in a single JSON file.


```python
 # Assuming the images are stored in a subfolder named 'images'
img_dir = dataset_path/'images/'

# Assuming the annotation file is in JSON format and located in a subdirectory of the dataset
annotation_file_path = list(dataset_path.glob('*/*.json'))[0]

# Creating a Series with the paths and converting it to a DataFrame for display
pd.Series({
    "Image Folder": img_dir, 
    "Annotation File": annotation_file_path}).to_frame().style.hide(axis='columns')
```

<div style="overflow-x:auto; max-height:500px">
<table id="T_3ae81">
  <thead>
  </thead>
  <tbody>
    <tr>
      <th id="T_3ae81_level0_row0" class="row_heading level0 row0" >Image Folder</th>
      <td id="T_3ae81_row0_col0" class="data row0 col0" >Datasets/coco-bounding-box-toy-dataset/images</td>
    </tr>
    <tr>
      <th id="T_3ae81_level0_row1" class="row_heading level0 row1" >Annotation File</th>
      <td id="T_3ae81_row1_col0" class="data row1 col0" >Datasets/coco-bounding-box-toy-dataset/annotations/instances_default.json</td>
    </tr>
  </tbody>
</table>
</div>




### Get Image File Paths

Each image file has a unique name that we can use to locate the corresponding annotation data. We can make a dictionary that maps image names to file paths. The dictionary will allow us to retrieve the file path for a given image more efficiently.


```python
# Get all image files in the 'img_dir' directory
img_dict = {
    file.stem : file # Create a dictionary that maps file names to file paths
    for file in get_img_files(img_dir) # Get a list of image files in the image directory
}

# Print the number of image files
print(f"Number of Images: {len(img_dict)}")

# Display the first five entries from the dictionary using a Pandas DataFrame
pd.DataFrame.from_dict(img_dict, orient='index').head()
```

```text
Number of Images: 28
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
      <th>258421</th>
      <td>Datasets/coco-bounding-box-toy-dataset/images/258421.jpg</td>
    </tr>
    <tr>
      <th>3075367</th>
      <td>Datasets/coco-bounding-box-toy-dataset/images/3075367.jpg</td>
    </tr>
    <tr>
      <th>3076319</th>
      <td>Datasets/coco-bounding-box-toy-dataset/images/3076319.jpg</td>
    </tr>
    <tr>
      <th>3145551</th>
      <td>Datasets/coco-bounding-box-toy-dataset/images/3145551.jpg</td>
    </tr>
    <tr>
      <th>3176048</th>
      <td>Datasets/coco-bounding-box-toy-dataset/images/3176048.jpg</td>
    </tr>
  </tbody>
</table>
</div>


### Get Image Annotations

Next, we read the content of the JSON annotation file into a Pandas DataFrame so we can easily query the annotations.

#### Load the annotation file into a DataFrame

We will transpose the DataFrame to store each section in the JSON file in a separate column.


```python
# Read the JSON file into a DataFrame, assuming the JSON is oriented by index
annotation_file_df = pd.read_json(annotation_file_path, orient='index').transpose()
annotation_file_df.head()
```


<div style="overflow-x:auto; max-height:500px">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>licenses</th>
      <th>info</th>
      <th>categories</th>
      <th>images</th>
      <th>annotations</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>{'name': '', 'id': 0, 'url': ''}</td>
      <td>contributor</td>
      <td>{'id': 1, 'name': 'person', 'supercategory': ''}</td>
      <td>{'id': 1, 'width': 768, 'height': 1152, 'file_name': '258421.jpg', 'license': 0, 'flickr_url': '', 'coco_url': '', 'date_captured': 0}</td>
      <td>{'id': 1, 'image_id': 1, 'category_id': 1, 'segmentation': [], 'area': 24904.862800000003, 'bbox': [386.08, 443.94, 74.74, 333.22], 'iscrowd': 0, 'attributes': {'occluded': False, 'rotation': 0.0}}</td>
    </tr>
    <tr>
      <th>1</th>
      <td>None</td>
      <td>date_created</td>
      <td>None</td>
      <td>{'id': 2, 'width': 1344, 'height': 768, 'file_name': '3075367.jpg', 'license': 0, 'flickr_url': '', 'coco_url': '', 'date_captured': 0}</td>
      <td>{'id': 2, 'image_id': 1, 'category_id': 1, 'segmentation': [], 'area': 24440.896000000004, 'bbox': [340.25, 466.94, 78.74, 310.4], 'iscrowd': 0, 'attributes': {'occluded': False, 'rotation': 0.0}}</td>
    </tr>
    <tr>
      <th>2</th>
      <td>None</td>
      <td>description</td>
      <td>None</td>
      <td>{'id': 3, 'width': 768, 'height': 1120, 'file_name': '3076319.jpg', 'license': 0, 'flickr_url': '', 'coco_url': '', 'date_captured': 0}</td>
      <td>{'id': 3, 'image_id': 2, 'category_id': 1, 'segmentation': [], 'area': 365660.4554999999, 'bbox': [413.32, 41.22, 506.49, 721.95], 'iscrowd': 0, 'attributes': {'occluded': False, 'rotation': 0.0}}</td>
    </tr>
    <tr>
      <th>3</th>
      <td>None</td>
      <td>url</td>
      <td>None</td>
      <td>{'id': 4, 'width': 1184, 'height': 768, 'file_name': '3145551.jpg', 'license': 0, 'flickr_url': '', 'coco_url': '', 'date_captured': 0}</td>
      <td>{'id': 4, 'image_id': 3, 'category_id': 1, 'segmentation': [], 'area': 363031.32340000005, 'bbox': [335.31, 151.75, 375.91, 965.74], 'iscrowd': 0, 'attributes': {'occluded': False, 'rotation': 0.0}}</td>
    </tr>
    <tr>
      <th>4</th>
      <td>None</td>
      <td>version</td>
      <td>None</td>
      <td>{'id': 5, 'width': 1152, 'height': 768, 'file_name': '3176048.jpg', 'license': 0, 'flickr_url': '', 'coco_url': '', 'date_captured': 0}</td>
      <td>{'id': 5, 'image_id': 3, 'category_id': 1, 'segmentation': [], 'area': 390988.36079999997, 'bbox': [8.11, 131.88, 396.09, 987.12], 'iscrowd': 0, 'attributes': {'occluded': False, 'rotation': 0.0}}</td>
    </tr>
  </tbody>
</table>
</div>
---

Let's examine the source JSON content corresponding to the first row in the DataFrame.

<div style="overflow-x:auto; max-height:500px">

```json
{
    "licenses": [
        {
            "name": "",
            "id": 0,
            "url": ""
        }
    ],
    "info": {
        "contributor": "",
        "date_created": "",
        "description": "",
        "url": "",
        "version": "",
        "year": ""
    },
    "categories": [
        {
            "id": 1,
            "name": "person",
            "supercategory": ""
        }
    ],
    "images": [
        {
            "id": 1,
            "width": 768,
            "height": 1152,
            "file_name": "258421.jpg",
            "license": 0,
            "flickr_url": "",
            "coco_url": "",
            "date_captured": 0
        }
    ],
    "annotations": [
        {
            "id": 1,
            "image_id": 1,
            "category_id": 1,
            "segmentation": [],
            "area": 24904.862800000003,
            "bbox": [
                386.08,
                443.94,
                74.74,
                333.22
            ],
            "iscrowd": 0,
            "attributes": {
                "occluded": false,
                "rotation": 0.0
            }
        }
    ]
}
```

</div>

---

The most relevant information for our purposes is in the following sections:

- `categories`: Stores the class names for the various object types in the dataset. Note that this toy dataset only has one object type.
- `images`: Stores the dimensions and file names for each image.
- `annotations`: Stores the image IDs, category IDs, and the bounding box annotations in `[Top-Left X, Top-Left Y, Width, Height]` format.





#### Extract the object classes

We first need to extract the class names from the `categories` column of the DataFrame.


```python
# Extract and transform the 'categories' section of the data
# This DataFrame contains category details like category ID and name
categories_df = annotation_file_df['categories'].dropna().apply(pd.Series)
categories_df.set_index('id', inplace=True)
categories_df
```


<div style="overflow-x:auto; max-height:500px">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>supercategory</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>person</td>
      <td></td>
    </tr>
  </tbody>
</table>
</div>
---

This toy dataset only contains a single object class, named `person`.



#### Extract the image information

Next, we will extract the file names, image dimensions, and Image IDs from the `images` column of the DataFrame.


```python
# Extract and transform the 'images' section of the data
# This DataFrame contains image details like file name, height, width, and image ID
images_df = annotation_file_df['images'].to_frame()['images'].apply(pd.Series)[['file_name', 'height', 'width', 'id']]
images_df.head()
```


<div style="overflow-x:auto; max-height:500px">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>file_name</th>
      <th>height</th>
      <th>width</th>
      <th>id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>258421.jpg</td>
      <td>1152.0</td>
      <td>768.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3075367.jpg</td>
      <td>768.0</td>
      <td>1344.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3076319.jpg</td>
      <td>1120.0</td>
      <td>768.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3145551.jpg</td>
      <td>768.0</td>
      <td>1184.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3176048.jpg</td>
      <td>768.0</td>
      <td>1152.0</td>
      <td>5.0</td>
    </tr>
  </tbody>
</table>
</div>


#### Extract the annotation information

Last, we must extract the Image IDs, bounding box annotations, and Category IDs from the `annotations` column in the DataFrame.


```python
# Extract and transform the 'annotations' section of the data
# This DataFrame contains annotation details like image ID, bounding box, and category ID
annotations_df = annotation_file_df['annotations'].to_frame()['annotations'].apply(pd.Series)[['image_id', 'bbox', 'category_id']]
annotations_df.head()
```


<div style="overflow-x:auto; max-height:500px">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>image_id</th>
      <th>bbox</th>
      <th>category_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>[386.08, 443.94, 74.74, 333.22]</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>[340.25, 466.94, 78.74, 310.4]</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>[413.32, 41.22, 506.49, 721.95]</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>[335.31, 151.75, 375.91, 965.74]</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3</td>
      <td>[8.11, 131.88, 396.09, 987.12]</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>
---



Now that we have extracted the relevant information from the JSON file, we can recombine it into a single DataFrame for convenience.



#### Add the class names to the annotations

We will first add a new `label` column to the `annotations_df` DataFrame containing the corresponding class name from the `categories_df` DataFrame for each bounding box annotation.


```python
# Map 'category_id' in annotations DataFrame to category name using categories DataFrame
annotations_df['label'] = annotations_df['category_id'].apply(lambda x: categories_df.loc[x]['name'])
annotations_df.head()
```


<div style="overflow-x:auto; max-height:500px">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>image_id</th>
      <th>bbox</th>
      <th>category_id</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>[386.08, 443.94, 74.74, 333.22]</td>
      <td>1</td>
      <td>person</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>[340.25, 466.94, 78.74, 310.4]</td>
      <td>1</td>
      <td>person</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>[413.32, 41.22, 506.49, 721.95]</td>
      <td>1</td>
      <td>person</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>[335.31, 151.75, 375.91, 965.74]</td>
      <td>1</td>
      <td>person</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3</td>
      <td>[8.11, 131.88, 396.09, 987.12]</td>
      <td>1</td>
      <td>person</td>
    </tr>
  </tbody>
</table>
</div>


#### Merge the image and  annotation information

Next, we will add the data from the `images_df` DataFrame and match it to the bounding box annotations using the Image IDs.


```python
# Merge annotations DataFrame with images DataFrame on their image ID
annotation_df = pd.merge(annotations_df, images_df, left_on='image_id', right_on='id')
annotation_df.head()
```


<div style="overflow-x:auto; max-height:500px">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>image_id</th>
      <th>bbox</th>
      <th>category_id</th>
      <th>label</th>
      <th>file_name</th>
      <th>height</th>
      <th>width</th>
      <th>id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>[386.08, 443.94, 74.74, 333.22]</td>
      <td>1</td>
      <td>person</td>
      <td>258421.jpg</td>
      <td>1152.0</td>
      <td>768.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>[340.25, 466.94, 78.74, 310.4]</td>
      <td>1</td>
      <td>person</td>
      <td>258421.jpg</td>
      <td>1152.0</td>
      <td>768.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>[413.32, 41.22, 506.49, 721.95]</td>
      <td>1</td>
      <td>person</td>
      <td>3075367.jpg</td>
      <td>768.0</td>
      <td>1344.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>[335.31, 151.75, 375.91, 965.74]</td>
      <td>1</td>
      <td>person</td>
      <td>3076319.jpg</td>
      <td>1120.0</td>
      <td>768.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3</td>
      <td>[8.11, 131.88, 396.09, 987.12]</td>
      <td>1</td>
      <td>person</td>
      <td>3076319.jpg</td>
      <td>1120.0</td>
      <td>768.0</td>
      <td>3.0</td>
    </tr>
  </tbody>
</table>
</div>


#### Use the image name as the index

Then, we will change the index for the `annotations_df` DataFrame to match the keys in the `img_dict` dictionary, allowing us to retrieve both the image paths and annotation data using the same index key.


```python
# Remove old 'id' column post-merge
annotation_df.drop('id', axis=1, inplace=True)

# Extract the image_id from the file_name (assuming file_name contains the image_id)
annotation_df['image_id'] = annotation_df['file_name'].apply(lambda x: x.split('.')[0])

# Set 'image_id' as the index for the DataFrame
annotation_df.set_index('image_id', inplace=True)
annotation_df.head()
```


<div style="overflow-x:auto; max-height:500px">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>bbox</th>
      <th>category_id</th>
      <th>label</th>
      <th>file_name</th>
      <th>height</th>
      <th>width</th>
    </tr>
    <tr>
      <th>image_id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>258421</th>
      <td>[386.08, 443.94, 74.74, 333.22]</td>
      <td>1</td>
      <td>person</td>
      <td>258421.jpg</td>
      <td>1152.0</td>
      <td>768.0</td>
    </tr>
    <tr>
      <th>258421</th>
      <td>[340.25, 466.94, 78.74, 310.4]</td>
      <td>1</td>
      <td>person</td>
      <td>258421.jpg</td>
      <td>1152.0</td>
      <td>768.0</td>
    </tr>
    <tr>
      <th>3075367</th>
      <td>[413.32, 41.22, 506.49, 721.95]</td>
      <td>1</td>
      <td>person</td>
      <td>3075367.jpg</td>
      <td>768.0</td>
      <td>1344.0</td>
    </tr>
    <tr>
      <th>3076319</th>
      <td>[335.31, 151.75, 375.91, 965.74]</td>
      <td>1</td>
      <td>person</td>
      <td>3076319.jpg</td>
      <td>1120.0</td>
      <td>768.0</td>
    </tr>
    <tr>
      <th>3076319</th>
      <td>[8.11, 131.88, 396.09, 987.12]</td>
      <td>1</td>
      <td>person</td>
      <td>3076319.jpg</td>
      <td>1120.0</td>
      <td>768.0</td>
    </tr>
  </tbody>
</table>
</div>


#### Group annotations by image

Each bounding box annotation is currently in a separate row in the DataFrame. We will want to group the annotations for each image into a single row for use with PyTorch and torchvision.


```python
# Group the data by 'image_id' and aggregate information
# This results in each image ID having a list of bounding boxes, category IDs, labels, and the respective file name, height, and width
annotation_df = annotation_df.groupby('image_id').agg({'bbox': list, 
                                           'category_id': list,
                                           'label' :list,
                                           'file_name': 'first', 
                                           'height': 'first', 
                                           'width': 'first'
                                                      })

# Rename columns for clarity 
# 'bbox' is renamed to 'bboxes' and 'label' to 'labels'
annotation_df.rename(columns={'bbox': 'bboxes', 'label': 'labels'}, inplace=True)
annotation_df.head()
```


<div style="overflow-x:auto; max-height:500px">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>bboxes</th>
      <th>category_id</th>
      <th>labels</th>
      <th>file_name</th>
      <th>height</th>
      <th>width</th>
    </tr>
    <tr>
      <th>image_id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>258421</th>
      <td>[[386.08, 443.94, 74.74, 333.22], [340.25, 466.94, 78.74, 310.4]]</td>
      <td>[1, 1]</td>
      <td>[person, person]</td>
      <td>258421.jpg</td>
      <td>1152.0</td>
      <td>768.0</td>
    </tr>
    <tr>
      <th>3075367</th>
      <td>[[413.32, 41.22, 506.49, 721.95]]</td>
      <td>[1]</td>
      <td>[person]</td>
      <td>3075367.jpg</td>
      <td>768.0</td>
      <td>1344.0</td>
    </tr>
    <tr>
      <th>3076319</th>
      <td>[[335.31, 151.75, 375.91, 965.74], [8.11, 131.88, 396.09, 987.12]]</td>
      <td>[1, 1]</td>
      <td>[person, person]</td>
      <td>3076319.jpg</td>
      <td>1120.0</td>
      <td>768.0</td>
    </tr>
    <tr>
      <th>3145551</th>
      <td>[[642.0, 289.85, 27.66, 109.04], [658.63, 281.25, 28.46, 117.36]]</td>
      <td>[1, 1]</td>
      <td>[person, person]</td>
      <td>3145551.jpg</td>
      <td>768.0</td>
      <td>1184.0</td>
    </tr>
    <tr>
      <th>3176048</th>
      <td>[[518.23, 338.97, 76.4, 127.11], [683.42, 356.48, -44.56, 81.34]]</td>
      <td>[1, 1]</td>
      <td>[person, person]</td>
      <td>3176048.jpg</td>
      <td>768.0</td>
      <td>1152.0</td>
    </tr>
  </tbody>
</table>
</div>
---

With the annotations loaded, we can start inspecting our dataset.



### Inspecting the Class Distribution

First, we get the names of all the classes in our dataset and inspect the distribution of samples among these classes. This step won't yield any insights for the toy dataset but is worth doing for real-world projects. A balanced dataset (where each class has approximately the same number of instances) is ideal for training a machine-learning model.



#### Get image classes


```python
# Get a list of unique labels in the 'annotation_df' DataFrame
class_names = annotation_df['labels'].explode().unique().tolist()

# Display labels using a Pandas DataFrame
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
      <td>person</td>
    </tr>
  </tbody>
</table>
</div>



#### Visualize the class distribution


```python
# Get the number of samples for each object class
class_counts = pd.DataFrame(annotation_df['labels'].explode().tolist()).value_counts()

plot_labels = [index[0] for index in class_counts.index]

# Plot the distribution
class_counts.plot(kind='bar', figsize=(12, 5))
plt.title('Class distribution')
plt.ylabel('Count')
plt.xlabel('Classes')
plt.xticks(range(len(class_counts.index)), plot_labels, rotation=75)  # Set the x-axis tick labels
plt.show()
```

![](./images/output_37_0.png){fig-align="center"}



### Visualizing Image Annotations

In this section, we will annotate a single image with its bounding boxes using torchvision's `BoundingBoxes` class and `draw_bounding_boxes` function.

#### Generate a color map

While not required, assigning a unique color to bounding boxes for each  object class enhances visual distinction, allowing for easier  identification of different objects in the scene. We can use the [`distinctipy`](https://distinctipy.readthedocs.io/en/latest/) package to generate a visually distinct colormap.


```python
# Generate a list of colors with a length equal to the number of labels
colors = distinctipy.get_colors(len(class_names))

# Make a copy of the color map in integer format
int_colors = [tuple(int(c*255) for c in color) for color in colors]

# Generate a color swatch to visualize the color map
distinctipy.color_swatch(colors)
```

![](./images/output_41_0.png){fig-align="center"}



#### Download a font file

The [`draw_bounding_boxes`](https://pytorch.org/vision/stable/generated/torchvision.utils.draw_bounding_boxes.html) function included with torchvision uses a pretty small font size. We  can increase the font size if we use a custom font. Font files are  available on sites like [Google Fonts](https://fonts.google.com/), or we can use one included with the operating system.


```python
# Set the name of the font file
font_file = 'KFOlCnqEu92Fr1MmEU9vAw.ttf'

# Download the font file
download_file(f"https://fonts.gstatic.com/s/roboto/v30/{font_file}", "./")
```

#### Define the bounding box annotation function

We can make a partial function using `draw_bounding_boxes` since we’ll use the same box thickness and font each time we visualize bounding boxes.


```python
draw_bboxes = partial(draw_bounding_boxes, fill=False, width=2, font=font_file, font_size=25)
```



### Selecting a Sample Image

We can use the unique ID for an image in the image dictionary to get the image file path and the associated annotations from the annotation DataFrame.

#### Load the sample image


```python
# Get the file ID of the first image file
file_id = list(img_dict.keys())[0]

# Open the associated image file as a RGB image
sample_img = Image.open(img_dict[file_id]).convert('RGB')

# Print the dimensions of the image
print(f"Image Dims: {sample_img.size}")

# Show the image
sample_img
```

```text
Image Dims: (768, 1152)
```


![](./images/output_48_1.png){fig-align="center"}
    



#### Inspect the corresponding annotation data


```python
# Get the row from the 'annotation_df' DataFrame corresponding to the 'file_id'
annotation_df.loc[file_id].to_frame()
```


<div style="overflow-x:auto; max-height:500px">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>258421</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>bboxes</th>
      <td>[[386.08, 443.94, 74.74, 333.22], [340.25, 466.94, 78.74, 310.4]]</td>
    </tr>
    <tr>
      <th>category_id</th>
      <td>[1, 1]</td>
    </tr>
    <tr>
      <th>labels</th>
      <td>[person, person]</td>
    </tr>
    <tr>
      <th>file_name</th>
      <td>258421.jpg</td>
    </tr>
    <tr>
      <th>height</th>
      <td>1152.0</td>
    </tr>
    <tr>
      <th>width</th>
      <td>768.0</td>
    </tr>
  </tbody>
</table>
</div>


#### Annotate sample image

The `draw_bounding_boxes` function expects bounding box annotations in `[top-left X, top-left Y, bottom-right X, bottom-right Y]` format, so we’ll use the [`box_convert`](https://pytorch.org/vision/stable/generated/torchvision.ops.box_convert.html#torchvision.ops.box_convert) function included with torchvision to convert the bounding box annotations from `[x,y,w,h]` to `[x,y,x,y]` format.


```python
# Extract the labels and bounding box annotations for the sample image
labels = annotation_df.loc[file_id]['labels']
bboxes = annotation_df.loc[file_id]['bboxes']

# Annotate the sample image with labels and bounding boxes
annotated_tensor = draw_bboxes(
    image=transforms.PILToTensor()(sample_img), 
    boxes=torchvision.ops.box_convert(torch.Tensor(bboxes), 'xywh', 'xyxy'),
    labels=labels, 
    colors=[int_colors[i] for i in [class_names.index(label) for label in labels]]
)

tensor_to_pil(annotated_tensor)
```

![](./images/output_52_0.png){fig-align="center"}



We have loaded the dataset, inspected its class distribution, and visualized the annotations for a sample image. In the final section, we will cover how to augment images using torchvision's Transforms V2 API and create a custom Dataset class for training.



## Preparing the Data

In this section, we will first walk through a single example of how to apply augmentations to a single annotated image using torchvision's Transforms V2 API before putting everything together in a custom Dataset class.

### Data Augmentation

Here, we will define some data augmentations to apply to images during training. I created a few custom image transforms to help streamline the code.

The [first](https://cj-mills.github.io/cjm-torchvision-tfms/core.html#customrandomioucrop) extends the [`RandomIoUCrop`](https://pytorch.org/vision/stable/generated/torchvision.transforms.v2.RandomIoUCrop.html#torchvision.transforms.v2.RandomIoUCrop) transform included with torchvision to give the user more control over how much it crops into bounding box areas. The [second](https://cj-mills.github.io/cjm-torchvision-tfms/core.html#resizemax) resizes images based on their largest dimension rather than their smallest. The [third](https://cj-mills.github.io/cjm-torchvision-tfms/core.html#padsquare) applies square padding and allows the padding to be applied equally on both sides or randomly split between the two sides.

All three are available through the [`cjm-torchvision-tfms`](https://cj-mills.github.io/cjm-torchvision-tfms/) package.

#### Set training image size

Next, we will specify the image size to use during training.


```python
# Set training image size
train_sz = 384
```

#### Initialize custom transforms

Now, we can initialize the transform objects.


```python
# Create a RandomIoUCrop object
iou_crop = CustomRandomIoUCrop(min_scale=0.3, 
                               max_scale=1.0, 
                               min_aspect_ratio=0.5, 
                               max_aspect_ratio=2.0, 
                               sampler_options=[0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
                               trials=400, 
                               jitter_factor=0.25)

# Create a `ResizeMax` object
resize_max = ResizeMax(max_sz=train_sz)

# Create a `PadSquare` object
pad_square = PadSquare(shift=True)
```

#### Test the transforms

Torchvision's V2 image transforms take an image and a `targets` dictionary. The `targets` dictionary contains the annotations and labels for the image.

We will pass input through the `CustomRandomIoUCrop` transform first and then through `ResizeMax` and `PadSquare`. We can pass the result through a final resize operation to ensure both sides match the `train_sz` value.


```python
# Prepare bounding box targets
targets = {
    'boxes': BoundingBoxes(torchvision.ops.box_convert(torch.Tensor(bboxes), 'xywh', 'xyxy'), 
                           format='xyxy', 
                           canvas_size=sample_img.size[::-1]), 
    'labels': torch.Tensor([class_names.index(label) for label in labels])
}

# Crop the image
cropped_img, targets = iou_crop(sample_img, targets)

# Resize the image
resized_img, targets = resize_max(cropped_img, targets)

# Pad the image
padded_img, targets = pad_square(resized_img, targets)

# Ensure the padded image is the target size
resize = transforms.Resize([train_sz] * 2, antialias=True)
resized_padded_img, targets = resize(padded_img, targets)
sanitized_img, targets = transforms.SanitizeBoundingBoxes()(resized_padded_img, targets)

# Annotate the augmented image with updated labels and bounding boxes
annotated_tensor = draw_bboxes(
    image=transforms.PILToTensor()(sanitized_img), 
    boxes=targets['boxes'], 
    labels=[class_names[int(label.item())] for label in targets['labels']], 
    colors=[int_colors[i] for i in [class_names.index(label) for label in labels]]
)

# Display the annotated image
display(tensor_to_pil(annotated_tensor))

pd.Series({
    "Source Image:": sample_img.size,
    "Cropped Image:": cropped_img.size,
    "Resized Image:": resized_img.size,
    "Padded Image:": padded_img.size,
    "Resized Padded Image:": resized_padded_img.size,
}).to_frame().style.hide(axis='columns')
```

![](./images/output_60_0.png){fig-align="center"}

<div style="overflow-x:auto; max-height:500px">
<table id="T_cc3ae">
  <thead>
  </thead>
  <tbody>
    <tr>
      <th id="T_cc3ae_level0_row0" class="row_heading level0 row0" >Source Image:</th>
      <td id="T_cc3ae_row0_col0" class="data row0 col0" >(768, 1152)</td>
    </tr>
    <tr>
      <th id="T_cc3ae_level0_row1" class="row_heading level0 row1" >Cropped Image:</th>
      <td id="T_cc3ae_row1_col0" class="data row1 col0" >(653, 941)</td>
    </tr>
    <tr>
      <th id="T_cc3ae_level0_row2" class="row_heading level0 row2" >Resized Image:</th>
      <td id="T_cc3ae_row2_col0" class="data row2 col0" >(266, 383)</td>
    </tr>
    <tr>
      <th id="T_cc3ae_level0_row3" class="row_heading level0 row3" >Padded Image:</th>
      <td id="T_cc3ae_row3_col0" class="data row3 col0" >(383, 383)</td>
    </tr>
    <tr>
      <th id="T_cc3ae_level0_row4" class="row_heading level0 row4" >Resized Padded Image:</th>
      <td id="T_cc3ae_row4_col0" class="data row4 col0" >(384, 384)</td>
    </tr>
  </tbody>
</table>
</div>
---

Now that we know how to apply data augmentations, we can put all the steps we've covered into a custom Dataset class.



### Training Dataset Class

The following custom Dataset class is responsible for loading a single image, preparing the associated annotations, applying any image transforms, and returning the final `image` tensor and its `target` dictionary during training.


```python
class COCOBBoxDataset(Dataset):
    """
    A dataset class for COCO-style datasets with bounding box annotations.

    This class is designed to handle datasets where images are annotated with bounding boxes,
    such as object detection tasks. It supports loading images, applying transformations, 
    and retrieving the associated bounding box annotations.

    Attributes:
        _img_keys (list): A list of keys (identifiers) for each image in the dataset.
        _annotation_df (DataFrame): A DataFrame containing annotations for the images. 
                                    Each row corresponds to an image, indexed by its key.
        _img_dict (dict): A dictionary mapping image keys to their file paths.
        _class_to_idx (dict): A dictionary mapping class names to their corresponding indices.
        _transforms (callable, optional): Optional transform to be applied on a sample.

    Methods:
        __len__: Returns the number of images in the dataset.
        __getitem__: Retrieves an image and its corresponding target (bounding boxes and labels) 
                     by index.
        _load_image_and_target: Helper function to load an image and its corresponding target.
    """

    def __init__(self, img_keys, annotation_df, img_dict, class_to_idx, transforms=None):
        """
        Initializes the COCOBBoxDataset instance.

        Parameters:
            img_keys (list): List of image keys.
            annotation_df (DataFrame): DataFrame containing image annotations.
            img_dict (dict): Dictionary mapping image keys to file paths.
            class_to_idx (dict): Dictionary mapping class names to indices.
            transforms (callable, optional): Optional transform to be applied on a sample.
        """
        super(Dataset, self).__init__()
        
        self._img_keys = img_keys  # List of image keys
        self._annotation_df = annotation_df  # DataFrame containing annotations
        self._img_dict = img_dict  # Dictionary mapping image keys to image paths
        self._class_to_idx = class_to_idx  # Dictionary mapping class names to class indices
        self._transforms = transforms  # Image transforms to be applied
        
    def __len__(self):
        """
        Returns the total number of images in the dataset.

        Returns:
            int: The number of images in the dataset.
        """
        return len(self._img_keys)
        
    def __getitem__(self, index):
        """
        Retrieves an image and its corresponding target (bounding boxes and labels) by index.

        Parameters:
            index (int): The index of the image in the dataset.

        Returns:
            tuple: A tuple containing the image and its target. The target is a dictionary with 
                   keys 'boxes' and 'labels'.
        """
        img_key = self._img_keys[index]
        annotation = self._annotation_df.loc[img_key]
        image, target = self._load_image_and_target(annotation)
        
        if self._transforms:
            # Apply the specified transformations to the image and target
            image, target = self._transforms(image, target)
        
        return image, target

    def _load_image_and_target(self, annotation):
        """
        Helper function to load an image and its corresponding target.

        The target includes bounding boxes and labels for the image.

        Parameters:
            annotation (pandas.Series): The annotation data for the image, typically a row from the DataFrame.

        Returns:
            tuple: A tuple containing the image and its target, where the target is a dictionary 
                   with keys 'boxes' and 'labels'.
        """
        # Load the image file using the path from the image dictionary
        filepath = self._img_dict[annotation.name]
        image = Image.open(filepath).convert('RGB')

        # Extract bounding box data from the annotations and convert to the desired format
        bbox_list = annotation['bboxes']
        bbox_tensor = torchvision.ops.box_convert(torch.Tensor(bbox_list), 'xywh', 'xyxy')
        boxes = BoundingBoxes(bbox_tensor, format='xyxy', canvas_size=image.size[::-1])

        # Convert class labels in the annotation to their corresponding indices
        annotation_labels = annotation['labels']
        labels = torch.Tensor([self._class_to_idx[label] for label in annotation_labels])

        return image, {'boxes': boxes, 'labels': labels}
```



### Image Transforms

Here, we will specify and organize all the image transforms to apply during training.


```python
# Compose transforms for data augmentation
data_aug_tfms = transforms.Compose(
    transforms=[
        iou_crop,
        transforms.ColorJitter(
                brightness = (0.875, 1.125),
                contrast = (0.5, 1.5),
                saturation = (0.5, 1.5),
                hue = (-0.05, 0.05),
        ),
        transforms.RandomGrayscale(),
        transforms.RandomEqualize(),
        transforms.RandomPosterize(bits=3, p=0.5),
        transforms.RandomHorizontalFlip(p=0.5),
    ],
)

# Compose transforms to resize and pad input images
resize_pad_tfm = transforms.Compose([
    resize_max, 
    pad_square,
    transforms.Resize([train_sz] * 2, antialias=True)
])

# Compose transforms to sanitize bounding boxes and normalize input data
final_tfms = transforms.Compose([
    transforms.ToImage(), 
    transforms.ToDtype(torch.float32, scale=True),
    transforms.SanitizeBoundingBoxes(),
])

# Define the transformations for training and validation datasets
train_tfms = transforms.Compose([
    data_aug_tfms, 
    resize_pad_tfm, 
    final_tfms
])
```

::: {.callout-important}

Always use the [`SanitizeBoundingBoxes`](https://pytorch.org/vision/stable/generated/torchvision.transforms.v2.SanitizeBoundingBoxes.html#torchvision.transforms.v2.SanitizeBoundingBoxes) transform to clean up annotations after using data augmentations that alter bounding boxes (e.g., cropping, warping, etc.).
:::



### Initialize Dataset

Now, we can create the dataset object using the image dictionary, the annotation DataFrame, and the image transforms.


```python
# Create a mapping from class names to class indices
class_to_idx = {c: i for i, c in enumerate(class_names)}

# Instantiate the dataset using the defined transformations
train_dataset = COCOBBoxDataset(list(img_dict.keys()), annotation_df, img_dict, class_to_idx, train_tfms)

# Print the number of samples in the training dataset
pd.Series({
    'Training dataset size:': len(train_dataset),
}).to_frame().style.hide(axis='columns')
```

<div style="overflow-x:auto; max-height:500px">
<table id="T_69f73">
  <thead>
  </thead>
  <tbody>
    <tr>
      <th id="T_69f73_level0_row0" class="row_heading level0 row0" >Training dataset size:</th>
      <td id="T_69f73_row0_col0" class="data row0 col0" >28</td>
    </tr>
  </tbody>
</table>
</div>




### Inspect Samples

To close out, we should verify the dataset object works as intended by inspecting the first sample.

#### Inspect training set sample


```python
dataset_sample = train_dataset[0]

annotated_tensor = draw_bboxes(
    image=(dataset_sample[0]*255).to(dtype=torch.uint8), 
    boxes=dataset_sample[1]['boxes'], 
    labels=[class_names[int(i.item())] for i in dataset_sample[1]['labels']], 
    colors=[int_colors[int(i.item())] for i in dataset_sample[1]['labels']]
)

tensor_to_pil(annotated_tensor)
```


![](./images/output_69_0.png){fig-align="center"}







## Conclusion

In this tutorial, we covered how to load custom COCO bounding box annotations and work with them using torchvision's Transforms V2 API. The skills and knowledge you acquired here provide a solid foundation for future object detection projects.

As a next step, perhaps try annotating a custom COCO object detection dataset with a tool like [CVAT](https://github.com/opencv/cvat) and loading it with this tutorial's code. Once you're comfortable with that, try adapting the code in the following tutorial to train an object detection model on your custom dataset.

- [Training YOLOX Models for Real-Time Object Detection in PyTorch](/posts/pytorch-train-object-detector-yolox-tutorial)



## Recommended Tutorials

- [**Working with COCO Segmentation Annotations in Torchvision**](/posts/torchvision-coco-annotation-tutorials/segmentation-polygons/)**:** Learn how to work with COCO segmentation annotations in torchvision for instance segmentation tasks.  
- [**Training YOLOX Models for Real-Time Object Detection in PyTorch**](/posts/pytorch-train-object-detector-yolox-tutorial)**:** Learn how to train YOLOX models for real-time object detection in PyTorch by creating a hand gesture detection model. 



{{< include /_tutorial-cta.qmd >}}
