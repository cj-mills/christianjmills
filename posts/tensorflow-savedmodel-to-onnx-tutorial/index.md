---
title: How to Convert TensorFlow Models to ONNX with tf2onnx
date: '2020-10-21'

title-block-categories: true
layout: post
toc: false
hide: false
search_exclude: false
comments:
  utterances:
    repo: cj-mills/christianjmills
description: This post covers how to use tf2onnx to convert a TensorFlow SavedModel
  to ONNX.
categories: [tensorflow, onnx, tutorial]

aliases:
-  /How-to-Convert-a-TensorFlow-SavedModel-to-ONNX/

---

## Requirements

To follow along with this example, you will need:

* A python environment with  [tensorflow](https://www.tensorflow.org/install) (**tip:** [Use conda](https://www.michaelphi.com/stop-installing-tensorflow-using-pip-for-performance-sake/))

  * ```bash
    conda install tensorflow
    ```

    or

  * ```bash
    conda install tensorflow-gpu
    ```

    

* A TensorFlow SavedModel ([download](https://drive.google.com/drive/folders/1RRuNOR4pj2tUw8VIBEgZ3gQJQf9nJk8T?usp=sharing))

The TensorFlow model used for this tutorial is a PoseNet model with a ResNet architecture. You can download the exact model [here](https://drive.google.com/drive/folders/1RRuNOR4pj2tUw8VIBEgZ3gQJQf9nJk8T?usp=sharing). 

## Usage

### Installation

You can install the library using pip:

```bash
pip install -U tf2onnx
```

### Steps

1. Make sure the SavedModel file is named `saved_model.pb`

2. At a minimum, you need to specify the source model format, the path to the folder containing the SavedModel, and a name for the ONNX file.

   For example:

   * Model Format: `--saved-model`

   * Model Folder: `./savedmodel` 

     **Note:** Do not include a `/` at the end of the path.

   * Output Name: `model.onnx`

```bash
python -m tf2onnx.convert --saved-model ./savedmodel --opset 10 --output model.onnx
```

With these parameters you might receive some warnings, but the output should include something like this.

```bash
2020-10-21 12:54:11,024 - INFO - Using tensorflow=2.3.0, onnx=1.7.0, tf2onnx=1.6.3/d4abc8
2020-10-21 12:54:11,024 - INFO - Using opset <onnx, 10>
2020-10-21 12:54:12,423 - INFO - Optimizing ONNX model
2020-10-21 12:54:14,047 - INFO - After optimization: Add -4 (20->16), Const -1 (115->114), Identity -4 (4->0), Transpose -117 (122->5)
2020-10-21 12:54:14,138 - INFO -
2020-10-21 12:54:14,138 - INFO - Successfully converted TensorFlow model ./savedmodel to ONNX
2020-10-21 12:54:14,215 - INFO - ONNX model is saved at model.onnx
```



## Next Steps

Be sure to check out the GitHub [repo](https://github.com/onnx/tensorflow-onnx) if you want to learn what else you can do with the tool. The README page goes in to greater detail about the following:

* Current TensorFlow support
* Parameter options
* Advanced use cases
* How the tool converts TensorFlow Models