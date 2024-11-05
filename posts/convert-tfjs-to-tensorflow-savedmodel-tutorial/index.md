---
categories:
- tensorflow
- tutorial
date: '2020-09-15'
description: A simple example of how to convert a TensorFlow.js graph model to a TensorFlow
  SavedModel.
title: How to Convert a TensorFlow.js Graph Model to a TensorFlow SavedModel
comments:
  utterances:
    repo: cj-mills/christianjmills

aliases:
- /How-to-Convert-a-TensorFlow-js-Graph-Model-to-a-Tensorflow-SavedModel/


twitter-card:
  creator: "@cdotjdotmills"
  site: "@cdotjdotmills"
  image: /images/default-preview-image-black.png
open-graph:
  image: /images/default-preview-image-black.png
---

## Motivation

The [Tensoflow.js](https://www.tensorflow.org/js) library is great for leveraging machine learning directly in a [web browser](https://pose-animator-demo.firebaseapp.com/static_image.html) or [Node.js application](https://medium.com/@andreas.schallwig/do-not-laugh-a-simple-ai-powered-game-3e22ad0f8166). However, TensorFlow does not currently provide any methods for converting TensorFlow.js models back into a standard TensorFlow format. This can be a problem if you need to change how your model is deployed at some point and don't have access to a standard TensorFlow format. In addition, standard TensorFlow formats have not been made available for most of the pretrained TFJS models. Fortunately, there is a third-party [library](https://github.com/patlevin/tfjs-to-tf) that provides this functionality. This post will cover how to use this library to convert a TFJS model to the standard SavedModel format.



## About the Tool

The aptly named [TensorFlow.js Graph Model Converter](https://github.com/patlevin/tfjs-to-tf) library allows you to convert a TFJS graph model to either a TensorFlow frozen graph or SavedModel format. Which format is best depends on your intended use case. However, the SavedModel format provides much more flexibility and can still be trained. The frozen graph format can only be used for inference. Therefore, we will use the SavedModel format for this tutorial.

The library can either be used as a command-line tool or accessed through an [API](https://github.com/patlevin/tfjs-to-tf/blob/master/docs/api.rst) within Python. The API provides more advanced functionality such as combining multiple TFJS models into a single SavedModel. However, using the command-line tool is faster for simple conversions. We will stick with the command-line tool for this tutorial.



## Requirements

To follow along with this example, you will need:

* A python environment with [tensorflow 2.1+](https://www.tensorflow.org/install) and [tensorflowjs 1.5.2+](https://pypi.org/project/tensorflowjs/) installed
* A pretrained TFJS model ([download](https://drive.google.com/drive/folders/1gxXxLpof1biBIU0_jgUazQ5L32C0GfCT?usp=sharing))

The TFJS model used for this tutorial is a [PoseNet](https://medium.com/tensorflow/real-time-human-pose-estimation-in-the-browser-with-tensorflow-js-7dd0bc881cd5) model with a [ResNet](https://arxiv.org/abs/1512.03385) architecture. You can download the folder containing the TFJS model [here](https://drive.google.com/drive/folders/1gxXxLpof1biBIU0_jgUazQ5L32C0GfCT?usp=sharing). Once you've downloaded the zip file, extract the `posenet-resnet-stride16` folder.

A pretrained TFJS model consists of the following files:

* A JSON file that defines the model topology
* One or more .bin files that contain the trained weights

**Note:** When downloading a TFJS model from from somewhere like [TensorFlow Hub](https://tfhub.dev/s?deployment-format=tfjs), make sure the JSON file isn't corrupted. If you open up the JSON file, you should see something like this:

```json
"format": "graph-model",
    "generatedBy": "1.13.1",
    "convertedBy": "TensorFlow.js Converter v1.1.2",
    "modelTopology": {
        "node": [{
            "name": "sub_2",
            "op": "Placeholder",
            "attr": {
                "shape": {
                    "shape": {
                        "dim": [{
                            "size": "1"
                        }, {
                            "size": "-1"
                        }, {
                            "size": "-1"
                        }, {
                            "size": "3"
                        }]
                    }
                },
                "dtype": {
                    "type": "DT_FLOAT"
                }
            }
        }
```



## Usage

### Installation

You can install the library using pip:

```bash
pip install tfjs-graph-converter
```

### Steps

1. Open the `posenet-resnet-stride16` folder in a terminal.

2. To convert the TFJS model into a SavedModel, you need to specify the path to the JSON file, the path to a folder that the SavedModel will be saved to, and the output format. The tool will create the folder if it doesn't exist.

   For example:

   * JSON file: `model-stride16.json`
   * Save Folder: `savedmodel`
   * Output Format: `tf_saved_model`
   \
   ```bash
   tfjs_graph_converter ./model-stride16.json ./savedmodel --output_format tf_saved_model
   ```

3. If all goes well, you should see something like this:

   ```bash
   TensorFlow.js Graph Model Converter
   
   Graph model:    ./model-stride16.json
   Output:         ./savedmodel
   Target format:  tf_saved_model
   
   Converting.... Done.
   Conversion took 2.778s
   ```
   **Note:** Some newer TFJS models released by Google use new types of layers in their Neural Network architecture that are not yet supported by the converter library at the time of writing.
   
   The `savedmodel` folder should contain:

   * A `variables` folder (which is empty for this example)
   * A `saved_model.pb` file.
   
4. (Optional) If you wish, you can examine the SavedModel using the following command:

   ```bash
   saved_model_cli show --dir ./savedmodel --all
   ```

   This command should return the following output:

   ```bash
   MetaGraphDef with tag-set: 'serve' contains the following SignatureDefs:
   
   signature_def['serving_default']:
     The given SavedModel SignatureDef contains the following input(s):
       inputs['sub_2'] tensor_info:
           dtype: DT_FLOAT
           shape: (1, -1, -1, 3)
           name: sub_2:0
     The given SavedModel SignatureDef contains the following output(s):
       outputs['float_heatmaps'] tensor_info:
           dtype: DT_FLOAT
           shape: (1, -1, -1, 17)
           name: float_heatmaps:0
       outputs['float_short_offsets'] tensor_info:
           dtype: DT_FLOAT
           shape: (1, -1, -1, 34)
           name: float_short_offsets:0
       outputs['resnet_v1_50/displacement_bwd_2/BiasAdd'] tensor_info:
           dtype: DT_FLOAT
           shape: (1, -1, -1, 32)
           name: resnet_v1_50/displacement_bwd_2/BiasAdd:0
       outputs['resnet_v1_50/displacement_fwd_2/BiasAdd'] tensor_info:
           dtype: DT_FLOAT
           shape: (1, -1, -1, 32)
           name: resnet_v1_50/displacement_fwd_2/BiasAdd:0
     Method name is: tensorflow/serving/predict
   ```

   Here, you can see descriptions for the input and output layers of the model.
   
5. (Optional) You can rename layers when converting the model by using the `--rename` option.

   ```bash
   tfjs_graph_converter ./model-stride16.json ./savedmodel --output_format tf_saved_model --rename float_short_offsets:offsets,float_heatmaps:heatmaps,sub_2:input
   ```

   

## Next Steps

Be sure to check out the GitHub [repo](https://github.com/patlevin/tfjs-to-tf) if you want to learn what else you can do with this library or request (or add) support for new layer types. 

Once you've successfully converted your TFJS model to standard TensorFlow, you have a lot more options for working with the model. A few of them are listed below.

* Work with the model in Python using standard [TensorFlow](https://www.tensorflow.org/tutorials).
* Convert the model to a [TensorFlow Lite](https://www.tensorflow.org/lite/convert) format and deploy it to mobile and IoT devices.
* Convert the model to non-TensorFlow formats such as [ONNX](https://github.com/onnx/tensorflow-onnx).






{{< include /_about-author-cta.qmd >}}
