---
title: OpenVINO Object Detection for Unity Tutorial Pt.1 (Outdated)
layout: post
toc: false
comments: true
description: This post covers the prerequisite software, pretrained object detection models, and test videos used in the tutorial.
categories: [openvino, object-detection, yolox, tutorial, unity]
hide: false
permalink: /:title/
search_exclude: false
---



### 8/11/2022:

* This tutorial is outdated. Use the new version at the link below.
* [End-to-End Object Detection for Unity With IceVision and OpenVINO Pt. 1](https://christianjmills.com/IceVision-to-OpenVINO-to-Unity-Tutorial-1/)

------



* [Overview](#overview)
* [Prerequisites](#prerequisites)
* [Download OpenVINO IR Models](#download-openvino-ir-models)
* [Download Test Videos](#download-test-videos)
* [Conclusion](#conclusion)



![yolox_example_1](..\images\openvino-yolox-unity\yolox_example_1.jpg)

# Overview

In this tutorial series, we will cover how to perform [object detection](https://www.fritz.ai/object-detection/) in the [Unity](https://unity.com/products/unity-platform) game engine with the [OpenVINO™ Toolkit](https://docs.openvinotoolkit.org/latest/index.html). As demonstrated above, object detection models allow us to locate and classify objects inside an image or video. Combining this functionality with Unity unlocks significant potential for interactive applications.

These models can be trained to detect arbitrary types of objects provided there is sufficient training data. They can even be trained to detect specific people, expressions, gestures, or poses. The models used in this tutorial have been trained on the [COCO (Common Objects in Context)](https://cocodataset.org/#home) dataset, which contains 80 different [object categories](https://cocodataset.org/#explore).

We will be using the [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) model developed by Megvii. YOLOX builds on the [YOLO (You Only Look Once)](https://www.youtube.com/watch?v=ag3DLKsl2vk) family of real-time object detection models and implements multiple recent advancements from object detection research. The YOLOX model provides one of the best tradeoffs between accuracy and inference speed at the time of writing.

Check out the video below to see how the model performs in different settings.

{% include youtube.html content="https://youtu.be/opClIrHumzI" %}

To access the [OpenVINO™ Toolkit](https://docs.openvinotoolkit.org/latest/index.html) inside Unity, we need to create a [Dynamic link library (DLL)](https://docs.microsoft.com/en-us/troubleshoot/windows-client/deployment/dynamic-link-library) in Visual Studio. This will contain the code to perform [inference](https://www.intel.com/content/www/us/en/artificial-intelligence/posts/deep-learning-training-and-inference.html) with an object detection model. We can then call functions from this DLL inside a Unity application by importing it as a [native plugin](https://docs.unity3d.com/Manual/NativePlugins.html).

In this first part, we will ensure the prerequisite software is installed on our system and download pretrained object detection models in the OpenVINO [Intermediate Representation](https://docs.openvinotoolkit.org/latest/openvino_docs_MO_DG_IR_and_opsets.html) format, along with some test videos.



# Prerequisites

The following prerequisites are required to complete this tutorial.

## Experience

This tutorial does not assume any prior experience with the OpenVINO™ Toolkit or Unity. However, some basic experience with Unity would be beneficial.

## System Requirements

The target platform for this project is Windows 10 64-bit. The OpenVINO™ Toolkit does not appear to support 32-bit versions. Given that the OpenVINO™ Toolkit is designed for Intel hardware, an Intel CPU and/or GPU is highly recommended.

## Unity

The first prerequisite we will want to set up is Unity. The Unity Editor can be installed through the Unity Hub, which can be downloaded from the link below.

* Unity Hub: ([download](https://store.unity.com/download?ref=personal))

We will be using **Unity 2020 LTS**. The exact version can be downloaded from the links below. 

* Unity LTS Releases: ([download](https://unity3d.com/unity/qa/lts-releases))

* Download Unity 2020.3.18 (LTS): ([download](about:blank))

**Note:** The installation process will also install Visual Studio, one of the other prerequisites.

The tutorial below walks through the basics of Unity, from the installation process all the way to making an Angry Birds clone.

* [How to Make a Game - Unity Beginner Tutorial](https://www.youtube.com/watch?v=Lu76c85LhGY)

 

## Visual Studio

Unity automatically includes Visual Studio when installing the Editor. However it can also be downloaded directly from the link below.

* Visual Studio Community 2019: ([download](https://visualstudio.microsoft.com/))

## Visual C++ Redistributables

The Visual C++ Redistributables should be installed along with Visual Studio. If not, they can be downloaded from the link below.

* Latest C++ Redistributables: ([link](https://support.microsoft.com/en-us/topic/the-latest-supported-visual-c-downloads-2647da03-1eea-4433-9aff-95f26a218cc0))

## CMake

The official OpenVINO™ installation guide lists CMake as a requirement. However, we do not need it for this project. Still, the latest release of CMake 64-bit is available at the link below.

* CMake: [link](https://cmake.org/download/)

**Note:** Make sure to select one of the Add CMake to the system PATH options during the installation process.

![cmake_install_add_to_path](..\images\openvino-unity-plugin\cmake_install_add_to_path.png)



## Python

Python 3.6, 3.7, or 3.8 64-bit are needed to convert a model from [ONNX format](https://onnx.ai/) to OpenVINO's [intermediate representation (IR)](https://docs.openvinotoolkit.org/latest/openvino_docs_MO_DG_IR_and_opsets.html). We can install Python 3.8 from the Windows Store. This method automatically configures the Python installation to be accessible from the command line.

* Windows Store Python 3.8: ([link](https://www.microsoft.com/en-us/p/python-38/9mssztt1n39l?activetab=pivot:overviewtab))

The YOLOX models are already available in OpenVINO IR format, so Python is not required for this tutorial. However, models trained on custom datasets will need to be converted. The steps for converting models from ONNX format to OpenVINO IR are covered in a [previous tutorial](https://christianjmills.com/OpenVINO-Plugin-for-Unity-Tutorial-1/#convert-onnx-model-to-openvino-ir). The YOLOX models are also available in ONNX format on [GitHub](https://github.com/Megvii-BaseDetection/YOLOX/tree/main/demo/ONNXRuntime).

 

## OpenVINO

We now have all the required prerequisites to install OpenVINO. We'll be using OpenVINO 2021.3 for this tutorial. First time users need to fill out a registration form to download the toolkit.

* [Registration Link](https://software.seek.intel.com/openvino-toolkit)

* [Download Link](https://registrationcenter.intel.com/en/products/postregistration/?sn=C5RC-BZX263HW&Sequence=632852&encema=Wg/bUFJY2qspv9ef8QA1f1BOLNxZ1m3iLsVPacdcuTnDhAsIxOgbt1LgCVHooFk3zSUt/6VQWTA=&dnld=t&pass=yes)

 

# Download OpenVINO IR Models

Megvii has already converted several variants of the YOLOX model to OpenVINO IR format and made them available on [GitHub](https://github.com/Megvii-BaseDetection/YOLOX/tree/main/demo/OpenVINO/cpp). Each variant provides a different tradeoff between accuracy and inference speed.

Download the folder containing the models from the link below. We will be using this folder in the final Unity application.

* [Google Drive Link](https://drive.google.com/file/d/1N4GuHcKyBpDzJQ1r0LulzD3KRE3GRnAe/view?usp=sharing)

Each variant of the model has three files associated with it:

* yolox_10.bin

* yolox_10.mapping

* yolox_10.xml

 

We will need the .bin and .xml files. The .xml files describe the network topology, including the layer operations and flow of data through the network. Here is a snippet from the top of an .xml file.

```xml
<?xml version="1.0" ?>
<net name="yolox_10" version="10">
    <layers>
        <layer id="0" name="inputs" type="Parameter" version="opset1">
            <data element_type="f16" shape="1, 3, 640, 640"/>
            <output>
                <port id="0" names="inputs" precision="FP16">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>640</dim>
                    <dim>640</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="Slice_4/Concat592120366" type="Const" version="opset1">
            <data element_type="i64" offset="0" shape="4" size="32"/>
            <output>
                <port id="0" precision="I64">
                    <dim>4</dim>
                </port>
            </output>
        </layer>
```

The `.bin` file stores the constant values for the model learned during the training process.

# Download Test Videos

We'll be using several different videos to test the models' performance on different object classes. These videos are available on [Pexels](https://www.pexels.com/), a free stock photos & videos site. Download the video in 1080p or Full HD when available. We will not be testing all 80 object classes, but these should provide a general idea of how the models perform in different settings. Feel free to try other videos as well.

1. [airplane](https://www.pexels.com/video/an-airplane-landing-5237590/)

2. [bicycle](https://www.pexels.com/video/man-riding-a-bike-854217/)

3. [bird](https://www.pexels.com/video/close-up-of-a-young-chick-2098873/)

4. [boat](https://www.pexels.com/video/drone-footage-of-a-boat-sailing-8916888/)

5.     [bus](https://www.pexels.com/video/footage-of-the-street-with-the-bus-passing-by-3474308/)

6.     [cat](https://www.pexels.com/video/a-playful-cute-kitten-1722593/)

7.     [person](https://www.pexels.com/video/a-woman-yoga-exercises-at-home-5381485/)

8.     [skateboard](https://www.pexels.com/video/teens-riding-skateboard-doing-grind-rail-5039831/)

9.     [train](https://www.pexels.com/video/a-train-leaving-a-station-3807783/)

10.   [zebra](https://www.pexels.com/video/drone-footage-of-zebra-running-on-the-field-6168935/)



# Conclusion

That takes care of the required setup. In the next part, we will cover how to create a [Dynamic link library (DLL)](https://docs.microsoft.com/en-us/troubleshoot/windows-client/deployment/dynamic-link-library) in Visual Studio to perform [inference](https://www.intel.com/content/www/us/en/artificial-intelligence/posts/deep-learning-training-and-inference.html) with an OpenVINO IR model.

**Project Resources:**

[GitHub Repository](https://github.com/cj-mills/Unity-OpenVINO-YOLOX)



### Next: [Part 2](https://christianjmills.com/OpenVINO-Object-Detection-for-Unity-Tutorial-2/)
