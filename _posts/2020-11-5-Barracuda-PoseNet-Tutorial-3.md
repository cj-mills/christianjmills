---
title: Barracuda PoseNet Tutorial Pt.3
layout: post
toc: false
description: This post covers how to perform inference with the PoseNet model.
categories: [unity, tutorial]
hide: true
search_exclude: false
---

### [Part 1](https://christianjmills.com/unity/tutorial/2020/10/25/Barracuda-PoseNet-Tutorial-1.html) [Part 2](https://christianjmills.com/unity/tutorial/2020/10/25/Barracuda-PoseNet-Tutorial-2.html)

* [Install Barracuda Package](#install-barracuda-package)
* [Import PoseNet Model](#import-posenet-model)

## Install Barracuda Package

Select the `Package Manager` tab in the Unity editor.

![select_package_manager_tab](\images\barracuda-posenet-tutorial\select_package_manager_tab.png)

Type `Barracuda` into the search box.

![barracuda_search](\images\barracuda-posenet-tutorial\barracuda_search.PNG)

Click the `Install` button to install the package.

![barracuda_install](\images\barracuda-posenet-tutorial\barracuda_install.PNG)

Wait for Unity to install the dependencies.

![barracuda_installation_progress](\images\barracuda-posenet-tutorial\barracuda_installation_progress.PNG)



## Import PoseNet Model

Now we can import the model into Unity. The Barracuda dev team has focused on supporting the [ONNX](https://onnx.ai/) format for models. We aren't able to directly import models from TensorFlow or PyTorch. I've already converted the PoseNet model to ONNX. You can check out my tutorial for converting TensorFlow SavedModels to ONNX ([here](https://christianjmills.com/tensorflow/onnx/tutorial/2020/10/21/How-to-Convert-a-TensorFlow-SavedModel-to-ONNX.html)). PyTorch provides built-in support for ONNX ([link](https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html)).

Unity supports directly loading [ONNX](https://onnx.ai/) models.

The model has been converted to the ONNX format.



You can download the PoseNet model from this ([link](https://drive.google.com/file/d/1oKrlraI3m3ecme-pAvAh25-Jzzu86sv_/view?usp=sharing)).