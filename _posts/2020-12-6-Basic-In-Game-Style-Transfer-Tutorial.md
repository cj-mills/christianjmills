---
title: Basic In-Game Style Transfer Tutorial
layout: post
toc: false
comments: true
description: This post provides a basic method for performing in-game style transfer.
categories: [unity, tutorial]
hide: true
permalink: /:title/
search_exclude: false
---

* [Introduction](#introduction)
* [Select a Unity Project](#select-a-unity-project)
* [Install Barracuda Package](#install-barracuda-package)
* [Create Style Transfer Folder](#create-style-transfer-folder)
* [Import Models](#import-models)
* [Summary](#summary)

## Introduction

Unity has finally released the in-game style transfer project they've been [teasing](https://docs.unity3d.com/Packages/com.unity.barracuda@1.0/manual/images/BarracudaLanding.png) in the Barracuda [documentation](https://docs.unity3d.com/Packages/com.unity.barracuda@1.0/manual/index.html). Their implementation is slightly more polished than my early attempts described in my initial [post](https://christianjmills.com/unity/style_transfer/2020/10/19/In-Game-Style-Transfer.html). And by slightly, I mean they seem to have addressed every major complaint I had with my implementation. Be sure to check out their sample [project](https://github.com/UnityLabs/barracuda-style-transfer) as well as the accompanying blog [post](https://blogs.unity3d.com/2020/11/25/real-time-style-transfer-in-unity-using-deep-neural-networks/).

It's exciting that Unity has started releasing projects that explore alternative uses for the Barracuda library. Hopefully, they'll explore other deep learning applications in future projects. I would love to see projects that use GANs for dynamically generating in-game content.

I plan to work on a more sophisticated implementation for in-game style transfer in the future, perhaps using some tricks from Unity's implementation. However, I wanted to start with a basic implementation to serve as a baseline. 

This tutorial will cover how to use trained models from the [`fast_neural_style`](https://github.com/pytorch/examples/tree/master/fast_neural_style) project provided by PyTorch. The models take in regular images and return stylized versions. We'll get our input images from the in-game camera and display the stylized output to the user.

**Important:** This is meant as a simple proof of concept and requires a powerful GPU to get playable frame rates. An RTX 20-series card or newer is recommended.



## Select a Unity Project

I'll be using the [Kinematica_Demo](https://github.com/Unity-Technologies/Kinematica_Demo/) project provided by Unity for this tutorial. It provides a great character model for testing different styles. However, feel free to follow along with a different project. This one is a bit large and takes a while to open the first time.

### Download Kinematica Demo

You can download the Unity project by clicking on the link below. The zipped folder is approximately 1.2 GB.

* Kinematica_Demo_0.8.0-preview: ([download](https://github.com/Unity-Technologies/Kinematica_Demo/releases/download/0.8.0-preview/Kinematica_Demo_0.8.0-preview.zip))

### Add Project to Unity Hub

Once downloaded, unzip the folder and add the project to Unity Hub using the `Add` button.

![unity_hub_add_project](\images\basic-in-game-style-transfer-tutorial\unity_hub_add_project.png)

### Set the Unity Version

Select a Unity version from the drop-down menu. The demo project was made using Unity `2019.4.5f1`. You can use a later `2019.4` release if you don't have that version installed ([download](unityhub://2019.4.16f1/e05b6e02d63e)).

![set-unity-version-0-0](\images\basic-in-game-style-transfer-tutorial\set-unity-version.png)

### Open the Project

Now we can open the project. We'll be prompted to upgrade the project to the selected Unity version. Click `Confirm` in the popup to upgrade the project. As mentioned earlier, this project takes a while to load the first time.

![set-unity-version](\images\basic-in-game-style-transfer-tutorial\upgrade-unity-version.png)

## Install Barracuda Package

We'll install the Barracuda package once the project has finished loading. Select the Package Manager tab in the Unity editor and type Barracuda into the search box.

![barracuda_search](\images\basic-in-game-style-transfer-tutorial\barracuda_search.PNG)

Click the `Install` button to install the package.

![barracuda_install](\images\basic-in-game-style-transfer-tutorial\barracuda_install.PNG)

## Create Style Transfer Folder

We'll place all our additions to the project in a new folder called `Style_Transfer`. This will help keep things organized.

![style_transfer_folder](..\images\basic-in-game-style-transfer-tutorial\style_transfer_folder.PNG)

## Import Models

Next, we need to add some style transfer models. PyTorch models need to be exported to the [ONNX](https://onnx.ai/) format before being imported to Unity. Fortunately, PyTorch provides built-in support for exporting to ONNX ([link](https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html)).

### Download ONNX Files

You can download some exported style transfer models from the links below.

* [Mosaic](https://drive.google.com/file/d/1gnWUCTkLmDyUFHzMl7fk9F64vSoZk5jK/view?usp=sharing)

  ![mosaic](\images\basic-in-game-style-transfer-tutorial\mosaic.jpg)

* [Van Gogh Starry Night](https://drive.google.com/file/d/1vL5-NZo0Dn0ijkX5u94WoP_WWnxFIU3o/view?usp=sharing)

![van-gogh-starry-night-google-art-project](\images\basic-in-game-style-transfer-tutorial\van-gogh-starry-night-google-art-project.jpg)

### Import ONNX Files to Assets

Open the `Style_Transfer` folder and make a new folder called `Models`.

![create-models-folder](\images\basic-in-game-style-transfer-tutorial\create-models-folder.png)

Drag and drop the ONNX files into the `Models` folder.

![imported_onnx_files](\images\basic-in-game-style-transfer-tutorial\imported_onnx_files.png)



## Create Render Texture Assets





## Create Compute Shader



## Create StyleTransfer Script







## Summary





