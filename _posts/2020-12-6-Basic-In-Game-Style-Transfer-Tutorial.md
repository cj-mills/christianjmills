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
* [Import Models](#import-models)
* [Summary](#summary)

## Introduction

Unity has finally released the in-game style transfer project they've been teasing in the Barracuda documentation. Their implementation is slightly more polished than my early attempts described in my initial post. And by slightly, I mean they seem to have addressed every major complaint I had with my initial implementation. Be sure to check out there accompanying blog post ([link](https://blogs.unity3d.com/2020/11/25/real-time-style-transfer-in-unity-using-deep-neural-networks/)).

It's exciting that Unity has started releasing projects that explore alternative uses for the Barracuda library. Hopefully, they'll explore other deep learning applications in future projects. I would love to see projects that use GANs for dynamically generating in-game content.

I plan to work on a more sophisticated implementation for in-game style transfer in the future, perhaps using some tricks from Unity's implementation. However, I wanted to start with a basic implementation to serve as a baseline. 

This tutorial will cover how to use models generated with this example [project](https://github.com/pytorch/examples/tree/master/fast_neural_style) provided by PyTorch. The models take in regular images and return stylized versions. We'll get our input images from the in-game camera and display the stylized output to the user.

**Important:** This is meant as a simple proof of concept and requires a powerful GPU to get playable frame rates. An RTX 20-series card or newer is recommended.



## Select a Unity Project

I'll be using the [Kinematica_Demo](https://github.com/Unity-Technologies/Kinematica_Demo/) project provided by Unity for this tutorial. It provides a great character model for testing different styles. However, feel free to follow along with a different project. This one is a bit large and takes a while to open the first time. 

### Download Kinematica Demo

You can download the Unity project by clicking on this ([link](https://github.com/Unity-Technologies/Kinematica_Demo/releases/download/0.8.0-preview/Kinematica_Demo_0.8.0-preview.zip)). The zipped folder is approximately 1.2 GB.

![download_kinematica_demo](\images\basic-in-game-style-transfer-tutorial\download_kinematica_demo.png)

### Add Project to Unity Hub

Once downloaded, unzip the folder and add the project to Unity Hub using the `Add` button.

![unity_hub_add_project](\images\basic-in-game-style-transfer-tutorial\unity_hub_add_project.png)

### Set the Unity Version

The demo project was made using Unity `2019.4.5f1`. You can just use the latest `2019.4` release if you don't have one already installed ([download](unityhub://2019.4.16f1/e05b6e02d63e)). Select the target Unity version from the drop-down menu.

![set-unity-version-0-0](\images\basic-in-game-style-transfer-tutorial\set-unity-version.png)

### Open the Project

Now we can open the project. We'll be prompted to upgrade the project to the selected Unity version. Click `Confirm` in the popup to upgrade the project. As mentioned earlier, this project takes a while to load the first time.

![set-unity-version](\images\basic-in-game-style-transfer-tutorial\upgrade-unity-version.png)

## Install Barracuda Package

We'll install the Barracuda package once the project has finished loading. Select the Package Manager tab in the Unity editor. Type Barracuda into the search box.

![barracuda_search](\images\basic-in-game-style-transfer-tutorial\barracuda_search.PNG)

Click the `Install` button to install the package.

![barracuda_install](\images\basic-in-game-style-transfer-tutorial\barracuda_install.PNG)



## Import Models

We'll place all our additions to the project in a new folder called `Style_Transfer`. This will help keep things organized. 

### Download ONNX Files

You can download some converted style transfer models from the links below.

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





