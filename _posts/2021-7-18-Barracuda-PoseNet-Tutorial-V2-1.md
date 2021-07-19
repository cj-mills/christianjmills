---
title: Barracuda PoseNet Tutorial 2nd Edition Pt. 1
layout: post
toc: false
comments: true
description: This tutorial series provides step-by-step instructions for how to perform human pose estimation in Unity with the Barracuda inference library.
categories: [unity,barracuda,tutorial]
hide: true
permalink: /:title/
search_exclude: false
---

* [Introduction](#introduction)
* [Prerequisites](#prerequisites)
* [Create a New Project](#create-a-new-project)
* [Import Video Files](#import-video-files)
* [Create the Video Player](#create-the-video-player)
* [Create the Video Screen](#create-the-video-screen)
* [Test the Video Player](#test-the-video-player)
* [Summary](#summary)





\> youtube: https://youtu.be/nEoCAVl6yAI



<video width=auto height=auto src="..\videos\multipose-demo-1.mp4" controls></video>

## Introduction

This tutorial series provides step-by-step instructions for how to perform human [pose estimation](https://www.fritz.ai/pose-estimation/) in [Unity](https://unity.com/) with the [Barracuda](https://docs.unity3d.com/Packages/com.unity.barracuda@2.1/manual/index.html) inference library. We'll be using a pretrained [PoseNet](https://medium.com/tensorflow/real-time-human-pose-estimation-in-the-browser-with-tensorflow-js-7dd0bc881cd5) model to estimate the 2D locations of key points on the bodies of one or more individuals in a video frame. We will then use the output from the model to control the locations of [`GameObjects`](https://docs.unity3d.com/ScriptReference/GameObject.html) in a scene.

## Overview

This post demonstrates how to play and view videos inside Unity. We'll later perform pose estimation on individual frames while the video is playing. We can gauge the model's accuracy by comparing the estimated key point locations to the source video.

## Prerequisites

I recommend checking the following prerequisites if you want to follow along on your own computer.

### Unity

This tutorial assumes that you have Unity installed. You can get acquainted with Unity by clicking on one of the tutorials listed below.

#### How to Make a Game - Unity Beginner Tutorial

* [Unity 2020.1](https://www.youtube.com/watch?v=Lu76c85LhGY)

**Note:** You can download the exact version of Unity used for this tutorial by clicking the link below. 

* [Unity 2020.3.14](unityhub://2020.3.14f1/d0d1bb862f9d)

### Hardware

We'll be performing [inference](https://www.intel.com/content/www/us/en/artificial-intelligence/posts/deep-learning-training-and-inference.html) on the GPU for this series. If possible, use a graphics card from a recent generation.

## Create a New Project

First, we need to create a new Unity project. We can use the default 3D template.

![create-project](..\images\barracuda-posenet-tutorial-v2\part-1\create-project.png)



## Install Barracuda Package



![open-package-manager](..\images\barracuda-posenet-tutorial-v2\part-1\open-package-manager.png)





![update-visual-studio-editor](..\images\barracuda-posenet-tutorial-v2\part-1\update-visual-studio-editor.png)





![package-manager-open-advanced-settings](..\images\barracuda-posenet-tutorial-v2\part-1\package-manager-open-advanced-settings.png)





![package-manager-enable-preview-packages](..\images\barracuda-posenet-tutorial-v2\part-1\package-manager-enable-preview-packages.png)





![package-manager-enable-preview-packages-popup](..\images\barracuda-posenet-tutorial-v2\part-1\package-manager-enable-preview-packages-popup.png)





![package-manager-add-git-package](..\images\barracuda-posenet-tutorial-v2\part-1\package-manager-add-git-package.png)





![package-manager-add-barracuda-git-package](..\images\barracuda-posenet-tutorial-v2\part-1\package-manager-add-barracuda-git-package.png)





![barracuda-package-see-other-versions](..\images\barracuda-posenet-tutorial-v2\part-1\barracuda-package-see-other-versions.png)







![barracuda-select-latest-version](..\images\barracuda-posenet-tutorial-v2\part-1\barracuda-select-latest-version.png)





![burst-package-update-detected](..\images\barracuda-posenet-tutorial-v2\part-1\burst-package-update-detected.png)











## Import Video Files

We'll be using these two videos available on [Pexels](https://www.pexels.com/), a free stock photos & videos site. The first one is for testing single pose estimation and only has one person in frame at a time. The second video is meant for testing multipose estimation and has several individuals in frame at varying distances from the camera. Download the videos in `Full HD` resolution.

1. [Two Young Men Doing a Boardslide Over a Railing](https://www.pexels.com/video/two-young-men-doing-a-boardslide-over-a-railing-4824358/)

   **Note:** Renamed to `pexels_boardslides`

2. [Teens Riding Skateboard Doing Grind Rail](https://www.pexels.com/video/teens-riding-skateboard-doing-grind-rail-5039831/)

   **Note:** Renamed to `pexels_teens_riding_skateboard_doing_grind_rail`

### Create the Videos Folder

In the `Assets` section, right-click an empty space, select the `Create` option, and click `Folder`. Name the folder `Videos`.

![unity-create-folder](..\images\barracuda-posenet-tutorial-v2\part-1\unity-create-folder.png)

Double-click the `Videos` folder to open it.

### Add Video Files

Drag and drop the two video files from the file explorer into the `Videos` folder.

![unity-add-video-files](..\images\barracuda-posenet-tutorial-v2\part-1\unity-add-video-files.png)







## Import ONNX Models





![unity-add-onnx-models](..\images\barracuda-posenet-tutorial-v2\part-1\unity-add-onnx-models.png)







## Create the Video Player

In the `Hierarchy` tab, right-click an empty area, select the `Video` section, and click `Video Player`. This will create a new `GameObject` called `Video Player`.

![unity-create-video-player](..\images\barracuda-posenet-tutorial-v2\part-1\unity-create-video-player.png)

### Set Video Clip

Select the `Video Player` object in the `Hierarchy` tab. Then, drag and drop the `pexels_boardslides` file into the `Video Clip` parameter in the `Inspector` tab.

![unity-assign-video-clip](..\images\barracuda-posenet-tutorial-v2\part-1\unity-assign-video-clip.png)

### Make the Video Loop

Tick the `Loop` checkbox in the `Inspector` tab to make the video repeat when the project is running.

![unity-loop-video](..\images\barracuda-posenet-tutorial-v2\part-1\unity-loop-video.png)



## Create the Video Screen

We need to make a "screen" in Unity to watch the video. We'll use a [`Quad`](https://docs.unity3d.com/Manual/PrimitiveObjects.html) object for the screen. Right click an empty space in the `Hierarchy` tab, select the `3D Object` section and click `Quad`. We can just name it `VideoScreen`.

![unity-create-quad](..\images\barracuda-posenet-tutorial-v2\part-1\unity-create-quad.png)

Since we are only working in 2D, we can switch the scene to 2D view by clicking the `2D` button in the scene tab.

![unity-toggle-2D-scene-view](..\images\barracuda-posenet-tutorial-v2\part-1\unity-toggle-2D-scene-view.png)





## Summary

We now have a video player that we can use to feed input to the PoseNet model. The next post covers how to prepare input for the model on the GPU.

### [GitHub Repository](https://github.com/cj-mills/Barracuda-PoseNet-Tutorial)

## Next: [Part 2](https://christianjmills.com/Barracuda-PoseNet-Tutorial-2/)

