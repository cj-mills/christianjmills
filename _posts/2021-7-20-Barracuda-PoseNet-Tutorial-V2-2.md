---
title: Barracuda PoseNet Tutorial 2nd Edition Pt. 2
layout: post
toc: false
comments: true
description: This pose covers how to set up a video player in Unity. We'll be using the video player to check the accuracy of the PoseNet model.
categories: [unity,barracuda,tutorial]
hide: true
permalink: /:title/
search_exclude: false
---

* [Overview](#overview)
* [Create the Video Player](#create-the-video-player)
* [Create the Video Screen](#create-the-video-screen)
* [Summary](#summary)



## Overview

This post demonstrates how to play and view videos inside Unity from both video files and a webcam. We'll later perform pose estimation on individual frames while the video is playing. We can gauge the model's accuracy by comparing the estimated key point locations to the source video.

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

We now have a video player that we can use to feed input to the PoseNet model. The next post covers how to ____.

### [GitHub Repository](https://github.com/cj-mills/Barracuda-PoseNet-Tutorial)

## Next: [Part 2](https://christianjmills.com/Barracuda-PoseNet-Tutorial-2/)

