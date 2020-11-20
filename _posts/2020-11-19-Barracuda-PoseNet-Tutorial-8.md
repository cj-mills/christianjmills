---
title: Barracuda PoseNet Tutorial Pt. 8
layout: post
toc: false
description: This post covers how to dynamically handle video input with different aspect ratios.
categories: [unity, tutorial]
hide: true
search_exclude: false
---

### Previous: [Part 7](https://christianjmills.com/unity/tutorial/2020/11/15/Barracuda-PoseNet-Tutorial-7.html)

* 

## Modify `PoseNet` Script

We can add the ability to handle input with different aspect ratio by once again making some modifications to the `PoseNet` script.



### Add Webcam Variables

#### Create `webcamHeight` Variable

![webcamHeight_variable](\images\barracuda-posenet-tutorial\webcamHeight_variable.png)

#### Create `webcamWidth` Variable

![webcamWidth_variable](\images\barracuda-posenet-tutorial\webcamWidth_variable.png)

#### Create `webcamFPS` Variable

![webcam_fps_variable](\images\barracuda-posenet-tutorial\webcam_fps_variable.png)



### Add Video Resolution Variables

#### Create `videoHeight` Variable

#### Create `videoWidth` Variable

![video_resolution_variables](\images\barracuda-posenet-tutorial\video_resolution_variables.png)



### Add `UnityEngine.Video` Dependency



### Get Webcam Resolution



### Get Video Clip Dimensions



### Replace `videoTexture`



### Adjust `VideoScreen`



### Adjust `Main Camera`

![start_method_dynamic_video](\images\barracuda-posenet-tutorial\start_method_dynamic_video_4.png)

