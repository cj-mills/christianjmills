---
title: Barracuda PoseNet Tutorial Pt. 7
layout: post
toc: false
description: This post covers how to use a webcam feed as input for the PoseNet model.
categories: [unity, tutorial]
hide: true
search_exclude: false
---

### Previous: [Part 6](https://christianjmills.com/unity/tutorial/2020/11/14/Barracuda-PoseNet-Tutorial-6.html)

* 

## Add Webcam Variables



### Create `useWebcam` Variable



![useWebcam_variable](\images\barracuda-posenet-tutorial\useWebcam_variable.png)





### Create `webcamTexture` Variable





![webcamTexture_variable](\images\barracuda-posenet-tutorial\webcamTexture_variable.png)





## Set Up Webcam Feed



![initialize_webcam_start_method](\images\barracuda-posenet-tutorial\initialize_webcam_start_method.png)





## Get `webcamTexture` Data



![useWebcam_update_method](\images\barracuda-posenet-tutorial\useWebcam_update_method.png)



## Flip Key Point Locations



![useWebcam_processOutput_method](\images\barracuda-posenet-tutorial\useWebcam_processOutput_method.png)



### A Inefficient Alternative



![flipImage_computeShader](\images\barracuda-posenet-tutorial\flipImage_computeShader.png)



![flipImage_method](\images\barracuda-posenet-tutorial\flipImage_method.png)







## Set Inspector Variable





![enable_useWebcam_inspector](\images\barracuda-posenet-tutorial\enable_useWebcam_inspector.PNG)







