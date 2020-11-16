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

* [Modify PoseNet Script](#modify-posenet-script)
* [Set Inspector Variable](#set-inspector-variable)



## Modify `PoseNet` Script

We can add the option to use a webcam feed by making some modifications to the `PoseNet` script.

### Create `useWebcam` Variable

Open the `PoseNet` script and create a new public `bool` variable. Name the variable `useWebcam` and set the default value to `false`. This will create a checkbox in the `Inspector` tab that we can use to enable and disable the webcam.

![useWebcam_variable](\images\barracuda-posenet-tutorial\useWebcam_variable.png)



### Create `webcamTexture` Variable

We'll use a [`WebCamTexture`](https://docs.unity3d.com/ScriptReference/WebCamTexture.html) variable to store the live video input from our webcam. Name the variable `webcamTexture`.

![webcamTexture_variable](\images\barracuda-posenet-tutorial\webcamTexture_variable.png)



### Set Up Webcam Feed

We'll prepare the webcam feed at the top of the `Start()` method.

#### Initialize the `webcamTexture`

First, initialize the `webcamTexture`. We'll use the first video input device Unity finds. If you have more than one webcam attached, you'll need to [specify](https://docs.unity3d.com/ScriptReference/WebCamTexture-ctor.html) the device name.

#### Flip the `VideoScreen`

Next, we need to adjust the rotation and scale of the `VideoScreen` object. The webcam feed doesn't mirror the user by default. For example, the user's right arm appears on the left side of the screen. This can be disorienting when looking at the generated pose skeleton. We'll flip the `VideoScreen` to compensate.

#### Start the Camera

We'll use the `webcamTexture.Play()` method to start the camera.

### Deactivate the Video Player

Finally, we'll deactivate the `Video Player` as it's not being used. 

#### Completed Code

![initialize_webcam_start_method](\images\barracuda-posenet-tutorial\initialize_webcam_start_method.png)





### Get `webcamTexture` Data



![useWebcam_update_method](\images\barracuda-posenet-tutorial\useWebcam_update_method.png)



### Flip Key Point Locations



![useWebcam_processOutput_method](\images\barracuda-posenet-tutorial\useWebcam_processOutput_method.png)



#### A Inefficient Alternative



![flipImage_computeShader](\images\barracuda-posenet-tutorial\flipImage_computeShader.png)



![flipImage_method](\images\barracuda-posenet-tutorial\flipImage_method.png)







## Set Inspector Variable





![enable_useWebcam_inspector](\images\barracuda-posenet-tutorial\enable_useWebcam_inspector.PNG)







