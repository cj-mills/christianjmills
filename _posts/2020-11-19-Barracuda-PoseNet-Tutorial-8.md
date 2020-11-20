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

* [Modify PoseNet Script](#modify-posenet-script)
* [Set Inspector Variables](#set-inspector-variables)

## Modify `PoseNet` Script

We can add the ability to handle input with different aspect ratios by once again making some modifications to the `PoseNet` script.

### Add `UnityEngine.Video` Dependency

We'll first add a new `using` statement at the top of the `PoseNet` script. We need the `UnityEngine.Video` package to work with the `Video Player` object.

![import_unityengine_video](\images\barracuda-posenet-tutorial\import_unityengine_video.png)

### Add Webcam Variables

Unity defaults to a resolution of `640 x 480` for webcams. There is no built-in method that returns the frame rate. However, the default does not appear to be over 30fps.  We can request a resolution and frame rate when initializing the `webcamTexture`. Unity should accept the requested settings as long as the camera supports them.

#### Create `webcamHeight` Variable

Add a new public `int` variable so we can adjust the camera height from the `Inspector` tab. Name the variable `webcamHeight`. My webcam supports 720p at 60fps so I've set the default value to `720`.

![webcamHeight_variable](\images\barracuda-posenet-tutorial\webcamHeight_variable.png)

#### Create `webcamWidth` Variable

Next, create a variable for the camera's width and name it `webcamWidth`. I've set the default value to `1280`.

![webcamWidth_variable](\images\barracuda-posenet-tutorial\webcamWidth_variable.png)

#### Create `webcamFPS` Variable

We'll also add a variable to set the frame rate for the camera and name it `webcamFPS`. Set the default value to `60`.

![webcam_fps_variable](\images\barracuda-posenet-tutorial\webcam_fps_variable.png)



### Add Video Resolution Variables

Next, we need to create a couple of private `int` variables to store the dimensions of the video source. Name the variables `videoHeight` and `videoWidth`.

![video_resolution_variables](\images\barracuda-posenet-tutorial\video_resolution_variables.png)



### Get Reference to `Video Player`

![find_video_player](\images\barracuda-posenet-tutorial\find_video_player.png)



### Get Webcam Resolution

![get_webcam_resolution](\images\barracuda-posenet-tutorial\get_webcam_resolution.png)



### Get Video Clip Dimensions

![get_video_clip_dimensions](\images\barracuda-posenet-tutorial\get_video_clip_dimensions.png)

### Replace `videoTexture`

![replace_videoTexture](\images\barracuda-posenet-tutorial\replace_videoTexture.png)

### Update `VideoScreen`

![update_videoScreen](\images\barracuda-posenet-tutorial\update_videoScreen.png)

### Adjust `Main Camera`

![adjust_main_camera](\images\barracuda-posenet-tutorial\adjust_main_camera.png)

### Complete `Start()` Method

![start_method_dynamic_video](\images\barracuda-posenet-tutorial\start_method_dynamic_video_4.png)



### Update `ProcessOutput()` Method



#### Update Scaling Calculations



![update_scaling_calculations](\images\barracuda-posenet-tutorial\update_scaling_calculations.png)





#### Update Key Point Calculations



![update_key_point_calculations_part1](\images\barracuda-posenet-tutorial\update_key_point_calculations_part1.png)



![update_key_point_calculations_part2](\images\barracuda-posenet-tutorial\update_key_point_calculations_part2.png)



## Set Inspector Variables

