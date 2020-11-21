---
title: Barracuda PoseNet Tutorial Pt. 8
layout: post
toc: false
description: This post covers how to handle video input with different aspect ratios.
categories: [unity, tutorial]
hide: false
search_exclude: false
---

### Previous: [Part 7](https://christianjmills.com/unity/tutorial/2020/11/15/Barracuda-PoseNet-Tutorial-7.html)

* [Add UnityEngine.Video Namespace](#add-unityenginevideo-namespace)
* [Add Video Resolution Variables](#add-video-resolution-variables)
* [Update Start() Method](#update-start-method)
* [Update ProcessOutput() Method](#update-processoutput-method)
* [Try It Out](#try-it-out)

## Add `UnityEngine.Video` Namespace

Open the `PoseNet` script and add a new [`using directive`](https://docs.microsoft.com/en-us/dotnet/csharp/language-reference/keywords/using-directive) at the top. We need the `UnityEngine.Video` namespace to work with the `Video Player` object.

![import_unityengine_video](\images\barracuda-posenet-tutorial\import_unityengine_video.png)



## Add Webcam Variables

Unity defaults to a resolution of `640 x 480` for webcams. There is no built-in method that returns the frame rate. However, the default does not appear to be over 30fps.  We can request a resolution and frame rate when initializing the `webcamTexture`. Unity should accept the requested settings as long as the camera supports them. Unity defaults to its closest preset if the requested settings are not supported.

### Create `webcamHeight` Variable

Add a new public `int` variable so we can adjust the camera height from the `Inspector` tab. Name the variable `webcamHeight`. My webcam supports 720p at 60fps, so I've set the default value to `720`.

![webcamHeight_variable](\images\barracuda-posenet-tutorial\webcamHeight_variable.png)

### Create `webcamWidth` Variable

Next, create a variable for the camera's width and name it `webcamWidth`. I've set the default value to `1280`.

![webcamWidth_variable](\images\barracuda-posenet-tutorial\webcamWidth_variable.png)

### Create `webcamFPS` Variable

We'll also add a variable to set the frame rate for the camera and name it `webcamFPS`. Set the default value to `60`.

![webcam_fps_variable](\images\barracuda-posenet-tutorial\webcam_fps_variable.png)



## Add Video Resolution Variables

Next, we need to create a couple of private `int` variables to store the dimensions of the video source. Name the variables `videoHeight` and `videoWidth`.

![video_resolution_variables](\images\barracuda-posenet-tutorial\video_resolution_variables.png)



## Update `Start()` Method

We'll make some modifications to the `Start()` method to prepare for input with different aspect ratios.

### Get Reference to `Video Player`

Create a new `GameObject` variable to access the `Video Player` object. Name the variable `videoPlayer` and place it at the top of the `Start()` method. We'll also move the `Transform videoScreen` variable outside of the `if (useWebcam)` statement.

![find_video_player](\images\barracuda-posenet-tutorial\find_video_player.png)

### Get Webcam Resolution

We can't access the webcam resolution until after the webcam has started. Therefore, we'll update the values for `videoHeight` and `videoWidth` after `webcamTexture.Play()` has been called.

![get_webcam_resolution](\images\barracuda-posenet-tutorial\get_webcam_resolution.png)

### Get Video Clip Dimensions

We need to get the video resolution from the `Video Player` object when we're not using a webcam. We can get the height and width by accessing the `VideoPlayer` component.

![get_video_clip_dimensions](\images\barracuda-posenet-tutorial\get_video_clip_dimensions.png)

### Replace `videoTexture`

The dimensions for `videoTexture` can't be updated once it's been created. We'll need to replace it with a new `RenderTexture` that uses the new `videoHeight` and `videoWidth` values.

![replace_videoTexture](\images\barracuda-posenet-tutorial\replace_videoTexture.png)

### Update `VideoScreen`

Next, we need to apply the new `videoTexture` to the `VideoScreen` object. We'll also resize and reposition the `VideoScreen` to account for any changes in the `videoTexture` dimensions. 

![update_videoScreen](\images\barracuda-posenet-tutorial\update_videoScreen_2.png)

### Adjust `Main Camera`

We'll also resize and reposition the `Main Camera` object to account for changes to the `VideoScreen`.

![adjust_main_camera](\images\barracuda-posenet-tutorial\adjust_main_camera.png)



## Update `ProcessOutput()` Method

Finally, we need to modify how we calculate the key point locations.

### Update Scaling Calculations

We'll account for different aspect ratios by first determining whether the height or width of `videoTexture` is larger. We then use that information to calculate the `scale` and `unsqueezeScale` values.

![update_scaling_calculations](\images\barracuda-posenet-tutorial\update_scaling_calculations.png)

### Update Key Point Calculations

We should only scale the `xPos` value by the `unsqueezeScale` if the width larger than the height. Remove that part from the initial `xPos` calculation. We also need to move the `if (useWebcam)` statement.

![update_key_point_calculations_part1](\images\barracuda-posenet-tutorial\update_key_point_calculations_part1.png)

We'll scale either `xPos` or `yPos` by `unsqueezeScale` depending on whether the width or height value for `videoTexture` is larger. Place the `if (useWebcam)` statement after this calculation.

![update_key_point_calculations_part2](\images\barracuda-posenet-tutorial\update_key_point_calculations_part2.png)



## Try It Out

Now we can test the modifications by using some vertical videos from [Pexels](https://www.pexels.com/).

1. [Woman Doing a Jump Rope Exercise](https://www.pexels.com/video/woman-doing-a-jump-rope-exercise-2785536/)
2. [Man Dancing Hip-Hop](https://www.pexels.com/video/man-dancing-hip-hop-2795742/)

![jump_rope_pose_estimation_4](\images\barracuda-posenet-tutorial\jump_rope_pose_estimation_4.gif)







