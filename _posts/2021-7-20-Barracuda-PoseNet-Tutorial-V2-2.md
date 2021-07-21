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
* [Create `PoseEstimator` Script](#create-poseestimator-script)
* [Summary](#summary)



## Overview

This post demonstrates how to play and view videos inside Unity from both video files and a webcam. We'll later perform pose estimation on individual frames while the video is playing. We can gauge the model's accuracy by comparing the estimated key point locations to the source video.



## Create the Video Player

To start, we will create new `GameObject` to play and view a video feed. 

### Create the Video Screen

We will use a [Quad](https://docs.unity3d.com/Manual/PrimitiveObjects.html) object for the screen. Right-click an empty space in the `Hierarchy` tab, select the `3D Object` section and click `Quad`. We can just name it `VideoScreen`.

![unity-create-quad](..\images\barracuda-posenet-tutorial-v2\part-2\unity-create-quad.png)

Since we are only working in 2D, we can switch the scene to 2D view by clicking the `2D` button in the scene tab.

![unity-toggle-2D-scene-view](..\images\barracuda-posenet-tutorial-v2\part-2\unity-toggle-2D-scene-view.png)

This will remove perspective from the scene view and align it with the `VideoScreen`.

![unity-2D-scene-view](..\images\barracuda-posenet-tutorial-v2\part-2\unity-2D-scene-view.png)

We will be updating the `VideoScreen` dimensions in code based on the resolution of the video or webcam feed.

### Add Video Player Component

Unity has a [Video Player component](https://docs.unity3d.com/Manual/class-VideoPlayer.html) that provides the functionality to attach video files to the `VideoScreen`. With the `VideoScreen` object selected in the Hierarchy tab, click the `Add Component` button in the Inspector tab.

![videoScreen-add-component](..\images\barracuda-posenet-tutorial-v2\part-2\videoScreen-add-component.png)

Type `video` into the search box and select `Video Player` from the search results.

![videoScreen-add-video-player-component](..\images\barracuda-posenet-tutorial-v2\part-2\videoScreen-add-video-player-component.png)





### Assign Video Clip

Video files can be assigned by dragging them from the Assets section into the `Video Clip` spot in the Inspector tab. We will start with the `pexels_boardslides` file.

![unity-assign-video-clip](..\images\barracuda-posenet-tutorial-v2\part-2\unity-assign-video-clip.png)

### Make the Video Loop

Tick the `Loop` checkbox in the `Inspector` tab to make the video repeat when the project is running.

![unity-loop-video](..\images\barracuda-posenet-tutorial-v2\part-2\unity-loop-video.png)







## Create `PoseEstimator` Script

We will be adjusting both the `VideoScreen` and `Main Camera` objects in the same script in which we will be executing the PoseNet model.

Create a new folder in the Assets section and name it `Scripts`. Enter the Scripts folder and right-click an empty space. Select `C# Script` in the `Create` submenu and name it `PoseEstimator`.

![create-csharp-script](..\images\barracuda-posenet-tutorial-v2\part-2\create-csharp-script.png)

Double-click the new script to open it in the code editor.

![create-pose-estimator-script](..\images\barracuda-posenet-tutorial-v2\part-2\create-pose-estimator-script.png)



### Add Required Namespace

We first need to add the `UnityEngine.Video` namespace to access the functionality for the `Video Player` component. Add the line `using UnityEngine.Video;` at the top of the script.

```c#
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Video;
```



### Define Public Variables

We can specify a desired resolution and framerate for webcams in Unity. If the provided resolution and framerate is not supported by the hardware, Unity will use a default resolution.

We will specify the desired webcam resolution using a `public Vector2Int` variable called `webcamDims`. Set the default values to `1280x720`.

Next, create a `public int` variable called `webcamFPS` and give it a default value of `60`.

We will use a `public bool` variable to toggle between using a video file or webcam as input for the model. Set the default value to `false` as we will be starting with a video file.

Lastly, create a `public Transform` variable called `videoScreen`. We will use this variable to access the `VideoScreen` object and its `Video Player` component.

```c#
public class PoseEstimator : MonoBehaviour
{
    [Tooltip("The requested webcam dimensions")]
    public Vector2Int webcamDims = new Vector2Int(1280, 720);

    [Tooltip("The requested webcam frame rate")]
    public int webcamFPS = 60;

    [Tooltip("Use webcam feed as input")]
    public bool useWebcam = false;

    [Tooltip("The screen for viewing preprocessed images")]
    public Transform videoScreen;
```

### Define Private Variables

We need a `private` [WebCamTexture](https://docs.unity3d.com/ScriptReference/WebCamTexture.html) variable to access the video feed from a webcam.

We will store the final dimensions from either the video or webcam feed in a `private Vector2Int` variable called `videoDims`.

The last variable we need is a `private RenderTexture` variable called `videoTexture`. This will store the pixel data for the current video or webcam frame.

```c#
// Live video input from a webcam
private WebCamTexture webcamTexture;

// The dimensions of the current video source
private Vector2Int videoDims;

// The source video texture
private RenderTexture videoTexture;
```



### Create `InitializeVideoScreen()` Method

We will update the position, orientation, and size of the `VideoScreen` object in a new method called `InitializeVideoScreen`. The method will take in width and height value along with a `bool` to indicate whether to mirror the screen. When using a webcam, we need to mirror the `VideoScreen` object so that the user's position is mirrored on screen (e.g. their right side is on the right side of the screen).

When `mirrorScreen` is set to `true` the `VideoScreen` will be rotated `180` around the Y-Axis and scaled by `-1` along the Z-Axis.

The default [shader](https://docs.unity3d.com/ScriptReference/Shader.html) assigned to the `VideoScreen` object needs to be replaced with an `Unlit/Texture` shader. This will remove the need for the screen to be lit by an in-game light.

We will then assign the `videoTexture` created earlier as the texture for the `VideoScreen`. This will allow us to access to pixel data for the current video frame.

We can adjust the dimensions of the `VideoScreen` object by updating it's [localScale](https://docs.unity3d.com/ScriptReference/Transform-localScale.html) attribute.

The last step is to reposition the screen based on the the new dimensions, so that the bottom left corner is at `X:0, Y:0, Z:0`. This will simplify the process for updating the positions of objects with the estimated key point locations.

```c#
/// <summary>
/// Prepares the videoScreen GameObject to display the chosen video source.
/// </summary>
/// <param name="width"></param>
/// <param name="height"></param>
/// <param name="mirrorScreen"></param>
private void InitializeVideoScreen(int width, int height, bool mirrorScreen)
{
    if (mirrorScreen)
    {
        // Flip the VideoScreen around the Y-Axis
        videoScreen.rotation = Quaternion.Euler(0, 180, 0);
        // Invert the scale value for the Z-Axis
        videoScreen.localScale = new Vector3(videoScreen.localScale.x, videoScreen.localScale.y, -1f);
    }

    // Apply the new videoTexture to the VideoScreen Gameobject
    videoScreen.gameObject.GetComponent<MeshRenderer>().material.shader = Shader.Find("Unlit/Texture");
    videoScreen.gameObject.GetComponent<MeshRenderer>().material.SetTexture("_MainTex", videoTexture);
    // Adjust the VideoScreen dimensions for the new videoTexture
    videoScreen.localScale = new Vector3(width, height, videoScreen.localScale.z);
    // Adjust the VideoScreen position for the new videoTexture
    videoScreen.position = new Vector3(width / 2, height / 2, 1);
}
```



### Create `InitializeCamera()` Method

Once the `VideoScreen` has been updated, we need to resize and reposition the in-game camera. We will do so in a new method called `InitializeCamera`. 

We can access the `Main Camera` object with `GameObject.Find("Main Camera")`.  We will set the `X` and `Y` coordinates to the same as the `VideoScreen`.

The camera also needs to be set to `orthographic` mode to remove perspective.

Lastly, we need to update the size of the camera. The `orthographicSize` attribute is actually the half size, so we need to divide `videoDims.y` by `2` as well.

```c#
/// <summary>
/// Resizes and positions the in-game Camera to accommodate the video dimensions
/// </summary>
private void InitializeCamera()
{
    // Get a reference to the Main Camera GameObject
    GameObject mainCamera = GameObject.Find("Main Camera");
    // Adjust the camera position to account for updates to the VideoScreen
    mainCamera.transform.position = new Vector3(videoDims.x / 2, videoDims.y / 2, 0f);
    // Render objects with no perspective (i.e. 2D)
    mainCamera.GetComponent<Camera>().orthographic = true;
    // Adjust the camera size to account for updates to the VideoScreen
    mainCamera.GetComponent<Camera>().orthographicSize = videoDims.y / 2;
}
```



### Modify `Start()` Method



```c#
// Start is called before the first frame update
void Start()
{
    if (useWebcam)
    {
        // Create a new WebCamTexture
        webcamTexture = new WebCamTexture(webcamDims.x, webcamDims.y, webcamFPS);

        // Start the Camera
        webcamTexture.Play();

        // Deactivate the Video Player
        videoScreen.gameObject.SetActive(false);

        // Update the videoDims.y
        videoDims.y = (int)webcamTexture.height;
        // Update the videoDims.x
        videoDims.x = (int)webcamTexture.width;
    }
    else
    {
        // Update the videoDims.y
        videoDims.y = (int)videoScreen.GetComponent<VideoPlayer>().height;
        // Update the videoDims.x
        videoDims.x = (int)videoScreen.GetComponent<VideoPlayer>().width;
    }

    // Create a new videoTexture using the current video dimensions
    videoTexture = RenderTexture.GetTemporary(videoDims.x, videoDims.y, 24, RenderTextureFormat.ARGBHalf);

    // Use new videoTexture for Video Player
    videoScreen.GetComponent<VideoPlayer>().targetTexture = videoTexture;

    // Initialize the videoScreen
    InitializeVideoScreen(videoDims.x, videoDims.y, useWebcam);

    // Adjust the camera based on the source video dimensions
    InitializeCamera();
}
```



### Modify `Update()` Method



```c#
// Update is called once per frame
void Update()
{
    // Copy webcamTexture to videoTexture if using webcam
    if (useWebcam) Graphics.Blit(webcamTexture, videoTexture);
}
```



### Create `OnDisable()` Method



```c#
// OnDisable is called when the MonoBehavior becomes disabled or inactive
private void OnDisable()
{
    RenderTexture.ReleaseTemporary(videoTexture);
}
```







![create-empty-gameobject](..\images\barracuda-posenet-tutorial-v2\part-2\create-empty-gameobject.png)





![attach-pose-estimator-script](..\images\barracuda-posenet-tutorial-v2\part-2\attach-pose-estimator-script.png)





![populate-pose-estimator-component](..\images\barracuda-posenet-tutorial-v2\part-2\populate-pose-estimator-component.png)







![video-player-test-3](..\images\barracuda-posenet-tutorial-v2\part-2\video-player-test-3.gif)





## Summary

We now have a video player that we can use to feed input to the PoseNet model. The next post covers how to ____.

**Previous:** [Part 1](https://christianjmills.com/Barracuda-PoseNet-Tutorial-V2-1/)

**Next:** [Part 3](https://christianjmills.com/Barracuda-PoseNet-Tutorial-V2-3/)

**Project Resources:** [GitHub Repository](https://github.com/cj-mills/Barracuda-PoseNet-Tutorial)

