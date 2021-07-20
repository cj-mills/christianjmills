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

![unity-create-video-player](..\images\barracuda-posenet-tutorial-v2\part-2\unity-create-video-player.png)

### Set Video Clip

Select the `Video Player` object in the `Hierarchy` tab. Then, drag and drop the `pexels_boardslides` file into the `Video Clip` parameter in the `Inspector` tab.

![unity-assign-video-clip](..\images\barracuda-posenet-tutorial-v2\part-2\unity-assign-video-clip.png)

### Make the Video Loop

Tick the `Loop` checkbox in the `Inspector` tab to make the video repeat when the project is running.

![unity-loop-video](..\images\barracuda-posenet-tutorial-v2\part-2\unity-loop-video.png)



## Create the Video Screen

We need to make a "screen" in Unity to watch the video. We'll use a [`Quad`](https://docs.unity3d.com/Manual/PrimitiveObjects.html) object for the screen. Right click an empty space in the `Hierarchy` tab, select the `3D Object` section and click `Quad`. We can just name it `VideoScreen`.

![unity-create-quad](..\images\barracuda-posenet-tutorial-v2\part-2\unity-create-quad.png)

Since we are only working in 2D, we can switch the scene to 2D view by clicking the `2D` button in the scene tab.

![unity-toggle-2D-scene-view](..\images\barracuda-posenet-tutorial-v2\part-2\unity-toggle-2D-scene-view.png)











![create-csharp-script](..\images\barracuda-posenet-tutorial-v2\part-2\create-csharp-script.png)







![create-pose-estimator-script](..\images\barracuda-posenet-tutorial-v2\part-2\create-pose-estimator-script.png)







```c#
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Video;
```





```c#
public class PoseEstimator : MonoBehaviour
{
    [Tooltip("The requested webcam dimensions")]
    public Vector2Int webcamDims = new Vector2Int(1280, 720);

    [Tooltip("The requested webcam frame rate")]
    public int webcamFPS = 60;

    [Tooltip("Use webcam feed as input")]
    public bool useWebcam = false;

    [Tooltip("The GameObject with the video player component")]
    public GameObject videoPlayer;

    [Tooltip("The screen for viewing preprocessed images")]
    public Transform videoScreen;
```





```c#
	// Live video input from a webcam
    private WebCamTexture webcamTexture;

    // The dimensions of the current video source
    private Vector2Int videoDims;

    // The source video texture
    private RenderTexture videoTexture;
```





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
            videoScreen.localScale = new Vector3(videoScreen.localScale.x,
                                                 videoScreen.localScale.y, -1f);
        }

        // Apply the new videoTexture to the VideoScreen Gameobject
        videoScreen.gameObject.GetComponent<MeshRenderer>().material.shader =
            Shader.Find("Unlit/Texture");
        videoScreen.gameObject.GetComponent<MeshRenderer>().material.SetTexture(
            "_MainTex", videoTexture);
        // Adjust the VideoScreen dimensions for the new videoTexture
        videoScreen.localScale = new Vector3(width, height, videoScreen.localScale.z);
        // Adjust the VideoScreen position for the new videoTexture
        videoScreen.position = new Vector3(width / 2, height / 2, 1);
    }
```





```c#
	/// <summary>
    /// Resizes and positions the in-game Camera to accommodate the video dimensions
    /// </summary>
    private void InitializeCamera()
    {
        // Get a reference to the Main Camera GameObject
        GameObject mainCamera = GameObject.Find("Main Camera");
        // Adjust the camera position to account for updates to the VideoScreen
        mainCamera.transform.position = new Vector3(videoDims.x / 2, 
                                                    videoDims.y / 2, 
                                                    -(videoDims.x / 2));
        // Increase draw distance for camera
        mainCamera.GetComponent<Camera>().farClipPlane = (videoDims.x / 2) + 100;
        // Render objects with no perspective (i.e. 2D)
        mainCamera.GetComponent<Camera>().orthographic = true;
        // Adjust the camera size to account for updates to the VideoScreen
        mainCamera.GetComponent<Camera>().orthographicSize = videoDims.y / 2;
    }
```





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
            videoPlayer.SetActive(false);

            // Update the videoDims.y
            videoDims.y = (int)webcamTexture.height;
            // Update the videoDims.x
            videoDims.x = (int)webcamTexture.width;

        }
        else
        {
            // Update the videoDims.y
            videoDims.y = (int)videoPlayer.GetComponent<VideoPlayer>().height;
            // Update the videoDims.x
            videoDims.x = (int)videoPlayer.GetComponent<VideoPlayer>().width;
        }

        // Create a new videoTexture using the current video dimensions
        videoTexture = RenderTexture.GetTemporary(videoDims.x, videoDims.y, 24,
                                                  RenderTextureFormat.ARGBHalf);

        // Use new videoTexture for Video Player
        videoPlayer.GetComponent<VideoPlayer>().targetTexture = videoTexture;

        // Initialize the videoScreen
        InitializeVideoScreen(videoDims.x, videoDims.y, useWebcam);

        // Adjust the camera based on the source video dimensions
        InitializeCamera();
    }
```





```c#
	// Update is called once per frame
    void Update()
    {
        // Copy webcamTexture to videoTexture if using webcam
        if (useWebcam) Graphics.Blit(webcamTexture, videoTexture);
    }
```





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













## Summary

We now have a video player that we can use to feed input to the PoseNet model. The next post covers how to ____.

**Previous:** [Part 1](https://christianjmills.com/Barracuda-PoseNet-Tutorial-V2-1/)

**Next:** [Part 3](https://christianjmills.com/Barracuda-PoseNet-Tutorial-V2-3/)

**Project Resources:** [GitHub Repository](https://github.com/cj-mills/Barracuda-PoseNet-Tutorial)

