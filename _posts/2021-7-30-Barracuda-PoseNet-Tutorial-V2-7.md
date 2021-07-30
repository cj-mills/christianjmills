---
title: Barracuda PoseNet Tutorial 2nd Edition Pt. 7 - Unpublished
layout: post
toc: false
comments: true
description: This post covers how to poses skeletons and manipulate them using output from the model.
categories: [unity,barracuda,tutorial]
hide: true
permalink: /:title/
search_exclude: false
---

* [Overview](#overview)
* [Create `PoseSkeleton` Script](#create-poseskeleton-script)
* [Update `PoseEstimator` Script](#update-poseestimator-script)
* [Summary](#summary)



## Overview

In this post, we will cover how to create pose skeletons so that we can compare the estimated key point locations to the source video feed.



## Create `PoseSkeleton` Script



### Add Required Namespace



```c#
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;
```





### Add Variables



```c#
// The list of key point GameObjects that make up the pose skeleton
public Transform[] keypoints;

// The GameObjects that contain data for the lines between key points
private GameObject[] lines;

// The line renderers the draw the lines between key points
private LineRenderer[] lineRenderers;

// The names of the body parts that will be detected by the PoseNet model
private static string[] partNames = new string[]{
    "nose", "leftEye", "rightEye", "leftEar", "rightEar", "leftShoulder",
    "rightShoulder", "leftElbow", "rightElbow", "leftWrist", "rightWrist",
    "leftHip", "rightHip", "leftKnee", "rightKnee", "leftAnkle", "rightAnkle"
};

private static int NUM_KEYPOINTS = partNames.Length;

// The pairs of key points that should be connected on a body
private Tuple<int, int>[] jointPairs = new Tuple<int, int>[]{
    // Nose to Left Eye
    Tuple.Create(0, 1),
    // Nose to Right Eye
    Tuple.Create(0, 2),
    // Left Eye to Left Ear
    Tuple.Create(1, 3),
    // Right Eye to Right Ear
    Tuple.Create(2, 4),
    // Left Shoulder to Right Shoulder
    Tuple.Create(5, 6),
    // Left Shoulder to Left Hip
    Tuple.Create(5, 11),
    // Right Shoulder to Right Hip
    Tuple.Create(6, 12),
    // Left Shoulder to Right Hip
    Tuple.Create(5, 12),
    // Rigth Shoulder to Left Hip
    Tuple.Create(6, 11),
    // Left Hip to Right Hip
    Tuple.Create(11, 12),
    // Left Shoulder to Left Elbow
    Tuple.Create(5, 7),
    // Left Elbow to Left Wrist
    Tuple.Create(7, 9), 
    // Right Shoulder to Right Elbow
    Tuple.Create(6, 8),
    // Right Elbow to Right Wrist
    Tuple.Create(8, 10),
    // Left Hip to Left Knee
    Tuple.Create(11, 13), 
    // Left Knee to Left Ankle
    Tuple.Create(13, 15),
    // Right Hip to Right Knee
    Tuple.Create(12, 14), 
    // Right Knee to Right Ankle
    Tuple.Create(14, 16)
};

// Colors for the skeleton lines
private Color[] colors = new Color[] {
    // Head
    Color.magenta, Color.magenta, Color.magenta, Color.magenta,
    // Torso
    Color.red, Color.red, Color.red, Color.red, Color.red, Color.red,
    // Arms
    Color.green, Color.green, Color.green, Color.green,
    // Legs
    Color.blue, Color.blue, Color.blue, Color.blue
};

// The width for the skeleton lines
private float lineWidth;

// The material for the key point objects
private Material keypointMat;
```





### Create Constructor



```c#
public PoseSkeleton(float pointScale = 10f, float lineWidth = 5f)
{
    this.keypoints = new Transform[NUM_KEYPOINTS];

    keypointMat = new Material(Shader.Find("Unlit/Color"));
    keypointMat.color = Color.yellow;

    for (int i = 0; i < NUM_KEYPOINTS; i++)
    {
        this.keypoints[i] = GameObject.CreatePrimitive(PrimitiveType.Sphere).transform;
        this.keypoints[i].position = new Vector3(0, 0, 0);
        this.keypoints[i].localScale = new Vector3(pointScale, pointScale, 0);
        this.keypoints[i].gameObject.GetComponent<MeshRenderer>().material = keypointMat;
        this.keypoints[i].gameObject.name = partNames[i];
    }

    this.lineWidth = lineWidth;

    // The number of joint pairs
    int numPairs = keypoints.Length + 1;
    // Initialize the lines array
    lines = new GameObject[numPairs];
    // Initialize the lineRenderers array
    lineRenderers = new LineRenderer[numPairs];

    // Initialize the pose skeleton
    InitializeSkeleton();
}
```





### Create `ToggleKeypoints` Method

```c#
/// <summary>
/// Toggles visibility for the keypoint gameObjects 
/// </summary>
/// <param name="show"></param>
public void ToggleKeypoints(bool show)
{
    foreach (Transform transform in keypoints)
    {
        transform.GetComponent<MeshRenderer>().enabled = show;
    }
}
```





### Create `ToggleLines` Method



```c#
/// <summary>
/// Toggles visibility for the skeleton lines
/// </summary>
/// <param name="show"></param>
public void ToggleLines(bool show)
{
    foreach (LineRenderer lineRenderer in lineRenderers)
    {
        lineRenderer.enabled = show;
    }
}
```





### Create `InitializeLine` Method



```c#
/// <summary>
/// Create a line between the key point specified by the start and end point indices
/// </summary>
/// <param name="pairIndex"></param>
/// <param name="startIndex"></param>
/// <param name="endIndex"></param>
/// <param name="width"></param>
/// <param name="color"></param>
private void InitializeLine(int pairIndex, float width, Color color)
{
    int startIndex = jointPairs[pairIndex].Item1;
    int endIndex = jointPairs[pairIndex].Item2;

    // Create new line GameObject
    string name = $"{keypoints[startIndex].name}_to_{keypoints[endIndex].name}";
    lines[pairIndex] = new GameObject(name);

    // Add LineRenderer component
    lineRenderers[pairIndex] = lines[pairIndex].AddComponent<LineRenderer>();
    // Make LineRenderer Shader Unlit
    lineRenderers[pairIndex].material = new Material(Shader.Find("Unlit/Color"));
    // Set the material color
    lineRenderers[pairIndex].material.color = color;

    // The line will consist of two points
    lineRenderers[pairIndex].positionCount = 2;

    // Set the width from the start point
    lineRenderers[pairIndex].startWidth = width;
    // Set the width from the end point
    lineRenderers[pairIndex].endWidth = width;
}
```





### Create `InitializeSkeleton` Method



```c#
/// <summary>
/// Initialize the pose skeleton
/// </summary>
private void InitializeSkeleton()
{
    for (int i = 0; i < jointPairs.Length; i++)
    {
        InitializeLine(i, lineWidth, colors[i]);
    }
}
```





### Create `UpdateKeyPointPositions` Method



```c#
/// <summary>
/// Update the positions for the key point GameObjects
/// </summary>
/// <param name="keypoints"></param>
/// <param name="sourceScale"></param>
/// <param name="sourceTexture"></param>
/// <param name="mirrorImage"></param>
/// <param name="minConfidence"></param>
public void UpdateKeyPointPositions(Utils.Keypoint[] keypoints,
                                    float sourceScale, RenderTexture sourceTexture, bool mirrorImage, float minConfidence)
{
    // Iterate through the key points
    for (int k = 0; k < keypoints.Length; k++)
    {
        // Check if the current confidence value meets the confidence threshold
        if (keypoints[k].score >= minConfidence / 100f)
        {
            // Activate the current key point GameObject
            this.keypoints[k].GetComponent<MeshRenderer>().enabled = true;
        }
        else
        {
            // Deactivate the current key point GameObject
            this.keypoints[k].GetComponent<MeshRenderer>().enabled = false;
        }

        // Scale the keypoint position to the original resolution
        Vector2 coords = keypoints[k].position * sourceScale;

        // Flip the keypoint position vertically
        coords.y = sourceTexture.height - coords.y;

        // Mirror the x position if using a webcam
        if (mirrorImage) coords.x = sourceTexture.width - coords.x;

        // Update the current key point location
        // Set the z value to -1f to place it in front of the video screen
        this.keypoints[k].position = new Vector3(coords.x, coords.y, -1f);
    }
}
```





### Create `RenderSkeleton` Method



```c#
/// <summary>
/// Draw the pose skeleton based on the latest location data
/// </summary>
public void RenderSkeleton()
{
    // Iterate through the joint pairs
    for (int i = 0; i < jointPairs.Length; i++)
    {
        // Set the GameObject for the starting key point
        Transform startingKeyPoint = keypoints[jointPairs[i].Item1];
        // Set the GameObject for the ending key point
        Transform endingKeyPoint = keypoints[jointPairs[i].Item2];

        // Check if both the starting and ending key points are active
        if (startingKeyPoint.GetComponent<MeshRenderer>().enabled &&
            endingKeyPoint.GetComponent<MeshRenderer>().enabled)
        {
            // Activate the line
            lineRenderers[i].gameObject.SetActive(true);
            // Update the starting position
            lineRenderers[i].SetPosition(0, startingKeyPoint.position);
            // Update the ending position
            lineRenderers[i].SetPosition(1, endingKeyPoint.position);
        }
        else
        {
            // Deactivate the line
            lineRenderers[i].gameObject.SetActive(false);
        }
    }
}
```







## Update `PoseEstimator` Script





### Add Public Variables



```c#
[Tooltip("The size of the pose skeleton key points")]
public float pointScale = 10f;

[Tooltip("The width of the pose skeleton lines")]
public float lineWidth = 5f;

[Tooltip("The minimum confidence level required to display the key point")]
[Range(0, 100)]
public int minConfidence = 70;
```





### Add Private Variables



```c#
// Array of pose skeletons
private PoseSkeleton[] skeletons;
```



### Modify `Start` Method



```c#
// Initialize the list of pose skeletons
if (estimationType == EstimationType.SinglePose) maxPoses = 1;
skeletons = new PoseSkeleton[maxPoses];

// Populate the list of pose skeletons
for (int i = 0; i < maxPoses; i++) skeletons[i] = new PoseSkeleton(pointScale, lineWidth);
```



#### Full Code

```c#
// Start is called before the first frame update
void Start()
{
    if (useWebcam)
    {
        // Limit application framerate to the target webcam framerate
        Application.targetFrameRate = webcamFPS;

        // Create a new WebCamTexture
        webcamTexture = new WebCamTexture(webcamDims.x, webcamDims.y, webcamFPS);

        // Start the Camera
        webcamTexture.Play();

        // Deactivate the Video Player
        videoScreen.GetComponent<VideoPlayer>().enabled = false;

        // Update the videoDims.y
        videoDims.y = webcamTexture.height;
        // Update the videoDims.x
        videoDims.x = webcamTexture.width;
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

    // Initialize the videoScreen
    InitializeVideoScreen(videoDims.x, videoDims.y, useWebcam);

    // Adjust the camera based on the source video dimensions
    InitializeCamera();

    // Adjust the input dimensions to maintain the source aspect ratio
    aspectRatioScale = (float)videoTexture.width / videoTexture.height;
    targetDims.x = (int)(imageDims.y * aspectRatioScale);
    imageDims.x = targetDims.x;

    // Initialize the RenderTexture that will store the processed input image
    rTex = RenderTexture.GetTemporary(imageDims.x, imageDims.y, 24, RenderTextureFormat.ARGBHalf);

    // Initialize the Barracuda inference engine based on the selected model
    InitializeBarracuda();

    // Initialize the list of pose skeletons
    if (estimationType == EstimationType.SinglePose) maxPoses = 1;
    skeletons = new PoseSkeleton[maxPoses];

    // Populate the list of pose skeletons
    for (int i = 0; i < maxPoses; i++) skeletons[i] = new PoseSkeleton(pointScale, lineWidth);
}
```





### Modify `Update` Method



```c#
// The smallest dimension of the videoTexture
int minDimension = Mathf.Min(videoTexture.width, videoTexture.height);

// The value used to scale the key point locations up to the source resolution
float scale = (float)minDimension / Mathf.Min(imageDims.x, imageDims.y);

// Update the pose skeletons
for (int i = 0; i < skeletons.Length; i++)
{
    if (i <= poses.Length - 1)
    {
        skeletons[i].ToggleLines(true);

        // Update the positions for the key point GameObjects
        skeletons[i].UpdateKeyPointPositions(poses[i], scale, videoTexture, useWebcam, minConfidence);
        skeletons[i].RenderSkeleton();
    }
    else
    {
        skeletons[i].ToggleKeypoints(false);
        skeletons[i].ToggleLines(false);
    }
}
```



#### Full Code

```c#
// Update is called once per frame
void Update()
{
    // Copy webcamTexture to videoTexture if using webcam
    if (useWebcam) Graphics.Blit(webcamTexture, videoTexture);

    // Prevent the input dimensions from going too low for the model
    imageDims.x = Mathf.Max(imageDims.x, 64);
    imageDims.y = Mathf.Max(imageDims.y, 64);

    // Update the input dimensions while maintaining the source aspect ratio
    if (imageDims.x != targetDims.x)
    {
        aspectRatioScale = (float)videoTexture.height / videoTexture.width;
        targetDims.y = (int)(imageDims.x * aspectRatioScale);
        imageDims.y = targetDims.y;
        targetDims.x = imageDims.x;
    }
    if (imageDims.y != targetDims.y)
    {
        aspectRatioScale = (float)videoTexture.width / videoTexture.height;
        targetDims.x = (int)(imageDims.y * aspectRatioScale);
        imageDims.x = targetDims.x;
        targetDims.y = imageDims.y;
    }

    // Update the rTex dimensions to the new input dimensions
    if (imageDims.x != rTex.width || imageDims.y != rTex.height)
    {
        RenderTexture.ReleaseTemporary(rTex);
        // Assign a temporary RenderTexture with the new dimensions
        rTex = RenderTexture.GetTemporary(imageDims.x, imageDims.y, 24, rTex.format);
    }

    // Copy the src RenderTexture to the new rTex RenderTexture
    Graphics.Blit(videoTexture, rTex);

    // Prepare the input image to be fed to the selected model
    ProcessImage(rTex);

    // Reinitialize Barracuda with the selected model and backend 
    if (engine.modelType != modelType || engine.workerType != workerType)
    {
        engine.worker.Dispose();
        InitializeBarracuda();
    }

    // Execute neural network with the provided input
    engine.worker.Execute(input);
    // Release GPU resources allocated for the Tensor
    input.Dispose();

    // Decode the keypoint coordinates from the model output
    ProcessOutput(engine.worker);

    // The smallest dimension of the videoTexture
    int minDimension = Mathf.Min(videoTexture.width, videoTexture.height);

    // The value used to scale the key point locations up to the source resolution
    float scale = (float)minDimension / Mathf.Min(imageDims.x, imageDims.y);

    // Update the pose skeletons
    for (int i = 0; i < skeletons.Length; i++)
    {
        if (i <= poses.Length - 1)
        {
            skeletons[i].ToggleLines(true);

            // Update the positions for the key point GameObjects
            skeletons[i].UpdateKeyPointPositions(poses[i], scale, videoTexture, useWebcam, minConfidence);
            skeletons[i].RenderSkeleton();
        }
        else
        {
            skeletons[i].ToggleKeypoints(false);
            skeletons[i].ToggleLines(false);
        }
    }
}
```



## Summary

_.



**Previous:** [Part 6](https://christianjmills.com/Barracuda-PoseNet-Tutorial-V2-6/)

**Project Resources:** [GitHub Repository - Version 1](https://github.com/cj-mills/Barracuda-PoseNet-Tutorial)

