---
title: Barracuda PoseNet Tutorial 2nd Edition Pt. 5 - Unpublished
layout: post
toc: false
comments: true
description: This post covers how to implement the post processing steps for single pose estimation.
categories: [unity,barracuda,tutorial]
hide: true
permalink: /:title/
search_exclude: false
---

* [Overview](#overview)
* [Update `Utils` Script](#update-utils-script)
* [Update `PoseEstimator` Script](#update-poseestimator-script)
* [Summary](#summary)



## Overview

In this post, we will cover how to implement the post processing steps for single pose estimation.



## Update `Utils` Script



```c#
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Barracuda;
```





```c#
// The names of the body parts that will be detected by the PoseNet model
public static string[] partNames = new string[]{
    "nose", "leftEye", "rightEye", "leftEar", "rightEar", "leftShoulder",
    "rightShoulder", "leftElbow", "rightElbow", "leftWrist", "rightWrist",
    "leftHip", "rightHip", "leftKnee", "rightKnee", "leftAnkle", "rightAnkle"
};

public static int NUM_KEYPOINTS = partNames.Length;
```



```c#
/// <summary>
/// Stores the heatmap score, position, and partName index for a single keypoint
/// </summary>
public struct Keypoint
{
    public float score;
    public Vector2 position;
    public int id;

    public Keypoint(float score, Vector2 position, int id)
    {
        this.score = score;
        this.position = position;
        this.id = id;
    }
}
```





```c#
/// <summary>
/// Get the offset values for the provided heatmap indices
/// </summary>
/// <param name="y">Heatmap column index</param>
/// <param name="x">Heatmap row index</param>
/// <param name="keypoint">Heatmap channel index</param>
/// <param name="offsets">Offsets output tensor</param>
/// <returns></returns>
public static Vector2 GetOffsetVector(int y, int x, int keypoint, Tensor offsets)
{
    // Get the offset values for the provided heatmap coordinates
    return new Vector2(offsets[0, y, x, keypoint + NUM_KEYPOINTS], offsets[0, y, x, keypoint]);
}
```





```c#
 /// <summary>
/// Calculate the position of the provided keypoint in the input image
/// </summary>
/// <param name="part"></param>
/// <param name="stride"></param>
/// <param name="offsets"></param>
/// <returns></returns>
public static Vector2 GetImageCoords(Keypoint part, int stride, Tensor offsets)
{
    // The accompanying offset vector for the current coords
    Vector2 offsetVector = GetOffsetVector((int)part.position.y, (int)part.position.x,
                                           part.id, offsets);

    // Scale the coordinates up to the inputImage resolution
    // Add the offset vectors to refine the key point location
    return (part.position * stride) + offsetVector;
}
```







```c#
/// <summary>
/// Determine the estimated key point locations using the heatmaps and offsets tensors
/// </summary>
/// <param name="heatmaps">The heatmaps that indicate the confidence levels for key point locations</param>
/// <param name="offsets">The offsets that refine the key point locations determined with the heatmaps</param>
/// <returns>An array of keypoints for a single pose</returns>
public static Keypoint[] DecodeSinglePose(Tensor heatmaps, Tensor offsets, int stride)
{
    Keypoint[] keypoints = new Keypoint[heatmaps.channels];

    // Iterate through heatmaps
    for (int c = 0; c < heatmaps.channels; c++)
    {
        Keypoint part = new Keypoint();
        part.id = c;

        // Iterate through heatmap columns
        for (int y = 0; y < heatmaps.height; y++)
        {
            // Iterate through column rows
            for (int x = 0; x < heatmaps.width; x++)
            {
                if (heatmaps[0, y, x, c] > part.score)
                {
                    // Update the highest confidence for the current key point
                    part.score = heatmaps[0, y, x, c];

                    // Update the estimated key point coordinates
                    part.position.x = x;
                    part.position.y = y;
                }
            }
        }

        // Calcluate the position in the input image for the current (x, y) coordinates
        part.position = GetImageCoords(part, stride, offsets);

        // Add the current keypoint to the list
        keypoints[c] = part;
    }

    return keypoints;
}
```











## Update `PoseEstimator` Script



### Add Public Variables

```c#
public enum EstimationType
{
    MultiPose,
    SinglePose
}

[Tooltip("The type of pose estimation to be performed")]
public EstimationType estimationType = EstimationType.SinglePose;
```





### Add Private Variables

```c#
// Stores the current estimated 2D keypoint locations in videoTexture
private Utils.Keypoint[][] poses;

// The value used to scale the key point locations up to the source resolution
private float scale;
```







### Create `ProcessOutput` Method



```c#
/// <summary>
/// Obtains the model output and either decodes single or mutlple poses
/// </summary>
/// <param name="engine"></param>
private void ProcessOutput(IWorker engine)
{
    // Get the model output
    Tensor heatmaps = engine.PeekOutput(predictionLayer);
    Tensor offsets = engine.PeekOutput(offsetsLayer);
    Tensor displacementFWD = engine.PeekOutput(displacementFWDLayer);
    Tensor displacementBWD = engine.PeekOutput(displacementBWDLayer);

    // Calculate the stride used to scale down the inputImage
    int stride = (imageDims.y - 1) / (heatmaps.shape.height - 1);
    stride -= (stride % 8);

    if (estimationType == EstimationType.SinglePose)
    {
        poses = new Utils.Keypoint[1][];

        // Determine the key point locations
        poses[0] = Utils.DecodeSinglePose(heatmaps, offsets, stride);
    }
    else
    {
        
    }

    heatmaps.Dispose();
    offsets.Dispose();
    displacementFWD.Dispose();
    displacementBWD.Dispose();
}
```











### Modify `Update` Method



```c#
// The smallest dimension of the videoTexture
int minDimension = Mathf.Min(videoTexture.width, videoTexture.height);

// The value used to scale the key point locations up to the source resolution
scale = (float)minDimension / Mathf.Min(imageDims.x, imageDims.y);

// Decode the keypoint coordinates from the model output
ProcessOutput(engine.worker);
```



#### Full Code



```c#
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


    if (modelType == ModelType.MobileNet)
    {
        preProcessFunction = Utils.PreprocessMobileNet;
    }
    else
    {
        preProcessFunction = Utils.PreprocessResNet;
    }

    // Prepare the input image to be fed to the selected model
    ProcessImage(rTex);

    // Update the rTex dimensions to the new input dimensions
    if (engine.modelType != modelType || engine.workerType != workerType)
    {
        engine.worker.Dispose();
        InitializeBarracuda();
    }

    // Execute neural network with the provided input
    engine.worker.Execute(input);
    // Release GPU resources allocated for the Tensor
    input.Dispose();

    // The smallest dimension of the videoTexture
    int minDimension = Mathf.Min(videoTexture.width, videoTexture.height);

    // The value used to scale the key point locations up to the source resolution
    scale = (float)minDimension / Mathf.Min(imageDims.x, imageDims.y);

    // Decode the keypoint coordinates from the model output
    ProcessOutput(engine.worker);
}
```









## Summary

In the next post we will implement the post processing steps for multi-pose estimation. 



**Previous:** [Part 4](https://christianjmills.com/Barracuda-PoseNet-Tutorial-V2-4/)

**Project Resources:** [GitHub Repository - Version 1](https://github.com/cj-mills/Barracuda-PoseNet-Tutorial)

