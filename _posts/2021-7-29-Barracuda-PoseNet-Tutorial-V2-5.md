---
title: Barracuda PoseNet Tutorial 2nd Edition Pt. 5
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







## Summary

In the next post we will implement the post processing steps for multi-pose estimation. 



**Previous:** [Part 4](https://christianjmills.com/Barracuda-PoseNet-Tutorial-V2-4/)

**Project Resources:** [GitHub Repository - Version 1](https://github.com/cj-mills/Barracuda-PoseNet-Tutorial)

