---
title: Barracuda PoseNet Tutorial 2nd Edition Pt. 6 - Unpublished
layout: post
toc: false
comments: true
description: This post covers how to implement the post processing steps for multi-pose estimation.
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

In this post, we will cover how to implement the post processing steps for multi-pose estimation. This method is more complex than what is required to perform multi-pose estimation. However, it can produce more reliable results.

> **Note:** The original JavaScript code for decoding multiple poses can be found in the official [tfjs-models](https://github.com/tensorflow/tfjs-models/tree/master/posenet/src/multi_pose) repository on GitHub. The code has been modified for this tutorial to better take advantage of functionality provided by Unity.

## Update `Utils` Script

There are a couple new variables and several methods that we will need to add to decode multiple poses from the model output.

### Add Required Namespace

First, we need to add the [`System`](https://docs.microsoft.com/en-us/dotnet/api/system?view=net-5.0) namespace to access the [`Tuple`](https://docs.microsoft.com/en-us/dotnet/api/system.tuple-2?view=net-5.0) class. We also need to access the [`System.Linq`](https://docs.microsoft.com/en-us/dotnet/api/system.linq?view=net-5.0) namespace to access classes and interfaces for querying data structures.

```c#
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Barracuda;
using System;
using System.Linq;
```



### Add Public Variables

When iterating through the heatmaps from the model output, we will only be considering heatmap indices with the highest confidence score within a local radius called `kLocalMaximumRadius`. Naturally, setting this value to be larger than the dimensions of the heatmap would not do any good. The [original code](https://github.com/tensorflow/tfjs-models/blob/c3f5aa3ff74787457082f6683655c9d0c7cf3df7/posenet/src/multi_pose/decode_multiple_poses.ts#L54) sets this radius to a constant value of `1` so we will do the same.

When decoding the key point locations for a single body, we will need to traverse from the current key point to its neighboring key point. For example, the nose neighbors both the left eye and right eye. We will keep track of which key points neighbor each other in a `TupleTuple<int, int>` array, where the values are the key point id numbers.

```c#
/// <summary>
/// Defines the size of the local window in the heatmap to look for
/// confidence scores higher than the one at the current heatmap coordinate
/// </summary>
const int kLocalMaximumRadius = 1;

/// <summary>
/// Defines the parent->child relationships used for multipose detection.
/// </summary>
public static Tuple<int, int>[] parentChildrenTuples = new Tuple<int, int>[]{
    // Nose to Left Eye
    Tuple.Create(0, 1),
    // Left Eye to Left Ear
    Tuple.Create(1, 3),
    // Nose to Right Eye
    Tuple.Create(0, 2),
    // Right Eye to Right Ear
    Tuple.Create(2, 4),
    // Nose to Left Shoulder
    Tuple.Create(0, 5),
    // Left Shoulder to Left Elbow
    Tuple.Create(5, 7),
    // Left Elbow to Left Wrist
    Tuple.Create(7, 9), 
    // Left Shoulder to Left Hip
    Tuple.Create(5, 11),
    // Left Hip to Left Knee
    Tuple.Create(11, 13), 
    // Left Knee to Left Ankle
    Tuple.Create(13, 15),
    // Nose to Right Shoulder
    Tuple.Create(0, 6), 
    // Right Shoulder to Right Elbow
    Tuple.Create(6, 8),
    // Right Elbow to Right Wrist
    Tuple.Create(8, 10), 
    // Right Shoulder to Right Hip
    Tuple.Create(6, 12),
    // Right Hip to Right Knee
    Tuple.Create(12, 14), 
    // Right Knee to Right Ankle
    Tuple.Create(14, 16)
};
```





### Create `GetStridedIndexNearPoint` Method





```c#
/// <summary>
/// Calculate the heatmap indices closest to the provided point
/// </summary>
/// <param name="point"></param>
/// <param name="stride"></param>
/// <param name="height"></param>
/// <param name="width"></param>
/// <returns>A vector with the nearest heatmap coordinates</returns>
static Vector2Int GetStridedIndexNearPoint(Vector2 point, int stride, int height, int width)
{
    // Downscale the point coordinates to the heatmap dimensions
    return new Vector2Int(
        (int)Mathf.Clamp(Mathf.Round(point.x / stride), 0, width - 1),
        (int)Mathf.Clamp(Mathf.Round(point.y / stride), 0, height - 1)
    );
}
```







### Create `GetDisplacement` Method



```c#
/// <summary>
/// Retrieve the displacement values for the provided point
/// </summary>
/// <param name="edgeId"></param>
/// <param name="point"></param>
/// <param name="displacements"></param>
/// <returns>A vector witht he displacement values for the provided point</returns>
static Vector2 GetDisplacement(int edgeId, Vector2Int point, Tensor displacements)
{
    // Calculate the number of edges for the pose skeleton
    int numEdges = (int)(displacements.channels / 2);
    // Get the displacement values for the provided heatmap coordinates
    return new Vector2(
        displacements[0, point.y, point.x, numEdges + edgeId],
        displacements[0, point.y, point.x, edgeId]
    );
}
```



### Create `TraverseToTargetKeypoint` Method



```c#
/// <summary>
/// Get a new keypoint along the provided edgeId for the pose instance.
/// </summary>
/// <param name="edgeId"></param>
/// <param name="sourceKeypoint"></param>
/// <param name="targetKeypointId"></param>
/// <param name="scores"></param>
/// <param name="offsets"></param>
/// <param name="stride"></param>
/// <param name="displacements"></param>
/// <returns>A new keypoint with the displaced coordinates</returns>
static Keypoint TraverseToTargetKeypoint(
    int edgeId, Keypoint sourceKeypoint, int targetKeypointId,
    Tensor scores, Tensor offsets, int stride,
    Tensor displacements)
{
    // Get heatmap dimensions
    int height = scores.height;
    int width = scores.width;

    // Get neareast heatmap indices for source keypoint
    Vector2Int sourceKeypointIndices = GetStridedIndexNearPoint(
        sourceKeypoint.position, stride, height, width);
    // Retrieve the displacement values for the current indices
    Vector2 displacement = GetDisplacement(edgeId, sourceKeypointIndices, displacements);
    // Add the displacement values to the keypoint position
    Vector2 displacedPoint = sourceKeypoint.position + displacement;
    // Get neareast heatmap indices for displaced keypoint
    Vector2Int displacedPointIndices =
        GetStridedIndexNearPoint(displacedPoint, stride, height, width);
    // Get the offset vector for the displaced keypoint indices
    Vector2 offsetVector = GetOffsetVector(
        displacedPointIndices.y, displacedPointIndices.x, targetKeypointId,
        offsets);
    // Get the heatmap value at the displaced keypoint location
    float score = scores[0, displacedPointIndices.y, displacedPointIndices.x, targetKeypointId];
    // Calculate the position for the displaced keypoint
    Vector2 targetKeypoint = (displacedPointIndices * stride) + offsetVector;

    return new Keypoint(score, targetKeypoint, targetKeypointId);
}
```



### Create `DecodePose` Method



```c#
/// <summary>
/// Follows the displacement fields to decode the full pose of the object
/// instance given the position of a part that acts as root.
/// </summary>
/// <param name="root"></param>
/// <param name="scores"></param>
/// <param name="offsets"></param>
/// <param name="stride"></param>
/// <param name="displacementsFwd"></param>
/// <param name="displacementsBwd"></param>
/// <returns>An array of keypoints for a single pose</returns>
static Keypoint[] DecodePose(Keypoint root, Tensor scores, Tensor offsets,
                             int stride, Tensor displacementsFwd, Tensor displacementsBwd)
{

    Keypoint[] instanceKeypoints = new Keypoint[scores.channels];

    // Start a new detection instance at the position of the root.
    Vector2 rootPoint = GetImageCoords(root, stride, offsets);

    instanceKeypoints[root.id] = new Keypoint(root.score, rootPoint, root.id);

    int numEdges = parentChildrenTuples.Length;

    // Decode the part positions upwards in the tree, following the backward
    // displacements.
    for (int edge = numEdges - 1; edge >= 0; --edge)
    {
        int sourceKeypointId = parentChildrenTuples[edge].Item2;
        int targetKeypointId = parentChildrenTuples[edge].Item1;
        if (instanceKeypoints[sourceKeypointId].score > 0.0f &&
            instanceKeypoints[targetKeypointId].score == 0.0f)
        {
            instanceKeypoints[targetKeypointId] = TraverseToTargetKeypoint(
                edge, instanceKeypoints[sourceKeypointId], targetKeypointId, scores,
                offsets, stride, displacementsBwd);
        }
    }

    // Decode the part positions downwards in the tree, following the forward
    // displacements.
    for (int edge = 0; edge < numEdges; ++edge)
    {
        int sourceKeypointId = parentChildrenTuples[edge].Item1;
        int targetKeypointId = parentChildrenTuples[edge].Item2;
        if (instanceKeypoints[sourceKeypointId].score > 0.0f &&
            instanceKeypoints[targetKeypointId].score == 0.0f)
        {
            instanceKeypoints[targetKeypointId] = TraverseToTargetKeypoint(
                edge, instanceKeypoints[sourceKeypointId], targetKeypointId, scores,
                offsets, stride, displacementsFwd);
        }
    }

    return instanceKeypoints;
}
```







### Create `WithinNmsRadiusOfCorrespondingPoint` Method



```c#
/// <summary>
/// Check if the provided image coordinates are too close to any keypoints in existing poses
/// </summary>
/// <param name="poses"></param>
/// <param name="squaredNmsRadius"></param>
/// <param name="vec"></param>
/// <param name="keypointId"></param>
/// <returns>True if there are any existing poses too close to the provided coords</returns>
static bool WithinNmsRadiusOfCorrespondingPoint(
    List<Keypoint[]> poses, float squaredNmsRadius, Vector2 vec, int keypointId)
{
    // SquaredDistance
    return poses.Any(pose => (vec - pose[keypointId].position).sqrMagnitude <= squaredNmsRadius);
}
```





### Create `ScoreIsMaximumInLocalWindow` Method



```c#
/// <summary>
/// Compare the value at the current heatmap location to the surrounding values
/// </summary>
/// <param name="keypointId"></param>
/// <param name="score"></param>
/// <param name="heatmapY"></param>
/// <param name="heatmapX"></param>
/// <param name="localMaximumRadius"></param>
/// <param name="scores"></param>
/// <returns>True if the value is the highest within a given radius</returns>
static bool ScoreIsMaximumInLocalWindow(int keypointId, float score, int heatmapY, int heatmapX,
                                        int localMaximumRadius, Tensor heatmaps)
{
    bool localMaximum = true;
    // Calculate the starting heatmap colummn index
    int yStart = Mathf.Max(heatmapY - localMaximumRadius, 0);
    // Calculate the ending heatmap colummn index
    int yEnd = Mathf.Min(heatmapY + localMaximumRadius + 1, heatmaps.height);

    // Iterate through calulated range of heatmap columns
    for (int yCurrent = yStart; yCurrent < yEnd; ++yCurrent)
    {
        // Calculate the starting heatmap row index
        int xStart = Mathf.Max(heatmapX - localMaximumRadius, 0);
        // Calculate the ending heatmap row index
        int xEnd = Mathf.Min(heatmapX + localMaximumRadius + 1, heatmaps.width);

        // Iterate through calulated range of heatmap rows
        for (int xCurrent = xStart; xCurrent < xEnd; ++xCurrent)
        {
            // Check if the score for at the current heatmap location
            // is the highest within the specified radius
            if (heatmaps[0, yCurrent, xCurrent, keypointId] > score)
            {
                localMaximum = false; 
                break;
            }
        }
        if (!localMaximum) break;
    }
    return localMaximum;
}
```





### Create `BuildPartList` Method



Much like the `DecodeSinglePose` method, we will iterate through the entire heatmap Tensor. This time we will only consider heatmap indices with a value above the provided score threshold. When we get to an index with a value that meets this threshold, we will call the `ScoreIsMaximumInLocalWindow` method to confirm that it is the highest score in its local area. If it is the highest

```c#
/// <summary>
/// Iterate through the heatmaps and create a list of indicies 
/// with the highest values within the provided radius.
/// </summary>
/// <param name="scoreThreshold"></param>
/// <param name="localMaximumRadius"></param>
/// <param name="scores"></param>
/// <returns>A list of keypoints with the highest values in their local area</returns>
static List<Keypoint> BuildPartList(float scoreThreshold, int localMaximumRadius, Tensor heatmaps)
{
    List<Keypoint> list = new List<Keypoint>();

    // Iterate through heatmaps
    for (int c = 0; c < heatmaps.channels; c++)
    {
        // Iterate through heatmap columns
        for (int y = 0; y < heatmaps.height; y++)
        {
            // Iterate through column rows
            for (int x = 0; x < heatmaps.width; x++)
            {
                float score = heatmaps[0, y, x, c];

                // Skip parts with score less than the scoreThreshold
                if (score < scoreThreshold) continue;

                // Only add keypoints with the highest score in a local window.
                if (ScoreIsMaximumInLocalWindow(c, score, y, x, localMaximumRadius, heatmaps))
                {
                    list.Add(new Keypoint(score, new Vector2(x, y), c));
                }
            }
        }
    }

    return list;
}
```





### Create `DecodeMultiplePoses` Method



```c#
/// <summary>
/// Detects multiple poses and finds their parts from part scores and displacement vectors. 
/// </summary>
/// <param name="heatmaps"></param>
/// <param name="offsets"></param>
/// <param name="displacementsFwd"></param>
/// <param name="displacementBwd"></param>
/// <param name="stride"></param>
/// <param name="maxPoseDetections"></param>
/// <param name="scoreThreshold"></param>
/// <param name="nmsRadius"></param>
/// <returns>An array of poses up to maxPoseDetections in size</returns>
public static Keypoint[][] DecodeMultiplePoses(
    Tensor heatmaps, Tensor offsets,
    Tensor displacementsFwd, Tensor displacementBwd,
    int stride, int maxPoseDetections,
    float scoreThreshold = 0.5f, int nmsRadius = 20)
{
    // Stores the final poses
    List<Keypoint[]> poses = new List<Keypoint[]>();
    // 
    float squaredNmsRadius = (float)nmsRadius * nmsRadius;

    // Get a list of indicies with the highest values within the provided radius.
    List<Keypoint> list = BuildPartList(scoreThreshold, kLocalMaximumRadius, heatmaps);
    // Order the list in descending order based on score
    list = list.OrderByDescending(x => x.score).ToList();

    // Decode poses until the max number of poses has been reach or the part list is empty
    while (poses.Count < maxPoseDetections && list.Count > 0)
    {
        // Get the part with the highest score in the list
        Keypoint root = list[0];
        // Remove the keypoint from the list
        list.RemoveAt(0);

        // Calculate the input image coordinates for the current part
        Vector2 rootImageCoords = GetImageCoords(root, stride, offsets);

        // Skip parts that are too close to existing poses
        if (WithinNmsRadiusOfCorrespondingPoint(
            poses, squaredNmsRadius, rootImageCoords, root.id))
        {
            continue;
        }

        // Find the keypoints in the same pose as the root part
        Keypoint[] keypoints = DecodePose(
            root, heatmaps, offsets, stride, displacementsFwd,
            displacementBwd);

        // The current list of keypoints
        poses.Add(keypoints);
    }

    return poses.ToArray();
}
```







## Update `PoseEstimator` Script





### Add Public Variables



```c#
[Tooltip("The maximum number of posees to estimate")]
[Range(1, 20)]
public int maxPoses = 20;

[Tooltip("The score threshold for multipose estimation")]
[Range(0, 1.0f)]
public float scoreThreshold = 0.25f;

[Tooltip("Non-maximum suppression part distance")]
public int nmsRadius = 20;
```





### Modify `ProcessOutput` Method



```c#
// Determine the key point locations
poses = Utils.DecodeMultiplePoses(
    heatmaps, offsets,
    displacementFWD, displacementBWD,
    stride: stride, maxPoseDetections: maxPoses,
    scoreThreshold: scoreThreshold, 
    nmsRadius: nmsRadius);
```



#### Full Code

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
        // Determine the key point locations
        poses = Utils.DecodeMultiplePoses(
            heatmaps, offsets,
            displacementFWD, displacementBWD,
            stride: stride, maxPoseDetections: maxPoses,
            scoreThreshold: scoreThreshold, 
            nmsRadius: nmsRadius);
    }

    heatmaps.Dispose();
    offsets.Dispose();
    displacementFWD.Dispose();
    displacementBWD.Dispose();
}
```





## Summary

In the next post, we will demonstrate how to add pose skeletons so that we can compare the estimated key point locations to the source video feed. 



**Previous:** [Part 5](https://christianjmills.com/Barracuda-PoseNet-Tutorial-V2-5/)

**Project Resources:** [GitHub Repository - Version 1](https://github.com/cj-mills/Barracuda-PoseNet-Tutorial)

