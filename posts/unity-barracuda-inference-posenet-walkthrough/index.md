---
title: "Code Walkthrough: Unity Barracuda Inference PoseNet Package"
date: 2023-5-7
image: /images/empty.gif
hide: false
search_exclude: false
categories: [unity, walkthrough]
description: "Walk through the code for the Unity Barracuda Inference PoseNet package, which extends the functionality of `unity-barracuda-inference-base` to perform 2D human pose estimation using PoseNet models."

twitter-card:
  creator: "@cdotjdotmills"
  site: "@cdotjdotmills"
  image: ../social-media/cover.jpg
open-graph:
  image: ../social-media/cover.jpg
---





* [Introduction](#introduction)
* [Package Overview](#package-overview)
* [Code Explanation](#code-explanation)
* [Conclusion](#conclusion)




## Introduction

The [Barracuda Inference PoseNet](https://github.com/cj-mills/unity-barracuda-inference-posenet) package extends the functionality of [`unity-barracuda-inference-base`](https://github.com/cj-mills/unity-barracuda-inference-base) to perform 2D human pose estimation using PoseNet models. 

Pose estimation has numerous potential uses in Unity applications, including motion capture and animation, educational apps, and augmented reality, to name a few. Here is a demo video from a project that uses this package.



![](./videos/barracuda-inference-posenet-demo.mp4){fig-align="center"}



In this post, I'll walk through the package code, providing a solid understanding of its components and their roles.




## Package Overview

The package contains two C# scripts.

1. `PoseNetPoseEstimator.cs`: This script provides functionality to perform 2D human pose estimation with PoseNet models using the Barracuda inference engine.
2. `PackageInstaller.cs`: An Editor utility script for automatically installing a list of dependency packages defined in a JSON file.



## Code Explanation

In this section, we will delve deeper into the Barracuda Inference PoseNet package by examining the purpose and functionality of each C# script.



### `PoseNetPoseEstimator.cs`

This script defines the PoseNetPoseEstimator class, which extends the BarracudaModelRunner class from the Barracuda Inference Base package to perform 2D human pose estimation using PoseNet models. This class also depends on the `human-pose-2d-toolkit` package. The complete code is available on GitHub at the link below.

- [PoseNetPoseEstimator.cs](https://github.com/cj-mills/unity-barracuda-inference-posenet/blob/main/Runtime/Scripts/PoseNetPoseEstimator.cs)





#### Serialized Fields

The class has several serialized fields for configuring the model, and processing output.

```c#
[Header("PoseNet Model Configuration")]
[SerializeField, Tooltip("Index of the heatmap layer in the neural network")]
private int heatmapLayerIndex = 0;

[SerializeField, Tooltip("Index of the offsets layer in the neural network")]
private int offsetsLayerIndex = 1;

[SerializeField, Tooltip("Index of the forward displacement layer in the neural network")]
private int displacementFWDLayerIndex = 3;

[SerializeField, Tooltip("Index of the backward displacement layer in the neural network")]
private int displacementBWDLayerIndex = 2;

[Header("Output Processing")]
[SerializeField, Tooltip("TextAsset containing the class labels for pose estimation")]
private TextAsset classLabels;
```



It also includes a field to control how often to unload memory assets when using Barracuda's Pixel Shader backend. The Pixel Shader backend enables GPU inference on platforms that don't support Compute Shaders. However, there seems to be a bug in the current version of Barracuda, which does not release unused assets when using this backend. Left unchecked, this can fill up both system and GPU memory. We can address this by manually freeing memory. Doing that every frame can hurt performance, so we'll only do it at set intervals.



```c#
[Header("Settings")]
[SerializeField, Tooltip("Interval at which pixel shaders are unloaded")]
private int pixelShaderUnloadInterval = 100;
```





#### Private Variables

There are internal variables for handling class labels and parent-children relationships of pose skeleton points.

```c#
// Internal Variables
private int frameCounter = 0;
private class ClassLabels { public string[] classes; }
private const int kLocalMaximumRadius = 1;

// Parent-children relationships of the pose keypoints
private readonly (int, int)[] parentChildrenTuples = {
    (0, 1), (1, 3), (0, 2), (2, 4), (0, 5), (5, 7),
    (7, 9), (5, 11), (11, 13), (13, 15), (0, 6),
    (6, 8), (8, 10), (6, 12), (12, 14), (14, 16)
};

private const int maxStride = 32;
private const string SigmoidLayer = "sigmoid";

// Layer names for the neural network
private string offsetsLayer;
private string displacementFWDLayer;
private string displacementBWDLayer;

// Class labels array
private string[] classes;

// Smallest dimension of the input image
private int minDim = 0;
```





#### `Start`

This method initializes the pose estimation component by calling the Start() method of the parent class and loading class labels.

```c#
/// <summary>
/// Initializes the pose estimation component.
/// </summary>
protected override void Start()
{
    base.Start();
    LoadClassLabels();
}
```





#### `LoadAndPrepareModel`

This method loads and prepares the PoseNet model for pose estimation. It sets the worker type to PixelShader if running on WebGL, gets the output layers for the heatmap, offsets, forward displacement, and backward displacement, and adds a sigmoid layer if the last layer is not sigmoid.

```c#
/// <summary>
/// Loads and prepares the model for pose estimation.
/// </summary>
protected override void LoadAndPrepareModel()
{
    base.LoadAndPrepareModel();

    // Set worker type to PixelShader if running on WebGL
    if (Application.platform == RuntimePlatform.WebGLPlayer)
    {
        workerType = WorkerFactory.Type.PixelShader;
    }

    // Get the output layer of the heatmap
    string outputLayer = modelBuilder.model.outputs[heatmapLayerIndex];

    // Find the heatmap layer in the model
    Layer heatmapLayer = FindLayerByName(modelBuilder.model, outputLayer);
    bool lastLayerIsSigmoid = heatmapLayer.activation == Layer.Activation.Sigmoid;

    // Add a sigmoid layer if the last layer is not sigmoid
    if (!lastLayerIsSigmoid)
    {
        modelBuilder.Sigmoid(SigmoidLayer, outputLayer);
    }

    // Get the names of the output layers for offsets, forward, and backward displacements
    offsetsLayer = modelBuilder.model.outputs[offsetsLayerIndex];
    displacementFWDLayer = modelBuilder.model.outputs[displacementFWDLayerIndex];
    displacementBWDLayer = modelBuilder.model.outputs[displacementBWDLayerIndex];
}
```





#### `FindLayerByName`

This method searches for a layer in the model by its name and returns the layer if found.

```c#
/// <summary>
/// Finds a layer in the model by its name.
/// </summary>
/// <param name="model">The model to search for the layer.</param>
/// <param name="layerName">The name of the layer to find.</param>
/// <returns>The layer with the given name or null if not found.</returns>
Layer FindLayerByName(Model model, string layerName)
{
    foreach (Layer layer in model.layers)
    {
        if (layer.name == layerName)
        {
            return layer;
        }
    }

    return null;
}
```



#### `LoadClassLabels`

This method loads the class labels from the TextAsset, deserializes the JSON, and updates the `classes` array.

```c#
/// <summary>
/// Loads the class labels from the TextAsset and updates the classes array.
/// </summary>
private void LoadClassLabels()
{
    if (IsClassLabelsJsonNullOrEmpty())
    {
        Debug.LogError("Class labels JSON is null or empty.");
        return;
    }

    ClassLabels classLabelsObj = DeserializeClassLabels(classLabels.text);
    UpdateClassLabels(classLabelsObj);
}
```



#### `IsClassLabelsJsonNullOrEmpty`

This method checks if the provided class label JSON file is null or empty.

```c#
/// <summary>
/// Checks if the class labels JSON is null or empty.
/// </summary>
/// <returns>True if the JSON is null or empty, false otherwise.</returns>
private bool IsClassLabelsJsonNullOrEmpty()
{
    return classLabels == null || string.IsNullOrWhiteSpace(classLabels.text);
}
```



#### `DeserializeClassLabels`

This method deserializes the provided class label JSON string to a `ClassLabels` object.

```c#
/// <summary>
/// Deserializes the class labels JSON into a ClassLabels object.
/// </summary>
/// <param name="json">The class labels JSON string.</param>
/// <returns>A ClassLabels object, or null if deserialization fails.</returns>
private ClassLabels DeserializeClassLabels(string json)
{
    try
    {
        return JsonUtility.FromJson<ClassLabels>(json);
    }
    catch (Exception ex)
    {
        Debug.LogError($"Failed to deserialize class labels JSON: {ex.Message}");
        return null;
    }
}
```



#### `UpdateClassLabels`

This method updates the `classes` array with the provided `ClassLabels` object.

```c#
/// <summary>
/// Updates the classes array with the contents of the given ClassLabels object.
/// </summary>
/// <param name="classLabelsObj">The ClassLabels object containing class labels.</param>
private void UpdateClassLabels(ClassLabels classLabelsObj)
{
    if (classLabelsObj == null)
    {
        return;
    }

    classes = classLabelsObj.classes;
}
```



#### `ExecuteModel`

This method executes the PoseNet model with the given input texture.

```c#
/// <summary>
/// Executes the model with the given input texture.
/// </summary>
/// <param name="inputTexture">The input texture to process.</param>
public void ExecuteModel(RenderTexture inputTexture)
{
    minDim = Mathf.Min(inputTexture.width, inputTexture.height);

    using (Tensor input = new Tensor(inputTexture, channels: 3))
    {
        base.ExecuteModel(input);
    }
}
```



#### `ProcessOutput`

This method processes the output tensors and returns an array of detected human poses. It can use either single-pose decoding or multiple-pose decoding.

```c#
/// <summary>
/// Processes the output tensors and returns an array of detected human poses.
/// </summary>
/// <param name="useMultiPoseDecoding">True to use multiple pose decoding, false to use single pose decoding.</param>
/// <param name="maxPoses">The maximum number of poses to detect.</param>
/// <returns>An array of detected human poses.</returns>
public HumanPose2D[] ProcessOutput(float scoreThreshold, int nmsRadius, int maxPoses = 20, bool useMultiPoseDecoding = true)
{
    // Initialize a list to store the detected human poses
    List<HumanPose2D> humanPoses = new List<HumanPose2D>();

    // Get the output tensors from the neural network
    using Tensor heatmaps = engine.PeekOutput(SigmoidLayer);
    using Tensor offsets = engine.PeekOutput(offsetsLayer);
    using Tensor displacementFWD = engine.PeekOutput(displacementFWDLayer);
    using Tensor displacementBWD = engine.PeekOutput(displacementBWDLayer);

    // Calculate the stride based on the dimensions of the heatmaps
    int minHeatMapDim = Mathf.Min(heatmaps.width, heatmaps.height);
    int stride = (minDim - 1) / (minHeatMapDim - 1);
    stride -= (stride % 8);

    // Decide whether to use single pose decoding or multiple pose decoding
    if (useMultiPoseDecoding)
    {
        // Decode multiple poses and store them in the humanPoses list
        humanPoses = DecodeMultiplePoses(
            heatmaps, offsets,
            displacementFWD, displacementBWD,
            stride, maxPoses, scoreThreshold, nmsRadius);   
    }
    else
    {
        // Decode a single pose and add it to the humanPoses list
        HumanPose2D pose = new HumanPose2D
        {
            index = 0,
            bodyParts = DecodeSinglePose(heatmaps, offsets, stride)
        };
        humanPoses.Add(pose);
    }

    // Unload unused assets if needed
    UnloadUnusedAssetsIfNeeded();

    // Convert the list of human poses to an array and return it
    return humanPoses.ToArray();
}
```



#### `UnloadUnusedAssetsIfNeeded`

This method unloads unused assets if needed based on the worker type and frame counter.


```c#
/// <summary>
/// Unloads unused assets if needed based on the worker type and frame counter.
/// </summary>
private void UnloadUnusedAssetsIfNeeded()
{
    if (workerType != WorkerFactory.Type.PixelShader) return;

    frameCounter++;
    if (frameCounter % pixelShaderUnloadInterval == 0)
    {
        Resources.UnloadUnusedAssets();
        frameCounter = 0;
    }
}
```




#### `DecodeSinglePose`

This method decodes a single human pose from the given `heatmaps` and `offsets` tensors and returns an array of body parts.

```c#
/// <summary>
/// Decodes a single human pose from the given heatmaps and offsets tensors.
/// </summary>
/// <param name="heatmaps">The heatmaps tensor.</param>
/// <param name="offsets">The offsets tensor.</param>
/// <param name="stride">The stride for decoding the pose.</param>
/// <returns>An array of body parts for the decoded pose.</returns>
public BodyPart2D[] DecodeSinglePose(Tensor heatmaps, Tensor offsets, int stride)
{
    int numBodyParts = heatmaps.channels;
    BodyPart2D[] bodyParts = new BodyPart2D[numBodyParts];

    for (int c = 0; c < numBodyParts; c++)
    {
        BodyPart2D part = FindHighestConfidenceBodyPart(heatmaps, c);
        part.coordinates = GetImageCoords(part, stride, offsets);
        bodyParts[c] = part;
    }

    return bodyParts;
}
```



#### `FindHighestConfidenceBodyPart`

This method finds the body part with the highest confidence for the given channel in the heatmaps tensor and returns the body part.


```c#
/// <summary>
/// Finds the body part with the highest confidence for the given channel in the heatmaps tensor.
/// </summary>
/// <param name="heatmaps">The heatmaps tensor.</param>
/// <param name="channel">The channel representing the body part to search for.</param>
/// <returns>The body part with the highest confidence.</returns>
private BodyPart2D FindHighestConfidenceBodyPart(Tensor heatmaps, int channel)
{
    BodyPart2D part = new BodyPart2D { index = channel, prob = 0 };

    for (int y = 0; y < heatmaps.height; y++)
    {
        for (int x = 0; x < heatmaps.width; x++)
        {
            float confidence = heatmaps[0, y, x, channel];
            if (confidence > part.prob)
            {
                part.prob = confidence;
                part.coordinates.x = x;
                part.coordinates.y = y;
            }
        }
    }

    return part;
}
```




#### `GetOffsetVector`

This method returns the offset vector for the given coordinates and keypoint in the offsets tensor.

```c#
/// <summary>
/// Returns the offset vector for the given coordinates and keypoint in the offsets tensor.
/// </summary>
/// <param name="y">The y-coordinate.</param>
/// <param name="x">The x-coordinate.</param>
/// <param name="keypoint">The keypoint index.</param>
/// <param name="offsets">The offsets tensor.</param>
/// <returns>The offset vector for the specified keypoint.</returns>
public Vector2 GetOffsetVector(int y, int x, int keypoint, Tensor offsets)
{
    int channelOffset = offsets.channels / 2;
    return new Vector2(offsets[0, y, x, keypoint + channelOffset], offsets[0, y, x, keypoint]);
}
```




#### `GetImageCoords`

This method converts body part coordinates to image coordinates using the given stride and offsets tensor.

```c#
/// <summary>
/// Converts body part coordinates to image coordinates using the given stride and offsets tensor.
/// </summary>
/// <param name="part">The body part with heatmap coordinates.</param>
/// <param name="stride">The stride for decoding the pose.</param>
/// <param name="offsets">The offsets tensor.</param>
/// <returns>The image coordinates for the given body part.</returns>
public Vector2 GetImageCoords(BodyPart2D part, int stride, Tensor offsets)
{
    Vector2 offsetVector = GetOffsetVector((int)part.coordinates.y, (int)part.coordinates.x, part.index, offsets);
    return (part.coordinates * stride) + offsetVector;
}
```



#### `GetStridedIndexNearPoint`

This method gets the stridden index near a given point, given the stride, tensor height, and tensor width.

```c#
/// <summary>
/// Gets the strided index near a given point.
/// </summary>
/// <param name="point">The point for which the strided index is calculated.</param>
/// <param name="stride">The stride for decoding the pose.</param>
/// <param name="height">The height of the tensor.</param>
/// <param name="width">The width of the tensor.</param>
/// <returns>The strided index as a Vector2Int.</returns>
public Vector2Int GetStridedIndexNearPoint(Vector2 point, int stride, int height, int width)
{
    return new Vector2Int(
        Mathf.Clamp(Mathf.RoundToInt(point.x / stride), 0, width - 1),
        Mathf.Clamp(Mathf.RoundToInt(point.y / stride), 0, height - 1)
    );
}
```



#### `GetDisplacement`

This method gets the displacement for the specified edge and point in the `displacements` tensor and returns it as a Vector2.

```c#
/// <summary>
/// Gets the displacement for the specified edge and point in the displacements tensor.
/// </summary>
/// <param name="edgeId">The edge index.</param>
/// <param name="point">The point as a Vector2Int.</param>
/// <param name="displacements">The displacements tensor.</param>
/// <returns>The displacement as a Vector2.</returns>
public Vector2 GetDisplacement(int edgeId, Vector2Int point, Tensor displacements)
{
    int numEdges = displacements.channels / 2;
    return new Vector2(
        displacements[0, point.y, point.x, numEdges + edgeId],
        displacements[0, point.y, point.x, edgeId]
    );
}
```



#### `TraverseToTargetBodyPart2D`

This method takes an edge index, a source body part, a target body part index, and tensors for scores, offsets, stride, and displacements. It calculates the displaced point by adding the displacement value to the source body part coordinates and returns the target body part as a `BodyPart2D` instance.

```c#
/// <summary>
/// Traverses to the target body part from the source body part using the given edge.
/// </summary>
/// <param name="edgeId">The edge index.</param>
/// <param name="sourceBodyPart2D">The source body part.</param>
/// <param name="targetBodyPart2DId">The target body part index.</param>
/// <param name="scores">The scores tensor.</param>
/// <param name="offsets">The offsets tensor.</param>
/// <param name="stride">The stride for decoding the pose.</param>
/// <param name="displacements">The displacements tensor.</param>
/// <returns>The target body part as a BodyPart2D.</returns>
public BodyPart2D TraverseToTargetBodyPart2D(
    int edgeId, BodyPart2D sourceBodyPart2D, int targetBodyPart2DId,
    Tensor scores, Tensor offsets, int stride,
    Tensor displacements)
{
    // Get height and width from the scores tensor
    int height = scores.height;
    int width = scores.width;

    // Calculate the source body part indices in the strided space
    Vector2Int sourceBodyPart2DIndices = GetStridedIndexNearPoint(sourceBodyPart2D.coordinates, stride, height, width);

    // Get the displacement for the given edge
    Vector2 displacement = GetDisplacement(edgeId, sourceBodyPart2DIndices, displacements);

    // Calculate the displaced point by adding the displacement to the source body part coordinates
    Vector2 displacedPoint = sourceBodyPart2D.coordinates + displacement;

    // Calculate the displaced point indices in the strided space
    Vector2Int displacedPointIndices = GetStridedIndexNearPoint(displacedPoint, stride, height, width);

    // Get the offset vector for the target body part
    Vector2 offsetVector = GetOffsetVector(displacedPointIndices.y, displacedPointIndices.x, targetBodyPart2DId, offsets);

    // Get the score for the target body part
    float score = scores[0, displacedPointIndices.y, displacedPointIndices.x, targetBodyPart2DId];

    // Calculate the target body part coordinates by adding the offset vector to the displaced point indices
    Vector2 targetBodyPart2D = (displacedPointIndices * stride) + offsetVector;

    // Return the target body part as a BodyPart2D instance
    return new BodyPart2D(targetBodyPart2DId, targetBodyPart2D, score);
}
```



#### `DecodePose`

This method takes a root body part, tensors for scores, offsets, stride, the forward and backward displacements, and returns an array of BodyPart2D instances for the decoded pose.

```c#
/// <summary>
/// Decodes the pose given a root body part, scores, offsets, stride, and displacements tensors.
/// </summary>
/// <param name="root">The root BodyPart2D.</param>
/// <param name="scores">The scores tensor.</param>
/// <param name="offsets">The offsets tensor.</param>
/// <param name="stride">The stride for decoding the pose.</param>
/// <param name="displacementsFwd">The forward displacements tensor.</param>
/// <param name="displacementsBwd">The backward displacements tensor.</param>
/// <returns>An array of BodyPart2D for the decoded pose.</returns>
public BodyPart2D[] DecodePose(
    BodyPart2D root, Tensor scores, Tensor offsets,
    int stride, Tensor displacementsFwd, Tensor displacementsBwd)
{
    // Get the number of body parts from the scores tensor
    int numBodyParts = scores.channels;

    // Initialize an array of BodyPart2D instances for storing the decoded pose
    BodyPart2D[] instanceBodyParts = new BodyPart2D[numBodyParts];

    // Compute the root point coordinates in the image and store it in the array
    Vector2 rootPoint = GetImageCoords(root, stride, offsets);
    instanceBodyParts[root.index] = new BodyPart2D(root.index, rootPoint, root.prob);

    // Get the number of edges from parentChildrenTuples
    int numEdges = parentChildrenTuples.Length;

    // Traverse the edges in both directions to decode the pose
    TraverseEdges(instanceBodyParts, scores, offsets, stride, displacementsBwd, numEdges, reverse: true);
    TraverseEdges(instanceBodyParts, scores, offsets, stride, displacementsFwd, numEdges, reverse: false);

    // Return the decoded pose as an array of BodyPart2D instances
    return instanceBodyParts;
}
```



#### `TraverseEdges`

This method traverses edges from the source to the target body part, updating the position and probability of the target body part in the `instanceBodyParts` array.

```c#
/// <summary>
/// Traverses edges from the source to the target body part.
/// </summary>
/// <param name="instanceBodyParts">An array of BodyPart2D instances.</param>
/// <param name="scores">The scores tensor.</param>
/// <param name="offsets">The offsets tensor.</param>
/// <param name="stride">The stride for decoding the pose.</param>
/// <param name="displacements">The displacements tensor.</param>
/// <param name="numEdges">The number of edges.</param>
/// <param name="reverse">Whether to reverse the traversal direction.</param>
private void TraverseEdges(
    BodyPart2D[] instanceBodyParts, Tensor scores, Tensor offsets,
    int stride, Tensor displacements, int numEdges, bool reverse)
{
    // Set the start, end, and step of the edge traversal based on the reverse flag
    int edgeStart = reverse ? numEdges - 1 : 0;
    int edgeEnd = reverse ? -1 : numEdges;
    int edgeStep = reverse ? -1 : 1;

    // Traverse the edges in the specified direction
    for (int edge = edgeStart; edge != edgeEnd; edge += edgeStep)
    {
        (int sourceBodyPartId, int targetBodyPartId) = parentChildrenTuples[edge];

        // Swap source and target body part IDs if traversing in reverse
        if (reverse)
        {
            (sourceBodyPartId, targetBodyPartId) = (targetBodyPartId, sourceBodyPartId);
        }

        // If the source body part has a probability greater than 0 and the target body part has not been detected,
        // traverse to the target body part and update its position and probability in the instanceBodyParts array
        if (instanceBodyParts[sourceBodyPartId].prob > 0.0f &&
            instanceBodyParts[targetBodyPartId].prob == 0.0f)
        {
            instanceBodyParts[targetBodyPartId] = TraverseToTargetBodyPart2D(
                edge, instanceBodyParts[sourceBodyPartId], targetBodyPartId,
                scores, offsets, stride, displacements);
        }
    }
}
```



#### `ScoreIsMaximumInLocalWindow`

The `ScoreIsMaximumInLocalWindow` method checks if a given score is the maximum in a local window around the pose skeleton point.

```c#
/// <summary>
/// Checks if a score is the maximum in a local window around the keypoint.
/// </summary>
/// <param name="keypointId">The keypoint index.</param>
/// <param name="score">The score to check.</param>
/// <param name="heatmapY">The y-coordinate of the keypoint in the heatmap.</param>
/// <param name="heatmapX">The x-coordinate of the keypoint in the heatmap.</param>
/// <param name="localMaximumRadius">The radius of the local window to search.</param>
/// <param name="heatmaps">The heatmaps tensor.</param>
/// <returns>True if the score is the maximum in the local window, false otherwise.</returns>
public bool ScoreIsMaximumInLocalWindow(int keypointId, float score, int heatmapY, int heatmapX,
    int localMaximumRadius, Tensor heatmaps)
{
    int yStart = Mathf.Max(heatmapY - localMaximumRadius, 0);
    int yEnd = Mathf.Min(heatmapY + localMaximumRadius + 1, heatmaps.height);

    // Iterate through the local window around the keypoint
    for (int yCurrent = yStart; yCurrent < yEnd; ++yCurrent)
    {
        int xStart = Mathf.Max(heatmapX - localMaximumRadius, 0);
        int xEnd = Mathf.Min(heatmapX + localMaximumRadius + 1, heatmaps.width);

        for (int xCurrent = xStart; xCurrent < xEnd; ++xCurrent)
        {
            // If any value in the local window is greater than the score,
            // it is not the maximum
            if (heatmaps[0, yCurrent, xCurrent, keypointId] > score)
            {
                return false;
            }
        }
    }

    // If none of the values in the local window are greater, the score is the maximum
    return true;
}
```



#### `BuildPartList`

This method builds a list of BodyPart2D instances with scores above the specified threshold and which are the maximum in their local window.

```c#
/// <summary>
/// Builds a list of BodyPart2D instances that have a score above the threshold and are the maximum in their local window.
/// </summary>
/// <param name="scoreThreshold">The minimum score threshold.</param>
/// <param name="localMaximumRadius">The radius of the local window to search.</param>
/// <param name="heatmaps">The heatmaps tensor.</param>
/// <returns>A list of BodyPart2D instances that meet the conditions.</returns>
public List<BodyPart2D> BuildPartList(float scoreThreshold, int localMaximumRadius, Tensor heatmaps)
{
    List<BodyPart2D> list = new List<BodyPart2D>();

    // Iterate through the channels, height, and width of the heatmaps tensor
    for (int c = 0; c < heatmaps.channels; c++)
    {
        for (int y = 0; y < heatmaps.height; y++)
        {
            for (int x = 0; x < heatmaps.width; x++)
            {
                float score = heatmaps[0, y, x, c];

                // If the score is greater or equal to the threshold and is the maximum in the local window,
                // add it to the list
                if (score >= scoreThreshold &&
                    ScoreIsMaximumInLocalWindow(c, score, y, x, localMaximumRadius, heatmaps))
                {
                    list.Add(new BodyPart2D(c, new Vector2(x, y), score));
                }
            }
        }
    }

    return list;
}
```



#### `WithinNmsRadiusOfCorrespondingPoint`

This method checks if a given vector is within the non-maximum suppression radius of a corresponding point in any pose.

```c#
/// <summary>
/// Checks if a given vector is within the non-maximum suppression radius of a corresponding point in any pose.
/// </summary>
/// <param name="poses">A list of HumanPose2D instances.</param>
/// <param name="squaredNmsRadius">The squared non-maximum suppression radius.</param>
/// <param name="vec">The vector to be checked.</param>
/// <param name="keypointId">The keypoint index.</param>
/// <returns>True if the vector is within the radius of a corresponding point in any pose, false otherwise.</returns>
public bool WithinNmsRadiusOfCorrespondingPoint(
    List<HumanPose2D> poses, float squaredNmsRadius, Vector2 vec, int keypointId)
{
    return poses.Any(pose => (vec - pose.bodyParts[keypointId].coordinates).sqrMagnitude <= squaredNmsRadius);
}
```



#### `DecodeMultiplePoses`

This method decodes multiple human poses from the model output.

```c#
/// <summary>
/// Decodes multiple human poses from the given heatmaps, offsets, and displacements tensors.
/// </summary>
/// <param name="heatmaps">The heatmaps tensor.</param>
/// <param name="offsets">The offsets tensor.</param>
/// <param name="displacementsFwd">The forward displacements tensor.</param>
/// <param name="displacementBwd">The backward displacements tensor.</param>
/// <param name="stride">The stride for decoding the pose.</param>
/// <param name="maxPoseDetections">The maximum number of pose detections.</param>
/// <param name="scoreThreshold">The minimum score threshold for a part to be considered.</param>
/// <param name="nmsRadius">The non-maximum suppression radius.</param>
/// <returns>A list of decoded HumanPose2D instances.</returns>
public List<HumanPose2D> DecodeMultiplePoses(
    Tensor heatmaps, Tensor offsets,
    Tensor displacementsFwd, Tensor displacementBwd,
    int stride, int maxPoseDetections,
    float scoreThreshold = 0.5f, int nmsRadius = 20)
{
    List<HumanPose2D> humanPoses = new List<HumanPose2D>();
    float squaredNmsRadius = nmsRadius * nmsRadius;

    List<BodyPart2D> bodyPartList = BuildPartList(scoreThreshold, kLocalMaximumRadius, heatmaps);
    bodyPartList.Sort((a, b) => b.prob.CompareTo(a.prob));

    // Continue decoding poses until the maximum number of detections is reached or the body part list is empty
    while (humanPoses.Count < maxPoseDetections && bodyPartList.Count > 0)
    {
        BodyPart2D root = bodyPartList[0];
        bodyPartList.RemoveAt(0);

        Vector2 rootImageCoords = GetImageCoords(root, stride, offsets);

        // If the root is not within the non-maximum suppression radius of any existing pose,
        // decode the pose and add it to the list of human poses
        if (!WithinNmsRadiusOfCorrespondingPoint(humanPoses, squaredNmsRadius, rootImageCoords, root.index))
        {
            HumanPose2D pose = new HumanPose2D
            {
                index = humanPoses.Count,
                bodyParts = DecodePose(root, heatmaps, offsets, stride, displacementsFwd, displacementBwd)
            };
            humanPoses.Add(pose);
        }
    }

    return humanPoses;
}
```



#### `CropInputDims`

This method crops input dimensions to be divisible by the maximum stride.

```c#
/// <summary>
/// Crop input dimensions to be divisible by the maximum stride.
/// </summary>
public Vector2Int CropInputDims(Vector2Int inputDims)
{
    inputDims[0] -= inputDims[0] % maxStride;
    inputDims[1] -= inputDims[1] % maxStride;

    return inputDims;
}
```











---



### `PackageInstaller.cs`

In this section, we will go through the `PackageInstaller.cs` script and explain how each part of the code works to install the required packages. The complete code is available on GitHub at the link below.

- [PackageInstaller.cs](https://github.com/cj-mills/unity-barracuda-inference-posenet/blob/main/Editor/PackageInstaller.cs)



#### Serializable Classes
The script defines two serializable classes to hold package data.

```c#
// Serializable class to hold package data
[System.Serializable]
public class PackageData
{
    public string packageName;
    public string packageUrl;
}

// Serializable class to hold a list of PackageData objects
[System.Serializable]
public class PackageList
{
    public List<PackageData> packages;
}
```

These classes are for deserializing the JSON file containing the list of packages to install.



#### `PackageInstaller` Class Variables
The `PackageInstaller` class contains several private static fields.

```c#
// Stores the AddRequest object for the current package to install.
private static AddRequest addRequest;
// A list of PackageData objects to install.
private static List<PackageData> packagesToInstall;
// The index of the current package to install.
private static int currentPackageIndex;

// GUID of the JSON file containing the list of packages to install
private const string PackagesJSONGUID = "0d78f4ab62d44aba8a8e95e6a8abfe8a";
```



#### `InstallDependencies`
The `InstallDependencies()` method executes when Unity loads without action from the user. It reads the package JSON file and calls the `InstallNextPackage()` method to install the packages.

```c#
// Method called on load to install packages from the JSON file
[InitializeOnLoadMethod]
public static void InstallDependencies()
{
    // Read the package JSON file
    packagesToInstall = ReadPackageJson().packages;
    // Initialize the current package index
    currentPackageIndex = 0;
    // Start installing the packages
    InstallNextPackage();
}
```



#### `InstallNextPackage`
This method installs the next package in the list.

```c#
// Method to install the next package in the list
private static void InstallNextPackage()
{
    // Iterate through package list
    if (currentPackageIndex < packagesToInstall.Count)
    {
        PackageData packageData = packagesToInstall[currentPackageIndex];

        // Check if the package is already installed
        if (!IsPackageInstalled(packageData.packageName))
        {
            // Attempt to install package
            addRequest = Client.Add(packageData.packageUrl);
            EditorApplication.update += PackageInstallationProgress;
        }
        else
        {
            // Increment the current package index
            currentPackageIndex++;
            // Recursively call InstallNextPackage
            InstallNextPackage();
        }
    }
}
```





#### `PackageInstallationProgress`

This method monitors the progress of the package installation and logs whether it was successful. It then triggers the installation process for the next package in the list.

```c#
// Method to monitor the progress of package installation
private static void PackageInstallationProgress()
{
    if (addRequest.IsCompleted)
    {
        // Log whether the package installation was successful
        if (addRequest.Status == StatusCode.Success)
        {
            UnityEngine.Debug.Log($"Successfully installed: {addRequest.Result.packageId}");
        }
        else if (addRequest.Status >= StatusCode.Failure)
        {
            UnityEngine.Debug.LogError($"Failed to install package: {addRequest.Error.message}");
        }

        // Unregister the method from the EditorApplication.update 
        EditorApplication.update -= PackageInstallationProgress;
        // Increment the current package index
        currentPackageIndex++;
        // Install the next package in the list
        InstallNextPackage();
    }
}
```



#### `IsPackageInstalled`
This method verifies whether a package has already been installed or not.

```c#
// Method to check if a package is already installed
private static bool IsPackageInstalled(string packageName)
{
    // List the installed packages
    var listRequest = Client.List(true, false);
    while (!listRequest.IsCompleted) { }

    if (listRequest.Status == StatusCode.Success)
    {
        // Check if the package is already installed
        return listRequest.Result.Any(package => package.name == packageName);
    }
    else
    {
        UnityEngine.Debug.LogError($"Failed to list packages: {listRequest.Error.message}");
    }

    return false;
}
```



#### `ReadPackageJson`
This method reads the JSON file containing the list of packages to install and returns a `PackageList` object.

```c#
// Method to read the JSON file and return a PackageList object
private static PackageList ReadPackageJson()
{
    // Convert the PackagesJSONGUID to an asset path
    string assetPath = AssetDatabase.GUIDToAssetPath(PackagesJSONGUID);
    // Read the JSON file content as a string
    string jsonString = File.ReadAllText(assetPath);
    // Deserialize the JSON string into a PackageList object
    return JsonUtility.FromJson<PackageList>(jsonString);
}
```







## Conclusion

This post provided an in-depth walkthrough of the code for the Barracuda Inference PoseNet package. The package extends the functionality of [`unity-barracuda-inference-base`](https://github.com/cj-mills/unity-barracuda-inference-base) to perform 2D human pose estimation using PoseNet models. 

You can continue to explore the package by going to its GitHub repository linked below, where you will also find instructions for installing it using the Unity Package Manager.

- GitHub Repository: [unity-barracuda-inference-posenet](https://github.com/cj-mills/unity-barracuda-inference-posenet)

You can find the code for the demo project shown in the video at the beginning of this post linked below.

- [Barracuda Inference PoseNet Demo](https://github.com/cj-mills/barracuda-inference-posenet-demo): A simple Unity project demonstrating how to perform 2D human pose estimation with the `barracuda-inference-posenet` package.

