---
title: "Code Walkthrough: Unity Human Pose 2D Toolkit Package"
date: 2023-5-6
image: /images/empty.gif
hide: false
search_exclude: false
categories: [unity, walkthrough]
description: "Walk through the code for the Unity Human Pose 2D Toolkit package, which provides an easy-to-use and customizable solution to work with and visualize 2D human poses on a Unity canvas."

toc-depth: 5

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

The [Unity Human Pose 2D Toolkit](https://github.com/cj-mills/unity-human-pose-2d-toolkit) provides an easy-to-use and customizable solution to work with and visualize 2D human poses on a Unity canvas. 

Some of my tutorials involve using 2D pose estimation models in Unity applications. This package makes that shared functionality more modular and reusable, allowing me to streamline my tutorial content. Check out the demo video below to see this package in action.



![](./videos/barracuda-inference-posenet-demo.mp4){fig-align="center"}



In this post, I'll walk through the package code, providing a solid understanding of its components and their roles.




## Package Overview

The package contains three C# scripts and prefabs to construct 2D human poses.



### C# Scripts

1. `HumanPose2DUtils.cs`: This script provides functionality to work with 2D pose skeletons for pose estimation tasks.
2. `HumanPose2DVisualizer.cs`: This script displays 2D human pose skeletons on a Unity canvas.
3. `AddCustomDefineSymbol.cs`: An Editor script that automatically adds a custom scripting define symbol to the project after the package installs.



### Prefabs

1. `BonePrefab.prefab`: The HumanPose2DVisualizer.cs script uses this prefab to construct the bones connecting points in pose skeletons.
2. `JointPrefab.prefab`: An Image prefab used to visualize the points in pose skeletons.
3. `PoseContainerPrefab.prefab`: This prefab is for pose containers that hold the joints and bones for pose skeletons.
4. `HumanPose2DVisualizer.prefab`: This prefab helps simplify adding 2D human pose visualization to a Unity scene. The prefab already has the HumanPose2DVisualizer script attached and has a child Canvas component.



## Code Explanation

In this section, we will delve deeper into the Unity Human Pose 2D Toolkit package by examining the purpose and functionality of each C# script.



### `HumanPose2DUtils.cs`

The HumanPose2DUtils.cs script provides functionality to work with 2D pose skeletons for pose estimation tasks. It contains utility classes and structs for managing 2D human pose data. The complete code is available on GitHub at the link below.

- [HumanPose2DUtils.cs](https://github.com/cj-mills/unity-human-pose-2d-toolkit/blob/main/Runtime/Scripts/HumanPose2DUtils.cs)





#### `BodyPart2D` struct
This struct represents a single body part in 2D space with its index, coordinates, and probability.

```c#
/// <summary>
/// Represents a single body part in 2D space with its index, coordinates, and probability.
/// </summary>
public struct BodyPart2D
{
    public int index; // The index of the body part
    public Vector2 coordinates; // The 2D coordinates of the body part
    public float prob; // The probability of the detected body part

    /// <summary>
    /// Initializes a new instance of the BodyPart2D struct.
    /// </summary>
    /// <param name="index">The index of the body part.</param>
    /// <param name="coordinates">The 2D coordinates of the body part.</param>
    /// <param name="prob">The probability of the detected body part.</param>
    public BodyPart2D(int index, Vector2 coordinates, float prob)
    {
        this.index = index;
        this.coordinates = coordinates;
        this.prob = prob;
    }
}
```



#### `HumanPose2D` struct
This struct represents a detected human pose in 2D space with its index and an array of body parts.

```c#
/// <summary>
/// Represents a detected human pose in 2D space with its index and an array of body parts.
/// </summary>
public struct HumanPose2D
{
    public int index; // The index of the detected human pose
    public BodyPart2D[] bodyParts; // An array of the body parts that make up the human pose

    /// <summary>
    /// Initializes a new instance of the HumanPose2D struct.
    /// </summary>
    /// <param name="index">The index of the detected human pose.</param>
    /// <param name="bodyParts">An array of body parts that make up the human pose.</param>
    public HumanPose2D(int index, BodyPart2D[] bodyParts)
    {
        this.index = index;
        this.bodyParts = bodyParts;
    }
}
```



#### `HumanPose2DUtility` static class
This class contains a single static method that scales and optionally mirrors the coordinates of a body part in a pose skeleton to match the in-game screen and display resolutions.

```c#
public static class HumanPose2DUtility
{

    /// <summary>
    /// Scales and optionally mirrors the coordinates of a body part in a pose skeleton to match the in-game screen and display resolutions.
    /// </summary>
    /// <param name="coordinates">The (x,y) coordinates for a BodyPart object.</param>
    /// <param name="inputDims">The dimensions of the input image used for pose estimation.</param>
    /// <param name="screenDims">The dimensions of the in-game screen where the body part will be displayed.</param>
    /// <param name="offset">An offset to apply to the body part coordinates when scaling.</param>
    /// <param name="mirrorScreen">A boolean flag to indicate if the body part coordinates should be mirrored horizontally (default is false).</param>
    public static Vector2 ScaleBodyPartCoords(Vector2 coordinates, Vector2Int inputDims, Vector2 screenDims, Vector2Int offset, bool mirrorScreen)
    {
        // The smallest dimension of the screen
        float minScreenDim = Mathf.Min(screenDims.x, screenDims.y);
        // The smallest input dimension
        int minInputDim = Mathf.Min(inputDims.x, inputDims.y);
        // Calculate the scale value between the in-game screen and input dimensions
        float minImgScale = minScreenDim / minInputDim;
        // Calculate the scale value between the in-game screen and display
        float displayScaleX = Screen.width / screenDims.x;
        float displayScaleY = Screen.height / screenDims.y;
        float displayScale = Mathf.Min(displayScaleX, displayScaleY);


        // Scale body part coordinates to in-game screen resolution and flip the coordinates vertically
        float x = (coordinates.x + offset.x) * minImgScale;
        float y = (inputDims.y - (coordinates.y - offset.y)) * minImgScale;

        // Mirror bounding box across screen
        if (mirrorScreen)
        {
            x = screenDims.x - x;
        }

        // Scale coordinates to display resolution
        coordinates.x = x * displayScale;
        coordinates.y = y * displayScale;

        // Offset the coordinates coordinates based on the difference between the in-game screen and display
        coordinates.x += (Screen.width - screenDims.x * displayScale) / 2;
        coordinates.y += (Screen.height - screenDims.y * displayScale) / 2;

        return coordinates;
    }
}
```





---



### `HumanPose2DVisualizer.cs`
The HumanPose2DVisualizer script is a Unity C# `MonoBehaviour` class that displays 2D human pose skeletons on a Unity canvas. It creates, updates, and manages UI elements for visualizing them based on the provided `HumanPose2D` array. The complete code is available on GitHub at the link below.

- [HumanPose2DVisualizer.cs](https://github.com/cj-mills/unity-human-pose-2d-toolkit/blob/main/Runtime/Scripts/HumanPose2DVisualizer.cs)



#### Serialized Fields

The script contains several fields for prefabs and configuring pose skeleton visualizations.

```c#
// Main canvas to display poses
[Header("UI Components")]
[Tooltip("The main canvas to display poses")]
[SerializeField] private Canvas canvas;

// Prefabs for pose containers, joints, and bones
[Tooltip("The prefab for the pose container, which holds the joints and bones")]
[SerializeField] private RectTransform poseContainerPrefab;
[Tooltip("The prefab for the joint image")]
[SerializeField] private Image jointPrefab;
[Tooltip("The prefab for the bone RectTransform")]
[SerializeField] private RectTransform bonePrefab;

// Configuration and styling
[Header("Configuration")]
[Tooltip("The JSON file containing body part connection information")]
[SerializeField] private TextAsset bodyPartConnectionsFile;
[Tooltip("The color of the bones")]
[SerializeField] private Color boneColor = Color.green;
[Tooltip("The color of the joints")]
[SerializeField] private Color jointColor = Color.green;
```



#### Serialized Classes
There are a couple of nested serialized classes to store body part connection information from a JSON file.

```c#
// Serializable classes to store body part connection information from JSON
[System.Serializable]
class BodyPartConnection
{
    public int from; // Index of the starting body part
    public int to;   // Index of the ending body part
}

[System.Serializable]
class BodyPartConnectionList
{
    public List<BodyPartConnection> bodyPartConnections; // List of body part connections
}
```





#### Private Variables

```c#
// Variables to store runtime instances and data
private List<BodyPartConnection> bodyPartConnections; // List of body part connections
private List<RectTransform> poseContainers = new List<RectTransform>(); // List of instantiated pose containers
private List<List<Image>> joints = new List<List<Image>>(); // Nested list of instantiated joint images
private List<List<RectTransform>> bones = new List<List<RectTransform>>(); // Nested list of instantiated bone RectTransforms
private float confidenceThreshold; // Confidence threshold for displaying poses
```



#### GUID Constants

These are the GUIDs of the default assets.

```c#
// GUIDs of the default assets
private const string PoseContainerPrefabGUID = "12c840be0a8d4adc879fc14fb79a316d";
private const string JointPrefabGUID = "d90f7f2e5b8f4daa885f9441f0f33427";
private const string BonePrefabGUID = "ed947d23b5354617b130aa8ee0cc610b";
private const string BodyPartConnectionsFileGUID = "0fc008c60a8e44589674b0f455384a5b";
```



#### `Reset`

This method sets the default assets from the project using their GUIDs. It uses `AssetDatabase` to find them and set the default values. This method will only work in the Unity Editor, not in a build.

```c#
/// <summary>
/// Reset is called when the user hits the Reset button in the Inspector's context menu
/// or when adding the component the first time. This function is only called in editor mode.
/// </summary>
private void Reset()
{
    // Load default assets only in the Unity Editor, not in a build
#if UNITY_EDITOR
    poseContainerPrefab = LoadDefaultAsset<RectTransform>(PoseContainerPrefabGUID);
    jointPrefab = LoadDefaultAsset<Image>(JointPrefabGUID);
    bonePrefab = LoadDefaultAsset<RectTransform>(BonePrefabGUID);
    bodyPartConnectionsFile = LoadDefaultAsset<TextAsset>(BodyPartConnectionsFileGUID);
#endif
}
```



#### `LoadDefaultAsset`

This method provides a generic way to load default assets for the specified fields using their GUIDs.

```c#
/// <summary>
/// Loads the default asset for the specified type using its GUID.
/// </summary>
/// <typeparam name="T">The type of asset to be loaded.</typeparam>
/// <param name="guid">The GUID of the default asset.</param>
/// <returns>The loaded asset of the specified type.</returns>
/// <remarks>
/// This method is only executed in the Unity Editor, not in builds.
/// </remarks>
private T LoadDefaultAsset<T>(string guid) where T : UnityEngine.Object
{
#if UNITY_EDITOR
    // Load the asset from the AssetDatabase using its GUID
    return UnityEditor.AssetDatabase.LoadAssetAtPath<T>(UnityEditor.AssetDatabase.GUIDToAssetPath(guid));
#else
    return null;
#endif
}
```





#### `Start`
This method runs when the script initializes and loads the body part connection list from the JSON file.

```c#
private void Start()
{
    LoadBodyPartConnectionList();
}
```



#### `LoadBodyPartConnectionList`
This method deserializes the JSON file specifying the body part connections for pose skeletons.

```c#
/// <summary>
/// Load the JSON file
/// <summary>
private void LoadBodyPartConnectionList()
{
    if (IsJsonNullOrEmpty())
    {
        Debug.LogError("JSON file is null or empty.");
        return;
    }

    bodyPartConnections = DeserializeBodyPartConnectionsList(bodyPartConnectionsFile.text).bodyPartConnections;
}
```



#### `IsJsonNullOrEmpty`
This method checks if the JSON file is null or empty.

```c#
/// <summary>
/// Check if JSON file is null or empty
/// <summary>
private bool IsJsonNullOrEmpty()
{
    return bodyPartConnectionsFile == null || string.IsNullOrWhiteSpace(bodyPartConnectionsFile.text);
}
```



#### `DeserializeBodyPartConnectionsList`
This method deserializes the JSON string into a `BodyPartConnectionList`.

```c#
/// <summary>
/// Deserialize the JSON string
/// <summary>
private BodyPartConnectionList DeserializeBodyPartConnectionsList(string json)
{
    try
    {
        return JsonUtility.FromJson<BodyPartConnectionList>(json);
    }
    catch (Exception ex)
    {
        Debug.LogError($"Failed to deserialize class labels JSON: {ex.Message}");
        return null;
    }
}
```



#### `UpdatePoseVisualizations`
This method updates pose visualizations based on the provided human poses and a confidence threshold.

```c#
/// <summary>
/// Updates the pose visualizations based on the provided human poses and a confidence threshold.
/// </summary>
/// <param name="humanPoses">An array of human poses to visualize</param>
/// <param name="confidenceThreshold">The minimum confidence required to display a pose (default is 0.5f)</param>
public void UpdatePoseVisualizations(HumanPose2D[] humanPoses, float confidenceThreshold = 0.5f)
{
    this.confidenceThreshold = confidenceThreshold;

    // Instantiate pose containers, joint images, and bone RectTransforms as needed to match the number of humanPoses
    while (poseContainers.Count < humanPoses.Length)
    {
        RectTransform newPoseContainer = Instantiate(poseContainerPrefab, canvas.transform);
        poseContainers.Add(newPoseContainer);
        joints.Add(new List<Image>());
        bones.Add(new List<RectTransform>());
    }

    for (int i = 0; i < poseContainers.Count; i++)
    {
        if (i < humanPoses.Length)
        {
            // Get references to joint and bone containers for the current pose
            RectTransform jointContainer = poseContainers[i].Find("JointContainer").GetComponent<RectTransform>();
            RectTransform boneContainer = poseContainers[i].Find("BoneContainer").GetComponent<RectTransform>();

            // Update the joint positions and visibility
            UpdateJoints(humanPoses[i].bodyParts, jointContainer, joints[i]);
            // Update the bone positions, rotations, and visibility
            UpdateBones(humanPoses[i].bodyParts, boneContainer, joints[i], bones[i]);

            // Set the pose container active
            poseContainers[i].gameObject.SetActive(true);
        }
        else
        {
            // Set the pose container inactive for unused containers
            poseContainers[i].gameObject.SetActive(false);
        }
    }
}
```



#### `ScreenToCanvasPoint`
This method convert a screen point to a local one within the given canvas `RectTransform`.

```c#
/// <summary>
/// Converts a screen point to a local point within the given canvas RectTransform.
/// </summary>
/// <param name="canvas">The canvas RectTransform to convert the point to</param>
/// <param name="screenPoint">The screen point to convert</param>
/// <returns>A Vector2 representing the local point within the canvas RectTransform</returns>
private Vector2 ScreenToCanvasPoint(RectTransform canvas, Vector2 screenPoint)
{
    RectTransformUtility.ScreenPointToLocalPointInRectangle(canvas, screenPoint, null, out Vector2 localPoint);
    return localPoint;
}
```



#### `UpdateJoints`
This method updates joint visualizations based on the provided body parts, adjusting their positions and visibility.

```c#
/// <summary>
/// Updates the joint visualizations based on the provided body parts, adjusting their positions and visibility.
/// </summary>
/// <param name="bodyParts">An array of body parts containing position and probability data</param>
/// <param name="jointContainer">The RectTransform containing joint images</param>
/// <param name="jointsList">A list of instantiated joint images</param>
private void UpdateJoints(BodyPart2D[] bodyParts, RectTransform jointContainer, List<Image> jointsList)
{
    // Instantiate joint images as needed to match the number of bodyParts
    while (jointsList.Count < bodyParts.Length)
    {
        Image newJoint = Instantiate(jointPrefab, jointContainer);
        jointsList.Add(newJoint);
    }

    for (int i = 0; i < jointsList.Count; i++)
    {
        if (bodyParts[i].prob >= confidenceThreshold)
        {
            Image joint = jointsList[i];
            RectTransform jointRect = joint.rectTransform;
            // Update joint position
            jointRect.anchoredPosition = ScreenToCanvasPoint(jointContainer, bodyParts[i].coordinates);
            // Update joint color
            joint.color = jointColor;
            // Set the joint game object active
            joint.gameObject.SetActive(true);
        }
        else
        {
            // Set the joint game object inactive if below the confidence threshold
            jointsList[i].gameObject.SetActive(false);
        }
    }
}
```



#### `UpdateBones`
This method updates bone visualizations based on the provided body parts and joint positions, adjusting their positions, rotations, and visibility.

```c#
/// <summary>
/// Updates the bone visualizations based on the provided body parts and joint positions, adjusting their positions, rotations, and visibility.
/// </summary>
/// <param name="bodyParts">An array of body parts containing position and probability data</param>
/// <param name="boneContainer">The RectTransform containing bone RectTransforms</param>
/// <param name="jointsList">A list of instantiated joint images</param>
/// <param name="bonesList">A list of instantiated bone RectTransforms</param>
private void UpdateBones(BodyPart2D[] bodyParts, RectTransform boneContainer, List<Image> jointsList, List<RectTransform> bonesList)
{
    // Instantiate bone RectTransforms as needed to match the number of bodyPartConnections
    while (bonesList.Count < bodyPartConnections.Count)
    {
        RectTransform newBone = Instantiate(bonePrefab, boneContainer);
        bonesList.Add(newBone);
    }

    for (int i = 0; i < bonesList.Count; i++)
    {
        Image fromJoint = jointsList[bodyPartConnections[i].from];
        Image toJoint = jointsList[bodyPartConnections[i].to];

        // If both connected joints are active, display the bone
        if (fromJoint.IsActive() && toJoint.IsActive())
        {
            RectTransform bone = bonesList[i];
            Vector2 fromJointPos = bodyParts[bodyPartConnections[i].from].coordinates;
            Vector2 toJointPos = bodyParts[bodyPartConnections[i].to].coordinates;
            Vector2 direction = toJointPos - fromJointPos;
            float distance = direction.magnitude;
            float angle = Mathf.Atan2(direction.y, direction.x) * Mathf.Rad2Deg;

            // Update bone size based on the distance between joints
            bone.sizeDelta = new Vector2(distance, bone.sizeDelta.y);

            // Calculate the bone position and update it
            Vector2 bonePos = new Vector2((fromJointPos.x + toJointPos.x) / 2, (fromJointPos.y + toJointPos.y) / 2);
            bone.anchoredPosition = ScreenToCanvasPoint(boneContainer, bonePos);

            // Update bone rotation based on the angle between joints
            bone.localEulerAngles = new Vector3(0, 0, angle);
            bone.GetComponent<Image>().color = boneColor;
            // Set the bone game object active
            bone.gameObject.SetActive(true);
        }
        else
        {
            // Set the bone game object inactive if below the confidence threshold
            bonesList[i].gameObject.SetActive(false);
        }
    }
}
```








---



### `AddCustomDefineSymbol.cs`

This Editor script contains a class that adds a custom define symbol  to the project. We can use this custom symbol to prevent code that  relies on this package from executing unless the Human Pose 2D Toolkit package is present. The complete code is available on GitHub at the link below.

* [AddCustomDefineSymbol.cs](https://github.com/cj-mills/unity-human-pose-2d-toolkit/blob/main/Editor/AddCustomDefineSymbol.cs)

```c#
using UnityEditor;
using UnityEngine;

namespace CJM.HumanPose2DToolkit
{
    public class DependencyDefineSymbolAdder
    {
        private const string CustomDefineSymbol = "CJM_HUMAN_POSE_2D_TOOLKIT";

        [InitializeOnLoadMethod]
        public static void AddCustomDefineSymbol()
        {
            // Get the currently selected build target group
            var buildTargetGroup = EditorUserBuildSettings.selectedBuildTargetGroup;
            // Retrieve the current scripting define symbols for the selected build target group
            var defines = PlayerSettings.GetScriptingDefineSymbolsForGroup(buildTargetGroup);

            // Check if the CustomDefineSymbol is already present in the defines string
            if (!defines.Contains(CustomDefineSymbol))
            {
                // Append the CustomDefineSymbol to the defines string, separated by a semicolon
                defines += $";{CustomDefineSymbol}";
                // Set the updated defines string as the new scripting define symbols for the selected build target group
                PlayerSettings.SetScriptingDefineSymbolsForGroup(buildTargetGroup, defines);
                // Log a message in the Unity console to inform the user that the custom define symbol has been added
                Debug.Log($"Added custom define symbol '{CustomDefineSymbol}' to the project.");
            }
        }
    }
}
```



## Conclusion

This post provided an in-depth walkthrough of the code for the Unity Human Pose 2D Toolkit package. The package provides an easy-to-use and customizable solution to work with and visualize 2D human poses on a Unity canvas.

You can continue to explore the package by going to its GitHub repository linked below, where you will also find instructions for installing it using the Unity Package Manager.

- GitHub Repository: [unity-human-pose-2d-toolkit](https://github.com/cj-mills/unity-barracuda-inference-posenet)

You can find the code for the demo project shown in the video at the beginning of this post linked below.

- [Barracuda Inference PoseNet Demo](https://github.com/cj-mills/barracuda-inference-posenet-demo): A simple Unity project demonstrating how to perform 2D human pose estimation with the `barracuda-inference-posenet` package.





