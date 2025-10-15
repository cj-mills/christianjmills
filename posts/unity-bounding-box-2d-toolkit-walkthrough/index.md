---
title: "Code Walkthrough: Unity Bounding Box 2D Toolkit Package"
date: 2023-5-5
image: /images/empty.gif
hide: false
search_exclude: false
categories: [unity, walkthrough]
description: "Walk through the code for the Unity Bounding Box 2D Toolkit package, which provides an easy-to-use and customizable solution to work with and visualize 2D bounding boxes on a Unity canvas."

toc-depth: 5

twitter-card:
  creator: "@cdotjdotmills"
  site: "@cdotjdotmills"
  image: /images/default-preview-image-black.png
open-graph:
  image: /images/default-preview-image-black.png
---





* [Introduction](#introduction)
* [Package Overview](#package-overview)
* [Code Explanation](#code-explanation)
* [Conclusion](#conclusion)




## Introduction

The [Unity Bounding Box 2D Toolkit](https://github.com/cj-mills/unity-bounding-box-2d-toolkit) package provides an easy-to-use and customizable solution to work with and visualize 2D bounding boxes on a Unity canvas. 

Some of my tutorials involve using 2D object detection models in Unity applications. This package makes that shared functionality more modular and reusable, allowing me to streamline my tutorial content. Check out the demo video below to see this package in action.



![](./videos/barracuda-inference-yolox-demo.mp4){fig-align="center"}



In this post, I'll walk through the package code, providing a solid understanding of its components and their roles.




## Package Overview

The package contains three C# scripts and prefabs to construct 2D bounding boxes.



### C# Scripts

1. `BBox2DUtils.cs`: This script provides functionality to work with 2D bounding boxes for object detection tasks.
2. `BoundingBox2DVisualizer.cs`: This script creates, updates, and manages UI elements for visualizing 2D bounding boxes.
3. `AddCustomDefineSymbol.cs`: An Editor script that automatically adds a custom scripting define symbol to the project after the package installs.



### Prefabs

1. `BoundingBoxBorderPrefab.prefab`: The BoundingBox2DVisualizer script uses this prefab to construct the sides of bounding boxes.
2. `DotPrefab.prefab`: An Image prefab used to display a dot at the center of bounding boxes
3. `LabelPrefab.prefab`: A text prefab used to display the class label associated probability score for a bounding box.
4. `LabelBackgroundPrefab.prefab`: An Image prefab used as the background for the LabelPrefab.
5. `BBox2DVisualizer.prefab`: This prefab helps simplify adding 2D bounding box visualization to a Unity scene. The prefab already has the BoundingBox2DVisualizer script attached and has a child Canvas component.





## Code Explanation

In this section, we will delve deeper into the Unity Bounding Box 2D Toolkit package by examining the purpose and functionality of each C# script.



### `BBox2DUtils.cs`

The BBox2DUtils.cs script provides functionality to work with 2D bounding boxes for object detection tasks. It contains two structs and a utility class. The complete code is available on GitHub at the link below.

- [BBox2DUtils.cs](https://github.com/cj-mills/unity-bounding-box-2d-toolkit/blob/main/Runtime/Scripts/BBox2DUtils.cs)



#### `BBox2D` struct

This struct contains the coordinates (x0, y0), width, height, class index, and probability value for a 2D bounding box.

```c#
/// <summary>
/// A struct that represents a 2D bounding box.
/// </summary>
public struct BBox2D
{
    public float x0;
    public float y0;
    public float width;
    public float height;
    public int index;
    public float prob;

    /// <summary>
    /// Initializes a new instance of the BBox2D struct.
    /// </summary>
    /// <param name="x0">The x-coordinate of the top-left corner.</param>
    /// <param name="y0">The y-coordinate of the top-left corner.</param>
    /// <param name="width">The width of the bounding box.</param>
    /// <param name="height">The height of the bounding box.</param>
    /// <param name="index">The class index of the object.</param>
    /// <param name="prob">The probability of the object belonging to the given class.</param>
    public BBox2D(float x0, float y0, float width, float height, int index, float prob)
    {
        this.x0 = x0;
        this.y0 = y0;
        this.width = width;
        this.height = height;
        this.index = index;
        this.prob = prob;
    }
}
```



#### `BBox2DInfo` struct

This struct contains a BBox2D object, class label, and color.

```c#
/// <summary>
/// A struct for 2D bounding box information.
/// </summary>
public struct BBox2DInfo
{
    public BBox2D bbox;
    public string label;
    public Color color;

    /// <summary>
    /// Initializes a new instance of the BBox2DInfo struct.
    /// </summary>
    /// <param name="boundingBox">The 2D bounding box.</param>
    /// <param name="label">The class label.</param>
    /// <param name="width">The bounding box color.</param>
    public BBox2DInfo(BBox2D boundingBox, string label = "", Color color = new Color())
    {
        this.bbox = boundingBox;
        this.label = label;
        this.color = color;
    }
}
```



#### `BBox2DUtility` class

This class provides various utility methods for working with bounding boxes.



##### `CalcUnionArea`

This method calculates the union area between two bounding boxes.

```c#
/// <summary>
/// Calculates the union area between two bounding boxes.
/// </summary>
/// <param name="a">The first bounding box.</param>
/// <param name="b">The second bounding box.</param>
/// <returns>The union area between the two bounding boxes.</returns>
public static float CalcUnionArea(BBox2D a, BBox2D b)
{
    // Calculate the coordinates and dimensions of the union area
    float x = Mathf.Min(a.x0, b.x0);
    float y = Mathf.Min(a.y0, b.y0);
    float w = Mathf.Max(a.x0 + a.width, b.x0 + b.width) - x;
    float h = Mathf.Max(a.y0 + a.height, b.y0 + b.height) - y;

    // Calculate the union area of two bounding boxes
    return w * h;
}
```



##### `CalcInterArea`

This method calculates the intersection area between two bounding boxes.

```c#
/// <summary>
/// Calculates the intersection area between two bounding boxes.
/// </summary>
/// <param name="a">The first bounding box.</param>
/// <param name="b">The second bounding box.</param>
/// <returns>The intersection area between the two bounding boxes.</returns>
public static float CalcInterArea(BBox2D a, BBox2D b)
{
    // Calculate the coordinates and dimensions of the intersection area
    float x = Mathf.Max(a.x0, b.x0);
    float y = Mathf.Max(a.y0, b.y0);
    float w = Mathf.Min(a.x0 + a.width, b.x0 + b.width) - x;
    float h = Mathf.Min(a.y0 + a.height, b.y0 + b.height) - y;

    // Calculate the intersection area of two bounding boxes
    return w * h;
}
```



##### `NMSSortedBoxes`

This method performs Non-Maximum Suppression (NMS) on a sorted list of bounding box proposals, retaining only those with an intersection over union (IoU) value below the threshold.

```c#
/// <summary>
/// Performs Non-Maximum Suppression (NMS) on a sorted list of bounding box proposals.
/// </summary>
/// <param name="proposals">A sorted list of BBox2D objects representing the bounding box proposals.</param>
/// <param name="nms_thresh">The NMS threshold for filtering proposals (default is 0.45).</param>
/// <returns>A list of integers representing the indices of the retained proposals.</returns>
public static List<int> NMSSortedBoxes(List<BBox2D> proposals, float nms_thresh = 0.45f)
{
    // Iterate through the proposals and perform non-maximum suppression
    List<int> proposal_indices = new List<int>();

    for (int i = 0; i < proposals.Count; i++)
    {
        // Calculate the intersection and union areas
        BBox2D a = proposals[i];
        bool keep = proposal_indices.All(j =>
        {
            BBox2D b = proposals[j];
            float inter_area = CalcInterArea(a, b);
            float union_area = CalcUnionArea(a, b);
            // Keep the proposal if its IoU with all previous proposals is below the NMS threshold
            return inter_area / union_area <= nms_thresh;
        });

        // If the proposal passes the NMS check, add its index to the list
        if (keep) proposal_indices.Add(i);
    }

    return proposal_indices;
}
```



##### `ScaleBoundingBox`

This method scales and optionally mirrors the bounding box of a detected object to match the in-game screen and display resolutions.

```c#
/// <summary>
/// Scales and optionally mirrors the bounding box of a detected object to match the in-game screen and display resolutions.
/// </summary>
/// <param name="boundingBox">A BBox2D object containing the bounding box information for a detected object.</param>
/// <param name="inputDims">The dimensions of the input image used for object detection.</param>
/// <param name="screenDims">The dimensions of the in-game screen where the bounding boxes will be displayed.</param>
/// <param name="offset">An offset to apply to the bounding box coordinates when scaling.</param>
/// <param name="mirrorScreen">A boolean flag to indicate if the bounding boxes should be mirrored horizontally (default is false).</param>
public static BBox2D ScaleBoundingBox(BBox2D boundingBox, Vector2Int inputDims, Vector2 screenDims, Vector2Int offset, bool mirrorScreen)
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


    // Scale bounding box to in-game screen resolution and flip the bbox coordinates vertically
    float x0 = (boundingBox.x0 + offset.x) * minImgScale;
    float y0 = (inputDims.y - (boundingBox.y0 - offset.y)) * minImgScale;
    float width = boundingBox.width * minImgScale;
    float height = boundingBox.height * minImgScale;

    // Mirror bounding box across screen
    if (mirrorScreen)
    {
        x0 = screenDims.x - x0 - width;
    }

    // Scale bounding box to display resolution
    boundingBox.x0 = x0 * displayScale;
    boundingBox.y0 = y0 * displayScale;
    boundingBox.width = width * displayScale;
    boundingBox.height = height * displayScale;

    // Offset the bounding box coordinates based on the difference between the in-game screen and display
    boundingBox.x0 += (Screen.width - screenDims.x * displayScale) / 2;
    boundingBox.y0 += (Screen.height - screenDims.y * displayScale) / 2;

    return boundingBox;
}
```



---



### `BoundingBox2DVisualizer.cs`
The `BoundingBox2DVisualizer` script is a Unity C# MonoBehaviour class that displays 2D bounding boxes and labels on a Unity canvas. It creates, updates, and manages UI elements for visualizing them based on the provided BBox2DInfo array. This class supports customizable settings such as bounding box transparency and the ability to toggle the display of bounding boxes. The complete code is available on GitHub at the link below.

* [BoundingBox2DVisualizer.cs](https://github.com/cj-mills/unity-bounding-box-2d-toolkit/blob/main/Runtime/Scripts/BoundingBox2DVisualizer.cs)



#### Serialized Fields

The BoundingBox2DVisualizer class contains serialized fields referencing UI components, prefabs, and settings.

```c#
// UI components
[Header("Components")]
[Tooltip("Container for holding the bounding box UI elements")]
[SerializeField] private RectTransform boundingBoxContainer;
[Tooltip("Container for holding the label UI elements")]
[SerializeField] private RectTransform labelContainer;

// Prefabs for creating UI elements
[Header("Prefabs")]
[Tooltip("Prefab for the bounding box UI element")]
[SerializeField] private RectTransform boundingBoxPrefab;
[Tooltip("Prefab for the label UI element")]
[SerializeField] private TMP_Text labelPrefab;
[Tooltip("Prefab for the label background UI element")]
[SerializeField] private Image labelBackgroundPrefab;
[Tooltip("Prefab for the dot UI element")]
[SerializeField] private Image dotPrefab;

// Settings for customizing the bounding box visualizer
[Header("Settings")]
[Tooltip("Flag to control whether bounding boxes should be displayed or not")]
[SerializeField] private bool displayBoundingBoxes = true;
[Tooltip("Transparency value for the bounding boxes, ranging from 0 (completely transparent) to 1 (completely opaque)")]
[SerializeField, Range(0f, 1f)] private float bboxTransparency = 1f;
```



#### GUID Constants

These are the GUIDs of the default assets. They are used to set default values for the bounding box, label, label background, and dot prefabs in the Unity Editor.

```c#
// GUIDs of the default assets for the bounding box, label, label background, and dot prefabs
private const string BoundingBoxPrefabGUID = "be0edeacc0f249fab31ac75426ad8a2a";
private const string LabelPrefabGUID = "4e39b47d4b984862aeab14255855fcc9";
private const string LabelBackgroundPrefabGUID = "9074ea186151430084312ba891bad58e";
private const string DotPrefabGUID = "3eb64b4f1a4e4e2595066ed269be9532";
```



#### Lists for UI Elements

```c#
// Lists for storing and managing instantiated UI elements
private List<RectTransform> boundingBoxes = new List<RectTransform>(); // List of instantiated bounding box UI elements
private List<TMP_Text> labels = new List<TMP_Text>(); // List of instantiated label UI elements
private List<Image> labelBackgrounds = new List<Image>(); // List of instantiated label background UI elements
private List<Image> dots = new List<Image>(); // List of instantiated dot UI elements
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
    boundingBoxPrefab = LoadDefaultAsset<RectTransform>(BoundingBoxPrefabGUID);
    labelPrefab = LoadDefaultAsset<TMP_Text>(LabelPrefabGUID);
    labelBackgroundPrefab = LoadDefaultAsset<Image>(LabelBackgroundPrefabGUID);
    dotPrefab = LoadDefaultAsset<Image>(DotPrefabGUID);
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



#### `UpdateBoundingBoxVisualizations`
This method updates the visualization of bounding boxes based on the given `BBox2DInfo` array.

```c#
/// <summary>
/// Update the visualization of bounding boxes based on the given BBox2DInfo array.
/// </summary>
/// <param name="bboxInfoArray">An array of BBox2DInfo objects containing bounding box information</param>
public void UpdateBoundingBoxVisualizations(BBox2DInfo[] bboxInfoArray)
{
    // Depending on the displayBoundingBoxes flag, either update or disable bounding box UI elements
    if (displayBoundingBoxes)
    {
        UpdateBoundingBoxes(bboxInfoArray);
    }
    else
    {
        // Disable bounding boxes, labels, and label backgrounds for all existing UI elements
        for (int i = 0; i < boundingBoxes.Count; i++)
        {
            boundingBoxes[i].gameObject.SetActive(false);
            labelBackgrounds[i].gameObject.SetActive(false);
            labels[i].gameObject.SetActive(false);
            dots[i].gameObject.SetActive(false);
        }
    }
}
```



#### `ScreenToCanvasPoint`
This method converts a screen point to a local one in the `RectTransform` space of the given canvas.

```c#
/// <summary>
/// Convert a screen point to a local point in the RectTransform space of the given canvas.
/// </summary>
/// <param name="canvas">The RectTransform object of the canvas</param>
/// <param name="screenPoint">The screen point to be converted</param>
/// <returns>A Vector2 object representing the local point in the RectTransform space of the canvas</returns>
private Vector2 ScreenToCanvasPoint(RectTransform canvas, Vector2 screenPoint)
{
    RectTransformUtility.ScreenPointToLocalPointInRectangle(canvas, screenPoint, null, out Vector2 localPoint);
    return localPoint;
}
```



#### `UpdateBoundingBoxes`
This method updates bounding box UI elements to match the provided `BBox2DInfo` array. It creates or removes bounding box UI elements to match the number of detected objects and updates bounding boxes, labels, and label backgrounds. It also disables UI elements if not needed.

```c#
/// <summary>
/// Update bounding box UI elements to match the provided BBox2DInfo array.
/// </summary>
/// <param name="bboxInfoArray">An array of BBox2DInfo objects containing bounding box information</param>
private void UpdateBoundingBoxes(BBox2DInfo[] bboxInfoArray)
{
    // Create or remove bounding box UI elements to match the number of detected objects
    while (boundingBoxes.Count < bboxInfoArray.Length)
    {
        RectTransform newBoundingBox = Instantiate(boundingBoxPrefab, boundingBoxContainer);
        boundingBoxes.Add(newBoundingBox);

        Image newLabelBackground = Instantiate(labelBackgroundPrefab, labelContainer);
        labelBackgrounds.Add(newLabelBackground);

        TMP_Text newLabel = Instantiate(labelPrefab, labelContainer);
        labels.Add(newLabel);

        Image newDot = Instantiate(dotPrefab, boundingBoxContainer);
        dots.Add(newDot);
    }

    // Update bounding boxes, labels, and label backgrounds for each detected object, or disable UI elements if not needed
    for (int i = 0; i < boundingBoxes.Count; i++)
    {
        if (i < bboxInfoArray.Length)
        {
            BBox2DInfo bboxInfo = bboxInfoArray[i];

            // Get UI elements for the current bounding box, label, and label background
            RectTransform boundingBox = boundingBoxes[i];
            TMP_Text label = labels[i];
            Image labelBackground = labelBackgrounds[i];
            Image dot = dots[i];

            UpdateBoundingBox(boundingBox, bboxInfo);
            UpdateLabelAndBackground(label, labelBackground, bboxInfo);
            UpdateDot(dot, bboxInfo);

            // Enable bounding box, label, and label background UI elements
            boundingBox.gameObject.SetActive(true);
            labelBackground.gameObject.SetActive(true);
            label.gameObject.SetActive(true);
            dots[i].gameObject.SetActive(true);
        }
        else
        {
            // Disable UI elements for extra bounding boxes, labels, and label backgrounds
            boundingBoxes[i].gameObject.SetActive(false);
            labelBackgrounds[i].gameObject.SetActive(false);
            labels[i].gameObject.SetActive(false);
            dots[i].gameObject.SetActive(false);
        }
    }
}
```



#### `UpdateBoundingBox`
This method updates the bounding box UI element with the information from the given `BBox2DInfo` object. It converts the screen point to a local one in the `RectTransform` space of the bounding box container and sets the color of the bounding box with the specified transparency.

```c#
/// <summary>
/// Update the bounding box UI element with the information from the given BBox2DInfo object.
/// </summary>
/// <param name="boundingBox">The RectTransform object representing the bounding box UI element</param>
/// <param name="bboxInfo">The BBox2DInfo object containing the information for the bounding box</param>
private void UpdateBoundingBox(RectTransform boundingBox, BBox2DInfo bboxInfo)
{
    // Convert the screen point to a local point in the RectTransform space of the bounding box container
    Vector2 localPosition = ScreenToCanvasPoint(boundingBoxContainer, new Vector2(bboxInfo.bbox.x0, bboxInfo.bbox.y0));
    boundingBox.anchoredPosition = localPosition;
    boundingBox.sizeDelta = new Vector2(bboxInfo.bbox.width, bboxInfo.bbox.height);

    // Set the color of the bounding box with the specified transparency
    Color color = GetColorWithTransparency(bboxInfo.color);
    Image[] sides = boundingBox.GetComponentsInChildren<Image>();
    foreach (Image side in sides)
    {
        side.color = color;
    }
}
```



#### `UpdateLabelAndBackground`
This method updates the label and label background UI elements with the information from the provided `BBox2DInfo` object. It sets the label text, position, color, and the label's background position, size, and color with the specified transparency.

```c#
/// <summary>
/// Update the label and label background UI elements with the information from the given BBox2DInfo object.
/// </summary>
/// <param name="label">The TMP_Text object representing the label UI element</param>
/// <param name="labelBackground">The Image object representing the label background UI element</param>
/// <param name="bboxInfo">The BBox2DInfo object containing the information for the label and label background</param>
private void UpdateLabelAndBackground(TMP_Text label, Image labelBackground, BBox2DInfo bboxInfo)
{
    // Convert the screen point to a local point in the RectTransform space of the bounding box container
    Vector2 localPosition = ScreenToCanvasPoint(boundingBoxContainer, new Vector2(bboxInfo.bbox.x0, bboxInfo.bbox.y0));

    // Set the label text and position
    label.text = $"{bboxInfo.label}: {(bboxInfo.bbox.prob * 100).ToString("0.##")}%";
    label.rectTransform.anchoredPosition = new Vector2(localPosition.x, localPosition.y - label.preferredHeight);

    // Set the label color based on the grayscale value of the bounding box color
    Color color = GetColorWithTransparency(bboxInfo.color);
    label.color = color.grayscale > 0.5 ? Color.black : Color.white;

    // Set the label background position and size
    labelBackground.rectTransform.anchoredPosition = new Vector2(localPosition.x, localPosition.y - label.preferredHeight);
    labelBackground.rectTransform.sizeDelta = new Vector2(Mathf.Max(label.preferredWidth, bboxInfo.bbox.width), label.preferredHeight);

    // Set the label background color with the specified transparency
    labelBackground.color = color;
}
```



#### `UpdateDot`
This method updates the dot UI element based on the provided BBox2DInfo object.

```c#
/// <summary>
/// Update the dot UI element with the information from the given BBox2DInfo object.
/// </summary>
/// <param name="dot">The Image object representing the dot UI element</param>
/// <param name="bboxInfo">The BBox2DInfo object containing the information for the bounding box</param>
private void UpdateDot(Image dot, BBox2DInfo bboxInfo)
{
    // Calculate the center of the bounding box
    Vector2 center = new Vector2(bboxInfo.bbox.x0 + bboxInfo.bbox.width / 2, bboxInfo.bbox.y0 - bboxInfo.bbox.height / 2);

    // Convert the screen point to a local point in the RectTransform space of the bounding box container
    Vector2 localPosition = ScreenToCanvasPoint(boundingBoxContainer, center);

    // Set the dot position
    dot.rectTransform.anchoredPosition = localPosition;

    // Set the dot color with the specified transparency
    Color color = GetColorWithTransparency(bboxInfo.color);
    dot.color = color;
}
```



#### `GetColorWithTransparency`
This method is a utility function that returns a new color based on the input color with the adjusted transparency.

```c#
/// <summary>
/// Get a new color based on the input color with the adjusted transparency.
/// </summary>
/// <param name="color">The input color to be modified</param>
/// <returns>A new color with the specified transparency</returns>
private Color GetColorWithTransparency(Color color)
{
    color.a = bboxTransparency;
    return color;
}
```



---



### `AddCustomDefineSymbol.cs`

This Editor script contains a class that adds a custom define symbol  to the project. We can use this custom symbol to prevent code that  relies on this package from executing unless the Bounding Box 2D Toolkit package is present. The complete code is available on GitHub at the link below.

* [Editor/AddCustomDefineSymbol.cs](https://github.com/cj-mills/unity-bounding-box-2d-toolkit/blob/main/Editor/AddCustomDefineSymbol.cs)

```c#
using UnityEditor;
using UnityEngine;

namespace CJM.BBox2DToolkit
{
    public class DependencyDefineSymbolAdder
    {
        private const string CustomDefineSymbol = "CJM_BBOX_2D_TOOLKIT";

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

This post provided an in-depth walkthrough of the code for the Unity Bounding Box 2D Toolkit package. The package provides an easy-to-use and customizable solution to work with and visualize 2D bounding boxes on a Unity canvas.



You can continue to explore the package by going to its GitHub repository linked below, where you will also find instructions for installing it using the Unity Package Manager.

- GitHub Repository: [unity-bounding-box-2d-toolkit](https://github.com/cj-mills/unity-bounding-box-2d-toolkit)



You can find the code for the demo project shown in the video at the beginning of this post linked below.

- [Barracuda Inference YOLOX Demo](https://github.com/cj-mills/barracuda-inference-yolox-demo): A simple Unity project demonstrating how to perform object detection with the `barracuda-inference-yolox` package.









{{< include /_about-author-cta.qmd >}}
