---
title: "Code Walkthrough: Unity YOLOX Utilities Package"
date: 2023-5-5
image: /images/empty.gif
hide: false
search_exclude: false
categories: [unity, walkthrough]
description: "Walk through the code for the Unity YOLOX Utilities package, which provides utility functions to work with YOLOX object detection models in Unity."

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

The [Unity YOLOX Utilities](https://github.com/cj-mills/unity-yolox-utils) package provides utility functions to work with YOLOX object detection models in Unity. 

I use YOLOX models in multiple tutorials. This package makes that shared functionality more modular and reusable, allowing me to streamline my tutorial content. Here is a demo video from a project that uses this package.



![](./videos/barracuda-inference-yolox-demo.mp4){fig-align="center"}



In this post, I'll walk through the package code, providing a solid understanding of its components and their roles.




## Package Overview

The package contains three C# scripts.

1. `YOLOXUtils.cs`: This script provides a utility class for YOLOX-related operations.
2. `AddCustomDefineSymbol.cs`: An Editor script that automatically adds a custom scripting define symbol to the project after the package installs.
3. `PackageInstaller.cs`: An Editor utility script for automatically installing a list of dependency packages defined in a JSON file.



## Code Explanation

In this section, we will delve deeper into the Unity YOLOX Utilities package by examining the purpose and functionality of each C# script.



### `YOLOXUtils.cs`

This script utilizes the [Unity Bounding Box 2D Toolkit](https://github.com/cj-mills/unity-bounding-box-2d-toolkit) package and contains two main components: the `GridCoordinateAndStride` struct and the `YOLOXUtility` class.

The complete code is available on GitHub at the link below.

* [YOLOXUtils.cs](https://github.com/cj-mills/unity-yolox-utils/blob/main/Runtime/Scripts/YOLOXUtils.cs)



#### `GridCoordinateAndStride` struct
This struct represents the grid coordinates (x and y) and the stride of the grid cell.

```c#
/// <summary>
/// A struct for grid coordinates and stride information.
/// </summary>
public struct GridCoordinateAndStride
{
    public int xCoordinate;
    public int yCoordinate;
    public int stride;

    /// <summary>
    /// Initializes a new instance of the GridCoordinateAndStride struct.
    /// </summary>
    /// <param name="xCoordinate">The x-coordinate of the grid.</param>
    /// <param name="yCoordinate">The y-coordinate of the grid.</param>
    /// <param name="stride">The stride value for the grid.</param>
    public GridCoordinateAndStride(int xCoordinate, int yCoordinate, int stride)
    {
        this.xCoordinate = xCoordinate;
        this.yCoordinate = yCoordinate;
        this.stride = stride;
    }
}
```



#### `YOLOXUtility` class
This static utility class provides methods for YOLOX-related operations.

##### `GenerateGridCoordinatesWithStrides`
This method generates a list of `GridCoordinateAndStride` objects based on the input strides, grid height, and grid width.

```c#
/// <summary>
/// Generates a list of GridCoordinateAndStride objects based on input strides, height, and width.
/// </summary>
/// <param name="strides">An array of stride values.</param>
/// <param name="height">The height of the grid.</param>
/// <param name="width">The width of the grid.</param>
/// <returns>A list of GridCoordinateAndStride objects.</returns>
public static List<GridCoordinateAndStride> GenerateGridCoordinatesWithStrides(int[] strides, int height, int width)
{
    // Generate a list of GridCoordinateAndStride objects by iterating through possible grid positions and strides
    return strides.SelectMany(stride => Enumerable.Range(0, height / stride)
                                                   .SelectMany(g1 => Enumerable.Range(0, width / stride)
                                                                                .Select(g0 => new GridCoordinateAndStride(g0, g1, stride)))).ToList();
}
```



##### `GenerateBoundingBoxProposals`
This method generates a list of bounding box proposals based on the model output, grid strides, and other parameters.

```c#
/// <summary>
/// Generates a list of bounding box proposals based on the model output, grid strides, and other parameters.
/// </summary>
/// <param name="modelOutput">The output of the YOLOX model.</param>
/// <param name="gridCoordsAndStrides">A list of GridCoordinateAndStride objects.</param>
/// <param name="numClasses">The number of object classes.</param>
/// <param name="numBBoxFields">The number of bounding box fields.</param>
/// <param name="confidenceThreshold">The confidence threshold for filtering proposals.</param>
/// <returns>A list of BBox2D objects representing the generated proposals.</returns>
public static List<BBox2D> GenerateBoundingBoxProposals(float[] modelOutput, List<GridCoordinateAndStride> gridCoordsAndStrides, int numClasses, int numBBoxFields, float confidenceThreshold)
{
    int proposalLength = numClasses + numBBoxFields;

    // Process the model output to generate a list of BBox2D objects
    return gridCoordsAndStrides.Select((grid, anchorIndex) =>
    {
        int startIndex = anchorIndex * proposalLength;

        // Calculate coordinates and dimensions of the bounding box
        float centerX = (modelOutput[startIndex] + grid.xCoordinate) * grid.stride;
        float centerY = (modelOutput[startIndex + 1] + grid.yCoordinate) * grid.stride;
        float w = Mathf.Exp(modelOutput[startIndex + 2]) * grid.stride;
        float h = Mathf.Exp(modelOutput[startIndex + 3]) * grid.stride;

        // Initialize BBox2D object
        BBox2D obj = new BBox2D(
            centerX - w * 0.5f,
            centerY - h * 0.5f,
            w, h, 0, 0);

        // Compute objectness and class probabilities for each bounding box
        float box_objectness = modelOutput[startIndex + 4];

        for (int classIndex = 0; classIndex < numClasses; classIndex++)
        {
            float boxClassScore = modelOutput[startIndex + numBBoxFields + classIndex];
            float boxProb = box_objectness * boxClassScore;

            // Update the object with the highest probability and class label
            if (boxProb > obj.prob)
            {
                obj.index = classIndex;
                obj.prob = boxProb;
            }
        }

        return obj;
    })
    .Where(obj => obj.prob > confidenceThreshold) // Filter by confidence threshold
    .OrderByDescending(x => x.prob) // Sort by probability
    .ToList();
}
```





---



### `AddCustomDefineSymbol.cs`

This Editor script contains a class that adds a custom define symbol  to the project. We can use this custom symbol to prevent code that  relies on this package from executing unless the YOLOX Utilities package is present. The complete code is available on GitHub at the link below.

* [AddCustomDefineSymbol.cs](https://github.com/cj-mills/unity-yolox-utils/blob/main/Editor/AddCustomDefineSymbol.cs)

```c#
using UnityEditor;
using UnityEngine;

namespace CJM.YOLOXUtils
{
    public class DependencyDefineSymbolAdder
    {
        private const string CustomDefineSymbol = "CJM_YOLOX_UTILS";

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



---



### `PackageInstaller.cs`

In this section, we will go through the `PackageInstaller.cs` script and explain how each part of the code works to install the required packages. The complete code is available on GitHub at the link below.

- [PackageInstaller.cs](https://github.com/cj-mills/unity-yolox-utils/blob/main/Editor/PackageInstaller.cs)



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



#### PackageInstaller Class Variables
The PackageInstaller class contains several private static fields.

```c#
// Stores the AddRequest object for the current package to install.
private static AddRequest addRequest;
// A list of PackageData objects to install.
private static List<PackageData> packagesToInstall;
// The index of the current package to install.
private static int currentPackageIndex;

// GUID of the JSON file containing the list of packages to install
private const string PackagesJSONGUID = "487301ab13cf457b9c2ed07a3ec5c004";
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

This post provided an in-depth walkthrough of the code for the Unity YOLOX Utilities package. The package provides utility functions to work with YOLOX object detection models in Unity.



You can continue to explore the package by going to its GitHub repository linked below, where you will also find instructions for installing it using the Unity Package Manager.

- GitHub Repository: [unity-yolox-utils](https://github.com/cj-mills/unity-yolox-utils)



You can find the code for the demo project shown in the video at the beginning of this post linked below.

- [Barracuda Inference YOLOX Demo](https://github.com/cj-mills/barracuda-inference-yolox-demo): A simple Unity project demonstrating how to perform object detection with the `barracuda-inference-yolox` package.





