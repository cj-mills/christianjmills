---
title: "Code Walkthrough: Unity CV Image Gallery Package"
date: 2023-5-3
image: /images/empty.gif
hide: false
search_exclude: false
categories: [unity, walkthrough]
description: "Walk through the code for the Unity CV Image Gallery package, an interactive image gallery to facilitate testing computer vision applications in Unity projects."

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

The [Unity CV Image Gallery](https://github.com/cj-mills/unity-cv-image-gallery) package provides an interactive image gallery and `Scroll View` prefab to facilitate the testing of computer vision applications, such as image classification, object detection, and pose estimation in Unity. 

Many of my tutorials involve using computer vision models in Unity applications. This package makes that shared functionality more modular and reusable, allowing me to streamline my tutorial content. Check out the demo video below to see this package in action.



![](./videos/unity-media-display-demo.mp4){fig-align="center"}



In this post, I'll walk through the package code, providing a solid understanding of its components and their roles.




## Package Overview

The package contains two C# scripts and a `Scroll View` prefab. It is designed to be easily integrated into your existing Unity projects and can be customized to suit your needs.

### C# Scripts

1. `ImageGallery.cs`: This script creates and manages the interactive image gallery. It attaches to a `Scroll View -> Viewport -> Content` object in a Unity scene.
2. `PackageInstaller.cs`: An Editor utility script for automatically installing a list of dependency packages defined in a JSON file. 

### Scroll View Prefab

The Scroll View prefab helps simplify adding an image gallery to a Unity scene. The prefab contains an `ImagePrefab` object and already has the `ImageGallery.cs` script attached to the `Scroll View -> Viewport -> Content` object. This prefab can be easily added to your Unity project, allowing you to quickly set up an interactive image gallery for testing computer vision applications. You can customize the prefab by modifying the serialized fields in the `ImageGallery.cs` script.

### Dependencies

The package depends on the Unity Media Display package. This package simplifies creating and managing demo screens for displaying test images, videos, and webcam streams in Unity projects.

- [**GitHub Repository**](https://github.com/cj-mills/unity-media-display)
- [**Code Walkthrough**](../unity-media-display-walkthrough/)



## Code Explanation

In this section, we will delve deeper into the Unity CV Image Gallery package by examining the purpose and functionality of each C# script.



### `ImageGallery.cs`
In this section, we will go through the `ImageGallery.cs` script and explain how each part of the code works to create and manage the interactive image gallery. The complete code is available on GitHub at the link below.

- [ImageGallery.cs](https://github.com/cj-mills/unity-cv-image-gallery/blob/main/Runtime/Scripts/ImageGallery.cs)



#### Class Variables
The `ImageGallery` class contains several serialized fields to configure the gallery.

```c#
[Header("Scene")]
[Tooltip("The screen GameObject where the selected image will be displayed")]
[SerializeField] private GameObject screenObject;
[Tooltip("The camera GameObject used to display the selected image")]
[SerializeField] private GameObject cameraObject;
[Tooltip("The content panel GameObject where the image gallery is located")]
[SerializeField] private GameObject contentPanel;
[Tooltip("The image prefab used to create each image in the gallery")]
[SerializeField] private GameObject imagePrefab;
[Tooltip("A list of sprites to populate the image gallery.")]
[SerializeField] private List<Sprite> imageSprites;
[Tooltip(" The spacing between images in the gallery.")]
[SerializeField] private float spacing = 5f;
[Tooltip("The specified width for each image in the gallery.")]
[SerializeField] private float specifiedWidth = 100f;
```



#### `Start`
Start() runs when the script first executes. It calls several methods to initialize the gallery.

```c#
private void Start()
{
    // Configures the content panel with a VerticalLayoutGroup component.
    SetupContentPanel();
    // Populates the gallery with images using the provided sprites.
    PopulateImageGallery();
    // Adjusts the content panel height by summing the vertical dimensions of all gallery images and spacing.
    AdjustContentHeight();
    // Assigns click events to the images in the gallery to update the screen texture.
    AssignButtonClickEvents();
}
```



#### `SetupContentPanel`
This method sets up the content panel with a `VerticalLayoutGroup` component. It configures the component's properties, such as `spacing`, `childAlignment`, and `childControlHeight`, to arrange the images in the gallery with proper spacing and alignment.

```c#
/// <summary>
/// Set up the content panel with a VerticalLayoutGroup component.
/// </summary>
private void SetupContentPanel()
{
    VerticalLayoutGroup verticalLayoutGroup = contentPanel.AddComponent<VerticalLayoutGroup>();
    verticalLayoutGroup.spacing = spacing;
    verticalLayoutGroup.childAlignment = TextAnchor.UpperCenter;
    verticalLayoutGroup.childControlHeight = false;
    verticalLayoutGroup.childControlWidth = false;
    verticalLayoutGroup.childForceExpandHeight = false;
    verticalLayoutGroup.childForceExpandWidth = false;
}
```



#### `PopulateImageGallery`
This method populates the image gallery with the sprites in the `imageSprites` list.

```c#
/// <summary>
/// Populate the image gallery with the sprites provided in imageSprites.
/// </summary>
private void PopulateImageGallery()
{
    foreach (Sprite sprite in imageSprites)
    {
        // Instantiates a new GameObject using the imagePrefab
        GameObject newImageObject = Instantiate(imagePrefab, contentPanel.transform);
        Image newImage = newImageObject.GetComponent<Image>();
        newImageObject.SetActive(true);
        // Assign the curent sprite
        newImage.sprite = sprite;
        // Preserves the aspect ratio
        newImage.preserveAspect = true;
        // Use  the sprite's name for easier identification
        newImageObject.name = sprite.name;

        // Adjust the image size based on the specified width
        RectTransform rectTransform = newImageObject.GetComponent<RectTransform>();
        float aspectRatio = sprite.rect.height / sprite.rect.width;
        rectTransform.sizeDelta = new Vector2(specifiedWidth, specifiedWidth * aspectRatio);
    }
}
```



#### `AdjustContentHeight`

This method adjusts the content panel's height based on the total vertical size of the images and spacing.

```c#
/// <summary>
/// Adjust the content panel height based on the total height of the images and spacing.
/// </summary>
private void AdjustContentHeight()
{
    RectTransform contentPanelRectTransform = contentPanel.GetComponent<RectTransform>();
    float totalHeight = 0f;

    // Calculate the total height of all the images in the gallery
    for (int i = 0; i < contentPanelRectTransform.childCount; i++)
    {
        RectTransform childRect = contentPanelRectTransform.GetChild(i).GetComponent<RectTransform>();
        totalHeight += childRect.sizeDelta.y;
    }

    // Add the spacing between the images to the total height
    totalHeight += spacing * (contentPanelRectTransform.childCount - 1);
    // Updates the content panel to accommodate the total height
    contentPanelRectTransform.sizeDelta = new Vector2(contentPanelRectTransform.sizeDelta.x, totalHeight);
}
```



#### `AssignButtonClickEvents`
This method assigns click events to the images in the gallery to update the screen texture to display the selected image.

```c#
/// <summary>
/// Assigns click events to the images in the gallery to update the screen texture when clicked.
/// </summary>
private void AssignButtonClickEvents()
{
    Image[] images = transform.GetComponentsInChildren<Image>();

    foreach (Image image in images)
    {
        // Add a Button component if the image doesn't already have one
        Button button = image.GetComponent<Button>();
        if (button == null)
        {
            button = image.gameObject.AddComponent<Button>();
        }

        // Add a listener to update the screen texture to display the selected image when clicked
        button.onClick.AddListener(() => MediaDisplayManager.UpdateScreenTexture(screenObject, image.mainTexture, cameraObject, false));
    }
}
```



---



### `PackageInstaller.cs`

In this section, we will go through the `PackageInstaller.cs` script and explain how each part of the code works to install the required packages. The complete code is available on GitHub at the link below.

- [PackageInstaller.cs](https://github.com/cj-mills/unity-cv-image-gallery/blob/main/Editor/PackageInstaller.cs)



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
private const string PackagesJSONGUID = "f0b282a4fbb4473584f52e3fd0ab3087";
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

This post provided an in-depth walkthrough of the code for the Unity CV Image Gallery package. The package helps facilitate testing computer vision applications in Unity projects by providing an interactive image gallery.



You can continue to explore the package by going to its GitHub repository linked below, where you will also find instructions for installing it using the Unity Package Manager.

- GitHub Repository: [unity-cv-image-gallery](https://github.com/cj-mills/unity-cv-image-gallery)



You can find the code for the demo project shown in the video at the beginning of this post linked below, along with links for other demo projects that use the Unity CV Image Gallery package.

- [Unity Media Display Demo](https://github.com/cj-mills/UnityMediaDisplay_Demo): A simple demo project demonstrating how to use `the unity-media-display` and `unity-cv-image-gallery` packages in Unity.
- [Barracuda Image Classification Demo](https://github.com/cj-mills/barracuda-image-classification-demo): A simple Unity project demonstrating how to perform image classification with the `barracuda-inference-image-classification` package.
- [Barracuda Inference PoseNet Demo](https://github.com/cj-mills/barracuda-inference-posenet-demo): A simple Unity project demonstrating how to perform 2D human pose estimation with the `barracuda-inference-posenet` package.
- [Barracuda Inference YOLOX Demo](https://github.com/cj-mills/barracuda-inference-yolox-demo): A simple Unity project demonstrating how to perform object detection with the `barracuda-inference-yolox` package.









{{< include /_about-author-cta.qmd >}}
