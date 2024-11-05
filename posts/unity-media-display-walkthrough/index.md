---
title: "Code Walkthrough: Unity Media Display Package"
date: 2023-4-25
image: /images/empty.gif
hide: false
search_exclude: false
categories: [unity, walkthrough]
description: "Walk through the code for the Unity Media Display package, a tool for displaying test images, videos, and webcam streams in Unity projects."

twitter-card:
  creator: "@cdotjdotmills"
  site: "@cdotjdotmills"
  image: /images/default-preview-image-black.png
open-graph:
  image: /images/default-preview-image-black.png
aliases:
- /posts/unity-media-display-tutorial/
---





* [Introduction](#introduction)
* [Package Overview](#package-overview)
* [Code Explanation](#code-explanation)
* [Conclusion](#conclusion)




## Introduction

The [Unity Media Display](https://github.com/cj-mills/unity-media-display) package simplifies the creation and management of demo screens for displaying test images, videos, and webcam streams in Unity projects. As many of my tutorials use a demo screen, this package makes that shared functionality more modular and reusable, allowing me to streamline my tutorial content. Check out the demo video below to see this package in action.



![](./videos/unity-media-display-demo.mp4){fig-align="center"}



In this post, I'll walk through the package code, providing a solid understanding of its components and their roles.




## Package Overview

The package contains several C# scripts that work together to enable easy management of demo screens.

1. `BaseScreenManager.cs`: This abstract class is the foundation for managing in-scene screen objects. It provides essential functionality and basic structure for derived classes, allowing you to create custom screen managers tailored to your project's needs.
2. `MediaDisplayManager.cs`: This script offers utility methods for setting up and managing a screen object in Unity. It handles initializing and updating screen textures, transformations, and webcam streams.
3. `TextureChangeEvent.cs`: This script triggers when a material's main texture changes, allowing for efficient texture updates and seamless transitions between different media sources.
4. `AddCustomDefineSymbol.cs`: An Editor script that automatically adds a custom scripting define symbol to the project after the package installs.
5. `ShaderIncludePostprocessor.cs`: This build postprocessor script includes the shader used by the screen material in the build process and only runs in the Unity Editor.



## Code Explanation

In this section, we will delve deeper into the Unity Media Display package by examining the purpose and functionality of each C# script. We'll discuss how these scripts work together to enable seamless management of demo screens for displaying test images, videos, and webcam streams.



### `BaseScreenManager.cs`
This script provides an abstract class for managing in-scene screen objects, allowing you to create custom screen managers tailored to your project's needs. The complete code is available on GitHub at the link below.

- [unity-media-display/Runtime/Scripts/BaseScreenManager.cs](https://github.com/cj-mills/unity-media-display/blob/main/Runtime/Scripts/BaseScreenManager.cs)



The `BaseScreenManager` class includes several serialized fields for configuring scene components and settings, webcam settings, and toggle-key settings.

```c#
// Scene components and settings
[Header("Scene")]
[Tooltip("Screen object in the scene")]
[SerializeField] protected GameObject screenObject;
[Tooltip("Camera object in the scene")]
[SerializeField] protected GameObject cameraObject;
[Tooltip("A test texture to display on the screen")]
[SerializeField] protected Texture testTexture;
[Tooltip("A framerate cap to reduce lag")]
[SerializeField] protected int maxFrameRate = 500;

// Webcam settings
[Header("Webcam")]
[Tooltip("Option to use webcam as input")]
[SerializeField] protected bool useWebcam = false;
[Tooltip("Requested webcam dimensions")]
[SerializeField] protected Vector2Int webcamDims = new Vector2Int(1280, 720);
[Tooltip("Requested webcam framerate")]
[SerializeField, Range(0, 60)] protected int webcamFrameRate = 60;

// Toggle key settings
[Header("Toggle Key")]
[Tooltip("Key to toggle between image and webcam feed")]
[SerializeField] protected KeyCode toggleKey = KeyCode.Space;
```



It also contains protected variables for managing the current texture, webcam devices, and the webcam texture.

```c#
protected Texture currentTexture;
protected WebCamDevice[] webcamDevices;
protected WebCamTexture webcamTexture;
protected string currentWebcam;
```



Lastly, the script has some constant values for working with webcam devices.

```c#
// The value to multiply the webcam frame rate by to determine the application's target framerate.
protected const int WebcamFrameRateMultiplier = 4;
// The value for the width of an uninitialized webcam device
protected const int UninitializedWebcamWidth = 16;
```



The script includes several methods:



#### `OnEnable`
Subscribes to the `TextureChangeEvent` for handling changes in the `mainTexture` of a `Material`.

```c#
// Subscribe to the texture change event
protected virtual void OnEnable()
{
    TextureChangeEvent.OnMainTextureChanged += HandleMainTextureChanged;
}
```



#### `Initialize`
Sets the application's target frame rate, configures the webcam devices, and initializes the current webcam if the `useWebcam` option is enabled.

```c#
// Initializes the application's target frame rate and configures the webcam devices.
protected virtual void Initialize()
{
    // Limit the application's target frame rate to reduce lag.
    Application.targetFrameRate = maxFrameRate;
    // Get the list of available webcam devices.
    webcamDevices = WebCamTexture.devices;
    // If no webcam devices are available, disable the useWebcam option.
    useWebcam = webcamDevices.Length > 0 ? useWebcam : false;
    // Set the current webcam device to the first available device, if any.
    currentWebcam = useWebcam ? webcamDevices[0].name : "";
}
```



#### `UpdateDisplay`
Updates the display with the current texture, sets up the webcam if necessary, and starts a coroutine to update the screen texture asynchronously.

```c#
// Updates the display with the current texture (either a test texture or the webcam feed).
protected virtual void UpdateDisplay()
{
    // Set up the webcam if necessary.
    SetupWebcam();
    // Update the current texture based on the useWebcam option.
    UpdateCurrentTexture();
    // Start a coroutine to asynchronously update the screen texture.
    StartCoroutine(UpdateScreenTextureAsync());
}
```



#### `SetupWebcam`
If the `useWebcam` option is enabled and there are available webcam devices, this method initializes the webcam device and stops it if the option is disabled.

```c#
// Sets up the webcam if the useWebcam option is enabled.
protected void SetupWebcam()
{
    // If there are no webcam devices, return immediately.
    if (webcamDevices.Length == 0) return;

    // If the useWebcam option is enabled, initialize the webcam.
    if (useWebcam)
    {
        // Initialize the webcam and check if it started playing.
        bool webcamPlaying = MediaDisplayManager.InitializeWebcam(ref webcamTexture, currentWebcam, webcamDims, webcamFrameRate);
        // If the webcam failed to start playing, disable the useWebcam option.
        useWebcam = webcamPlaying ? useWebcam : false;
    }
    // If the useWebcam option is disabled and the webcam texture is playing, stop the webcam.
    else if (webcamTexture != null && webcamTexture.isPlaying)
    {
        webcamTexture.Stop();
    }
}
```



#### `UpdateCurrentTexture`
This method updates the current texture and target frame rate based on the `useWebcam` option.

```c#
// Updates the current texture and target frame rate based on the useWebcam option.
protected void UpdateCurrentTexture()
{
    // Set the current texture to the webcam texture if useWebcam is enabled, otherwise use the test texture.
    currentTexture = useWebcam ? webcamTexture : testTexture;
    // Set the target frame rate based on whether the webcam is being used or not.
    Application.targetFrameRate = useWebcam ? webcamFrameRate * WebcamFrameRateMultiplier : maxFrameRate;
}
```



#### `UpdateScreenTextureAsync`
A coroutine that waits until the webcamTexture is ready (if `useWebcam` is enabled) and updates the screen texture with the current texture (either the test texture or the webcam feed).

```c#
// Coroutine to update the screen texture asynchronously.
protected IEnumerator UpdateScreenTextureAsync()
{
    // Wait until the webcamTexture is ready if useWebcam is enabled.
    while (useWebcam && webcamTexture.isPlaying && webcamTexture.width <= UninitializedWebcamWidth)
    {
        yield return null;
    }

    // Update the screen texture with the current texture (image or webcam feed).
    MediaDisplayManager.UpdateScreenTexture(screenObject, currentTexture, cameraObject, useWebcam);
}
```



#### `HandleMainTextureChanged`
This method handles the `TextureChangeEvent`, updating the current texture and stopping the webcam if the new main texture differs from the webcam texture.

```c#
// Handle the texture change event.
protected virtual void HandleMainTextureChanged(Material material)
{
    // Update the current texture.
    currentTexture = material.mainTexture;
    // If the new main texture is different from the webcam texture and the webcam is playing, stop the webcam.
    if (webcamTexture && material.mainTexture != webcamTexture && webcamTexture.isPlaying)
    {
        useWebcam = false;
        webcamTexture.Stop();
    }
}
```



#### `OnDisable`

Lastly, the `OnDisable()` method unsubscribes from the `TextureChangeEvent` when the script is disabled.

```c#
// Unsubscribe from the texture change event when the script is disabled.
protected virtual void OnDisable()
{
    TextureChangeEvent.OnMainTextureChanged -= HandleMainTextureChanged;
}
```



----



### `MediaDisplayManager.cs`
The `MediaDisplayManager` script contains a static utility class that provides various methods to set up and manage media displays in Unity. This script sets screen textures, updates screen transforms, and initializes camera and webcam configurations. The complete code is available on GitHub at the link below.

- [unity-media-display/Runtime/Scripts/MediaDisplayManager.cs](https://github.com/cj-mills/unity-media-display/blob/main/Runtime/Scripts/MediaDisplayManager.cs)



#### `SetScreenTexture`
This method sets the texture for the screen object and raises a texture change event.

```c#
/// <summary>
/// Sets the texture for the screen object and raises a texture change event.
/// </summary>
/// <param name="screenObject">The GameObject to be used as the screen.</param>
/// <param name="displayTexture">The Texture to display on the screen.</param>
public static void SetScreenTexture(GameObject screenObject, Texture displayTexture)
{
    // Attempt to get the MeshRenderer component from the screen object
    if (screenObject.TryGetComponent<MeshRenderer>(out MeshRenderer meshRenderer))
    {
        // Create a new material with the Unlit/Texture shader and set its main texture
        Material screenMaterial = new Material(Shader.Find("Unlit/Texture"))
        {
            mainTexture = displayTexture
        };

        // Assign the material to the mesh renderer
        meshRenderer.material = screenMaterial;

        // Raise a texture change event to inform other scripts
        TextureChangeEvent.RaiseMainTextureChangedEvent(meshRenderer.material);
    }
}
```



#### `UpdateScreenTransform`

This method updates the screen object's rotation, scale, and position based on the display texture and mirror settings.
```c#
/// <summary>
/// Updates the screen object's rotation, scale, and position based on the display texture and mirror settings.
/// </summary>
/// <param name="screenObject">The GameObject to be used as the screen.</param>
/// <param name="displayTexture">The Texture displayed on the screen.</param>
/// <param name="mirrorScreen">Optional parameter to mirror the screen horizontally. Default is false.</param>
public static void UpdateScreenTransform(GameObject screenObject, Texture displayTexture, bool mirrorScreen = false)
{
    // Get the width and height of the display texture
    float width = displayTexture.width;
    float height = displayTexture.height;

    // Set the rotation, scale, and position of the screen object
    screenObject.transform.rotation = Quaternion.Euler(0, mirrorScreen ? 180f : 0f, 0);
    screenObject.transform.localScale = new Vector3(width, height, mirrorScreen ? -1f : 1f);
    screenObject.transform.position = new Vector3(width / 2, height / 2, 1);
}
```



#### `InitializeCamera`

This method initializes the camera used for displaying the screen.
```c#
/// <summary>
/// Initializes the camera used for displaying the screen.
/// </summary>
/// <param name="cameraObject">The GameObject with a Camera component to be used for displaying the screen.</param>
/// <param name="screenDimensions">The dimensions of the screen.</param>
public static void InitializeCamera(GameObject cameraObject, Vector2Int screenDimensions)
{
    // Attempt to get the Camera component from the camera object
    if (cameraObject.TryGetComponent<Camera>(out Camera camera))
    {
        // Set the position of the camera object
        Vector3 position = new Vector3(screenDimensions.x / 2, screenDimensions.y / 2, -10f);
        cameraObject.transform.position = position;

        // Configure the camera for orthographic mode
        camera.orthographic = true;

        // Calculate the aspect ratios of the screen object and the camera's viewport
        float screenAspectRatio = (float)screenDimensions.x / screenDimensions.y;
        float cameraAspectRatio = (float)camera.pixelWidth / camera.pixelHeight;

        // Set the camera's orthographic size based on the aspect ratios
        if (screenAspectRatio > cameraAspectRatio)
        {
            // Wider screen object
            camera.orthographicSize = screenDimensions.x / 2 / camera.aspect;
        }
        else
        {
            // Taller screen object
            camera.orthographicSize = screenDimensions.y / 2;
        }
    }
}
```



#### `InitializeWebcam`

This method initializes and plays a webcam stream with the specified settings.
```c#
/// <summary>
/// Initializes and plays a webcam stream with the specified settings.
/// </summary>
/// <param name="webcamTexture">A reference to the WebCamTexture instance to be initialized and played.</param>
/// <param name="deviceName">The name of the webcam device to be used for streaming.</param>
/// <param name="webcamDimensions">The desired resolution of the webcam stream.</param>
/// <param name="webcamFrameRate">The desired frame rate of the webcam stream. Default is 60.</param>
/// <returns>Returns true if the webcam stream has started playing, false otherwise.</returns>
public static bool InitializeWebcam(ref WebCamTexture webcamTexture, string deviceName, Vector2Int webcamDimensions, int webcamFrameRate = 60)
{
    // If the webcam texture is not null and it is playing, stop it
    if (webcamTexture != null && webcamTexture.isPlaying)
    {
        webcamTexture.Stop();
    }

    // Create a new WebCamTexture instance with the specified settings
    webcamTexture = new WebCamTexture(deviceName, webcamDimensions.x, webcamDimensions.y, webcamFrameRate);

    // Start playing the webcam stream
    webcamTexture.Play();

    // Return true if the webcam stream has started playing, false otherwise
    return webcamTexture.isPlaying;
}
```



#### `UpdateScreenTexture`

This method updates the texture, transform, and camera object.
```c#
/// <summary>
/// Updates the texture, transform, and camera of the screen object.
/// </summary>
/// <param name="screenObject">The GameObject to be used as the screen.</param>
/// <param name="displayTexture">The Texture displayed on the screen.</param>
/// <param name="cameraObject">The GameObject with a Camera component to be used for displaying the screen.</param>
/// <param name="mirrorScreen">Optional parameter to mirror the screen horizontally. Default is false.</param>
public static void UpdateScreenTexture(GameObject screenObject, Texture displayTexture, GameObject cameraObject, bool mirrorScreen = false)
{
    // Update the texture of the screen object
    SetScreenTexture(screenObject, displayTexture);

    // Update the transform of the screen object based on the new texture dimensions and mirror settings
    UpdateScreenTransform(screenObject, displayTexture, mirrorScreen);

    // Get the screen dimensions from the updated screen object
    Vector2Int screenDimensions = new Vector2Int(displayTexture.width, displayTexture.height);

    // Initialize the camera for displaying the screen
    InitializeCamera(cameraObject, screenDimensions);
}
```



----



### `TextureChangeEvent.cs`
The `TextureChangeEvent` class defines and manages a custom event that triggers when the main texture of a Material object changes. This event allows other scripts to be notified and respond accordingly when the displayed texture updates. The complete code is available on GitHub at the link below.

- [unity-media-display/Runtime/Scripts/TextureChangeEvent.cs](https://github.com/cj-mills/unity-media-display/blob/main/Runtime/Scripts/TextureChangeEvent.cs)



1. `OnMainTextureChangedDelegate`: This delegate defines the signature for the event handlers that get called when the main texture changes.
2. `OnMainTextureChanged`: A  static event of the `OnMainTextureChangedDelegate` type. It allows other scripts to subscribe and respond to texture change events.
3. `RaiseMainTextureChangedEvent`: This static method gets called when the main texture of a Material object changes. The method invokes the `OnMainTextureChanged` event, calling any subscribed event handlers and passing the updated material as an argument.



```c#
public class TextureChangeEvent : MonoBehaviour
{
    // Define a delegate with the desired signature
    public delegate void OnMainTextureChangedDelegate(Material material);

    // Create a static event with the delegate type
    public static event OnMainTextureChangedDelegate OnMainTextureChanged;

    // Method to call when the mainTexture has been changed
    public static void RaiseMainTextureChangedEvent(Material material)
    {
        OnMainTextureChanged?.Invoke(material);
    }
}
```



---



### `AddCustomDefineSymbol.cs`
This Editor script contains a class that adds a custom define symbol to the project. We can use this custom symbol to prevent code that relies on this package from executing unless the Unity Media Display package is present. The complete code is available on GitHub at the link below.

- [unity-media-display/Editor/AddCustomDefineSymbol.cs](https://github.com/cj-mills/unity-media-display/blob/main/Editor/AddCustomDefineSymbol.cs)

```c#
public class DependencyDefineSymbolAdder
{
    // Constant string representing the custom define symbol.
    private const string CustomDefineSymbol = "CJM_UNITY_MEDIA_DISPLAY";

    // This method is called on Unity editor load to ensure the custom define symbol is added.
    [InitializeOnLoadMethod]
    public static void AddCustomDefineSymbol()
    {
        // Get the currently selected build target group.
        var buildTargetGroup = EditorUserBuildSettings.selectedBuildTargetGroup;

        // Retrieve the current scripting define symbols for the selected build target group.
        var defines = PlayerSettings.GetScriptingDefineSymbolsForGroup(buildTargetGroup);

        // Check if the custom define symbol is already in the list of define symbols.
        if (!defines.Contains(CustomDefineSymbol))
        {
            // Append the custom define symbol to the list of define symbols.
            defines += $";{CustomDefineSymbol}";

            // Set the updated list of define symbols for the selected build target group.
            PlayerSettings.SetScriptingDefineSymbolsForGroup(buildTargetGroup, defines);

            // Log the successful addition of the custom define symbol.
            Debug.Log($"Added custom define symbol '{CustomDefineSymbol}' to the project.");
        }
    }
}
```



---



### `ShaderIncludePostprocessor.cs`
The `ShaderIncludePostprocessor.cs` script includes the shader used by the screen material in the build by adding it to the list of `Always Included Shaders` in the Graphics Settings. This script implements the `IPostprocessBuildWithReport` interface and runs only in the Unity Editor. The complete code is available on GitHub at the link below.

- [unity-media-display/Runtime/Editor/ShaderIncludePostprocessor.cs](https://github.com/cj-mills/unity-media-display/blob/main/Editor/ShaderIncludePostprocessor.cs)

The `ShaderIncludePostprocessor` class includes a property to specify the order in which the build postprocessor gets called.

```c#
/// <summary>
/// The order in which this postprocessor is called.
/// </summary>
public int callbackOrder { get { return 0; } }
```



#### `OnPostprocessBuild`
This method gets called after the build completes. It calls the `AddShaderToAlwaysIncludedShaders` method with the `"Unlit/Texture"` shader name, ensuring the built project includes it.

```c#
/// <summary>
/// Called after the build has been completed.
/// </summary>
/// <param name="report">The BuildReport object containing information about the build.</param>
public void OnPostprocessBuild(BuildReport report)
{
    AddShaderToAlwaysIncludedShaders("Unlit/Texture");
}
```



#### `AddShaderToAlwaysIncludedShaders`
This method adds the specified shader to the `Always Included Shaders` list in the Graphics Settings.

```c#
/// <summary>
/// Adds the specified shader to the list of always included shaders in GraphicsSettings.
/// </summary>
/// <param name="shaderName">The name of the shader to add to the always included shaders list.</param>
private static void AddShaderToAlwaysIncludedShaders(string shaderName)
{
    Shader shader = Shader.Find(shaderName);
    if (shader == null)
    {
        Debug.LogWarning($"Shader '{shaderName}' not found.");
        return;
    }

    // Load the GraphicsSettings asset
    var graphicsSettings = AssetDatabase.LoadAssetAtPath<GraphicsSettings>("ProjectSettings/GraphicsSettings.asset");
    if (graphicsSettings == null)
    {
        Debug.LogWarning("GraphicsSettings.asset not found.");
        return;
    }

    // Create a serialized object from the GraphicsSettings asset
    SerializedObject serializedGraphicsSettings = new SerializedObject(graphicsSettings);
    // Find the "m_AlwaysIncludedShaders" property in the serialized object
    SerializedProperty alwaysIncludedShadersProp = serializedGraphicsSettings.FindProperty("m_AlwaysIncludedShaders");

    // Iterate through the shaders in the Always Included Shaders list
    for (int i = 0; i < alwaysIncludedShadersProp.arraySize; i++)
    {
        SerializedProperty shaderProp = alwaysIncludedShadersProp.GetArrayElementAtIndex(i);
        if (shaderProp.objectReferenceValue == shader)
        {
            Debug.Log($"Shader '{shaderName}' is already in the Always Included Shaders list.");
            return;
        }
    }

    // Add the specified shader to the Always Included Shaders list
    alwaysIncludedShadersProp.InsertArrayElementAtIndex(alwaysIncludedShadersProp.arraySize);
    SerializedProperty newShaderProp = alwaysIncludedShadersProp.GetArrayElementAtIndex(alwaysIncludedShadersProp.arraySize - 1);
    newShaderProp.objectReferenceValue = shader;
    serializedGraphicsSettings.ApplyModifiedProperties();

    Debug.Log($"Shader '{shaderName}' has been added to the Always Included Shaders list.");
}
```








## Conclusion

This post provided an in-depth walkthrough of the code for the Unity Media Display package. The package helps simplify creating and managing demo screens for displaying test images, video, and webcam streams in Unity projects.



You can continue to explore the package by going to its GitHub repository linked below, where you will also find instructions for installing it using the Unity Package Manager.

- GitHub Repository: [unity-media-display](https://github.com/cj-mills/unity-media-display)



You can find the code for the demo project shown in the video at the beginning of this post linked below, along with links for other demo projects that use the Unity Media Display package.

- [Unity Media Display Demo](https://github.com/cj-mills/UnityMediaDisplay_Demo): A simple demo project demonstrating how to use `the unity-media-display` and `unity-cv-image-gallery` packages in Unity.
- [Barracuda Image Classification Demo](https://github.com/cj-mills/barracuda-image-classification-demo): A simple Unity project demonstrating how to perform image classification with the `barracuda-inference-image-classification` package.
- [Barracuda Inference PoseNet Demo](https://github.com/cj-mills/barracuda-inference-posenet-demo): A simple Unity project demonstrating how to perform 2D human pose estimation with the `barracuda-inference-posenet` package.
- [Barracuda Inference YOLOX Demo](https://github.com/cj-mills/barracuda-inference-yolox-demo): A simple Unity project demonstrating how to perform object detection with the `barracuda-inference-yolox` package.









{{< include /_about-author-cta.qmd >}}
