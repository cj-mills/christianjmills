---
title: "Code Walkthrough: Unity Barracuda Inference Image Classification Package"
date: 2023-5-6
image: /images/empty.gif
hide: false
search_exclude: false
categories: [unity, walkthrough]
description: "Walk through the code for the Unity Barracuda Inference Image Classification package, which extends the functionality of `unity-barracuda-inference-base` to perform image classification using computer vision models."

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

The [Barracuda Inference Image Classification](https://github.com/cj-mills/unity-barracuda-inference-image-classification) package extends the functionality of [`unity-barracuda-inference-base`](https://github.com/cj-mills/unity-barracuda-inference-base) to perform image classification using computer vision models.

Image classification has numerous potential uses in Unity applications, from gesture recognition to analyzing user-generated content. This package makes it easy to add image classification functionality to Unity applications. Here is a demo video from a project that uses this package for gesture classification.



![](./videos/barracuda-inference-image-classification-demo.mp4){fig-align="center"}



In this post, I'll walk through the package code, providing a solid understanding of its components and their roles.




## Package Overview

The package contains two C# scripts.

1. `MultiClassImageClassifier.cs`: This script provides functionality to perform image classification using the Barracuda inference engine.
2. `PackageInstaller.cs`: An Editor utility script for automatically installing a list of dependency packages defined in a JSON file.



## Code Explanation

In this section, we will delve deeper into the Barracuda Inference Image Classification package by examining the purpose and functionality of each C# script.



### `MultiClassImageClassifier.cs`

This script defines the `MultiClassImageClassifier` class, which extends the `BarracudaModelRunner` class from the [Barracuda Inference Base](https://github.com/cj-mills/unity-barracuda-inference-base) package to perform image classification. The complete code is available on GitHub at the link below.

- [MultiClassImageClassifier.cs](https://github.com/cj-mills/unity-barracuda-inference-image-classification/blob/main/Runtime/Scripts/MultiClassImageClassifier.cs)



#### Serialized Fields
The `MultiClassImageClassifier` class includes a field to add class labels with a JSON file.

```c#
[Tooltip("JSON file with class labels")]
[SerializeField] private TextAsset classLabels;
```



It also includes a field to control how often to unload memory assets when using Barracuda's Pixel Shader backend. The Pixel Shader backend enables GPU inference on platforms that don't support Compute Shaders. However, there seems to be a bug in the current version of Barracuda, which does not release unused assets when using this backend. Left unchecked, this can fill up both system and GPU memory. We can address this by manually freeing memory. Doing that every frame can hurt performance, so we'll only do it at set intervals.

```c#
[Tooltip("Interval (in frames) for unloading unused assets with Pixel Shader backend")]
[SerializeField] private int pixelShaderUnloadInterval = 100;
```



#### Private Variables

```c#
// A counter for the number of frames processed.
private int frameCounter = 0;

// Indicates if the system supports asynchronous GPU readback
private bool supportsAsyncGPUReadback = false;

// The name of the transpose layer.
private const string TransposeLayer = "transpose";
// The softmax layer.
private string SoftmaxLayer = "softmaxLayer";
// The name of the output layer.
private string outputLayer;

// Helper class for deserializing class labels from the JSON file
private class ClassLabels { public string[] classes; }

// The class labels
private string[] classes;

// Texture formats for output processing
private TextureFormat textureFormat = TextureFormat.RGBA32;
private RenderTextureFormat renderTextureFormat = RenderTextureFormat.ARGB32;

// Output textures for processing on CPU and GPU
private Texture2D outputTextureCPU;
private RenderTexture outputTextureGPU;
```



#### `Start`
This method initializes necessary components at the start of the script, such as checking async GPU readback support, loading class labels, and creating output textures.

```c#
/// <summary>
/// Initialize necessary components during the start of the script.
/// </summary>
protected override void Start()
{
    base.Start();
    CheckAsyncGPUReadbackSupport(); // Check if async GPU readback is supported
    LoadClassLabels(); // Load class labels from JSON file
    CreateOutputTextures(); // Initialize output texture
}
```





#### `CheckAsyncGPUReadbackSupport`

This method checks if the system supports asynchronous GPU readback

```c#
// Check if the system supports async GPU readback
public bool CheckAsyncGPUReadbackSupport()
{
    supportsAsyncGPUReadback = SystemInfo.supportsAsyncGPUReadback && supportsAsyncGPUReadback;
    return supportsAsyncGPUReadback;
}
```





#### `LoadAndPrepareModel`
This method loads the model and prepares it for execution. It applies the softmax function to the output layer if it's not already a softmax layer. 

```c#
/// <summary>
/// Load the model and prepare it for execution by applying softmax to the output layer.
/// </summary>
protected override void LoadAndPrepareModel()
{
    // Load and prepare the model with the base implementation
    base.LoadAndPrepareModel();

    outputLayer = modelBuilder.model.outputs[0];

    // Set worker type for WebGL
    if (Application.platform == RuntimePlatform.WebGLPlayer)
    {
        workerType = WorkerFactory.Type.PixelShader;
    }

    // Check if the last layer is a Softmax layer
    Layer lastLayer = modelBuilder.model.layers[modelBuilder.model.layers.Count - 1];
    bool lastLayerIsSoftmax = lastLayer.activation == Layer.Activation.Softmax;

    // Add the Softmax layer if the last layer is not already a Softmax layer
    if (!lastLayerIsSoftmax)
    {
        // Add the Softmax layer
        modelBuilder.Softmax(SoftmaxLayer, outputLayer);
        outputLayer = SoftmaxLayer;
    }

    // Apply transpose operation on the output layer
    modelBuilder.Transpose(TransposeLayer, outputLayer, new[] { 0, 1, 3, 2 });
    outputLayer = TransposeLayer;
}
```



#### `InitializeEngine`
This method initializes the inference engine and checks if the model uses a Compute Shader backend.

```c#
/// <summary>
/// Initialize the inference engine and check if the model is using a Compute Shader backend.
/// </summary>
protected override void InitializeEngine()
{
    base.InitializeEngine();

    // Check if async GPU readback is supported by the engine
    supportsAsyncGPUReadback = engine.Summary().Contains("Unity.Barracuda.ComputeVarsWithSharedModel");
}
```



#### `LoadClassLabels`
This method loads the class labels from the provided JSON file.

```c#
/// <summary>
/// Load the class labels from the provided JSON file.
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
This method checks if the provided class labels JSON file is null or empty.

```c#
/// <summary>
/// Check if the provided class labels JSON file is null or empty.
/// </summary>
/// <returns>True if the file is null or empty, otherwise false.</returns>
private bool IsClassLabelsJsonNullOrEmpty()
{
    return classLabels == null || string.IsNullOrWhiteSpace(classLabels.text);
}
```



#### `DeserializeClassLabels`
This method deserializes the provided class labels JSON string to a `ClassLabels` object.

```c#
/// <summary>
/// Deserialize the provided class labels JSON string to a ClassLabels object.
/// </summary>
/// <param name="json">The JSON string to deserialize.</param>
/// <returns>A deserialized ClassLabels object.</returns>
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
/// Update the classes array with the provided ClassLabels object.
/// </summary>
/// <param name="classLabelsObj">The ClassLabels object containing the class labels.</param>
private void UpdateClassLabels(ClassLabels classLabelsObj)
{
    if (classLabelsObj == null)
    {
        return;
    }

    classes = classLabelsObj.classes;
}
```



#### `CreateOutputTextures`
This method creates the output textures that will store the model output.

```c#
/// <summary>
/// Create the output textures that will store the model output.
/// </summary>
private void CreateOutputTextures()
{
    outputTextureCPU = new Texture2D(classes.Length, 1, textureFormat, false);
    outputTextureGPU = RenderTexture.GetTemporary(classes.Length, 1, 0, renderTextureFormat);
}
```



#### `ExecuteModel`
This method executes the model on the provided input texture.

```c#
/// <summary>
/// Execute the model on the provided input texture and return the output array.
/// </summary>
/// <param name="inputTexture">The input texture for the model.</param>
public void ExecuteModel(RenderTexture inputTexture)
{
    using (Tensor input = new Tensor(inputTexture, channels: 3))
    {
        base.ExecuteModel(input);
    }
}
```



#### `CopyOutputToArray`
This method copies the model output to a float array.

```c#
/// <summary>
/// Copy the model output to a float array.
/// </summary>
public float[] CopyOutputToArray()
{
    // Retrieve the output tensor from the engine
    using (Tensor output = engine.PeekOutput(outputLayer))
    {
        if (workerType == WorkerFactory.Type.PixelShader)
        {
            frameCounter++;
            if (frameCounter % pixelShaderUnloadInterval == 0)
            {
                Resources.UnloadUnusedAssets();
                frameCounter = 0;
            }
        }
        // Download the data from the tensor
        return output.data.Download(output.shape);
    }
}
```



#### `CopyOutputToTexture`
This method copies the model output to a texture.

```c#
/// <summary>
/// Copy the model output to a texture.
/// </summary>
public void CopyOutputToTexture()
{
    using (Tensor output = engine.PeekOutput(outputLayer))
    {
        // Store output tensor data in a RenderTexture
        output.ToRenderTexture(outputTextureGPU);
    }
}
```



#### `CopyOutputWithAsyncReadback`
This method copies the model output using asynchronous GPU readback if the platform supports it.

```c#
/// <summary>
/// Copy the model output using async GPU readback. If not supported, defaults to synchronous readback.
/// </summary>
public float[] CopyOutputWithAsyncReadback()
{
    if (!supportsAsyncGPUReadback)
    {
        Debug.Log("Async GPU Readback not supported. Defaulting to synchronous readback");
        return CopyOutputToArray();
    }

    CopyOutputToTexture();

    AsyncGPUReadback.Request(outputTextureGPU, 0, textureFormat, OnCompleteReadback);

    // Extract the output tensor data from the texture
    Color[] outputColors = outputTextureCPU.GetPixels();
    return outputColors.Select(color => color.r).ToArray();
}
```



#### `GetClassName`
This method gets the class name corresponding to the provided class index.

```c#
/// <summary>
/// Get the class name corresponding to the provided class index.
/// </summary>
/// <param name="classIndex">The index of the class to retrieve.</param>
/// <returns>The class name corresponding to the class index.</returns>
public string GetClassName(int classIndex)
{
    return classes[classIndex];
}
```



#### `OnCompleteReadback`
This callback method handles the completion of async GPU readback.

```c#
/// <summary>
/// Callback method for handling the completion of async GPU readback.
/// </summary>
/// <param name="request">The async GPU readback request.</param>
private void OnCompleteReadback(AsyncGPUReadbackRequest request)
{
    if (request.hasError)
    {
        Debug.Log("GPU readback error detected.");
        return;
    }

    if (outputTextureCPU != null)
    {
        try
        {
            // Load readback data into the output texture and apply changes
            outputTextureCPU.LoadRawTextureData(request.GetData<uint>());
            outputTextureCPU.Apply();
        }
        catch (UnityException ex)
        {
            if (ex.Message.Contains("LoadRawTextureData: not enough data provided (will result in overread)."))
            {
                Debug.Log("Updating input data size to match the texture size.");
            }
            else
            {
                Debug.LogError($"Unexpected UnityException: {ex.Message}");
            }
        }
    }
}
```



#### `OnDisable`
This method cleans up resources when the script is disabled, such as releasing the temporary render texture.

```c#
/// <summary>
/// Clean up resources when the script is disabled.
/// </summary>
protected override void OnDisable()
{
    base.OnDisable();
    // Release the temporary render texture
    RenderTexture.ReleaseTemporary(outputTextureGPU);
}
```






---



### `PackageInstaller.cs`

In this section, we will go through the `PackageInstaller.cs` script and explain how each part of the code works to install the required packages. The complete code is available on GitHub at the link below.

- [PackageInstaller.cs](https://github.com/cj-mills/unity-barracuda-inference-image-classification/blob/main/Editor/PackageInstaller.cs)



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
The `PackageInstaller` class contains several private static fields.

```c#
// Stores the AddRequest object for the current package to install.
private static AddRequest addRequest;
// A list of PackageData objects to install.
private static List<PackageData> packagesToInstall;
// The index of the current package to install.
private static int currentPackageIndex;

// GUID of the JSON file containing the list of packages to install
private const string PackagesJSONGUID = "4a3b2c83681748b49d28cb6ed4f587d9";
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

This post provided an in-depth walkthrough of the code for the Barracuda Inference Image Classification package. The package extends the functionality of [`unity-barracuda-inference-base`](https://github.com/cj-mills/unity-barracuda-inference-base) to perform image classification using computer vision models.

You can continue to explore the package by going to its GitHub repository linked below, where you will also find instructions for installing it using the Unity Package Manager.

- GitHub Repository: [unity-barracuda-inference-image-classification](https://github.com/cj-mills/unity-barracuda-inference-image-classification)

You can find the code for the demo project shown in the video at the beginning of this post linked below.

- [Barracuda Image Classification Demo](https://github.com/cj-mills/barracuda-image-classification-demo): A simple Unity project demonstrating how to perform image classification with the `barracuda-inference-image-classification` package.





