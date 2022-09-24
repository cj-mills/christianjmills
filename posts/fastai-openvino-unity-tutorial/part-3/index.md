---
title: How to Create an OpenVINO Plugin for Unity on Windows Pt. 3
date: 2022-7-18
image: /images/empty.gif
title-block-categories: true
layout: post
toc: false
hide: false
search_exclude: false
description: Part 3 covers modifying the Unity project from the fastai-to-unity tutorial
  to classify images with a OpenVINO DLL.
categories: [fastai, openvino, unity]

aliases:
- /Fastai-to-OpenVINO-to-Unity-Tutorial-Windows-3/

twitter-card:
  creator: "@cdotjdotmills"
  site: "@cdotjdotmills"
open-graph:
---

* [Overview](#overview)
* [Open Unity Project](#open-unity-project)
* [Add New Asset Files](#add-new-asset-files)
* [Allow Unsafe Code](#allow-unsafe-code)
* [Modify Compute Shader](#modify-compute-shader)
* [Create `ImageClassifierOpenVINO` Script](#create-imageclassifieropenvino-script)
* [Modify GUI](#modify-gui)
* [Add `ImageClassifierOpenVINO` Component](#add-imageclassifieropenvino-component)
* [Update On Value Changed Events](#update-on-value-changed-events)
* [Summary](#summary)



## Overview

[Part 2](../part-2/) covered creating a dynamic link library ([DLL](https://docs.microsoft.com/en-us/troubleshoot/windows-client/deployment/dynamic-link-library)) file to perform image classification using [OpenVINO](https://docs.openvino.ai/latest/index.html). This post covers the required modifications for the Unity project from the [fastai-to-unity tutorial](../../fastai-to-unity-tutorial/part-1/) to use this DLL.



## Open Unity Project

Open the [Fastai-Unity-Tutorial](https://github.com/cj-mills/fastai-to-unity-tutorial/tree/main/Fastai-Unity-Tutorial) project in the Unity Editor. The project is available in the GitHub repository linked below for anyone who did not follow the original tutorial series.

* **[fastai-to-unity-tutorial GitHub repository](https://github.com/cj-mills/fastai-to-unity-tutorial)**



## Add New Asset Files

We'll store the [DLL files](../../fastai-openvino-unity-tutorial/part-2/#gather-dependencies) from part 2 in a new folder called `Plugins`. The DLL targets 64-bit x86 architectures, so we need to place the DLL files in a subfolder named `x86_64`.

* [Plugins Folder Google Drive](https://drive.google.com/drive/folders/1zPHOeYB9jOlPG_KSZyvUVipmBDbV2p76?usp=sharing)

  

![unity-create-plugins-folder](./images/unity-create-plugins-folder.png)



Copy all the DLL files and the `plugins.xml` file into the `Assets/Plugins/x86_64` folder. 

![unity-openvino-plugins-folder](./images/unity-openvino-plugins-folder.png)



We then need to close and reopen the project for Unity to load the plugin files.



Next, we'll create a folder to store the OpenVINO IR models. We need to place the XML and BIN files for the IR models in a [StreamingAssets](https://docs.unity3d.com/Manual/StreamingAssets.html) folder to include them in project builds. Create a new folder named `StreamingAssets`. We'll put the files in a new subfolder called `OpenVINOModels` to keep things organized.

* [OpenVINOModels Folder Google Drive](https://drive.google.com/drive/folders/1QE9d7pADn4THB91_WwwQgD7NsjbBUnE6?usp=sharing)

![unity-openvino-models-folder](./images/unity-openvino-models-folder.png)





The plugins.xml file included with the DLL files contains locations for the DLL files needed for using different types of devices.

**`plugins.xml` content:**

```xml
<ie>
    <plugins>
        <plugin name="AUTO" location="openvino_auto_plugin.dll">
            <properties>
                <property key="MULTI_WORK_MODE_AS_AUTO" value="YES"/>
            </properties>
        </plugin>
        <plugin name="BATCH" location="openvino_auto_batch_plugin.dll">
        </plugin>
        <plugin name="CPU" location="openvino_intel_cpu_plugin.dll">
        </plugin>
        <plugin name="GNA" location="openvino_intel_gna_plugin.dll">
        </plugin>
        <plugin name="GPU" location="openvino_intel_gpu_plugin.dll">
        </plugin>
        <plugin name="HETERO" location="openvino_hetero_plugin.dll">
        </plugin>
        <plugin name="MULTI" location="openvino_auto_plugin.dll">
        </plugin>
        <plugin name="MYRIAD" location="openvino_intel_myriad_plugin.dll">
        </plugin>
        <plugin name="HDDL" location="openvino_intel_hddl_plugin.dll">
        </plugin>
        <plugin name="VPUX" location="openvino_intel_vpux_plugin.dll">
        </plugin>
    </plugins>
</ie>
```



It needs to be in the same folder as the DLL files for the plugin to work. However, Unity does not include XML files in the Plugins folder when building the project. We need to store a copy of the plugins.xml file in the StreamingAssets folder and then copy it back to the Plugins/x86_64 folder when first running the built project. We can handle both steps automatically in code.







## Allow Unsafe Code

Rather than copying the input image from Unity to the OpenVINO plugin, we'll pass a pointer to the pixel data. First, we need to allow unsafe code for the Unity project. Select `Edit → Project Settings...` from the top menu.

![unity-open-project-settings](./images/unity-open-project-settings.png)



Open the `Player → Other Settings` dropdown and scroll down to the `Allow 'unsafe' Code` checkbox. Enable the setting and close the Project Settings window.



![unity-allow-unsafe-code](./images/unity-allow-unsafe-code.png)



Now we can start modifying the code.



## Modify Compute Shader

The input image gets flipped upside down when we send it to the plugin. We can pre-flip the image in the `ProcessingShader` compute shader before sending it to the plugin. We need to know the height of the input image, which we can access with the [Texture2D::GetDimensions](https://docs.microsoft.com/en-us/windows/win32/direct3dhlsl/sm5-object-texture2d-getdimensions) function.



```c#
// Each #kernel tells which function to compile; you can have many kernels
#pragma kernel NormalizeImageNet
#pragma kernel FlipXAxis

// The pixel data for the input image
Texture2D<float4> InputImage;
// The pixel data for the processed image
RWTexture2D<float4> Result;

// Flip the image around the x-axis
[numthreads(8, 8, 1)]
void FlipXAxis(uint3 id : SV_DispatchThreadID)
{
    // Stores the InputImage width
    uint width;
    // Stores the InputImage height
    uint height;
    // Get the dimensions of the InputImage
    InputImage.GetDimensions(width, height);

    // Update the y value for the pixel coordinates
    int2 coords = int2(id.x, height - id.y);
    Result[id.xy] = float4(InputImage[coords].x, InputImage[coords].y, InputImage[coords].z, 1.0f);
}

// Apply the ImageNet normalization stats from PyTorch to an image
[numthreads(8, 8, 1)]
void NormalizeImageNet(uint3 id : SV_DispatchThreadID)
{
    // Set the pixel color values for the processed image
    Result[id.xy] = float4(
        // Normalize the red color channel values
        (InputImage[id.xy].r - 0.4850f) / 0.2290f,
        // Normalize the green color channel values
        (InputImage[id.xy].g - 0.4560f) / 0.2240f,
        // Normalize the blue color channel values
        (InputImage[id.xy].b - 0.4060f) / 0.2250f,
        // Ignore the alpha/transparency channel
        InputImage[id.xy].a);
}
```





## Create `ImageClassifierOpenVINO` Script

Duplicate the `ImageClassifier` script and name the copy `ImageClassifierOpenVINO`.



![unity-create-image-classifier-openvino-script](./images/unity-create-image-classifier-openvino-script.png)



**Update class name**

Open the new script in the code editor and replace the class name with the new file name.

```c#
public class ImageClassifierOpenVINO : MonoBehaviour
```



**Update required namespaces**

We no longer need the Barracuda namespace. Instead, we need the [System.Runtime.InteropServices](https://docs.microsoft.com/en-us/dotnet/api/system.runtime.interopservices?view=net-5.0) namespace to handle interactions with the OpenVINO plugin.

```c#
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;
using System;
using UnityEngine.UI;
using System.Runtime.InteropServices;
```



**Remove Barracuda code**

We need to delete all the public and private Barracuda variables, along with the `InitializeWorker` and `OnDisable` methods.





**Add code to copy `plugins.xml` file to `StreamingAssets` folder**

Unity provides an [`InitializeOnLoad`](https://docs.unity3d.com/Manual/RunningEditorCodeOnLaunch.html) attribute to run code in the Unity Editor without requiring action from the user. This attribute requires the [`UnityEditor`](https://docs.unity3d.com/ScriptReference/UnityEditor.html) namespace. We can only use this while in the Editor, so we need to wrap the code in [Conditional compilation](https://docs.microsoft.com/en-us/dotnet/csharp/language-reference/preprocessor-directives#conditional-compilation) preprocessor directives. 

```c#
#if UNITY_EDITOR
using UnityEditor;

[InitializeOnLoad]
public class Startup
{
    static Startup()
    {
        // Get all files named "plugins.xml"
        string[] files = Directory.GetFiles("./Assets/", "plugins.xml", SearchOption.AllDirectories);
        // Iterate through each found file
        foreach (string file in files)
        {
            // Check if the file is in the "x86_64" folder
            if (file.Contains("x86_64"))
            {
                // Define file path for StreamingAssets folder
                string targetPath = $"{Application.streamingAssetsPath}/plugins.xml";
                // Print the source file path
                Debug.Log(file);
                // Only copy the file to the StreamingAssets folder if it is not already present
                if (!File.Exists(targetPath)) File.Copy(file, targetPath);
            }
        }
    }
}
#endif
```

We use the `UNITY_EDITOR` [scripting symbol](https://docs.unity3d.com/Manual/PlatformDependentCompilation.html) to check whether we are in the Unity Editor. We are in the Editor, so it returns true, and the code executes.

![unity_scripting_symbol_in_editor](./images/unity_scripting_symbol_in_editor.png)

If we check if we are not in the Unity Editor, it returns false, and the code block does not execute.

![unity_scripting_symbol_not_in_editor](./images/unity_scripting_symbol_not_in_editor.png)



We can verify the code works by saving the script and going to the `StreamingAssets` folder in the Editor. The plugins.xml file should be present.



![unity-verify-initializeonload](./images/unity-verify-initializeonload.png)







**Update data processing variables**

Next, we can remove the `processingMaterial` variable from the Data Processing section. We no longer need to download model output from the GPU to the CPU. However, we now need to download the input image to the CPU before sending it to the plugin. We can do this asynchronously to help reduce the GPU-to-CPU performance bottleneck.

```c#
[Header("Data Processing")]
[Tooltip("The target minimum model input dimensions")]
public int targetDim = 216;
[Tooltip("The compute shader for GPU processing")]
public ComputeShader processingShader;
[Tooltip("Asynchronously download input image from the GPU to the CPU.")]
public bool useAsyncGPUReadback = true;
```



**Update variables for user interface**

We'll add new dropdown menus so that we can switch between the available OpenVINO models and devices at runtime.

```c#
[Header("GUI")]
[Tooltip("Display predicted class")]
public bool displayPredictedClass = true;
[Tooltip("Display fps")]
public bool displayFPS = true;
[Tooltip("The on-screen text color")]
public Color textColor = Color.red;
[Tooltip("The scale value for the on-screen font size")]
[Range(0, 99)]
public int fontScale = 50;
[Tooltip("The number of seconds to wait between refreshing the fps value")]
[Range(0.01f, 1.0f)]
public float fpsRefreshRate = 0.1f;
[Tooltip("The toggle for using a webcam as the input source")]
public Toggle useWebcamToggle;
[Tooltip("The dropdown menu that lists available webcam devices")]
public Dropdown webcamDropdown;
[Tooltip("The dropdown menu that lists available OpenVINO models")]
public Dropdown modelDropdown;
[Tooltip("The dropdown menu that lists available OpenVINO devices")]
public Dropdown deviceDropdown;
```





**Define public variables for the OpenVINO plugin**



```c#
[Header("OpenVINO")]
[Tooltip("The name of the openvino models folder")]
public string openvinoModelsDir = "OpenVINOModels";
```



**Update input variables**

Like in the previous tutorial series, when using asynchronous GPU readback, we need one Texture that stores data on the GPU and one that stores data on the CPU.

```c#
// The test image dimensions
private Vector2Int imageDims;
// The test image texture
private Texture imageTexture;
// The current screen object dimensions
private Vector2Int screenDims;
// The model GPU input texture
private RenderTexture inputTextureGPU;
// The model CPU input texture
private Texture2D inputTextureCPU;
```



**Define private variables for the OpenVINO plugin**



```c#
// File paths for the available OpenVINO models
private List<string> modelPaths = new List<string>();
// Names of the available OpenVINO models
private List<string> modelNames = new List<string>();
// Names of the available OpenVINO devices
private List<string> openvinoDevices = new List<string>();
```



**Import functions from the OpenVINO plugin**

We pass the pointer to the input pixel data as an [IntPtr](https://docs.microsoft.com/en-us/dotnet/api/system.intptr?view=net-6.0).

```c#
// Name of the DLL file
const string dll = "OpenVINO_Image_Classifier_DLL";

[DllImport(dll)]
private static extern int GetDeviceCount();

[DllImport(dll)]
private static extern IntPtr GetDeviceName(int index);

[DllImport(dll)]
private static extern int LoadModel(string model, int index, int[] inputDims);

[DllImport(dll)]
private static extern int PerformInference(IntPtr inputData);
```



**Define method to get the available OpenVINO models**



```c#
/// <summary>
/// Get the file paths for available OpenVION models
/// </summary>
private void GetOpenVINOModels()
{
    // Get the paths for the XML file for each model
    foreach (string file in System.IO.Directory.GetFiles($"{Application.streamingAssetsPath}/{openvinoModelsDir}"))
    {
        if (file.EndsWith(".xml"))
        {
            modelPaths.Add(file);
            string modelName = file.Split('\\')[1].Split('.')[0];
            modelNames.Add(modelName.Substring(0, modelName.Length));
        }
    }
}
```





**Define method to get the names of available OpenVINO devices**



```c#
/// <summary>
/// Get the names of the available OpenVINO devices
/// </summary>
private void GetOpenVINODevices()
{
    // Get the number of available OpenVINO devices
    int deviceCount = GetDeviceCount();

    for (int i = 0; i < deviceCount; i++)
    {
        openvinoDevices.Add(Marshal.PtrToStringAnsi(GetDeviceName(i)));
    }
}
```



**Update method to initialize GUI dropdown menu options**



```c#
/// <summary>
/// Initialize the GUI dropdown list
/// </summary>
private void InitializeDropdown()
{
    // Create list of webcam device names
    List<string> webcamNames = new List<string>();
    foreach(WebCamDevice device in webcamDevices) webcamNames.Add(device.name);

    // Remove default dropdown options
    webcamDropdown.ClearOptions();
    // Add webcam device names to dropdown menu
    webcamDropdown.AddOptions(webcamNames);
    // Set the value for the dropdown to the current webcam device
    webcamDropdown.SetValueWithoutNotify(webcamNames.IndexOf(currentWebcam));

    // Remove default dropdown options
    modelDropdown.ClearOptions();
    // Add OpenVINO model names to menu
    modelDropdown.AddOptions(modelNames);
    // Select the first option in the dropdown
    modelDropdown.SetValueWithoutNotify(0);

    // Remove default dropdown options
    deviceDropdown.ClearOptions();
    // Add OpenVINO device names to menu
    deviceDropdown.AddOptions(openvinoDevices);
    // Select the first option in the dropdown
    deviceDropdown.SetValueWithoutNotify(0);
}
```







**Define Awake Method**

We'll implement the code to copy the plugins.xml file from the `StreamingAssets` folder to the `Plugins/x86_64` folder in the build folder in the [Awake()](https://docs.unity3d.com/ScriptReference/MonoBehaviour.Awake.html) method. The code should be inactive since we are in the Editor.

```c#
// Awake is called when the script instance is being loaded
private void Awake()
{
    #if !UNITY_EDITOR
        // Define the path for the plugins.xml file in the StreamingAssets folder
        string sourcePath = $"{Application.streamingAssetsPath}/plugins.xml";
    // Define the destination path for the plugins.xml file
    string targetPath = $"{Application.dataPath}/Plugins/x86_64/plugins.xml";
    // Only copy the file if it is not already present at the destination
    if (!File.Exists(targetPath)) File.Copy(sourcePath, targetPath);
    #endif
}
```





**Update Start method**



```c#
// Start is called before the first frame update
void Start()
{
    // Get the source image texture
    imageTexture = screen.gameObject.GetComponent<MeshRenderer>().material.mainTexture;
    // Get the source image dimensions as a Vector2Int
    imageDims = new Vector2Int(imageTexture.width, imageTexture.height);

    // Initialize list of available webcam devices
    webcamDevices = WebCamTexture.devices;
    foreach (WebCamDevice device in webcamDevices) Debug.Log(device.name);
    currentWebcam = webcamDevices[0].name;
    useWebcam = webcamDevices.Length > 0 ? useWebcam : false;
    // Initialize webcam
    if (useWebcam) InitializeWebcam(currentWebcam);

    // Resize and position the screen object using the source image dimensions
    InitializeScreen();
    // Resize and position the main camera using the source image dimensions
    InitializeCamera(screenDims);

    // Initialize list of class labels from JSON file
    classes = JsonUtility.FromJson<ClassLabels>(classLabels.text).classes;

    // Get the file paths for available OpenVINO models
    GetOpenVINOModels();
    // Get the names of available OpenVINO devices
    GetOpenVINODevices();

    // Initialize the webcam dropdown list
    InitializeDropdown();
}
```





**Update method to process images using a compute shader**



```c#
/// <summary>
/// Process the provided image using the specified function on the GPU
/// </summary>
/// <param name="image">The target image RenderTexture</param>
/// <param name="computeShader">The target ComputerShader</param>
/// <param name="functionName">The target ComputeShader function</param>
/// <returns></returns>
private void ProcessImageGPU(RenderTexture image, ComputeShader computeShader, string functionName)
{
    // Specify the number of threads on the GPU
    int numthreads = 8;
    // Get the index for the specified function in the ComputeShader
    int kernelHandle = computeShader.FindKernel(functionName);
    // Define a temporary HDR RenderTexture
    RenderTexture result = new RenderTexture(image.width, image.height, 24, RenderTextureFormat.ARGBHalf);
    // Enable random write access
    result.enableRandomWrite = true;
    // Create the HDR RenderTexture
    result.Create();

    // Set the value for the Result variable in the ComputeShader
    computeShader.SetTexture(kernelHandle, "Result", result);
    // Set the value for the InputImage variable in the ComputeShader
    computeShader.SetTexture(kernelHandle, "InputImage", image);

    // Execute the ComputeShader
    computeShader.Dispatch(kernelHandle, result.width / numthreads, result.height / numthreads, 1);

    // Copy the result into the source RenderTexture
    Graphics.Blit(result, image);

    // Release RenderTexture
    result.Release();
}
```



**Update method to handle asynchronous GPU readback**



```c#
/// <summary>
/// Called once AsyncGPUReadback has been completed
/// </summary>
/// <param name="request"></param>
private void OnCompleteReadback(AsyncGPUReadbackRequest request)
{
    if (request.hasError)
    {
        Debug.Log("GPU readback error detected.");
        return;
    }

    // Make sure the Texture2D is not null
    if (inputTextureCPU)
    {
        // Fill Texture2D with raw data from the AsyncGPUReadbackRequest
        inputTextureCPU.LoadRawTextureData(request.GetData<uint>());
        // Apply changes to Textur2D
        inputTextureCPU.Apply();
    }
}
```



**Define method to send the input texture data to the plugin**



```c#
/// <summary>
/// Pin memory for the input data and pass a reference to the plugin for inference
/// </summary>
/// <param name="texture">The input texture</param>
/// <returns></returns>
public unsafe int UploadTexture(Texture2D texture)
{
    int classIndex = -1;

    //Pin Memory
    fixed (byte* p = texture.GetRawTextureData())
    {
        // Perform inference and get the predicted class index
        classIndex = PerformInference((IntPtr)p);
    }

    return classIndex;
}
```



**Modify Update method**



```c#
// Update is called once per frame
void Update()
{
    useWebcam = webcamDevices.Length > 0 ? useWebcam : false;
    if (useWebcam)
    {
        // Initialize webcam if it is not already playing
        if (!webcamTexture || !webcamTexture.isPlaying) InitializeWebcam(currentWebcam);

        // Skip the rest of the method if the webcam is not initialized
        if (webcamTexture.width <= 16) return;

        // Make sure screen dimensions match webcam resolution when using webcam
        if (screenDims.x != webcamTexture.width)
        {
            // Resize and position the screen object using the source image dimensions
            InitializeScreen();
            // Resize and position the main camera using the source image dimensions
            InitializeCamera(screenDims);
        }
    }
    else if (webcamTexture && webcamTexture.isPlaying)
    {
        // Stop the current webcam
        webcamTexture.Stop();

        // Resize and position the screen object using the source image dimensions
        InitializeScreen();
        // Resize and position the main camera using the source image dimensions
        InitializeCamera(screenDims);
    }

    // Scale the source image resolution
    Vector2Int inputDims = CalculateInputDims(screenDims, targetDim);
    if (printDebugMessages) Debug.Log($"Input Dims: {inputDims.x} x {inputDims.y}");

    // Initialize the input texture with the calculated input dimensions
    inputTextureGPU = RenderTexture.GetTemporary(inputDims.x, inputDims.y, 24, RenderTextureFormat.ARGBHalf);

    if (!inputTextureCPU || inputTextureCPU.width != inputTextureGPU.width)
    {
        inputTextureCPU = new Texture2D(inputDims.x, inputDims.y, TextureFormat.RGBA32, false);
        // Update the selected OpenVINO model
        UpdateOpenVINOModel();
    }

    if (printDebugMessages) Debug.Log($"Input Dims: {inputTextureGPU.width}x{inputTextureGPU.height}");

    // Copy the source texture into model input texture
    Graphics.Blit((useWebcam ? webcamTexture : imageTexture), inputTextureGPU);

    // Flip image before sending to DLL
    ProcessImageGPU(inputTextureGPU, processingShader, "FlipXAxis");

    // Download pixel data from GPU to CPU
    if (useAsyncGPUReadback)
    {
        AsyncGPUReadback.Request(inputTextureGPU, 0, TextureFormat.RGBA32, OnCompleteReadback);
    }
    else
    {
        RenderTexture.active = inputTextureGPU;
        inputTextureCPU.ReadPixels(new Rect(0, 0, inputTextureGPU.width, inputTextureGPU.height), 0, 0);
        inputTextureCPU.Apply();
    }

    // Send reference to inputData to DLL
    classIndex = UploadTexture(inputTextureCPU);
    if (printDebugMessages) Debug.Log($"Class Index: {classIndex}");

    // Check if index is valid
    bool validIndex = classIndex >= 0 && classIndex < classes.Length;
    if (printDebugMessages) Debug.Log(validIndex ? $"Predicted Class: {classes[classIndex]}" : "Invalid index");

    // Release the input texture
    RenderTexture.ReleaseTemporary(inputTextureGPU);
}
```



**Define a method to update the current OpenVINO model and device**



```c#
/// <summary>
/// Update the selected OpenVINO model
/// </summary>
public void UpdateOpenVINOModel()
{
    int[] inputDims = new int[] {
        inputTextureCPU.width,
        inputTextureCPU.height
    };

    Debug.Log($"Selected Device: {openvinoDevices[deviceDropdown.value]}");

    // Load the specified OpenVINO model
    int return_val = LoadModel(modelPaths[modelDropdown.value], deviceDropdown.value, inputDims);

    string[] return_messages = {
        "Model loaded and reshaped successfully", 
        "Failed to load model",
        "Failed to reshape model input",
    };

    Debug.Log(return_messages[return_val]);
}
```



That covers the required code changes.





## Modify GUI

As mentioned earlier, we'll add new dropdown menus to the GUI so we can switch between available OpenVINO models and devices at runtime. 



**Create new GUI objects**

Select the `WebcamDeviceText` and `WebcamDropdown` objects and press Ctrl-d to duplicate them. Rename the duplicates to `OpenVINOModelText` and `OpenVINOModelDropdown`, respectively. Then select `OpenVINOModelText` and `OpenVINOModelDropdown` and press Ctrl-d. Rename the copies to `OpenVINODevicelText` and `OpenVINODevicelDropdown`, respectively.



![unity-add-openvino-gui-objects](./images/unity-add-openvino-gui-objects.png)



**Update GUI Settings Panel Height**

Select the `SettingsPanel` object and update the `Height` value to `255`.

![unity-update-settings-panel-height](./images/unity-update-settings-panel-height.png)



**Update GUI object positions**

Select the `OpenVINOModelText` object and update the `Pos Y` value to `-145` and the Text value to `OpenVINO Model:` in the Inspector tab.

![unity-update-openvinomodeltext-position](./images/unity-update-openvinomodeltext-position-and-text.png)



Then, select the `OpenVINOModelDropdown` object and update the `Pos Y` value to `-165` in the Inspector tab.

![unity-update-openvinomodeldropdown-position](./images/unity-update-openvinomodeldropdown-position.png)



Next, select the `OpenVINODevicelText` object and update the `Pos Y` value to `-200` and the Text value to `OpenVINO Device:` in the Inspector tab.

![unity-update-openvinodevicetext-position](./images/unity-update-openvinodevicetext-position-and-text.png)



Last but not least, select the `OpenVINODevicelDropdown` object and update the `Pos Y` value to `-220` in the Inspector tab.

![unity-update-openvinodevicedropdown-position](./images/unity-update-openvinodevicedropdown-position.png)





The updated GUI should look like the image below.



![unity-view-updated-gui](./images/unity-view-updated-gui.png)







## Add `ImageClassifierOpenVINO` Component

Now we can add the new `ImageClassifierOpenVINO` script to the `InferenceManager` object. Make sure to disable the existing `ImageClassifier` component, as shown below.



![unity-add-image-classifier-openvino-component](./images/unity-add-image-classifier-openvino-component.png)



## Update On Value Changed Events

With the `ImageClassifierOpenVINO` component added, we can update the On Value Changed events for the `WebcamToggle`, `WebcamDropdown`, `OpenVINOModelDropdown`, and `OpenVINODevicelDropdown` objects.

**Update the `WebcamToggle` On Value Changed Event**

![unity-webcam-toggle-update-on-value-changed](./images/unity-webcam-toggle-update-on-value-changed.png)



**Update the `WebcamDropdown` On Value Changed Event**

![unity-webcam-dropdown-update-on-value-changed](./images/unity-webcam-dropdown-update-on-value-changed.png)



**Update the `OpenVINOModelDropdown` On Value Changed Event**

![unity-update-openvino-model-dropdown-on-value-changed](./images/unity-update-openvino-model-dropdown-on-value-changed.png)



**Update the `OpenVINODeviceDropdown` On Value Changed Event**

![unity-update-openvino-device-dropdown-on-value-changed](./images/unity-update-openvino-device-dropdown-on-value-changed.png)







## Summary

This tutorial series covered creating an OpenVINO plugin to improve inference speed in the Unity game engine.



**Previous:** [How to Create an OpenVINO Plugin for Unity on Windows Pt. 2](../part-2/)



**Project Resources:** [GitHub Repository](https://github.com/cj-mills/fastai-to-openvino-to-unity-tutorial)





<!-- Cloudflare Web Analytics --><script defer src='https://static.cloudflareinsights.com/beacon.min.js' data-cf-beacon='{"token": "56b8d2f624604c4891327b3c0d9f6703"}'></script><!-- End Cloudflare Web Analytics -->

