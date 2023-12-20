---
title: "Code Walkthrough: Unity Deep Learning Image Preprocessor Package"
date: 2023-5-4
image: /images/empty.gif
hide: false
search_exclude: false
categories: [unity, walkthrough]
description: "Walk through the code for the Unity Deep Learning Image Preprocessor package, a utility for preparing image input to perform inference with deep learning models in Unity."

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

The [Deep Learning Image Preprocessor](https://github.com/cj-mills/unity-deep-learning-image-preprocessor) package provides Shaders and Compute Shaders for various image processing tasks, such as cropping, normalizing, and flipping images in Unity.

Many of my tutorials involve using computer vision models in Unity applications. This package makes that shared functionality more modular and reusable, allowing me to streamline my tutorial content. 

In this post, I walk through the package code, providing a solid understanding of its components and their roles.




## Package Overview

The package contains a C# script and various processing shaders. 

### C# Script

- `ImageProcessor.cs`: This script provides utility functions to process images using compute shaders or materials.

### Shaders

1. `CropImage.shader`: This shader is responsible for cropping images based on provided offset and size values.
2. `NormalizeImage.shader`: This shader normalizes the color values of an input texture using the provided mean, standard deviation, and scaling values.
3. `ProcessingShader.compute`: This compute shader offers various image processing functionality, including normalizing input images, cropping images based on the provided offset and size values, and flipping images around the x-axis.



## Code Explanation

In this section, we will delve deeper into the Deep Learning Image Preprocessor package by examining the purpose and functionality of the C# script and shaders.



### `ImageProcessor.cs`

The script defines a public class `ImageProcessor` that inherits from `MonoBehaviour`. This class handles the processing of images using shaders. The complete code is available on GitHub at the link below.

- [ImageProcessor.cs](https://github.com/cj-mills/unity-deep-learning-image-preprocessor/blob/main/Runtime/Scripts/ImageProcessor.cs)



#### Serialized Fields
The `ImageProcessor` class contains a set of serialized fields for the shaders and normalization parameters.

```c#
[Header("Processing Shaders")]
[Tooltip("The compute shader for image processing")]
[SerializeField] private ComputeShader processingComputeShader;
[Tooltip("The shader for image normalization")]
[SerializeField] private Shader normalizeShader;
[Tooltip("The shader for image cropping")]
[SerializeField] private Shader cropShader;

[Header("Normalization Parameters")]
[Tooltip("JSON file with the mean and std values for normalization")]
[SerializeField] private TextAsset normStatsJson = null;
```





#### Private Fields and Constants
The script also defines several private fields and constants related to shaders and normalization parameters.

```c#
// GUIDs of the default assets used for shaders and normalization
private const string ProcessingComputeShaderGUID = "2c418cec15ae44419d94328d0e8dcea8";
private const string NormalizeShaderGUID = "45d8405a4cc64ecfa477b712e0465c05";
private const string CropShaderGUID = "0685d34a035b4cefa942d94390282c12";
private const string NormStatsJsonGUID = "9c8f1a57cb884c9b8a4439cae327a2f8";

// The material for image normalization
private Material normalizeMaterial;
// The material for image cropping
private Material cropMaterial;

[System.Serializable]
private class NormStats
{
    public float[] mean;
    public float[] std;
    public float scale;
}

// The mean values for normalization
private float[] mean = new float[] { 0f, 0f, 0f };
// The standard deviation values for normalization
private float[] std = new float[] { 1f, 1f, 1f };
// Value used to scale normalized input
private float scale = 1f;

// Buffer for mean values used in compute shader
private ComputeBuffer meanBuffer;
// Buffer for standard deviation values used in compute shader
private ComputeBuffer stdBuffer;
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
    processingComputeShader = LoadDefaultAsset<ComputeShader>(ProcessingComputeShaderGUID);
    normalizeShader = LoadDefaultAsset<Shader>(NormalizeShaderGUID);
    cropShader = LoadDefaultAsset<Shader>(CropShaderGUID);
    normStatsJson = LoadDefaultAsset<TextAsset>(NormStatsJsonGUID);
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
The script initializes the shaders and normalization parameters in the Start() method, which runs when the script initializes.

```c#
/// <summary>
/// Called when the script is initialized.
/// </summary>
private void Start()
{
    normalizeMaterial = new Material(normalizeShader);
    cropMaterial = new Material(cropShader);

    LoadNormStats();
    InitializeProcessingShaders();
}
```



#### `LoadNormStats`
This method loads the normalization statistics from the provided JSON file by deserializing the JSON content and updating the normalization statistics.

```c#
/// <summary>
/// Load the normalization stats from the provided JSON file.
/// </summary>
private void LoadNormStats()
{
    if (IsNormStatsJsonNullOrEmpty())
    {
        return;
    }

    NormStats normStats = DeserializeNormStats(normStatsJson.text);
    UpdateNormalizationStats(normStats);
}
```



#### `IsNormStatsJsonNullOrEmpty`
This method checks if the provided JSON file (normStatsJson) is null or empty.

```c#
/// <summary>
/// Check if the provided JSON file is null or empty.
/// </summary>
/// <returns>True if the file is null or empty, otherwise false.</returns>
private bool IsNormStatsJsonNullOrEmpty()
{
    return normStatsJson == null || string.IsNullOrWhiteSpace(normStatsJson.text);
}
```



#### `DeserializeNormStats`
This method deserializes the provided JSON string into a `NormStats` object. It catches any exceptions that might occur during the deserialization process and logs an error message, if any.

```c#
/// <summary>
/// Deserialize the provided JSON string to a NormStats object.
/// </summary>
/// <param name="json">The JSON string to deserialize.</param>
/// <returns>A deserialized NormStats object.</returns>
private NormStats DeserializeNormStats(string json)
{
    try
    {
        return JsonUtility.FromJson<NormStats>(json);
    }
    catch (Exception ex)
    {
        Debug.LogError($"Failed to deserialize normalization stats JSON: {ex.Message}");
        return null;
    }
}
```



#### `UpdateNormalizationStats`
This method updates the mean and standard deviation arrays with the provided `NormStats` object.

```c#
/// <summary>
/// Update the mean and standard deviation with the provided NormStats object.
/// </summary>
/// <param name="normStats">The NormStats object containing the mean and standard deviation.</param>
private void UpdateNormalizationStats(NormStats normStats)
{
    if (normStats == null)
    {
        return;
    }

    mean = normStats.mean;
    std = normStats.std;
    // Disable scaling if no scale value is provided
    scale = normStats.scale == 0f ? 1f : normStats.scale;
}
```



#### `InitializeProcessingShaders`
This method initializes the processing shaders by setting the mean and standard deviation values for normalization and creating compute buffers for those values.

```c#
/// <summary>
/// Initializes the processing shaders by setting the mean and standard deviation values.
/// </summary>
private void InitializeProcessingShaders()
{
    normalizeMaterial.SetVector("_Mean", new Vector4(mean[0], mean[1], mean[2], 0));
    normalizeMaterial.SetVector("_Std", new Vector4(std[0], std[1], std[2], 0));
    normalizeMaterial.SetFloat("_Scale", scale);

    if (SystemInfo.supportsComputeShaders)
    {
        int kernelIndex = processingComputeShader.FindKernel("NormalizeImage");

        meanBuffer = CreateComputeBuffer(mean);
        stdBuffer = CreateComputeBuffer(std);

        processingComputeShader.SetBuffer(kernelIndex, "_Mean", meanBuffer);
        processingComputeShader.SetBuffer(kernelIndex, "_Std", stdBuffer);
        processingComputeShader.SetFloat("_Scale", scale);
    }
}
```



#### `CreateComputeBuffer`
This method creates a `ComputeBuffer` and sets the provided data (a float array). It returns the created compute buffer.

```c#
/// <summary>
/// Creates a compute buffer and sets the provided data.
/// </summary>
/// <param name="data">The data to set in the compute buffer.</param>
/// <returns>A compute buffer with the provided data.</returns>
private ComputeBuffer CreateComputeBuffer(float[] data)
{
    ComputeBuffer buffer = new ComputeBuffer(data.Length, sizeof(float));
    buffer.SetData(data);
    return buffer;
}
```



#### `ProcessImageComputeShader`
This method prepares an image with a compute shader with the specified function name. It creates a temporary render texture, binds the source and destination textures to the compute shader, dispatches the shader, and blits the processed image back to the original image.

```c#
/// <summary>
/// Processes an image using a compute shader with the specified function name.
/// </summary>
/// <param name="image">The image to be processed.</param>
/// <param name="functionName">The name of the function in the compute shader to use for processing.</param>
public void ProcessImageComputeShader(RenderTexture image, string functionName)
{
    int kernelHandle = processingComputeShader.FindKernel(functionName);
    // Create a temporary render texture
    RenderTexture result = GetTemporaryRenderTexture(image);

    // Bind the source and destination textures to the compute shader
    BindTextures(kernelHandle, image, result);
    // Dispatche the shader
    DispatchShader(kernelHandle, result);
    // Blit the processed image back to the original image
    Graphics.Blit(result, image);

    RenderTexture.ReleaseTemporary(result);
}
```





#### `ProcessImageShader`
This method processes an image using a material. It creates a temporary render texture, applies the normalization material to the input image, and copies the resulting image back to the original image.

```c#
/// <summary>
/// Processes an image using a material.
/// </summary>
/// <param name="image">The image to be processed.</param>
public void ProcessImageShader(RenderTexture image)
{
    // Create a temporary render texture
    RenderTexture result = GetTemporaryRenderTexture(image, false);
    RenderTexture.active = result;
    // Apply the normalization material to the input image
    Graphics.Blit(image, result, normalizeMaterial);
    // Copy the result back to the original image
    Graphics.Blit(result, image);

    RenderTexture.ReleaseTemporary(result);
}
```





#### `GetTemporaryRenderTexture`
This method creates a temporary render texture with identical dimensions to the image. It takes an optional boolean parameter enableRandomWrite to enable or disable random access write into the RenderTexture.

```c#
/// <summary>
/// Creates a temporary render texture with the same dimensions as the given image.
/// </summary>
/// <param name="image">The image to match dimensions with.</param>
/// <param name="enableRandomWrite">Enable random access write into the RenderTexture.</param>
/// <returns>A temporary render texture.</returns>
private RenderTexture GetTemporaryRenderTexture(RenderTexture image, bool enableRandomWrite = true)
{
    // Create a temporary render texture
    RenderTexture result = RenderTexture.GetTemporary(image.width, image.height, 24, RenderTextureFormat.ARGBHalf);
    // Set random write access
    result.enableRandomWrite = enableRandomWrite;
    result.Create();
    return result;
}
```





#### `BindTextures`
This method binds the source and destination textures to the compute shader with the provided kernel handle. It sets the `_OutputImage` and `_InputImage` properties of the compute shader with the destination and source textures, respectively.

```c#
/// <summary>
/// Binds the source and destination textures to the compute shader.
/// </summary>
/// <param name="kernelHandle">The kernel handle of the compute shader.</param>
/// <param name="source">The source texture to be processed.</param>
/// <param name="destination">The destination texture for the processed result.</param>
private void BindTextures(int kernelHandle, RenderTexture source, RenderTexture destination)
{
    processingComputeShader.SetTexture(kernelHandle, "_OutputImage", destination);
    processingComputeShader.SetTexture(kernelHandle, "_InputImage", source);
}
```





#### `DispatchShader`
This method dispatches the compute shader based on the dimensions of the `result` texture. It calculates the thread groups in the X and Y dimensions and runs the compute shader using the provided kernel handle.

```c#
/// <summary>
/// Dispatches the compute shader based on the dimensions of the result texture.
/// </summary>
/// <param name="kernelHandle">The kernel handle of the compute shader.</param>
/// <param name="result">The result render texture.</param>
private void DispatchShader(int kernelHandle, RenderTexture result)
{
    // Calculate the thread groups in the X and Y dimensions
    int threadGroupsX = Mathf.CeilToInt((float)result.width / 8);
    int threadGroupsY = Mathf.CeilToInt((float)result.height / 8);
    // Execute the compute shader
    processingComputeShader.Dispatch(kernelHandle, threadGroupsX, threadGroupsY, 1);
}
```



#### `CalculateInputDims`
This method calculates the input dimensions of the processed image based on the original image dimensions, given a target dimension.

```c#
/// <summary>
/// Calculates the input dimensions of the processed image based on the original image dimensions.
/// </summary>
/// <param name="imageDims">The dimensions of the original image.</param>
/// <returns>The calculated input dimensions for the processed image.</returns>
public Vector2Int CalculateInputDims(Vector2Int imageDims, int targetDim = 224)
{
    targetDim = Mathf.Max(targetDim, 64);
    float scaleFactor = (float)targetDim / Mathf.Min(imageDims.x, imageDims.y);
    return Vector2Int.RoundToInt(new Vector2(imageDims.x * scaleFactor, imageDims.y * scaleFactor));
}
```



#### `CropImageComputeShader`
This method crops an image using a compute shader. It binds the source and destination textures to the compute shader, sets the offset and size parameters, dispatches the shader, and copies the result to the cropped image.

```c#
/// <summary>
/// Crops an image using a compute shader with the given offset and size.
/// </summary>
/// <param name="image">The original image to be cropped.</param>
/// <param name="croppedImage">The cropped output image.</param>
/// <param name="offset">The offset for the crop area in the original image.</param>
/// <param name="size">The size of the crop area.</param>
public void CropImageComputeShader(RenderTexture image, RenderTexture croppedImage, Vector2Int offset, Vector2Int size)
{
    int kernelHandle = processingComputeShader.FindKernel("CropImage");
    RenderTexture result = GetTemporaryRenderTexture(croppedImage);

    // Bind the source and destination textures to the compute shader
    BindTextures(kernelHandle, image, result);
    // Set the offset and size parameters
    processingComputeShader.SetInts("_CropOffset", new int[] { offset.x, offset.y });
    processingComputeShader.SetInts("_CropSize", new int[] { size.x, size.y });
    // Execute the compute shader
    DispatchShader(kernelHandle, result);
    // Copy the result to the cropped image texture
    Graphics.Blit(result, croppedImage);

    RenderTexture.ReleaseTemporary(result);
}
```



#### `CropImageShader`
This method crops an image using a material. It sets the offset and size parameters on the crop material, creates a temporary render texture, applies the crop material to the input image, and blits the result back to the cropped image.

```c#
/// <summary>
/// Crops an image using a shader with the given offset and size.
/// </summary>
/// <param name="image">The original image to be cropped.</param>
/// <param name="croppedImage">The cropped output image.</param>
/// <param name="offset">The offset for the crop area in the original image (float array with two elements).</param>
/// <param name="size">The size of the crop area (float array with two elements).</param>

public void CropImageShader(RenderTexture image, RenderTexture croppedImage, float[] offset, float[] size)
{
    // Set the offset and size parameters on the crop material
    cropMaterial.SetVector("_Offset", new Vector4(offset[0], offset[1], 0, 0));
    cropMaterial.SetVector("_Size", new Vector4(size[0], size[1], 0, 0));

    // Create a temporary render texture
    RenderTexture result = GetTemporaryRenderTexture(croppedImage, false);
    RenderTexture.active = result;

    // Apply the crop material to the input image
    Graphics.Blit(image, result, cropMaterial);
    // Copy the result to the cropped image texture
    Graphics.Blit(result, croppedImage);

    RenderTexture.ReleaseTemporary(result);
}
```



#### `OnDisable`
This method runs when the script is disabled. If the current platform supports compute shaders, it releases the compute buffers.

```c#
/// <summary>
/// Called when the script is disabled.
/// </summary>
private void OnDisable()
{
    ReleaseComputeBuffers();
}
```



#### `ReleaseComputeBuffers`
This method releases the compute buffers when compute shaders are supported.

```c#
/// <summary>
/// Releases the compute buffers if compute shaders are supported.
/// </summary>
private void ReleaseComputeBuffers()
{
    if (SystemInfo.supportsComputeShaders)
    {
        meanBuffer?.Release();
        stdBuffer?.Release();
    }
}
```






---



### `CropImage.shader`

This shader crops an input image based on the specified `offset` and `size` values. The complete code is available on GitHub at the link below.

- [CropImage.shader](https://github.com/cj-mills/unity-deep-learning-image-preprocessor/blob/main/Shaders/CropImage.shader)



#### Define Properties

```glsl
Properties {
    // The input texture to crop
    _MainTex ("Texture", 2D) = "white" {}
    // A vector representing the x and y offsets for the cropping area
    _Offset ("Offset", Vector) = (0, 0, 0, 0)
    // A vector representing the width and height of the cropping area
    _Size ("Size", Vector) = (0, 0, 0, 0)
}
```



#### SubShader Configuration

In the SubShader block, culling and depth are disabled to ensure that the shader will always render regardless of the camera's position and orientation.

```glsl
// No culling or depth
Cull Off ZWrite Off ZTest Always
```



#### Pass Block

The Pass block contains the vertex and fragment shaders that process the input texture and crops it based on the provided `_Offset` and `_Size` values.

The vertex shader receives the input vertex data and passes it through to the fragment shader. It is a simple pass-through shader that does not modify the input data.

The fragment shader is responsible for cropping the input texture based on the provided `_Offset` and `_Size` values.

```glsl
Pass {
    CGPROGRAM
    #pragma vertex vert
    #pragma fragment frag

    #include "UnityCG.cginc"

    // Uniform variables for the offset and size of the cropping area
    float2 _Offset;
    float2 _Size;

    // Contains the vertex position and texture coordinates
    struct appdata {
        float4 vertex : POSITION;
        float2 uv : TEXCOORD0;
    };

    // Contains the transformed vertex position and texture coordinates
    struct v2f {
        float2 uv : TEXCOORD0;
        float4 vertex : SV_POSITION;
    };

    v2f vert (appdata v) {
        v2f o;
        // Transform the input vertex position to clip space
        o.vertex = UnityObjectToClipPos(v.vertex);
        // Copy the input texture coordinates to the output structure
        o.uv = v.uv;
        return o;
    }

    sampler2D _MainTex;

    // Fragment shader function
    fixed4 frag (v2f i) : SV_Target {
        // Calculate the input position based on the offset and size
        float2 inputPos = i.uv * _Size + _Offset;
        // Sample the input image and return the cropped color values
        return tex2D(_MainTex, inputPos);
    }
    ENDCG
}
```



---



### `NormalizeImage.shader`

This shader normalizes an input image's color channels based on the specified mean, standard deviation, and scale values. The complete code is available on GitHub at the link below.

- [NormalizeImage.shader](https://github.com/cj-mills/unity-deep-learning-image-preprocessor/blob/main/Shaders/NormalizeImage.shader)



#### Define Properties

```glsl
Properties
{
    // The input image texture
    _MainTex("Texture", 2D) = "white" {}
    // A vector representing the mean of the color channels (r, g, b, a).
    _Mean("Mean", Vector) = (0, 0, 0, 0)
    // A vector representing the standard deviation of the color channels (r, g, b, a).
    _Std("Std", Vector) = (1, 1, 1, 1)
    // A float range to control the scaling of the output color values.
    _Scale("Scale", Range(0, 10)) = 1
}
```




#### SubShader Configuration

The shader disables culling and depth testing since it's for 2D image processing.

```glsl
// No culling or depth
Cull Off ZWrite Off ZTest Always
```



#### Pass Block

The Pass block contains the vertex and fragment shaders that process the input texture and normalizes it based on the provided `_Mean`, `_Std`, and `_Scale` values.

The vertex shader receives the input vertex data and passes it through to the fragment shader. It is a simple pass-through shader that does not modify the input data.

The fragment shader is responsible for cropping the input texture based on the provided `_Mean`, `_Std`, and `_Scale` values.

```glsl
Pass
{
    CGPROGRAM
    #pragma vertex vert
    #pragma fragment frag

    #include "UnityCG.cginc"

    // Uniform variables to hold the mean and standard deviation values for each color channel (r, g, b)
    float4 _Mean;
    float4 _Std;
    float _Scale;

    // Contains the vertex position and texture coordinates
    struct appdata {
        float4 vertex : POSITION;
        float2 uv : TEXCOORD0;
    };

    // Contains the transformed vertex position and texture coordinates
    struct v2f {
        float2 uv : TEXCOORD0;
        float4 vertex : SV_POSITION;
    };

    v2f vert (appdata v) {
        v2f o;
        // Transform the input vertex position to clip space
        o.vertex = UnityObjectToClipPos(v.vertex);
        // Copy the input texture coordinates to the output structure
        o.uv = v.uv;
        return o;
    }

    sampler2D _MainTex;

    // Fragment shader function
    float4 frag(v2f i) : SV_Target
    {
        // Sample the input image
        float4 col = tex2D(_MainTex, i.uv);
        // Normalize each color channel (r, g, b) and scale
        col.rgb = ((col.rgb - _Mean.rgb) / _Std.rgb) * _Scale;
        // Return the normalized color values
        return col;
    }
    ENDCG
}
```





---



### `ProcessingShader.compute`

This Compute Shader implements multiple image processing operations. The complete code is available on GitHub at the link below.

- [ProcessingShader.compute](https://github.com/cj-mills/unity-deep-learning-image-preprocessor/blob/main/Shaders/ProcessingShader.compute)



#### Resources

```glsl
// Input image texture
Texture2D<float4> _InputImage;

// Output image texture
RWTexture2D<float4> _OutputImage;

// Structured buffer to hold the mean values for each color channel (r, g, b)
RWStructuredBuffer<float> _Mean;

// Structured buffer to hold the standard deviation values for each color channel (r, g, b)
RWStructuredBuffer<float> _Std;

// Float variable that represents the scaling factor to apply to the normalized pixel values
float _Scale;

// The (x, y) coordinates of the top-left corner of the cropping region
int2 _CropOffset;
// The size (width, height) of the cropping region
int2 _CropSize;
```



#### `NormalizeImage`
This kernel normalizes the input image by applying a mean and standard deviation to each color channel (red, green, blue). It also scales the normalized pixel values by a given scale factor.

```glsl
// Normalize the input image
[numthreads(8, 8, 1)]
void NormalizeImage(uint3 id : SV_DispatchThreadID)
{
    float4 inputPixel = _InputImage[id.xy];
    
    // Create float4 variables for mean and standard deviation
    float4 mean = float4(_Mean[0], _Mean[1], _Mean[2], 0.0);
    float4 std = float4(_Std[0], _Std[1], _Std[2], 1.0);

    float4 normalizedPixel = (inputPixel - mean) / std;

    // Apply scaling and leave the alpha channel unchanged
    _OutputImage[id.xy] = float4(normalizedPixel.rgb * _Scale, inputPixel.a);
}
```



#### `CropImage`
This kernel crops the input image by applying a given offset and size vector to produce a smaller image.

```glsl
// Crop the input image
[numthreads(8, 8, 1)]
void CropImage(uint3 id : SV_DispatchThreadID)
{
    if (id.x < (uint)_CropSize.x && id.y < (uint)_CropSize.y)
    {
        int2 inputPos = id.xy + _CropOffset;
        _OutputImage[id.xy] = _InputImage[inputPos];
    }
}
```



#### `FlipXAxis`
This kernel flips the input image along the x-axis, effectively mirroring it.

```glsl
// Flip the input image around the x-axis
[numthreads(8, 8, 1)]
void FlipXAxis(uint3 id : SV_DispatchThreadID)
{
    uint width;
    uint height;
    _InputImage.GetDimensions(width, height);

    // Compute the flipped pixel coordinates
    int2 flippedCoords = int2(id.x, height - id.y - 1);
    _OutputImage[id.xy] = _InputImage[flippedCoords];
}
```




## Conclusion

This post provided an in-depth walkthrough of the code for the Deep Learning Image Preprocessor package. The package contains utility functions and shaders for preparing image input to perform inference with deep learning models in Unity.

You can continue to explore the package by going to its GitHub repository linked below, where you will also find instructions for installing it using the Unity Package Manager.

- GitHub Repository: [unity-deep-learning-image-preprocessor](https://github.com/cj-mills/unity-deep-learning-image-preprocessor)

You can explore demo projects that use this package at the links below.

- [Barracuda Image Classification Demo](https://github.com/cj-mills/barracuda-image-classification-demo): A simple Unity project demonstrating how to perform image classification with the `barracuda-inference-image-classification` package.
- [Barracuda Inference PoseNet Demo](https://github.com/cj-mills/barracuda-inference-posenet-demo): A simple Unity project demonstrating how to perform 2D human pose estimation with the `barracuda-inference-posenet` package.
- [Barracuda Inference YOLOX Demo](https://github.com/cj-mills/barracuda-inference-yolox-demo): A simple Unity project demonstrating how to perform object detection with the `barracuda-inference-yolox` package.





