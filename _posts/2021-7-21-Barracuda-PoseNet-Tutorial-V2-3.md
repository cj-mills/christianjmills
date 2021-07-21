---
title: Barracuda PoseNet Tutorial 2nd Edition Pt. 3
layout: post
toc: false
comments: true
description: This post covers how to implement the preprocessing steps for the PoseNet models.
categories: [unity,barracuda,tutorial]
hide: true
permalink: /:title/
search_exclude: false
---

* [Overview](#overview)
* [Create Compute Shader](#create-compute-shader)
* [Create Utils Script](#create-utils-script)
* [Update `PoseEstimator` Script](#update-poseestimator-script)
* [Summary](#summary)



## Overview

The MobileNet and ResNet50 versions of the PoseNet model require different preprocessing steps. While it is more efficient to perform theses steps on a GPU with [Compute shaders](https://docs.unity3d.com/Manual/class-ComputeShader.html), this may not be supported on the target platform. Therefore, we will also cover how to perform the preprocessing steps on the CPU as well. 

**Note:** We will be manually toggling between using the CPU and GPU in this tutorial. For real-world applications, we can determine if the target system supports compute shaders with the [SystemInfo](https://docs.unity3d.com/ScriptReference/SystemInfo.html).[supportsComputeShaders](https://docs.unity3d.com/ScriptReference/SystemInfo-supportsComputeShaders.html) property.



## Create Compute Shader

In the Assets section, create a new folder called `Shaders`. Enter the Shaders folder and right-click an empty space. Select the `Create` submenu and select `Shader`. Inside the Shader submenu, select `Compute Shader`. We can name the new shader `PoseNetShader`.

![unity-create-compute-shader](..\images\barracuda-posenet-tutorial-v2\part-3\unity-create-compute-shader.png)





Double-click the new shader to open it in the code editor. By default, Compute shaders contain the following code.

```c#
// Each #kernel tells which function to compile; you can have many kernels
#pragma kernel CSMain

// Create a RenderTexture with enableRandomWrite flag and set it
// with cs.SetTexture
RWTexture2D<float4> Result;

[numthreads(8,8,1)]
void CSMain (uint3 id : SV_DispatchThreadID)
{
    // TODO: insert actual code here!

    Result[id.xy] = float4(id.x & id.y, (id.x & 15)/15.0, (id.y & 15)/15.0, 0.0);
}
```



### Specify Function Names



```c#
// Each #kernel tells which function to compile; you can have many kernels
#pragma kernel PreprocessResNet
#pragma kernel PreprocessMobileNet
```



### Define Variables



```c#
// The pixel data for the input image
Texture2D<half4> InputImage;
// The pixel data for the processed image
RWTexture2D<half4> Result;
```





### Create PreprocessMobileNet Function



```c#
[numthreads(8, 8, 1)]
void PreprocessMobileNet(uint3 id : SV_DispatchThreadID)
{
    // Normalize the color values to the range [-1,1]
    //2 * (value - min) / (max - min) - 1
    Result[id.xy] = half4((2.0h * (InputImage[id.xy].x) / (1.0h) - 1.0h),
        (2.0h * (InputImage[id.xy].y) / (1.0h) - 1.0h),
        (2.0h * (InputImage[id.xy].z) / (1.0h) - 1.0h), 1.0h);

}
```





### Create PreprocessResNet Function



```c#
[numthreads(8, 8, 1)]
void PreprocessResNet(uint3 id : SV_DispatchThreadID)
{
    // Scale each color value to the range [0,255] and add the ImageNet mean value
    Result[id.xy] = half4((InputImage[id.xy].x * 255.0h) + (-123.15h),
        (InputImage[id.xy].y * 255.0h) + (-115.90h),
        (InputImage[id.xy].z * 255.0h) + (-103.06h), 1.0h);
}
```







### Final Code

```c#
// Each #kernel tells which function to compile; you can have many kernels
#pragma kernel PreprocessResNet
#pragma kernel PreprocessMobileNet

// The pixel data for the input image
Texture2D<half4> InputImage;
// The pixel data for the processed image
RWTexture2D<half4> Result;

[numthreads(8, 8, 1)]
void PreprocessMobileNet(uint3 id : SV_DispatchThreadID)
{
    // Normalize the color values to the range [-1,1]
    //2 * (value - min) / (max - min) - 1
    Result[id.xy] = half4((2.0h * (InputImage[id.xy].x) / (1.0h) - 1.0h),
        (2.0h * (InputImage[id.xy].y) / (1.0h) - 1.0h),
        (2.0h * (InputImage[id.xy].z) / (1.0h) - 1.0h), 1.0h);

}

[numthreads(8, 8, 1)]
void PreprocessResNet(uint3 id : SV_DispatchThreadID)
{
    // Scale each color value to the range [0,255] and add the ImageNet mean value
    Result[id.xy] = half4((InputImage[id.xy].x * 255.0h) + (-123.15h),
        (InputImage[id.xy].y * 255.0h) + (-115.90h),
        (InputImage[id.xy].z * 255.0h) + (-103.06h), 1.0h);
}
```











## Create Utils Script



### Remove MonoBehaviour Inheritance

```c#
public class Utils
```





### Create PreprocessMobilenet Method

```c#
/// <summary>
    /// Applies the preprocessing steps for the MobileNet model on the CPU
    /// </summary>
    /// <param name="tensor">Pixel data from the input tensor</param>
    public static void PreprocessMobilenet(float[] tensor)
    {
        // Normaliz the values to the range [-1, 1]
        System.Threading.Tasks.Parallel.For(0, tensor.Length, (int i) =>
        {

            tensor[i] = (float)(2.0f * tensor[i] / 1.0f) - 1.0f;

        });
    }
```



### Create PreprocessResnet Method

```c#
///// <summary>
    ///// Applies the preprocessing steps for the ResNet50 model on the CPU
    ///// </summary>
    ///// <param name="tensor">Pixel data from the input tensor</param>
    public static void PreprocessResnet(float[] tensor)
    {

        float[] imagenetMean = new float[] { -123.15f, -115.90f, -103.06f };

        System.Threading.Tasks.Parallel.For(0, tensor.Length / 3, (int i) =>
        {

            tensor[i * 3 + 0] = (float)tensor[i * 3 + 0] * 255f + imagenetMean[0];
            tensor[i * 3 + 1] = (float)tensor[i * 3 + 1] * 255f + imagenetMean[1];
            tensor[i * 3 + 2] = (float)tensor[i * 3 + 2] * 255f + imagenetMean[2];

        });
    }
```





## Update PoseEstimator Script





### Add Barracuda Namespace

```c#
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Video;
using Unity.Barracuda;
```



### Add Public Variables

```c#
public class PoseEstimator : MonoBehaviour
{
    public enum ModelType
    {
        MobileNet,
        ResNet50
    }
```







```c#
    [Tooltip("The ComputeShader that will perform the model-specific preprocessing")]
    public ComputeShader posenetShader;

    [Tooltip("The model architecture used")]
    public ModelType modelType = ModelType.ResNet50;

    [Tooltip("Use GPU for preprocessing")]
    public bool useGPU = true;

    [Tooltip("The dimensions of the image being fed to the model")]
    public Vector2Int imageDims = new Vector2Int(256, 256);
```





### Add Private Variables

```c#
    // Target dimensions for model input
    private Vector2Int targetDims;

    // Used to scale the input image dimensions while maintaining aspect ratio
    private float aspectRatioScale;

    // The texture used to create input tensor
    private RenderTexture rTex;

    // The name of the compute shader function to proces model input
    private string preProcessFunction;

    // Stores the input data for the model
    private Tensor input;
```









```c#
// Start is called before the first frame update
    void Start()
    {
        if (useWebcam)
        {
            // Limit application framerate to the target webcam framerate
            Application.targetFrameRate = webcamFPS;

            // Create a new WebCamTexture
            webcamTexture = new WebCamTexture(webcamDims.x, webcamDims.y, webcamFPS);

            // Start the Camera
            webcamTexture.Play();

            // Deactivate the Video Player
            videoScreen.GetComponent<VideoPlayer>().enabled = false;

            // Update the videoDims.y
            videoDims.y = (int)webcamTexture.height;
            // Update the videoDims.x
            videoDims.x = (int)webcamTexture.width;

        }
        else
        {
            // Update the videoDims.y
            videoDims.y = (int)videoScreen.GetComponent<VideoPlayer>().height;
            // Update the videoDims.x
            videoDims.x = (int)videoScreen.GetComponent<VideoPlayer>().width;
        }

        // Create a new videoTexture using the current video dimensions
        videoTexture = RenderTexture.GetTemporary(videoDims.x, videoDims.y, 24, RenderTextureFormat.ARGBHalf);

        // Initialize the videoScreen
        InitializeVideoScreen(videoDims.x, videoDims.y, useWebcam);

        // Adjust the camera based on the source video dimensions
        InitializeCamera();

        // Adjust the input dimensions to maintain the source aspect ratio
        aspectRatioScale = (float)videoTexture.width / videoTexture.height;
        targetDims.x = (int)(imageDims.y * aspectRatioScale);
        imageDims.x = targetDims.x;

        // Initialize the RenderTexture that will store the processed input image
        rTex = RenderTexture.GetTemporary(imageDims.x, imageDims.y, 24, RenderTextureFormat.ARGBHalf);
    }
```















## Summary

__.



**Previous:** [Part 2](https://christianjmills.com/Barracuda-PoseNet-Tutorial-V2-2/)

**Project Resources:** [GitHub Repository](https://github.com/cj-mills/Barracuda-PoseNet-Tutorial)

