---
title: Getting Started With Deep Learning in Unity
layout: post
toc: false
comments: true
description: This post provides an overview of Unity's Barracuda inference library for executing deep learning models on user devices.
categories: [unity, barracuda]
hide: false
permalink: /:title/
search_exclude: false
---



* [Introduction](#introduction)
* [The Barracuda Library](#the-barracuda-library)
* [Exporting Models to ONNX](#exporting-models-to-onnx)
* [Loading Models](#loading-models)
* [Executing Models](#executing-models)
* [Working with Data](#working-with-data)
* [Summary](#summary)



## Introduction

Most deep learning models get deployed to servers instead of user devices. Server-side inference comes with many advantages, like complete control over the runtime environment and the option to scale computing resources up and down as needed. It can also be the only feasible way to run extremely-large models like [GPT-3](https://openai.com/api/).

However, running models on user devices can provide compelling cost, latency, and privacy benefits. There are no servers to maintain, no lag from poor internet connections, and no user data to protect. The latency benefits can be particularly significant for interactive real-time applications.

[Unity](https://unity.com/) is one of the best platforms for developing real-time 2D, 3D, VR, and AR applications. Its core competency is game development, but it also works well for other immersive and interactive applications.

There are many potential ways to leverage deep learning in Unity applications, including mapping user movement to virtual avatars, generating character dialogue, and powering enemy AI to name a few. Below are some examples from personal projects.

#### In-Game Style Transfer


<center>
	<video style="width:720px;max-width:100%;height:auto;" controls loop>
		<source src="../videos/deep-learning-unity-intro/in-game-style-transfer.mp4" type="video/mp4">
	</video>
</center>
#### Pose Estimation

<center>
	<video style="width:720px;max-width:100%;height:auto;" controls loop>
		<source src="../videos/multipose-demo-1.mp4" type="video/mp4">
	</video>
</center>
#### Object Detection

<center>
	<video style="width:720px;max-width:100%;height:auto;" controls loop>
		<source src="../videos/deep-learning-unity-intro/openvino-yolox-object-detection-short.mp4" type="video/mp4">
	</video>
</center>

These examples only scratch the surface of what's possible by combining deep learning models with powerful real-time creation tools like Unity and [Unreal Engine](https://www.unrealengine.com/en-US). The Barracuda library makes it easy to start exploring these possibilities.



## The Barracuda Library

[Barracuda](https://docs.unity3d.com/Packages/com.unity.barracuda@3.0/manual/index.html) is a neural network inference library for the Unity game engine. It initially focused on models trained with Unity's Deep Reinforcement Learning toolkit, [ML-Agents](https://github.com/Unity-Technologies/ml-agents), but has expanded support over time.

Barracuda provides [multiple](https://docs.unity3d.com/Packages/com.unity.barracuda@3.0/api/Unity.Barracuda.WorkerFactory.Type.html) backends for both CPU and GPU inference. The fastest CPU backend uses the [Burst compiler](https://docs.unity3d.com/Packages/com.unity.burst@1.7/manual/index.html), which translates IL/.NET bytecode into highly-optimized native code using [LLVM](https://llvm.org/). The most performant GPU backend uses [Compute shaders](https://docs.unity3d.com/Manual/class-ComputeShader.html). Compute shaders are programs written in [High-level shader language (HLSL)](https://docs.microsoft.com/en-us/windows/win32/direct3dhlsl/dx-graphics-hlsl?redirectedfrom=MSDN) that run on the GPU, outside the standard [rendering pipeline](https://docs.unity3d.com/Manual/Glossary.html#Renderpipeline).

Some platforms don't support Compute shaders, so Unity recently added a Pixel Shader backend to enable GPU inference on platforms where Compute shaders are not supported. While faster than CPU inference, it is significantly slower than the Compute shader backend in my testing.

One of Barracuda's greatest strengths is its cross-platform support. As of writing, Barracuda does not support specialized inference hardware, quantization, or even FP16 precision. However, it runs wherever Unity does, which is [nearly everywhere](https://support.unity.com/hc/en-us/articles/206336795-What-platforms-are-supported-by-Unity-).



## Exporting Models to ONNX

Barracuda works with models in the [ONNX](https://onnx.ai/) file format. PyTorch provides [built-in support](https://pytorch.org/docs/stable/onnx.html) to export models to ONNX.

```python
torch.onnx.export(learn.model.cpu(),
                  batched_tensor,
                  onnx_file_name,
                  export_params=True,
                  opset_version=9,
                  do_constant_folding=True,
                  input_names = ['input'],
                  output_names = ['output'],
                 )
```

We can use the [tf2onnx](https://github.com/onnx/tensorflow-onnx) python package to convert TensorFlow models.

```bash
python -m tf2onnx.convert --saved-model ./savedmodel --opset 10 --output model.onnx
```

Barracuda maps [ONNX operators](https://github.com/onnx/onnx/blob/main/docs/Operators.md) to backend-specific implementations, so model support depends on what operators Unity [implements](https://docs.unity3d.com/Packages/com.unity.barracuda@3.0/manual/SupportedOperators.html) for a given inference backend. One could theoretically implement missing operations themselves, but it would probably make more sense to explore other inference options at that point. Another option is to tweak the model architecture to ensure it only uses supported operations.



## Loading Models

Unity imports ONNX models as an [NNModel](https://docs.unity3d.com/Packages/com.unity.barracuda@3.0/api/Unity.Barracuda.NNModel.html) asset.

```c#
[Tooltip("The Barracuda/ONNX asset file")]
public NNModel modelAsset;
```

These then compile into a [Model](https://docs.unity3d.com/Packages/com.unity.barracuda@3.0/api/Unity.Barracuda.Model.html#methods) object at runtime.

```c#
// Get an object oriented representation of the model
m_RunTimeModel = ModelLoader.Load(modelAsset);
```



## Executing Models

Barracuda has an [IWorker](https://docs.unity3d.com/Packages/com.unity.barracuda@3.0/api/Unity.Barracuda.IWorker.html) interface that abstracts implementation details for different inference backends. It is responsible for translating the `Model` object into a set of operations and executing them.

```c#
// Create a worker to execute the model using the selected backend
IWorker engine = WorkerFactory.CreateWorker(workerType, m_RunTimeModel);
```

Barracuda can run models in a single frame or across multiple using [Coroutines](https://docs.unity3d.com/Manual/Coroutines.html). The latter option can help maintain smooth frame rates when using more demanding models.

```c#
// Execute the model with the input Tensor
engine.Execute(input);
```



## Working with Data

Barracuda stores data in multi-dimensional array-like objects called [Tensors](https://docs.unity3d.com/Packages/com.unity.barracuda@3.0/api/Unity.Barracuda.Tensor.html). 

### Initializing Tensors

We can initialize an input Tensor from an array for CPU data, a [ComputeBuffer](https://docs.unity3d.com/ScriptReference/ComputeBuffer.html) for general GPU data, or a [Texture2D](https://docs.unity3d.com/ScriptReference/Texture2D.html) or [RenderTexture](https://docs.unity3d.com/ScriptReference/RenderTexture.html) for image data.

**Initialize from an array**

```c#
// Normal single-dimensional array
float[] tensorData = new float[]
{
    0f, 1f, 2f, 
    3f, 4f, 5f,
    6f, 7f, 8f 
};

Tensor tensor = new Tensor(n: 1, h: 3, w: 3, c: 1, tensorData);
```

**Initialize from image data**

```c#
// Initialize a Tensor using the inputTexture
Tensor input = new Tensor(inputTexture, channels: 3);
```

### Accessing Tensor Elements

We can [access](https://docs.unity3d.com/Packages/com.unity.barracuda@3.0/manual/TensorHandling.html#data-access) Tensor elements using multi-dimensional array operators. 

```c#
// Normal single-dimensional array
float[] tensorData = new float[]
{
    0f, 1f, 2f, 
    3f, 4f, 5f,
    6f, 7f, 8f 
};

// Batch size: 1, Height: 3, Width: 3, Channels: 1
Tensor tensor = new Tensor(n: 1, h: 3, w: 3, c: 1, tensorData);

Debug.Log($"Tensor shape: {tensor.shape}");
Debug.Log($"First element in flat array: {tensor[0]}");
Debug.Log($"Second row, third column: {tensor[0, 1, 2, 0]}");

tensor.Dispose();

// Batch size: 1, Height: 1, Width: 3, Channels: 3
tensor = new Tensor(n: 1, h: 1, w: 3, c: 3, tensorData);

Debug.Log($"Tensor shape: {tensor.shape}");
Debug.Log($"First element in flat array: {tensor[0]}");
Debug.Log($"First row, first column, second channel: {tensor[0, 0, 0, 1]}");
Debug.Log($"First row, second column, third channel: {tensor[0, 0, 1, 2]}");

tensor.Dispose();
```

**Output**

```text
Tensor shape: (n:1, h:3, w:3, c:1)
First element in flat array: 0
Second row, third column: 5

Tensor shape: (n:1, h:1, w:3, c:3)
First element in flat array: 0
First row, first column, second channel: 1
First row, second column, third channel: 5
```



### Retrieving Model Output

We can download model output to the CPU or copy it to a RenderTexture to keep the data on the GPU, as shown below.

```c#
// Get raw model output
Tensor output = engine.PeekOutput(outputLayer);

// Copy model output to a RenderTexture
output.ToRenderTexture(outputTextureGPU);
```

Reading model output from the GPU to the CPU causes a [pipeline stall](https://en.wikipedia.org/wiki/Pipeline_stall) as Unity prevents any execution on the main thread to prevent the data from changing before it has finished downloading to the CPU. The pipeline stall can cause a noticeable performance bottleneck that increases with the amount of data we need to download.

This performance bottleneck is not an issue when the model output can stay on the GPU like when performing artistic style transfer. However, reading the prediction of a simple image classifier to the CPU can cap GPU utilization from approximately 100% to around 60%. 

#### Standard GPU Readback

![image-classifier-without-async](..\images\deep-learning-unity-intro\image-classifier-without-async.png)



Fortunately, Unity provides a method to read data from the GPU asynchronously called [AsyncGPUReadback.Request()](https://docs.unity3d.com/ScriptReference/Rendering.AsyncGPUReadback.Request.html). The one drawback to this method is that it adds a few frames of latency. That should not be noticeable as long as the frame rate is high enough.

#### Asynchronous GPU Readback

![image-classifier-with-async](..\images\deep-learning-unity-intro\image-classifier-with-async.png)



### Processing Data

We typically need to manually implement preprocessing steps like applying ImageNet normalization to input images. We can implement these preprocessing steps on the CPU using C# scripts or on the GPU using Compute shaders (when supported) or [Fragment Shaders](https://docs.unity3d.com/Manual/SL-VertexFragmentShaderExamples.html). Naturally, we want to perform image preprocessing on the GPU when possible.

#### ImageNet Normalization Compute shader

```c#
// Each #kernel tells which function to compile; you can have many kernels
#pragma kernel NormalizeImageNet

// The pixel data for the input image
Texture2D<float4> InputImage;
// The pixel data for the processed image
RWTexture2D<float4> Result;

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



We can sometimes use Barracuda to handle postprocessing by adding additional layers to the end of models, like Sigmoid, Softmax, and Argmax, [at runtime](https://docs.unity3d.com/Packages/com.unity.barracuda@3.0/api/Unity.Barracuda.ModelBuilder.html#Unity_Barracuda_ModelBuilder_Upsample2D_System_String_System_Object_Int32___System_Boolean_). 

```c#
// Create a model builder to modify the m_RunTimeModel
ModelBuilder modelBuilder = new ModelBuilder(m_RunTimeModel);

// Add a new Softmax layer
modelBuilder.Softmax(softmaxLayer, outputLayer);
// Add a new Argmax layer
modelBuilder.Reduce(Layer.Type.ArgMax, argmaxLayer, softmaxLayer);

// Create a worker to execute the model using the selected backend
IWorker engine = WorkerFactory.CreateWorker(workerType, modelBuilder.model);
```

Otherwise, we need to implement those manually as well.

```c#
// Get raw model output
Tensor output = engine.PeekOutput(outputLayer);

// Initialize vector for coordinates
Vector2 coords = new Vector2();

// Process estimated point coordinates
for (int i = 0; i < output.length; i++)
{
    coords[i] = ((output[i] + 1) / 2) * inputDims[i] * (imageDims[i] / inputDims[i]);
}
```



## Summary

This post introduced the Barracuda inference library for the Unity game engine. Barracuda is not the only option to perform inference in Unity, but it provides a good starting point. A follow-up tutorial series will walk through training a model using the [fastai library](https://docs.fast.ai/), exporting it to ONNX format, and performing inference with it in a Unity project using the Barracuda library.



**Next:** [Fastai to Unity Tutorial Pt. 1](https://christianjmills.com/Fastai-to-Unity-Tutorial-1/)













