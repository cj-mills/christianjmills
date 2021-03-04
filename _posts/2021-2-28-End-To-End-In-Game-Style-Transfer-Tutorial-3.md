---
title: End-to-End In-Game Style Transfer Tutorial Pt.3
layout: post
toc: false
comments: true
description: This post covers how implement the style transfer model in Unity with the Barracuda library.
categories: [style_transfer, pytorch, unity, tutorial]
hide: true
permalink: /:title/
search_exclude: false
---

### Previous: [Part 2](https://christianjmills.com/End-To-End-In-Game-Style-Transfer-Tutorial-2/)

* [Introduction](#introduction)

* [Create Style Transfer Folder](#)

* [Import Model](#import-model)

* [Create Compute Shader](#create-compute-shader)

* [Create `StyleTransfer` Script](#create-styletransfer-script)

* [Attach Script to Camera](#attach-script-to-camera)

* [Test it Out](#test-it-out)

* [Conclusion](#conclusion)

  

## Introduction

In this post we'll implement our trained style transfer model in Unity. 

## Create Style Transfer Folder

We'll place all our additions to the project in a new asset folder called `Style_Transfer`. This will help keep things organized.

![style_transfer_folder](..\images\end-to-end-in-game-style-transfer-tutorial\unity_style_transfer_folder.png)

## Import Model

Next, we need to import the trained ONNX file that we created in Part 2.

### Download ONNX Files

Right-click the `final.onnx` in your Google Drive project folder and click `Download`.

![gdrive-download-onnx-file](..\images\end-to-end-in-game-style-transfer-tutorial\gdrive-download-onnx-file.png)

### Import ONNX Files to Assets

Open the `Style_Transfer` folder and make a new folder called `Models`.

![create-models-folder](..\images\end-to-end-in-game-style-transfer-tutorial\create-models-folder.png)

Drag and drop the ONNX file into the `Models` folder.

![unity-import-onnx-file](..\images\end-to-end-in-game-style-transfer-tutorial\unity-import-onnx-file.png)



## Create Compute Shader

We can perform both the preprocessing and postprocessing operations on the GPU since both the input and output are images. We'll implement these steps in a [compute shader](https://docs.unity3d.com/Manual/class-ComputeShader.html).

### Create the Asset File

Open the `Style_Transfer` folder and create a new folder called `Shaders`. Enter the `Shaders` folder and right-click an empty space. Select `Shader` in the `Create` submenu and click `Compute Shader`. We’ll name it `StyleTransferShader`.

![unity-create-compute-shader](..\images\end-to-end-in-game-style-transfer-tutorial\unity-create-compute-shader.png)

### Remove the Default Code

Open the `StyleTransferShader` in your code editor. By default, the `ComputeShader` will contain the following. 

![default_compute_shader](..\images\end-to-end-in-game-style-transfer-tutorial\default_compute_shader.png)

Delete the `CSMain` function along with the `#pragma kernel CSMain`. Next, we need to add a `Texture2D` variable to store the input image. Name it `InputImage` and give it a data type of `<half4>`. Use the same data type for the `Result` variable as well.

![styleTransfer_shader_part1](..\images\end-to-end-in-game-style-transfer-tutorial\styleTransfer_shader_part1.png)

### Create `ProcessInput` Function

The style transfer models expect RGB channel values to be in the range `[0, 255]`. Color values in Unity are in the range `[0,1]`. Therefore, we need to scale the three channel values for the `InputImage` by `255`. We'll perform this step in a new function called `ProcessInput` as shown below.

![processInput_compute_shader](..\images\end-to-end-in-game-style-transfer-tutorial\processInput_compute_shader.png)

### Create `ProcessOutput` Function

The models are supposed to output an image with RGB channel values in the range `[0, 255]`. However, it can sometimes return values a little outside that range. We can use the built-in [`clamp()`](https://docs.microsoft.com/en-us/windows/win32/direct3dhlsl/dx-graphics-hlsl-clamp) method to make sure all values are in the correct range. We'll then scale the values back down to `[0, 1]` for Unity. We'll perform these steps in a new function called `ProcessOutput` as shown below.

![processOutput_compute_shader](..\images\end-to-end-in-game-style-transfer-tutorial\processOutput_compute_shader.png)

Now that we’ve created our `ComputeShader`, we need to execute it using a `C#` script.



## Create `StyleTransfer` Script

We need to make a new `C#` script to perform inference with the style transfer model. This script will load the model, process the input, run the model, and process the output.

### Create the Asset File

Open the `Style_Transfer` folder and create a new folder called `Scripts`. In the `Scripts` folder, right-click an empty space and select `C# Script` in the `Create` submenu.

<img src="..\images\end-to-end-in-game-style-transfer-tutorial\unity-create-csharp-script.png" alt="unity-create-csharp-script"  />

Name the script `StyleTransfer`.

![styleTransfer_script_new](..\images\basic-in-game-style-transfer-tutorial\styleTransfer_script_new.png)

### Add `Unity.Barracuda` Namespace

Open the `StyleTransfer` script and add the `Unity.Barracuda` namespace at the top of the script.

<img src="..\images\basic-in-game-style-transfer-tutorial\add_barracuda_namespace.png" alt="add_barracuda_namespace" style="zoom: 50%;" />

### Create `StyleTransferShader` Variable

Next, we need to add a public variable to access our compute shader.

![unity-declare-computeshader-variable](..\images\end-to-end-in-game-style-transfer-tutorial\unity-declare-computeshader-variable.png)



### Create Style Transfer Toggle

We'll also add a public `bool` variable to indicate whether we want to stylize the scene. This  will create a checkbox in the `Inspector` tab that we can use to toggle the style transfer on and off while the game is running.

![unity-stylizeImage-variable](..\images\end-to-end-in-game-style-transfer-tutorial\unity-stylizeImage-variable.png)



### Create TargetHeight Variable

Getting playable frame rates at higher resolutions can be difficult even when using a smaller model. We can help out our GPU by scaling down the camera input to a lower resolution before feeding it to the model. We would then scale the output image back up to the source resolution. This can also yield results closer to the test results during training if you trained the model with lower resolution images.

Create a new public `int` variable named `targetHeight`. We'll set the default value to `540` which is the same as the test image used in the Colab Notebook.

![unity-targetHeight-variable](..\images\end-to-end-in-game-style-transfer-tutorial\unity-targetHeight-variable.png)



### Create Barracuda Variables

Now we need to add a few variables to perform inference with the style transfer model.

#### Create `modelAsset` Variable

Make a new public `NNModel` variable called `modelAsset`. We’ll assign the ONNX file to this variable in the Unity Editor.

![unity-modelAsset-variable](..\images\end-to-end-in-game-style-transfer-tutorial\unity-modelAsset-variable.png)

#### Create `workerType` Variable

We’ll also add a variable that let’s us choose which [backend](https://docs.unity3d.com/Packages/com.unity.barracuda@1.0/manual/Worker.html) to use when performing inference. The options are divided into `CPU` and `GPU`. Our preprocessing pipeline runs entirely on the `GPU` so we’ll be sticking with the `GPU` options for this tutorial series.

Make a new public `WorkerFactory.Type` called `workerType`. Give it a default value of `WorkerFactory.Type.Auto`.

![unity-workerType-variable](..\images\end-to-end-in-game-style-transfer-tutorial\unity-workerType-variable.png)

#### Create `m_RuntimeModel` Variable

We need to compile the `modelAsset` into a run-time model to perform inference. We’ll store the compiled model in a new private `Model` variable called `m_RuntimeModel`.

![unity-m_RuntimModel-variable](..\images\end-to-end-in-game-style-transfer-tutorial\unity-m_RuntimModel-variable.png)

#### Create `engine` Variable

Next, we’ll create a new private `IWorker` variable to store our inference engine. Name the variable `engine`.

![unity-engine-variable](..\images\end-to-end-in-game-style-transfer-tutorial\unity-engine-variable.png)

### Compile the Model

We need to get an object oriented representation of the model before we can work with it. We’ll do this in the `Start()` method and store it in the `m_RuntimeModel`.

![compile_model](..\images\basic-in-game-style-transfer-tutorial\compile_model.png)

### Initialize Inference Engine

Now we can create a worker to execute the modified model using the selected backend. We’ll do this using the [`WorkerFactory.CreateWorker()`](https://docs.unity3d.com/Packages/com.unity.barracuda@1.0/api/Unity.Barracuda.WorkerFactory.html#Unity_Barracuda_WorkerFactory_CreateWorker_Unity_Barracuda_WorkerFactory_Type_Unity_Barracuda_Model_System_Boolean_) method.

![initialize_inference_engine](..\images\basic-in-game-style-transfer-tutorial\initialize_inference_engine.png)

### Release Inference Engine Resources

We need to manually release the resources that get allocated for the inference `engine`. This should be one of the last actions performed. Therefore, we’ll do it in the `OnDisable()` method. This method gets called when the Unity project exits.

![onDisable_method](..\images\basic-in-game-style-transfer-tutorial\onDisable_method.png)



### Create `ProcessImage()` Method

Next, we'll make a new method to execute the `ProcessInput()` and `ProcessOutput()` functions in our `ComputeShader`. This method will take in the image that needs to be processed as well as a function name to indicate which function we want to execute. We'll need to store the processed images in textures with HDR formats. This will allow us to use color values outside the default range of `[0, 1]`. As mentioned previously, the model expects values in the range of `[0, 255]`. 

#### Method Steps

1. Get the `ComputeShader` index for the specified function
2. Create a temporary `RenderTexture` with random write access enabled to store the processed image
3. Execute the `ComputeShader`
4. Copy the processed image back into the original `RenderTexture`
5. Release the temporary `RenderTexture`

#### Method Code

![unity-ProcessImage-method](..\images\end-to-end-in-game-style-transfer-tutorial\unity-ProcessImage-method.png)



### Create `StylizeImage()` Method

We'll create a new method to handle stylizing individual frames from the camera. This method will take in the `src` `RenderTexture` from the game camera and copy the stylized image back into that same `RenderTexture`.

#### Method Steps:

1. Resize the camera input to the `targetHeight`

   If the height of `src` is larger than the `targetHeight`, we'll calculate the new dimensions to downscale the camera input. We'll then adjust the new dimensions to be multiples of 8. This is to make sure we don't loose parts of the image after applying the processing steps with the `Compute shader`.

2. Apply preprocessing steps to the image

   We'll call the `ProcessImage()` method and pass `rTex` along with the name for the `ProcessInput()` function in the `ComputeShader`. The result will be stored in `rTex`.

3. Execute the model

   We'll use the `engine.Execute()` method to run the model with the current `input`. We can store the raw output from the model in a new `Tensor`.

4. Apply the postprocessing steps to the model output

   We'll call the `ProcessImage()` method and pass `rTex` along with the name for the `ProcessOutput()` function in the `ComputeShader`. The result will be stored in `rTex`.

5. Copy the stylized image to the `src` `RenderTexture`

   We'll use the `Graphics.Blit()` method to copy the final stylized image into the `src` `RenderTexure`.

6. Release the temporary `RenderTexture`

   Finally, we'll release the temporary `RenderTexture`.

#### Method Code

![unity-StylizeImage-method](..\images\end-to-end-in-game-style-transfer-tutorial\unity-StylizeImage-method.png)



### Define `OnRenderImage()` Method

We'll be calling the `StylizeImage()` method from the `OnRenderImage()` method instead of the `Update()` method. This gives us access to the `RenderTexture` for the game camera as well as the `RenderTexture` for the target display. We'll only call the the `StylizeImage()` method if `stylizeImage` is set to `true`. You can delete the empty `Update()` method as it's not needed in this tutorial.

#### Method Steps:

1. Stylize the `RenderTexture` for the game camera
2. Copy the `RenderTexture` for the camera to the `RenderTexture` for the target display. 

#### Method Code

<img src="..\images\end-to-end-in-game-style-transfer-tutorial\unity-OnRenderImage-method.png" alt="unity-OnRenderImage-method" style="zoom: 30%;" />

That completes the `StyleTransfer` script. Next, we'll attach it to the active camera in the scene.



## Attach Script to Camera

To run the `StyleTransfer` script, we need to attach it to the active `Camera` in the scene.

### Select the Camera

Open the `Biped` scene and expand the `_Scene` object in the `Hierarchy` tab. Select the `Main Camera` object from the dropdown list.

![unity-select-main-camera-object](..\images\end-to-end-in-game-style-transfer-tutorial\unity-select-main-camera-object.png)

**Note:** If you're following along with the FPS Microgame, the Main Camera is a child of the `Player` object. However, the active camera is actually the `WeaponCamera` object which is a child of the `Main Camera`.

### Attach the `StyleTransfer` Script

With the `Main Camera` object still selected, drag and drop the `StyleTransfer` script into the bottom of the `Inspector` tab.

![unity-attach-StyleTransfer-script](..\images\end-to-end-in-game-style-transfer-tutorial\unity-attach-StyleTransfer-script.png)

### Assign the Assets

Now we just need to assign the ComputeShader and model assets as well as set the inference backend. Drag and drop the `StyleTransferShader` asset into the `StyleTransferShader` spot in the `Inspector` tab. Then, drag and drop the `final.onnx` asset into the `Model Asset` spot in the `Inspector` tab. Finally, select `Compute Precompiled` from the `WorkerType` dropdown.

![unity-configure-StyleTransfer-component](..\images\end-to-end-in-game-style-transfer-tutorial\unity-configure-StyleTransfer-component.png)



### Reduce Flickering

The style transfer model used in this tutorial series does not account for consistency between frames. This results in a flickering effect that can be distracting. Getting rid of this flickering entirely would require using a different (and likely less efficient) model. However, we can minimize flickering when the camera isn't moving by disabling the `Post-process Layer` attached to the `Main Camera` object.

![unity-disable-post-process-layer](..\images\end-to-end-in-game-style-transfer-tutorial\unity-disable-post-process-layer.png)

## Test it Out

At last, we can press the play button and see how it runs.

![unity-style-transfer-screenshot](..\images\end-to-end-in-game-style-transfer-tutorial\unity-style-transfer-screenshot.png)



## Conclusion



