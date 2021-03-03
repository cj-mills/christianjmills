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

* [](#)

* [Create Compute Shader](#create-compute-shader)

* [Create `StyleTransfer` Script](#create-styletransfer-script)

* [Attach Script to Camera](#attach-script-to-camera)

* [](#)

* [](#)

* [](#)

* [Test it Out](#test-it-out)

* [Conclusion](#conclusion)

  

## Introduction

In this post we'll implement our trained style transfer model in Unity. 

## Create Style Transfer Folder

We'll place all our additions to the project in a new asset folder called `Style_Transfer`. This will help keep things organized.

![style_transfer_folder](..\images\end-to-end-in-game-style-transfer-tutorial\unity_style_transfer_folder.png)

## Import Model

Next, we need to import the trained ONNX file that we trained in Part 2.

### Download ONNX Files

Right-click the `final.onnx` in your Google Drive project folder and click `Download`.

![gdrive-download-onnx-file](..\images\end-to-end-in-game-style-transfer-tutorial\gdrive-download-onnx-file.png)

### Import ONNX Files to Assets

Open the `Style_Transfer` folder and make a new folder called `Models`.

![create-models-folder](..\images\basic-in-game-style-transfer-tutorial\create-models-folder.png)

Drag and drop the ONNX files into the `Models` folder.





## Prepare Render Textures





## Create Compute Shader

We can perform both the preprocessing and postprocessing operations on the GPU since both the input and output are images. We'll implement these steps in a [compute shader](https://docs.unity3d.com/Manual/class-ComputeShader.html).

### Create the Asset File

Open the `Style_Transfer` folder and create a new folder called `Shaders`. Enter the `Shaders` folder and right-click an empty space. Select `Shader` in the `Create` submenu and click `Compute Shader`. We’ll name it `StyleTransferShader`.

![create-compute-shader](..\images\basic-in-game-style-transfer-tutorial\create-compute-shader.png)

### Remove the Default Code

Open the `PoseNetShader` in your code editor. By default, the `ComputeShader` will contain the following. 

![default_compute_shader](..\images\basic-in-game-style-transfer-tutorial\default_compute_shader.png)

Delete the `CSMain` function along with the `#pragma kernel CSMain`. Next, we need to add a `Texture2D` variable to store the input image. Name it `InputImage` and give it a data type of `<half4>`. Use the same data type for the `Result` variable as well.

![styleTransfer_shader_part1](..\images\basic-in-game-style-transfer-tutorial\styleTransfer_shader_part1.png)

### Create `ProcessInput` Function

The style transfer models expect RGB channel values to be in range `[0, 255]`. Color values in Unity are in the range `[0,1]`. Therefore, we need to scale the three channel values for the `InputImage` by `255`. We'll perform this step in a new function called `ProcessInput` as shown below.

![processInput_compute_shader](..\images\basic-in-game-style-transfer-tutorial\processInput_compute_shader.png)

### Create `ProcessOutput` Function

The models are supposed to output an image with RGB channel values in the range `[0, 255]`. However, it can sometimes return values a little outside that range. We can use the built-in [`clamp()`](https://docs.microsoft.com/en-us/windows/win32/direct3dhlsl/dx-graphics-hlsl-clamp) method to make sure all values are in the correct range. We'll then scale the values back down to `[0, 1]` for Unity. We'll perform these steps in a new function called `ProcessOutput` as shown below.

![processOutput_compute_shader](..\images\basic-in-game-style-transfer-tutorial\processOutput_compute_shader.png)

Now that we’ve created our `ComputeShader`, we need to execute it using a `C#` script.



## Create `StyleTransfer` Script

We need to make a new `C#` script to perform inference with the style transfer models. This script will load the model, process the input, runt the model, and process the output.

### Create the Asset File

Open the `Style_Transfer` folder and create a new folder called `Scripts`. In the `Scripts` folder, right-click an empty space and select `C# Script` in the `Create` submenu.

![create_c_sharp_script](..\images\basic-in-game-style-transfer-tutorial\create_c_sharp_script.png)

Name the script `StyleTransfer`.

![styleTransfer_script_new](..\images\basic-in-game-style-transfer-tutorial\styleTransfer_script_new.png)

### Add `Unity.Barracuda` Namespace

Open the `StyleTransfer` script and add the `Unity.Barracuda` namespace at the top of the script.

![add_barracuda_namespace](..\images\basic-in-game-style-transfer-tutorial\add_barracuda_namespace.png)

### Create `RenderTexture` Variables

We need to create some public variables that we can use to access our two render texture assets in the script.

![renderTexture_variables](..\images\basic-in-game-style-transfer-tutorial\renderTexture_variables_2.png)

### Create `StyleTransferShader` Variable

Next, we'll add a public variable to access our compute shader.

![styleTransferShader_variable](..\images\basic-in-game-style-transfer-tutorial\styleTransferShader_variable_2.png)

### Create Barracuda Variables

Now we need to add a few variables to perform inference with the style transfer models.

#### Create `modelAsset` Variable

Make a new public `NNModel` variable called `modelAsset`. We’ll assign one of the ONNX files to this variable in the Unity Editor.

![modelAsset_variable](..\images\basic-in-game-style-transfer-tutorial\modelAsset_variable_2.png)

#### Create `workerType` Variable

We’ll also add a variable that let’s us choose which [backend](https://docs.unity3d.com/Packages/com.unity.barracuda@1.0/manual/Worker.html) to use when performing inference. The options are divided into `CPU` and `GPU`. Our preprocessing pipeline runs entirely on the `GPU` so we’ll be sticking with the `GPU` options for this tutorial series.

Make a new public `WorkerFactory.Type` called `workerType`. Give it a default value of `WorkerFactory.Type.Auto`.

![workerType_variable](..\images\basic-in-game-style-transfer-tutorial\workerType_variable.png)

#### Create `m_RuntimeModel` Variable

We need to compile the `modelAsset` into a run-time model to perform inference. We’ll store the compiled model in a new private `Model` variable called `m_RuntimeModel`.

![m_RuntimeModel_variable](..\images\basic-in-game-style-transfer-tutorial\m_RuntimeModel_variable.png)

#### Create `engine` Variable

Next, we’ll create a new private `IWorker` variable to store our inference engine. Name the variable `engine`.

![engine_variable](..\images\basic-in-game-style-transfer-tutorial\engine_variable.png)

### Compile the Model

We need to get an object oriented representation of the model before we can work with it. We’ll do this in the `Start()` method and store it in the `m_RuntimeModel`.

![compile_model](..\images\basic-in-game-style-transfer-tutorial\compile_model.png)

### Initialize Inference Engine

Now we can create a worker to execute the modified model using the selected backend. We’ll do this using the [`WorkerFactory.CreateWorker()`](https://docs.unity3d.com/Packages/com.unity.barracuda@1.0/api/Unity.Barracuda.WorkerFactory.html#Unity_Barracuda_WorkerFactory_CreateWorker_Unity_Barracuda_WorkerFactory_Type_Unity_Barracuda_Model_System_Boolean_) method.

![initialize_inference_engine](..\images\basic-in-game-style-transfer-tutorial\initialize_inference_engine.png)

### Release Inference Engine Resources

We need to manually release the resources that get allocated for the inference `engine`. This should be one of the last actions performed. Therefore, we’ll do it in the `OnDisable()` method. This method gets called when the Unity project exits.

![onDisable_method](..\images\basic-in-game-style-transfer-tutorial\onDisable_method.png)



### Create `ToTexture2D()` Method

We'll make a new method to copy the data from a `RenderTexture` to a new `Texture2D`. We'll need to call this method before performing both the preprocessing and postprocessing steps. The method will take in the source `RenderTexture` and the format for the new `Texture2D`.

![toTexture2D_method](..\images\basic-in-game-style-transfer-tutorial\toTexture2D_method.png)

### Create `ProcessImage()` Method

Next, we'll make a new method to execute the `ProcessInput()` and `ProcessOutput()` functions in our `ComputeShader`. This method will take in the image that needs to be processed as well as a function name to indicate which function we want to execute. We'll need to store the processed images in textures with HDR formats. This will allow us to use color values outside the default range of `[0, 1]`. As mentioned previously, the model expects values in the range of `[0, 255]`.

![processImage_method](..\images\basic-in-game-style-transfer-tutorial\processImage_method.png)

### Process Input Image

Now we can process the current camera frame. We'll call the `ToTexture2D()` method at the top of the `Update` method. The `cameraInput` is not an HDR texture so we'll use an SDR format for the new `Texture2D`. We'll then call the `ProcessImage()` method with new `Texture2D` as input.

![process_input_image](..\images\basic-in-game-style-transfer-tutorial\process_input_image.png)

### Perform Inference

Next, we'll feed the `processedImage` to the model and get the output. We first need to convert the `processedImage` to a `Tensor`.

![perform_inference_pt1](..\images\basic-in-game-style-transfer-tutorial\perform_inference_pt1.png)

We'll then use the `engine.Execute()` method to run the model with the current `input`. We can store the raw output from the model in a new `Tensor`.

![perform_inference_pt2](..\images\basic-in-game-style-transfer-tutorial\perform_inference_pt2.png)

### Process the  Output

We need to process the raw output from the model before we can display it to the user. We'll first copy the model output to a new HDR `RenderTexture`.

![process_output_pt1](..\images\basic-in-game-style-transfer-tutorial\process_output_pt1.png)

We'll then copy the data to a `Texture2D` and pass it to the `ProcessImage()` method. This time we'll be executing the `ProcessOutput()` function on the `ComputeShader`.

![process_output_pt2](..\images\basic-in-game-style-transfer-tutorial\process_output_pt2.png)

### Display the Processed Output

We can finally display the stylized image by using the `Graphics.Blit()` method to copy the final image to `processedOutput`.

![display_output](..\images\basic-in-game-style-transfer-tutorial\display_output.png)

Next, we'll need to modify the project scene to use the `StyleTransfer` script. 







## Attach Script to Camera

To run the `StyleTransfer` script, we need to attach it to a `GameObject` in the scene.



### Open the `Biped` Scene

In the `Assets` window, open the `Scenes` folder and double-click on the `Biped.unity` asset. You don't need to save the current scene if you get prompted to do so.

![select_biped_scene](..\images\basic-in-game-style-transfer-tutorial\select_biped_scene.png)



### Create an Empty `GameObject`

In the Hierarchy tab, right-click an empty space and select `Create Empty` from the menu. Name the empty GameObject `StyleConverter`.

![create_empyt_gameObject](..\images\basic-in-game-style-transfer-tutorial\create_empyt_gameObject.png)

### Attach the `StyleTransfer` Script

With the `StyleConverter` object selected, drag and drop the `StyleTransfer` script into the `Inspector` tab.

![attach_styleTransfer_script](..\images\basic-in-game-style-transfer-tutorial\attach_styleTransfer_script.png)

### Assign the Assets

We need to assign the render textures, compute shader and one of the ONNX files to their respective parameters in the `Inspector` tab. I'll start with the mosaic model. We'll also set the `Worker Type` to `Compute Precompiled`. 

![attach_styleTransfer_script_full](..\images\basic-in-game-style-transfer-tutorial\attach_styleTransfer_script_full.png)

## Set Camera Target Texture





## Test it Out

We can finally press the play button and see how it looks.





## Conclusion



#### [GitHub Repository]()

