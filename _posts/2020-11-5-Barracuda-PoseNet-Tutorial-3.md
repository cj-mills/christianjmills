---
title: Barracuda PoseNet Tutorial Pt. 3
layout: post
toc: false
description: This post covers how to load the the PoseNet model in a script.
categories: [unity, tutorial]
hide: false
search_exclude: false
---

### Previous: [Part 1](https://christianjmills.com/unity/tutorial/2020/10/25/Barracuda-PoseNet-Tutorial-1.html) [Part 2](https://christianjmills.com/unity/tutorial/2020/10/25/Barracuda-PoseNet-Tutorial-2.html) [Part 2.5](https://christianjmills.com/unity/tutorial/2020/11/05/Barracuda-PoseNet-Tutorial-2-5.html)

* [Install Barracuda Package](#install-barracuda-package)
* [Import PoseNet Model](#import-posenet-model)
* [Load the Model](#load-the-model)
* [Set Inspector Variables](#set-inspector-variables)

## Install Barracuda Package

Select the `Package Manager` tab in the Unity editor.

![select_package_manager_tab](\images\barracuda-posenet-tutorial\select_package_manager_tab.png)

Type `Barracuda` into the search box.

![barracuda_search](\images\barracuda-posenet-tutorial\barracuda_search.PNG)

Click the `Install` button to install the package.

![barracuda_install](\images\barracuda-posenet-tutorial\barracuda_install.PNG)

Wait for Unity to install the dependencies.

![barracuda_installation_progress](\images\barracuda-posenet-tutorial\barracuda_installation_progress.PNG)

## Import PoseNet Model

Now we can import the model into Unity. The Barracuda dev team has focused on supporting the [ONNX](https://onnx.ai/) format for models. We aren't able to directly import models from TensorFlow or PyTorch. I've already converted the PoseNet model to ONNX. You can check out my tutorial for converting TensorFlow SavedModels to ONNX ([here](https://christianjmills.com/tensorflow/onnx/tutorial/2020/10/21/How-to-Convert-a-TensorFlow-SavedModel-to-ONNX.html)). PyTorch provides built-in support for ONNX ([link](https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html)).

### Download the ONNX File

You can download the converted PoseNet model from this ([link](https://drive.google.com/file/d/1oKrlraI3m3ecme-pAvAh25-Jzzu86sv_/view?usp=sharing)).

### Import Model to Assets

Create a new folder in the `Assets` window and name it `Models`. Drag and drop the ONNX file into the `Models` folder.

If you select the `resnet50` asset, you should see the following in the `Inspector` tab.

![resnet50_inspector_tab](\images\barracuda-posenet-tutorial\resnet50_inspector_tab.PNG)

## Load the Model

Next, we need to implement the code for loading the model in the `PoseNet` [script](https://christianjmills.com/unity/tutorial/2020/11/04/Barracuda-PoseNet-Tutorial-2.html#create-the-posenet-script).

### Create `modelAsset` Variable

Open the `PoseNet` script and make a new public `NNModel` variable called `modelAsset`. We'll assign the `resnet50` asset to this variable in the Unity Editor.

### Create `workerType` Variable

We'll also add a variable that let's us choose which [backend](https://docs.unity3d.com/Packages/com.unity.barracuda@1.0/manual/Worker.html) to use when performing inference. The options are divided into `CPU` and `GPU`. I believe there are plans to add support for specialized hardware such as Neural Processing Units in the future. Our preprocessing pipeline runs entirely on the `GPU` so we'll be sticking with the `GPU` options for this tutorial series.

Make a new public `WorkerFactory.Type` called `workerType`. Give it a default value of `WorkerFactory.Type.Auto`.

![load_model_variables_1](\images\barracuda-posenet-tutorial\load_model_variables_1.png)

### Create `m_RuntimeModel` Variable

We need to compile the `modelAsset` into a run-time model to perform inference. We'll store the compiled model in a new private `Model` variable called `m_RuntimeModel`. This is the naming convention used in the Barracuda [documentation](https://docs.unity3d.com/Packages/com.unity.barracuda@1.0/manual/Loading.html). 

### Create `engine` Variable

Next, we'll create a new private `IWorker` variable to store our inference engine. Name the variable `engine`.

![load_model_variables_2](\images\barracuda-posenet-tutorial\load_model_variables_2.png)

### Create `heatmapLayer` Variable

Add a new private `string` variable to store the  name of the heatmap layer in the `resnet50` model. We'll need the output of this layer to determine the location of key points (e.g. nose, elbows, knees, etc.) in the input image. We can find the name for the model's output layers in the `Inspector` tab. For our model, the heatmap layer is named `float_heamap`.

![resnet50_output_layers](\images\barracuda-posenet-tutorial\resnet50_output_layers.PNG)

**Note:** The last two output layers, `resnet_v1_50/displacement_bwd_2/BiasAd` and `resnet_v1_50/displacement_fwd_2/BiasAd`, are used when estimating the pose of multiple people. We'll be sticking to single pose estimation for this series. 

### Create `offsetsLayer` Variable

We'll go ahead and create a variable for the `float_short_offsets` layer as well since we'll need it later. The output from this layer is used to refine the estimated key point locations determined with the heatmap layer. 

![layer_name_variables](\images\barracuda-posenet-tutorial\layer_name_variables.png)

### Compile the Model

We need to get an object oriented representation of the model before we can work with it. We'll do this in the `Start()` method and store it in the `m_RuntimeModel`.

![compile_model](\images\barracuda-posenet-tutorial\compile_model.png)

### Modify the Model

We need to add a [`Sigmoid`](https://docs.unity3d.com/Packages/com.unity.barracuda@1.0/api/Unity.Barracuda.ModelBuilder.html#Unity_Barracuda_ModelBuilder_Sigmoid_System_String_System_Object_) layer to the end of the model before creating our inference engine. This will map the output values to the range `[0,1]`. We'll use these values to measure the model's confidence that a given key point is in a given spot in the input image. A value of `1` would indicate that the model is `100%` confident the key point is in that location. We won't be getting any `1`'s.

First, we need to make a new private `string` variable to store the name of this new layer. We'll name the variable `predictionLayer` and name the layer `heatmap_predictions`.

![predictionLayer_name](\images\barracuda-posenet-tutorial\predictionLayer_name.png)

We'll add the new layer using a `ModelBuilder`.

![add_sigmoid_layer](\images\barracuda-posenet-tutorial\add_sigmoid_layer.png)

### Initialize the Inference Engine

Now we can create a worker to execute the modified model using the selected backend. We'll do this using the `WorkerFactory.CreateWorker()` method.

![create_worker](\images\barracuda-posenet-tutorial\create_worker.png)

### Release Inference Engine Resources

We need to manually release the resources that get allocated for the inference `engine`. This should be one of the last actions performed. Therefore, we'll do it in the `OnDisable()` method. This method gets called when the Unity project exits. We need to implement this method in the `PoseNet` script.

![onDisable_method](\images\barracuda-posenet-tutorial\onDisable_method.png)

## Set Inspector Variables

Now we just need to set the values for the `Model Asset` and select the inference backend.

### Assign the Model Asset

With the `PoseEstimator` object selected, drag and drop the `resnet50` asset into the `Model Asset` variable.

### Select Inference Backend

Set the backend to the `Compte Precompiled` option in the `Worker Type` drop-down. This is the most efficient GPU backend.

![assign_model_asset_and_backend](\images\barracuda-posenet-tutorial\assign_model_asset_and_backend.PNG)