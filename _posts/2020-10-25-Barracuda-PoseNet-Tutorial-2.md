---
title: Barracuda PoseNet Tutorial Pt.2
layout: post
toc: false
description: This post covers how to implement the preprocessing steps for the PoseNet model.
categories: [unity, tutorial]
hide: true
search_exclude: false
---

[Part 1](https://christianjmills.com/unity/tutorial/2020/10/25/Barracuda-PoseNet-Tutorial-1.html)



## Create a Compute Shader

We can perform the preprocessing steps more quickly by executing them on the GPU rather than the CPU. In Unity, we accomplish this using a [compute shader](https://docs.unity3d.com/Manual/class-ComputeShader.html). Compute shaders are pieces of code that can run parallel tasks on the graphics card. This is beneficial since we need to perform the same preprocessing operation on every pixel in an input image.

### Create a New Folder

Create a new folder in the assets windows and name it `Shaders`.

### Create a New ComputeShader

In the `Shader` folder, right-click an empty space, select `Shader` under the `Create` option and click `Compute Shader`. We'll name it `PoseNetShader`.

![create_compute_shader](\images\barracuda-posenet-tutorial\create_compute_shader.PNG)

### Replace the ComputeShader Code

Open the `PoseNetShader` in your code editor. By default, the `ComputeShader` will contain the following

 ![default_compute_shader](\images\barracuda-posenet-tutorial\default_compute_shader.png)

We don't need the `CSMain` function so we can delete it along with the `#pragma kernel CSMain`. We need to make a new function to apply the ResNet specific preprocessing. We can just name the new function `PreprocessResNet()`. We also need to add a `Texture2D` variable below the `Result` variable. We can name the new variable `InputImage`.

The updated `ComputeShader` should look like this. 

![posenet_compute_shader](\images\barracuda-posenet-tutorial\posenet_compute_shader_2.png)

The `PreprocessResNet` function scales the RGB channel values of every pixel in the `InputImage` by `255`. This is necessary because color values are in the range of `[0,1]` by default in Unity. The function then adds the ImageNet mean specific to the RGB channels. The updated image is returned in the `Result` variable.

Now that we've created our `ComputeShader`, we need to access it in a `C#` script. 





## Create PoseNet Script

Create a new folder in the `Assets` window and name it `Scripts`.

In the `Scripts` folder, right-click an empty space and click the `C# Script` option in the `Create` section.

![create_new_script](\images\barracuda-posenet-tutorial\create_new_script.PNG)

Name the script `PoseNet`.

![new_posenet_script](\images\barracuda-posenet-tutorial\new_posenet_script.PNG)





Open the script in your code editor.

### Create `videoTexture` Variable

Above the start method, create a new public RenderTexture named `videoTexture`. This is the variable to which we'll assign the `video_texture` RenderTexture that we made in part 1.

![create_videoTexture_variable](\images\barracuda-posenet-tutorial\create_videoTexture_variable.png)



### Create the Preprocessing Method

Next, we need to make a new method to handle the preprocessing steps for the `inputTexture`.

We'll name this method `PreprocessImage` and place it below the `Update` method. We want this method to be called for every frame so we'll call it in the `Update()` method.

#### Create a New Texture2D

To prepare the image for the model, we need to first create a new `Texture2D` using the `inputTexture`. We don't want to alter the `inputTexture` directly.

We can use the `Graphics.CopyTexture()` method to copy the data from the `RenderTexture` on the GPU. This is more efficient than downloading the data to the CPU.



#### Resize the Image

Now that we a `Texture2D` we need to resize it to a more reasonable resolution. Lowering the resolution does decrease the model's accuracy. Unfortunately, using a higher resolution can significantly impact inference speed. We'll examine this trade-off in a later post.

The `Graphics.Copy()` method requires that the source and destination textures be the same size. That means we need to destroy the current `imageTexture` and make a new one with the smaller dimensions.

The resizing method will squish our input image from a 16:9 aspect ration to a square aspect ratio. We'll need to account for this when we get to the post processing section.

#### Apply Model Specific Preprocessing

Finally, we need to modify the RGB channel values of the image so that they are in the same range of values that the model was trained on. The model that we'll be using has a ResNet50 architecture and was pretrained on the ImageNet dataset. This means we need to first scale the RGB channel values and then add the ImageNet mean for each channel.

This is where we'll make use of the `PoseNetShader` we made earlier.

Create a new public `ComputeShader` variable and name it `posenetShader`. We'll assign the `PoseNetShader` to this variable in the Unity Editor. 

![create_posenetShader_variable](\images\barracuda-posenet-tutorial\create_posenetShader_variable.png)

Next, we need to create a new method for executing the `ComputeShader`. We'll name the new method `PreprocessResNet` to match the function in the `PoseNetShader`.

The new method looks like this.

![preprocessResNet_method](\images\barracuda-posenet-tutorial\preprocessResNet_method_2.png)







## Create the Pose Estimator  GameObject

In the Hierarchy tab, right-click an empty space and select `Create Empty` from the menu. Name the empty GameObject `PoseEstimator`.





With the `PoseEstimator` object selected, drag and drop the `PoseNet` script into the `Inspector` tab.

Next, we need to assign the `video_texture` object to the `inputTexture` parameter. With the `PoseEstimator` object selected, drag and drop the `video_texture` object into `inputTexture` spot in the `Inspector` tab.



 













