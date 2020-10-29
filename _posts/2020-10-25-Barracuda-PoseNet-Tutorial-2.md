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



## Create PoseNet Script

Create a new folder in the `Assets` window and name it `Scripts`.

In the `Scripts` folder, right-click an empty space and click the `C# Script` option in the `Create` section.

![create_new_script](\images\barracuda-posenet-tutorial\create_new_script.PNG)

Name the script `PoseNet`.

![new_posenet_script](\images\barracuda-posenet-tutorial\new_posenet_script.PNG)





Open the script in your code editor.

### Create `inputTexture` Variable

Above the start method, create a new public RenderTexture named `videoTexture`. This is the variable to which we'll assign the `video_texture` RenderTexture that we made in part 1.

![create_inputTexture_variable](\images\barracuda-posenet-tutorial\create_videoTexture_variable.png)



### Create the Preprocessing Method

Next, we need to make a new method to handle the preprocessing steps for the `inputTexture`.

We'll name this method `PreprocessImage` and place it below the `Update` method. We want this method to be called for every frame so we'll call it in the `Update()` method.

#### Create a New Texture2D

To prepare the image for the model, we need to first create a new `Texture2D` using the `inputTexture`. We don't want to alter the `inputTexture` directly.

**Old**

We'll create a new method called `ToTexture2D` to handle this. This method will take in a RenderTexture and return a Texture2D. The method will copy the data from the `RenderTexture` without needing to download it from the GPU. This is more efficient than downloading the data to the CPU and then sending back to the GPU.

**New**

We can use the `Graphics.CopyTexture()` method to copy the data from the `RenderTexture` on the GPU. This is more efficient than downloading the data to the CPU and then sending it to the GPU.



#### Resize the Image

Now that we a `Texture2D` we need to resize it to a more reasonable resolution. Lowering the resolution does decrease the model's accuracy. Unfortunately, using a higher resolution can significantly impact inference speed. We'll examine this trade-off in a later post.

The resizing method will squish our input image from a 16:9 aspect ration to a 1:1 aspect ratio. We'll need to account for this when we get to the post processing method.

#### Apply Model Specific Preprocessing

Finally, we need to modify the RGB channel values of the image so that they are in the same range of values that the model was trained on. The model that we'll be using has a ResNet50 architecture and was pretrained on the ImageNet dataset. This means we need to first scale the RGB channel values and then add the ImageNet mean for each channel.







## Create the Pose Estimator  GameObject

In the Hierarchy tab, right-click an empty space and select `Create Empty` from the menu. Name the empty GameObject `PoseEstimator`.





With the `PoseEstimator` object selected, drag and drop the `PoseNet` script into the `Inspector` tab.

Next, we need to assign the `video_texture` object to the `inputTexture` parameter. With the `PoseEstimator` object selected, drag and drop the `video_texture` object into `inputTexture` spot in the `Inspector` tab.



 



