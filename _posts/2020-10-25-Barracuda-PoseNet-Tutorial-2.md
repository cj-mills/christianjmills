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

![create_new_script](..\images\barracuda-posenet-tutorial\create_new_script.PNG)

Name the script `PoseNet`.

![new_posenet_script](..\images\barracuda-posenet-tutorial\new_posenet_script.PNG)





Open the script in your code editor.



Above the start method, create a new public RenderTexture named `inputTexture`. This is the variable to which we'll assign the `video_texture` RenderTexture that we made in part 1.

![create_inputTexture_variable](..\images\barracuda-posenet-tutorial\create_inputTexture_variable_short.png)

Below the `Update()` method create a new method called `PreprocessImage`





## Create the Pose Estimator  GameObject

In the Hierarchy tab, right-click an empty space and select `Create Empty` from the menu. Name the empty GameObject `PoseEstimator`.





With the `PoseEstimator` object selected, drag and drop the `PoseNet` script into the `Inspector` tab.

Next, we need to assign the `video_texture` object to the `inputTexture` parameter. With the `PoseEstimator` object selected, drag and drop the `video_texture` object into `inputTexture` spot in the `Inspector` tab.



 



