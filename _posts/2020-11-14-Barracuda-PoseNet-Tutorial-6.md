---
title: Barracuda PoseNet Tutorial Pt. 6
layout: post
toc: false
description: This post covers how to create a pose skeleton by drawing lines between key points.
categories: [unity, tutorial]
hide: false
search_exclude: false
---

### Previous: [Part 5](https://christianjmills.com/unity/tutorial/2020/11/14/Barracuda-PoseNet-Tutorial-5.html)

* [Create DrawSkeleton Script](#create-drawskeleton-script)
* [Create SkeletonRenderer](#create-skeletonrenderer)
* [Try It Out](#try-it-out)

## Create `DrawSkeleton` Script

We'll complete our pose skeleton by drawing lines connecting the appropriate key points. Create a new `C#` script and name it `DrawSkeleton`.

![create_drawSkeleton_script](\images\barracuda-posenet-tutorial\create_drawSkeleton_script.PNG)

### Create `keypoints` Variable

We need to access the key point objects so make another public `GameObject` array just like in the `PoseNet` script.

![keypoints_variable_drawSkeleton](\images\barracuda-posenet-tutorial\keypoints_variable_drawSkeleton.png)



### Create `lines` Variable

Next, create a private `GameObject` array to hold the lines themselves. Name the variable `lines`.

![lines_variable](\images\barracuda-posenet-tutorial\lines_variable.png)



### Create `lineRenderers` Variable

We'll use [`LineRenderer`](https://docs.unity3d.com/Manual/class-LineRenderer.html) components to draw the skeleton.

![lineRenderers_variable](\images\barracuda-posenet-tutorial\lineRenderers_variable.png)



### Create `jointPairs` Variable

The next variable will contain pairs of key point indices. The corresponding key points indicate the start and end points for the skeleton lines.

![jointPairs_variable](\images\barracuda-posenet-tutorial\jointPairs_variable.png)



### Create `lineWidth` Variable

The last variable we'll make defines the line width for the skeleton lines.

![lineWidth_variable](\images\barracuda-posenet-tutorial\lineWidth_variable.png)



### Initialize Variables

We need initialize the `lines`, `lineRenderers`, and `jointPairs` variables in the `Start()` method.

![initialize_drawSkeleton_variables](\images\barracuda-posenet-tutorial\initialize_drawSkeleton_variables.png)



### Create `InitializeLine()` Method

We'll create a new method to set up each of the lines in the pose skeleton. The method will create an empty `GameObject` for a line and add a `LineRenderer` component to it. We won't set the start and end positions as none of the key points would have updated yet.

![initializeLine_method](\images\barracuda-posenet-tutorial\initializeLine_method.png)



### Create `InitializeSkeleton()` Method

Next we need to call `InitializeLine()` in a new method for each part of the pose skeleton. We'll give each region of the skeleton a different color.

![initializeSkeleton_method](\images\barracuda-posenet-tutorial\initializeSkeleton_method.png)



### Create `RenderSkeleton()` Method

The last method we need to define will handle updating the position of the each of the lines in the pose skeleton. The method will iterate through each of the joint pairs and update the start and end positions for the associated `LineRenderer`. We'll only display a given line if both of the key point objects are currently active. 

![renderSkeleton_method](\images\barracuda-posenet-tutorial\renderSkeleton_method.png)



### Call `InitializeSkeleton()` Method

We'll initialize the pose skeleton lines in the `Start()` method.

![call_initializeSkeleton_method](\images\barracuda-posenet-tutorial\call_initializeSkeleton_method.png)



### Call `RenderSkeleton()` Method

We'll render the skeleton lines in the `LateUpdate()` method instead of `Update()`. This will ensure the PoseNet model has run for the latest frame before updating the pose skeleton.

![call_renderSkeleton_method](\images\barracuda-posenet-tutorial\call_renderSkeleton_method.png)



## Create `SkeletonRenderer`

We'll attach the `DrawSkeleton` script to a new `GameObject`. Create an empty `GameObject` in the `Hierarchy` tab and name it `SkeletonRenderer`.

![create_skeletonRenderer_object](\images\barracuda-posenet-tutorial\create_skeletonRenderer_object.PNG)



### Attach the `DrawSkeleton` Script

With `SkeletonRenderer` selected in the `Hierarchy`, drag and drop the `DrawSkeleton` script into the `Inspector` tab.

![skeletonRenderer_inspector_empty](\images\barracuda-posenet-tutorial\skeletonRenderer_inspector_empty.PNG)

### Assign Key Points

Drag and drop the key point objects onto the `Keypoints` parameter just like with the `PoseNet` script.

![skeletonRenderer_inspector_full](\images\barracuda-posenet-tutorial\skeletonRenderer_inspector_full.PNG)

## Try It Out

If you press the play button, you should see something like this.

![pose_skeleton_480p_90c](\images\barracuda-posenet-tutorial\pose_skeleton_480p_90c.gif)