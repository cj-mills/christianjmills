---
title: Barracuda PoseNet Tutorial Pt. 4
layout: post
toc: false
description: This post covers how to process the output of the PoseNet model.
categories: [unity, tutorial]
hide: true
search_exclude: false
---

### Previous: [Part 1](https://christianjmills.com/unity/tutorial/2020/10/25/Barracuda-PoseNet-Tutorial-1.html) [Part 2](https://christianjmills.com/unity/tutorial/2020/10/25/Barracuda-PoseNet-Tutorial-2.html) [Part 2.5](https://christianjmills.com/unity/tutorial/2020/11/05/Barracuda-PoseNet-Tutorial-2-5.html) [Part 3](https://christianjmills.com/unity/tutorial/2020/11/05/Barracuda-PoseNet-Tutorial-3.html)

* [Create ProcessOutput() Method](#create-processoutput-method)
* [Calculate Scaling Values](#calculate-scaling-values)
* [Locate Key Point Indices](#locate-key-point-indices)
* [Calculate Key Point Positions](#calculate-key-point-positions)

## Create `ProcessOutput()` Method

The post processing steps will be handled in a new method called `ProcessOutput`. The method will take in the output `Tensors` from the `predictionLayer` and the `offsetsLayer`. Before defining the function, we need to create a new constant and a new variable.

### Create `numKeypoints` Constant

The PoseNet model estimates the 2D locations of `17` key points on a human body.

| Index | Name           |
| ----- | -------------- |
| 0     | Nose           |
| 1     | Left Eye       |
| 2     | Right Eye      |
| 3     | Left Ear       |
| 4     | Right Ear      |
| 5     | Left Shoulder  |
| 6     | Right Shoulder |
| 7     | Left Elbow     |
| 8     | Right Elbow    |
| 9     | Left Wrist     |
| 10    | Right Wrist    |
| 11    | Left Hip       |
| 12    | Right Hip      |
| 13    | Left Knee      |
| 14    | Right Knee     |
| 15    | Left Ankle     |
| 16    | Right Ankle    |

Since the number of key points never changes, we'll store it in an `int` constant. Name the constant `numKeypoints` and set the value to `17`.

![numKeypoints_constant](\images\barracuda-posenet-tutorial\numKeypoints_constant.png)

### Create `keypointLocations` Variable

The processed output from the model will be stored in a new variable called `keypointLocations`. This variable will contain the `(X,Y)` coordinates for each key point. For this tutorial, the coordinates will be scaled to the original resolution of `1920x1080`.

This variable will also store the confidence values associated with the coordinates. The model predicts key point locations even when there isn't a human in the input image. In such situations, the confidence values will likely be quite low. We can decide how to handle the latest coordinates based on a confidence threshold that we pick.

There are many ways we can store this information. For simplicity, we'll stick with an array of arrays. The array will have `17` elements. Each element will contain the location information for the key point that matches their index.

![numKeyPoints_and_keypointLocations](\images\barracuda-posenet-tutorial\keypointLocations_variable.png)

### Create Method Signature

![processoutput_method_empty](\images\barracuda-posenet-tutorial\processoutput_method_empty.png)

### Retrieve Output Tenors

![update_method_processoutput](\images\barracuda-posenet-tutorial\update_method_processoutput.png)

## Calculate Scaling Values

### Calculate Model Stride

![calculate_stride](\images\barracuda-posenet-tutorial\calculate_stride.png)

### Calculate Image Scale Value

![calculate_image_scale](\images\barracuda-posenet-tutorial\calculate_image_scale.png)

### Calculate Aspect Ratio Scale Value

![calculate_aspect_ratio_scale](\images\barracuda-posenet-tutorial\calculate_aspect_ratio_scale.png)

![calculate_scaling_values](\images\barracuda-posenet-tutorial\calculate_scaling_values.png)



## Locate Key Point Indices

![locateKeyPointIndex_method](\images\barracuda-posenet-tutorial\locateKeyPointIndex_method.png)



![processOutput_locateIndices](\images\barracuda-posenet-tutorial\processOutput_locateIndices.png)



### Another Way



## Calculate Key Point Positions

![calculate_position](\images\barracuda-posenet-tutorial\calculate_position.png)



### Store Key Point Positions

![store_position](\images\barracuda-posenet-tutorial\store_position_2.png)







 