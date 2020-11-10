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

### Create `numKeypoints` Variable

### Create `keypointLocations` Variable



![numKeyPoints_and_keypointLocations](\images\barracuda-posenet-tutorial\numKeyPoints_and_keypointLocations.png)



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







 