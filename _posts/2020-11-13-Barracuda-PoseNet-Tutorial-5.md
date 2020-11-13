---
title: Barracuda PoseNet Tutorial Pt. 5
layout: post
toc: false
description: This post covers how to map the key point locations to a pose skeleton.
categories: [unity, tutorial]
hide: true
search_exclude: false
---

### Previous: [Part 1](https://christianjmills.com/unity/tutorial/2020/10/25/Barracuda-PoseNet-Tutorial-1.html) [Part 2](https://christianjmills.com/unity/tutorial/2020/10/25/Barracuda-PoseNet-Tutorial-2.html) [Part 2.5](https://christianjmills.com/unity/tutorial/2020/11/05/Barracuda-PoseNet-Tutorial-2-5.html) [Part 3](https://christianjmills.com/unity/tutorial/2020/11/05/Barracuda-PoseNet-Tutorial-3.html) [Part 4](https://christianjmills.com/unity/tutorial/2020/11/12/Barracuda-PoseNet-Tutorial-4.html)

* [Create Pose Skeleton](#create-pose-skeleton)

## Create Pose Skeleton

We'll create the pose skeleton using `GameObjects` rather than altering the `videoTexture`. We need to create a separate `GameObjects` for each of the `17` key points. 

### Create Container

In the `Hierarchy` tab, create an empty `GameObject` and name it `Key Points`. We'll store the key point objects in here to keep things organized. 

**Optional:** With `Key Points` selected, right-click the `Transform` component in the `Inspector` tab. Click `Reset` in the pop-up menu. This will reset the object's position to the origin.



![reset_transform](\images\barracuda-posenet-tutorial\reset_transform.PNG)

### Create GameObjects

Right-click the `Key Points` object and select `Sphere` under `3D Object`. This will create a nested `GameObject` inside `Key Points`.

![create_keypoint_gameobject](\images\barracuda-posenet-tutorial\create_keypoint_gameobject.PNG)



Select the new `Sphere` object and press Ctrl-d to duplicate it. We'll need `16` duplicates for the remaining key points.

Rename the `Sphere` objects according to the image below.

![keypoint_gameobjects](\images\barracuda-posenet-tutorial\keypoint_gameobjects.PNG)

