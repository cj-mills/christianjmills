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

Rename the `Sphere` objects according to the table below.

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

#### Result

![keypoint_gameobjects](\images\barracuda-posenet-tutorial\keypoint_gameobjects.PNG)

### Resize GameObjects

Next, we'll make the key point objects larger so that they're easier to see. Select the `Nose` object in the `Hierachy`. Then,   hold Shift and click `RightAnkle` to select all 17 objects at once.

![select_all_keypoint_objects](\images\barracuda-posenet-tutorial\select_all_keypoint_objects.PNG)

We need to increase the `X` and `Y` values for the `Scale` parameter in the `Inspector` table. I'm setting them to 10, but use whatever size works best for you.

### Change GameObject Material

#### Create Yellow Material

Open the `Materials` folder in the `Assets` window. Right-click an empty space and select `Material` in the the `Create` sub-menu.

![create_material](\images\barracuda-posenet-tutorial\create_material.PNG)

Name the material `Yellow` since that's the color we'll be giving it.

#### Change Material Color

With the material selected click the small white box in the `Inspector` tab.

![select_material_color](\images\barracuda-posenet-tutorial\select_material_color_3.png)

Set the value for `B` to `0` in the popup `Color` window. This will change the color to pure yellow.

![change_material_color_to_yellow](\images\barracuda-posenet-tutorial\change_material_color_to_yellow.PNG)

#### Make Material Unlit

We'll change the `Shader` for the material to `Unlit/Color`.

![change_material_shader_to_unlit_color](\images\barracuda-posenet-tutorial\change_material_shader_to_unlit_color.PNG)