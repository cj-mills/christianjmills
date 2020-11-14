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

* [Create Key Points](#create-key-points)
* [Map Key Point Locations](#map-key-point-locations)

## Create Key Points

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

The default color for the `GameObjects` doesn't stand out much against the background. Apparently, yellow is really easy for humans to spot so we'll go with that.

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

#### Assign Yellow Material

Select all the key point objects in `Hierarchy` tab. Then, drag and drop the `Yellow` material into the `Inspector` tab.

![assign_yellow_material](\images\barracuda-posenet-tutorial\assign_yellow_material.PNG)



## Map Key Point Locations

Now we can update the positions of the key point objects using the location data obtained from the `PoseNet` model. Ordinarily, we would implement this in a separate `C#` script. This script would access the `keypointLocations[][]` array in the `PoseNet` script. However, we'll do it in the `PoseNet` script to keep things simple. 

### Create `keypoints` Variable

Open the `PoseNet` script and add a public `GameObject` array. Name the variable `keypoints`.

![keypoints_variable](\images\barracuda-posenet-tutorial\keypoints_variable.png)

### Assign the Key Point Objects

Select the `PoseEstimator` object in the `Hierarchy` tab. Then, click the small lock icon above the `Inspector` tab. This will lock the current selected object in the `Inspector` tab.

![lock_inspector_2](\images\barracuda-posenet-tutorial\lock_inspector_2.png)



Make sure the `Size` value for the `Keypoints` variable is set to `0`.

![initialize_keypoints_parameter](\images\barracuda-posenet-tutorial\initialize_keypoints_parameter.png)

Select all the key point objects in the `Hierarchy`. Then, drag and drop them onto the `Keypoints` variable in the `Inspector` tab.

![assign_keypoint_objects](\images\barracuda-posenet-tutorial\assign_keypoint_objects.PNG)

Go ahead and unlock the `Inspector` tab by clicking the lock icon again.

### Create `minConfidence` Variable

Next, we'll add a public `int` variable. This variable will be the confidence threshold for deciding whether or not to display a given key point object. Name the variable `minConfidence` and set the default value to 70. You can add a `Range` attribute to create a slider in the `Inspector` tab. Set the range to `[0, 100]`.

 ![minConfidence_variable](\images\barracuda-posenet-tutorial\minConfidence_variable_2.png)

### Create `UpdateKeyPointPositions()` Method

We need to define a new method to update the key point positions. Name the method `UpdateKeyPointPositions()`.

![updateKeyPointPositions_method](\images\barracuda-posenet-tutorial\updateKeyPointPositions_method.png)

### Call the Method

We'll call the method in `Update()` just after `ProcessOutput()`.

![call_updateKeyPointPositions_method](\images\barracuda-posenet-tutorial\call_updateKeyPointPositions_method.png)



## Create `DrawSkeleton` Script

The last step for creating our pose skeleton is to draw lines connecting the appropriate key points. We'll implement this step in a new `C#` script called `DrawSkeleton`.

![create_drawSkeleton_script](\images\barracuda-posenet-tutorial\create_drawSkeleton_script.PNG)



### Define Variables

We'll need to create a few variables in the `DrawSkeleton` script so open it up in your editor.

#### Create `keypoints` Variable

We need to access the key point objects so make another `GameObject` array just like in the `PoseNet` script.

![keypoints_variable_drawSkeleton](\images\barracuda-posenet-tutorial\keypoints_variable_drawSkeleton.png)



#### Create `lines` Variable

Next create a private `GameObject` array to hold the lines themselves. Name the variable `lines`.

![lines_variable](\images\barracuda-posenet-tutorial\lines_variable.png)



#### Create `lineRenderers` Variable

We'll use [`LineRenderer`](https://docs.unity3d.com/Manual/class-LineRenderer.html) components to draw the skeleton.

![lineRenderers_variable](\images\barracuda-posenet-tutorial\lineRenderers_variable.png)



#### Create `jointPairs` Variable

The last variable will contain pairs of key point indices. These pairs indicate the start and end points for the skeleton lines.

![jointPairs_variable](\images\barracuda-posenet-tutorial\jointPairs_variable.png)