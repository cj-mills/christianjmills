---
title: Barracuda PoseNet Tutorial Pt. 2.5 (Optional)
layout: post
toc: false
description: This post covers how to view preprocessed images during runtime.
categories: [unity, tutorial]
hide: false
search_exclude: false
---

### [Part 1](https://christianjmills.com/unity/tutorial/2020/10/25/Barracuda-PoseNet-Tutorial-1.html)

* [Optional: View Preprocessed Input](#optional-view-preprocessed-input)

## Optional: View Preprocessed Input

If you want to see what the preprocessed images look like before they get fed into the model, we can make a second screen to view them during runtime.

### Make a Quad

In the `Hierarchy` tab, make an new Quad and name it `InputScreen`.

#### Set the Scale

We'll set both the `X` and `Y` scale values to `360` to match our current input resolution.

#### Set the Position

We also need to set the `X` and `Y` position values to `180`. Set the `Z` position value to `-1` so that it's in front of the `VideoScreen`.

![set_inputScreen_transform](\images\barracuda-posenet-tutorial\set_inputScreen_transform.PNG)

### Make a RenderTexture

Make another `RenderTexture` and name it `input_texture`.

#### Set the Size

Set the `Size` to `360 x 360`.

![set_input_texture_size](\images\barracuda-posenet-tutorial\set_input_texture_size.PNG)

#### Apply the `input_texture`

Drag and drop the `input_texture` onto the `InputScreen` in the `Scene` tab.

### Make the Shader Unlit

Set the `Shader` for the `InputScreen` to `Unlit/Texture` just like the `VideoScreen`.



### Update the PoseNet Script

Next, we need to create a few new public variables in the PoseNet script.

#### Add `displayInput` variable

Create a new public `bool` variable called `displayInput`. This will add a checkbox in the `Inpsector` tab that we can use to turn the `InputScreen` on and off.

#### Add `inputScreen` variable

Create a new public `GameObject` variable called `inputScreen`. We need to access the `InputScreen` object to activate and deactivate it.

#### Add `inputTexture` variable 

Create a new public `RenderTexture` variable called `inputTexture`. We'll assign the `input_texture` asset to this variable in the Unity Editor.

![preview_preprocessed_input_variables](\images\barracuda-posenet-tutorial\preview_preprocessed_input_variables.png)



#### Modify the `Update()` Method

We can use the `Graphics.Blit()` method to copy the `processedImage` data to the `inputTexture` variable. We'll use the `inputScreen.SetActive()` method to activate and deactivate the `InputScreen`.

![update_method_with_displayInput](\images\barracuda-posenet-tutorial\update_method_with_displayInput.png)



### Assign the Variables

With the `PoseEstimator` selected in the `Hierarchy` tab, drag and drop the `InputScreen` and `input_texture` to their respective variables in the `Inspector` tab.

![pose_estimator_displayInput](\images\barracuda-posenet-tutorial\pose_estimator_displayInput.PNG)

### Test the InputScreen

Make sure the `Display Input` checkbox is ticked in the `Inspector` tab. It will be easier to see the changes to the preprocessed images if we use a full color video. We can set the `Video Clip` for the `Video Player` to the `pexels_woman_dancing` file that we downloaded in [Part 1](https://christianjmills.com/unity/tutorial/2020/10/25/Barracuda-PoseNet-Tutorial-1.html#import-video-files).

![preprocessed_image_preview6](\images\barracuda-posenet-tutorial\preprocessed_image_preview6.gif)