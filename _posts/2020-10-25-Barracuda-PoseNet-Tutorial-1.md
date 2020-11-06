---
title: Barracuda PoseNet Tutorial Pt. 1
layout: post
toc: false
description: This first post covers how to set up a video player in Unity. We'll be using the video player to check the performance of the PoseNet model.
categories: [unity,tutorial]
hide: false
search_exclude: false
---

* [Prerequisites](#prerequisites)
* [Create a New Project](#create-a-new-project)
* [Import Video Files](#import-video-files)
* [Create the Video Player](#create-the-video-player)
* [Create the Video Screen](#create-the-video-screen)
* [Test the Video Player](#test-the-video-player)

## Prerequisites

This tutorial assumes that you have Unity installed. If you're completely new to Unity, you can get your feet wet with this [tutorial](https://www.youtube.com/watch?v=Lu76c85LhGY).

**Note:** This tutorial was made using Unity 2019.4.13. If you want to use the exact same version, you can download it with this ([link](unityhub://2019.4.13f1/518737b1de84)).

## Create a New Project

First, we need to create a new Unity project. Since the PoseNet model only estimates 2D poses, we'll select the 2D template.

![create_project](\images\barracuda-posenet-tutorial\create_project.PNG)

## Import Video Files

We'll be using these two videos available on [Pexels](https://www.pexels.com/), a free stock photos & videos site. The first one is easier for the PoseNet model. The second one is a bit more challenging. We'll be using the **Full HD** resolution for the videos.

1. [Two Young Men Doing a Boardslide Over a Railing](https://www.pexels.com/video/two-young-men-doing-a-boardslide-over-a-railing-4824358/)

   **Note:** Renamed to `pexels_boardslides`

2. [Woman Dancing](https://www.pexels.com/video/woman-dancing-2873755/)

   **Note:** Renamed to `pexels_woman_dancing`

### Create the Videos Folder

In the `Assets` window, right-click an empty space, select the `Create` option, and click `Folder`. Name the folder `Videos`.

![create_folder](\images\barracuda-posenet-tutorial\create_folder.PNG)

Double-click the `Videos` folder to open it.

### Add Video Files

Drag and drop the two video files from the file explorer into the `Videos` folder.

![video_file_assets](\images\barracuda-posenet-tutorial\video_file_assets.PNG)



## Create the Video Player

In the `Hierarchy` tab, right click an empty area, select the `Video` section, and click `Video Player`. This will create a new `GameObject` called `Video Player`. The default name works well enough so we'll leave it as is.

![create_video_player](\images\barracuda-posenet-tutorial\create_video_player.PNG)

### Set Video Clip

Select the `Video Player` object in the `Hierarchy` tab. Then drag and drop the `pexels_boardslides` file into the `Video Clip` parameter in the `Inspector` tab.

#### Before:

![video_clip_empty](\images\barracuda-posenet-tutorial\video_clip_empty.png)

#### After:

![video_clip_filled](\images\barracuda-posenet-tutorial\video_clip_filled.png)

### Make Video Loop

Tick the `Loop` checkbox in the `Inspector` tab to make the video repeat when the project is running.

![loop_video_checkbox](\images\barracuda-posenet-tutorial\loop_video_checkbox.png)



## Create the Video Screen

We need to make a "screen" in Unity to watch the video. To make the screen, we'll use a [`Render Texture`](https://docs.unity3d.com/ScriptReference/RenderTexture.html) to store the data for the current frame and attach it to the surface of a `GameObject`. 

### Create a Render Texture

Create a new folder in the `Assets` window and name it `Textures`.

![textures_folder](\images\barracuda-posenet-tutorial\textures_folder.PNG)

Open the folder and right click an empty space. In the `Create` section, click `Render Texture`. You can name it something like `video_texture`.

![create_render_texture](\images\barracuda-posenet-tutorial\create_render_texture.PNG)

### Resize the Render Texture

With the `video_texture` object selected, adjust the values for the `Size` parameter in the `Inspector` tab. We'll set the size parameter to the resolution of the videos. In our case, the resolution is 1920 x 1080.

#### Before:

![rt_size_parameter](\images\barracuda-posenet-tutorial\rt_size_parameter.png)

#### After:

![rt_set_resolution](\images\barracuda-posenet-tutorial\rt_set_resolution.png)

### Assign the Render Texture

With the resolution set, select the `Video Player` object in the `Hierarchy` tab again. Drag and drop the `video_texture` object into the `Target Texture` parameter in the `Inspector` tab.

#### Before:

![target_texture_empty](\images\barracuda-posenet-tutorial\target_texture_empty.png)

#### After:

![target_texture_filled](\images\barracuda-posenet-tutorial\target_texture_filled.png)

### Create the Screen GameObject

Now, we need to create the screen itself. We'll use a [`Quad`](https://docs.unity3d.com/Manual/PrimitiveObjects.html) object for the screen. Right click an empty space in the `Hierarchy` tab, select the `3D Object` section and click `Quad`. We can just name it `VideoScreen`.

![create_quad](\images\barracuda-posenet-tutorial\create_quad.PNG)

### Resize the Screen

With the `VideoScreen` object selected, we need to adjust the `Scale` parameter in the `Inspector` tab. Set the `X` value to 1920 and the `Y` value to 1080. Leave the `Z` value at 1.

#### Before:

![quad_scale_default](\images\barracuda-posenet-tutorial\quad_scale_default.png)

#### After:

![quad_scale_set](\images\barracuda-posenet-tutorial\quad_scale_set.png)

### Set the Screen Position

Next, we need to adjust the position of the `VideoScreen` object so that the bottom left corner is at `X: 0, Y: 0, Z: 0`. This will make things easier when handling the output from the PoseNet model. To do this, we'll adjust the `Position` value in the `Inspector` tab. Set the `X` value to half the `X` value for the `Scale` parameter. Do the same for the `Y` value. The `X` value for `Position` should be 960 and the `Y` value should be set to 540.

#### Before:

![quad_position_default](\images\barracuda-posenet-tutorial\quad_position_default.png)

#### After:

![quad_position_set](\images\barracuda-posenet-tutorial\quad_position_set.png)

### Reset the Scene Perspective

With the parameters for the `VideoScreen` object set, we need to zoom out and re-center our perspective. We can easily do this by selecting the `VideoScreen` object and pressing the `F` key on our keyboard. If you want to zoom back in a bit, you can scroll up with your mouse wheel.

![recentered_video_screen](\images\barracuda-posenet-tutorial\recentered_video_screen.PNG)

### Apply the Render Texture to the Screen

Drag and drop the `video_texture` object from the `Textures` folder onto the `VideoScreen` object in the `Scene` tab. The `VideoScreen` object should turn completely black.

![empty_screen](\images\barracuda-posenet-tutorial\empty_screen.PNG)

### Make Video Screen Unlit

With the `VideoScreen` object selected, click the `Shader` dropdown in the `Inspector` tab. Select the Unlit option. In the `Unlit` section, select `Texture`. This removes the need for a separate light source. Without this setting, the video will look extremely dim.

![select_unlit_shader](\images\barracuda-posenet-tutorial\select_unlit_shader.PNG)

![select_unlit_texture_shader](\images\barracuda-posenet-tutorial\select_unlit_texture_shader.PNG)

## Camera Setup

Before playing the video, we need to reposition and resize the `Main Camera` object. 

### Set Camera Position

With the `Main Camera` object selected in the `Hierarchy` tab, set the values for the `Position` parameter to same values as the `VideoScreen` object. 

Next, we need to adjust the `Z` value for the `Position` parameter for the `Main Camera` object. Set it to the opposite of the `X` value.

![set_main_camera_position](\images\barracuda-posenet-tutorial\set_main_camera_position_new.png)

### Resize the Camera

Finally, we need to adjust the Size parameter to 540 in the `Inspector` tab.

![set_camera_size_new](\images\barracuda-posenet-tutorial\set_camera_size_new.png)

## Test the Video Player

Now we can finally click the play button and watch the video.

![play_button](\images\barracuda-posenet-tutorial\play_button.png)

### Result

![barracuda_posenet_tutorial_420p](\images\barracuda-posenet-tutorial\barracuda_posenet_tutorial_420p.gif)

## Next: [Part 2](https://christianjmills.com/unity/tutorial/2020/11/04/Barracuda-PoseNet-Tutorial-2.html)