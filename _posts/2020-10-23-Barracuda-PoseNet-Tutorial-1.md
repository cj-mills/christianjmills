---
title: Barracuda PoseNet Tutorial Pt.1
layout: post
toc: true
description: This is part 1 of a step-by-step-guide for running a PoseNet model with Unity's Barracuda library.
categories: []
hide: true
search_exclude: false
---

## Prerequisites

This tutorial assumes that you have used Unity before. If you've never used Unity, you can get your feet wet with this [tutorial](https://www.youtube.com/watch?v=OR0e-1UBEOU&list=PLB5_EOMkLx_VHKn4IISeNwhlDrb1948ZX&index=3).



We'll be using Unity 2019.4.13 ([download](unityhub://2019.4.13f1/518737b1de84)).



## Create a New Project

First, we need to create a new Unity project. Since the PoseNet model only estimates 2D poses, we'll select the 2D template.

![create_project](\images\barracuda-posenet-tutorial\create_project.PNG)



## Setup Video Player

Before getting into any code, we'll setup a video player. We'll cover how to use a webcam as well, but a prerecorded video is better for testing purposes.

In the `Hierarchy` tab, right click an empty area, select the `Video` section, and click `Video Player`. This will create a new `GameObject` called `Video Player`. The default name works well enough so we'll leave it as is.

![create_video_player](\images\barracuda-posenet-tutorial\create_video_player.PNG)



## Get Test Videos

https://www.pexels.com/videos/

For this tutorial we'll be using these two videos available on [Pexels](https://www.pexels.com/), a free stock photos & videos site.



[Woman Dancing](https://www.pexels.com/video/woman-dancing-2873755/)



[Boardslides](https://www.pexels.com/video/two-young-men-doing-a-boardslide-over-a-railing-4824358/)



In the `Assets` window, right-click an empty space, select the `Create` option, and click `Folder`. Name the folder `Videos`.

![create_folder](\images\barracuda-posenet-tutorial\create_folder.PNG)

Double-click the `Videos` folder to open it.

Drag and drop the two video files into the `Videos` folder.

![video_file_assets](\images\barracuda-posenet-tutorial\video_file_assets.PNG)



Select the `Video Player` object in the `Hierarchy` tab. Then click and drag the `pexels_boardslides` file into the `Video Clip` value in the `Inspector` tab.

![video_clip_empty](\images\barracuda-posenet-tutorial\video_clip_empty.png)

![video_clip_filled](\images\barracuda-posenet-tutorial\video_clip_filled.png)

Tick the `Loop` checkbox in the `Inspector` tab to make the video repeat when the project is running.

![loop_video_checkbox](\images\barracuda-posenet-tutorial\loop_video_checkbox.png)





Create a new folder in the `Assets` window and name it `Textures`.

![textures_folder](\images\barracuda-posenet-tutorial\textures_folder.PNG)

Open the folder and right click an empty space. In the `Create` section, click `Render Texture`. You can name it something like `video_texture`.

![create_render_texture](\images\barracuda-posenet-tutorial\create_render_texture.PNG)

With the `video_texture` object selected, the adjust values for `Size` parameter in the `Inspector` tab. We'll set the size parameter to the resolution of the videos. In our case, the resolution is 1920 x 1080.

![rt_size_parameter](\images\barracuda-posenet-tutorial\rt_size_parameter.png)

![rt_set_resolution](\images\barracuda-posenet-tutorial\rt_set_resolution.png)

With the resolution set, select the `Video Player` object in the `Hierarchy` tab again. Click and drag the `video_texture` object into the `Target Texture` parameter option in the `Inspector` tab.

![target_texture_empty](\images\barracuda-posenet-tutorial\target_texture_empty.png)



![target_texture_filled](\images\barracuda-posenet-tutorial\target_texture_filled.png)



Now, we need to create a screen to watch the video play. We'll use a `Quad` object for the screen. Right click an empty space in the `Hierarchy` tab, select the `3D Object` section and click `Quad`. We can just name it `VideoScreen`.

![create_quad](\images\barracuda-posenet-tutorial\create_quad.PNG)



With the `VideoScreen` object selected, we need to adjust the `Scale` parameter in the `Inspector` tab. Set the `X` value to 1920 and the `Y` value to 1080. Leave the `Z` value at 1.

![quad_scale_default](\images\barracuda-posenet-tutorial\quad_scale_default.png)

![quad_scale_set](\images\barracuda-posenet-tutorial\quad_scale_set.png)



Next, we need to adjust the position of the `VideoScreen` object so that the bottom left corner is at `X: 0, Y: 0, Z: 0`. This will make things easier when handling the output from the PoseNet model. To do this we'll set update the `Position` value in the `Inspector` tab. Set the `X` value to half the `X` value for the `Scale` parameter. Do the same for the `Y` value. The `X` value for `Position` should be 960 and the `Y` value should be set to 540.

![quad_position_default](\images\barracuda-posenet-tutorial\quad_position_default.png)

![quad_position_set](\images\barracuda-posenet-tutorial\quad_position_set.png)



With the parameters for the `VideoScreen` object set, we need to zoom out and re-center our perspective. We can easily do this by selecting the `VideoScreen` object and pressing the `F` key on our keyboard.

![recentered_video_screen](\images\barracuda-posenet-tutorial\recentered_video_screen.PNG)

If you want to zoom back in a bit, you can scroll up with you mouse wheel.



Click and drag the `video_texture` object from the `Textures` folder onto the `VideoScreen` object in the `Scene` tab. The `VideoScreen` object should turn completely black.



With the `VideoScreen` object selected, click the `Shader` dropdown in the `Inspector` tab. Select the Unlit option. In the `Unlit` section, select `Texture`. Choosing this setting means that out `VideoScreen` object does not require a separate light source. Without this setting, the video will look extremely dim.

![select_unlit_shader](\images\barracuda-posenet-tutorial\select_unlit_shader.PNG)

![select_unlit_texture_shader](\images\barracuda-posenet-tutorial\select_unlit_texture_shader.PNG)

Before playing the video we need to reorient the `Main Camera` object. With the `Main Camera` object selected in the `Hierarchy` tab, set the values for the `Position` parameter to same values as the `VideoScreen` object. 

Next, we need to adjust the `Z` value for the `Position` parameter for the `Main Camera` object. Set it to the opposite of the `X` value.

![set_main_camera_position_new](\images\barracuda-posenet-tutorial\set_main_camera_position_new.png)



Finally, we need to adjust the Size parameter to 540 in the `Inspector` tab.

![set_camera_size_new](\images\barracuda-posenet-tutorial\set_camera_size_new.png)

Now we can finally click the play button and watch the video.

![play_button](\images\barracuda-posenet-tutorial\play_button.png)



You should see something like this.

![barracuda-posenet-tutorial_2](\images\barracuda-posenet-tutorial\barracuda-posenet-tutorial_420p.gif)