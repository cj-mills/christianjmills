---
title: How to Crop Images on the GPU in Unity
layout: post
toc: false
comments: true
description: This post covers how to efficiently crop images in Unity with a GPU.
categories: [unity, tutorial]
hide: true
permalink: /:title/
search_exclude: false
---

* [Introduction](#introduction)
* [Train the Model](#train-the-model)
* [Modify the Unity Project](#modify-the-unity-project)
* [Test it Out](#test-it-out)
* [Conclusion](#conclusion)



## Introduction

In this post, we'll cover how to create a square crop of an image in Unity. The approach used in this tutorial can be adapted to crop other sections of an image as well.



## Create a 2D Unity Project

Open the Unity Hub and create a new 2D project. I'm using `Unity 2019.4.20f1`, but you should be fine using other versions.

![unity-hub-create-new-project](..\images\crop-images-on-gpu-tutorial.png\unity-hub-create-new-project.png)





## Create `Crop.cs` Script

In Unity, right-click an empty space in the Assets folder and select `C# Script` in the `Create` submenu. Name the new script, `Crop` and open it in your code editor.

![unity-create-flip-script](..\images\crop-images-on-gpu-tutorial.png\unity-create-flip-script.png)



### Define Variables

Create a public `GameObject` called `screen`. We'll be using this screen to confirm our script is correctly cropping the test images. Add a public `bool` variable called `cropImage` as well. This will let us toggle whether to crop the image during runtime. Lastly, we'll create a private `RenderTexture` called `image` to store a copy of the original  test image.

![crop-script-public-variables](..\images\crop-images-on-gpu-tutorial.png\crop-script-define-variables.png)



### Define `Start()` Method

In the `Start()` method, we'll store a copy the original test image in the `image` `RenderTexture`. We can do so by getting a reference to the `Texture` attached to the `screen` and using the `Graphics.Blit()` method. We'll also adjust the camera so that we can see the entire image. 

![crop-script-start-method](..\images\crop-images-on-gpu-tutorial.png\crop-script-start-method.png)



### Define `Update()` Method

First, we need make another copy of the original image so that we can edit it. We'll store this copy in a temporary `RenderTexture` that will get released at the end of the method. We'll then check if `cropImage` is set to `true`.



We can use the `Graphics.CopyTexture()` method included with Unity. We need to specify several parameters in order to use this method to crop images.

1. `src`: The original image
2. `dst`: An empty square `RenderTexture`
3. `srcMip`: The mipmap level for the image `RenderTexture`, set to `0`
   * Not relevant for our use case
4. `dstElement`: The destination texture element, set to `0`
   * Not relevant to our use case
5. `srcX`: The X coordinate of the original image to start copying from
   * This will vary depending on if the tall or wide
6. `srcY`: The Y coordinate of the original image to start copying from
   * This will vary depending on if the tall or wide
7. `srcWidth`: Width of the new square image
8. `srcHeight`: Height of the new square image
9. `dstX`: The X coordinate of the new square image to start copying to
10. `dstY`: The Y coordinate of the new square image to start copying to

![crop-script-update-method](..\images\crop-images-on-gpu-tutorial.png\crop-script-update-method.png)



## Create Screen GameObject





## Create ImageCropper







## Test it Out



 

## Conclusion

