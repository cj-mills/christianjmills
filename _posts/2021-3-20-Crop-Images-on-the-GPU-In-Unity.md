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

First, we need to make another copy of the original image so that we can edit it. We'll store this copy in a temporary `RenderTexture` called `rTex` that will get released at the end of the method. 

We can't change the dimensions of a `RenderTexture` after it's been created. Instead, we'll create a cropped image by copying part of `rTex` to another temporary `RenderTexture` called `tempTex` that will be square. We can copy the square image to `rTex` after we release the current `RenderTexture` assigned to `rTex` and make a new square one.

The size of `tempTex` will depend on whether the original image is wider or taller. We want to use the smallest side of the original image. 

We'll determine what part of `rTex` we need to copy by calculating either `(image.width - image.height) / 2f` or `(image.height - image.width) / 2f` depending on whether the image is wider or taller.

We can copy part of `rTex` to `tempTex` using the `Graphics.CopyTexture()` method. We need to specify several parameters in order to use this method to crop images.

1. `src`: The original image
2. `dst`: An empty square `RenderTexture`
3. `srcMip`: The mipmap level for the image `RenderTexture`, set to `0`
   * Not relevant for our use case
4. `dstElement`: The destination texture element, set to `0`
   * Not relevant to our use case
5. `srcX`: The X coordinate of the top left corner of the center square of the original image
6. `srcY`: The Y coordinate of the top left corner of the center square of the original image
7. `srcWidth`: Width of the new square image
8. `srcHeight`: Height of the new square image
9. `dstX`: The X coordinate of the new square image to start copying to
10. `dstY`: The Y coordinate of the new square image to start copying to

After we copy `tempTex` back to `rTex` we'll update the `Texture` for the `screen` with the new square image and adjust the shape of the screen to fit the new image. 

![crop-script-update-method](..\images\crop-images-on-gpu-tutorial.png\crop-script-update-method.png)



## Create Screen GameObject

Back in Unity, right-click an empty space In the `Hierarchy` tab and select `Quad` from the `3D Object` submenu. Name the new object `Screen`. The size will be updated automatically by the `Crop.cs` script.

![unity-create-screen-object](..\images\crop-images-on-gpu-tutorial.png\unity-create-screen-object.png)

## Create ImageCropper

Right-click an empty space in the `Hierarchy` tab and select `Create Empty` from the pop-up menu. Name the empty object `ImageCropper`

![unity-create-image-cropper-object](..\images\crop-images-on-gpu-tutorial.png\unity-create-image-cropper-object.png)

With the `ImageCropper` selected drag and drop the `Crop.cs` script into the `Inspector` tab.

![unity-attach-crop-script](..\images\crop-images-on-gpu-tutorial.png\unity-attach-crop-script.png)

Drag and drop the `Screen` object from the `Hierarchy` tab onto the `Screen` parameter in the `Inspector` tab.

![unity-inspector-tab-assign-screen](..\images\crop-images-on-gpu-tutorial.png\unity-inspector-tab-assign-screen.png)



## Test it Out

We'll need some test images to try out the `ImageCropper`. You can use your own or download the ones I used for this tutorial.

* [Wide Image](https://drive.google.com/file/d/1abd1RJTu5GvyRqrRfrNjePNX7WPq8mBQ/view?usp=sharing)
* [Tall Image](https://drive.google.com/file/d/1gQZr0vlPYFbvccRSryv0Zou1mPKd5wHj/view?usp=sharing)

 Drag and drop the test images into the `Assets` folder. Select one of the images and drag it onto the `Screen` in the `Scene`. 



![unity-import-images](..\images\crop-images-on-gpu-tutorial.png\unity-import-images.png)



Next, we need to set our Screen to use an `Unlit` shader. Otherwise it will be a bit dim. With the Screen object selected, open the `Shader` drop-down in the `Inspector` tab and select `Unlit`. 



![unity-inspector-tab-shader-drop-down](..\images\crop-images-on-gpu-tutorial.png\unity-inspector-tab-shader-drop-down.png)



Select `Texture` from the `Unlit` submenu.

![unity-inspector-tab-unlit-texture](..\images\crop-images-on-gpu-tutorial.png\unity-inspector-tab-unlit-texture.png)



Now we can click the Play button and toggle the `Crop Image` checkbox to confirm our script is working properly. If you check the performance stats, you should see that there is basically no performance hit from cropping the image.

![crop_image_on_gpu_unity_1](..\images\crop-images-on-gpu-tutorial.png\crop_image_on_gpu_unity_1.gif)



![crop_image_on_gpu_unity_2](..\images\crop-images-on-gpu-tutorial.png\crop_image_on_gpu_unity_2.gif)



## Conclusion

That is one method to efficiently crop images on the GPU in Unity. As mentioned earlier, this method can be adapted to crop different parts of the image. You do so by changing the values for the `Graphics.CopyTexture()` method to adjust what part of the source image gets copied and where in the target image it gets copied to.