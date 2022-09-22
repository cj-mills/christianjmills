---
title: How to Flip an Image With a Compute Shader
date: 2021-3-21
image: /images/empty.gif
title-block-categories: false
layout: post
toc: false
hide: false
search_exclude: false
description: This post covers how to flip an image with a compute shader in Unity.
categories: [unity, tutorial]

---

* [Introduction](#introduction)
* [Create a 2D Unity Project](#create-a-2d-unity-project)
* [Create Compute Shader](#create-compute-shader)
* [Create `Flip` Script](#create-flip-script)
* [Create Screen GameObject](#create-screen-gameobject)
* [Create ImageFlipper](#create-imagecropper)
* [Test it Out](#test-it-out)
* [Conclusion](#conclusion)



## Introduction

In this post, we'll cover how to use a [compute shader](https://docs.unity3d.com/Manual/class-ComputeShader.html) to flip an image across the x-axis, y-axis, and diagonal axis. We will also demonstrate how these operations can be combined to rotate an image.



## Create a 2D Unity Project

Open the Unity Hub and create a new 2D project. I'm using `Unity 2019.4.20f1`, but you should be fine using other versions.

![unity-hub-create-project](./images/unity-hub-create-project.png)



## Create Compute Shader

In Unity, right-click an empty space in the Assets folder and open the `Create` submenu. Select `ComputeShader` from the `Shader` submenu and name it `FlipShader`.

![unity-create-compute-shader](./images/unity-create-compute-shader.png)

Open the new compute shader in your code editor. By default, compute shaders contain the following code.

![flipshader-default-code](./images/flipshader-default-code.png)

We'll delete the `CSMain` function and create a new one for each of our three flip operations. 

### Define Variables

Before we create our functions, we need to define some extra variables.

* `Texture2D<float4> InputImage`: stores the original image
* `int height`: the height of the input image
* `int width`: the width of the input image
* `int2 coords`: stores the new `(x,y)` coordinates for individual pixel values

![flipshader-define-variables](./images/flipshader-define-variables.png)

### Define Flip Functions

The individual flip operations quite simple. They determine the coordinates of the pixel that will replace the values for a given pixel in the image. The RGB pixel values at the calculated coordinates will be stored at the current coordinates in the `Result` variable.

* `Flip x-axis`: subtract the y value for the current pixel's `(x,y)` coordinates from the height of the image
* `Flip y-axis`: subtract the x value for the current pixel's `(x,y)` coordinates from the width of the image
* `Flip diagonal`: swap the x and y values for the current pixel's `(x,y)` coordinates

These operations are performed on each pixel in parallel on the GPU. We'll use the default `numthreads(8, 8, 1)` for each function.

![flip-shader-define-functions](./images/flip-shader-define-functions-2.png)







## Create `Flip` Script

Back in Unity, right-click an empty space in the Assets folder and select `C# Script` in the `Create` submenu. Name the new script, `Flip` and open it in your code editor.

![unity-create-flip-script](./images/unity-create-flip-script.png)



### Define Variables

We'll define the following variables at the top of the script.

* `public ComputeShader computeShader`: The compute shader that contains the flip operations
* `public GameObject screen`: The screen to which the test image is attached
* `public bool flipXAxis`: Toggle whether to flip the image across the x-axis
* `public bool flipYAxis`: Toggle whether to flip the image across the y-axis
* `public bool flipDiag`: Toggle whether to flip the image across the diagonal axis
* `private GameObject mainCamera`: Stores a reference to the Main Camera object
* `private RenderTexture image`: A copy of the original test image

![flip-script-define-variables](./images/flip-script-define-variables.png)



### Define `Start()` Method

In the `Start()` method, we'll store a copy the original test image in the `image` `RenderTexture`. We can do so by getting a reference to the `Texture` attached to the `screen` and using the [`Graphics.Blit()`](https://docs.unity3d.com/ScriptReference/Graphics.Blit.html) method. We'll also get a reference to the camera so that we can adjust the view to fit the current image. 

![flip-script-start-method](./images/flip-script-start-method.png)



### Define `FlipImage()` Method

Next, we'll define a new method called `FlipImage` to handle executing the compute shader. This method will take in the image to be flipped, an empty `RenderTexture` to store the flipped image, and the name of the function to execute on the compute shader.

To execute the compute shader, we need to first get the kernel index for the specified function and initialize the variables we defined in the compute shader. Once we execute the compute shader using the `computeShader.Dispatch()` method, we can copy the result to the empty `RenderTexture` we passed in. We could copy the result directly to the `RenderTexture` containing the original image. However, this would cause an error when flipping non-square images across the diagonal axis. This is because a `RenderTexture` can not dynamically change dimensions.

![flip-script-flipImage-method](./images/flip-script-flipImage-method.png)



### Define `Update()` Method

First, we need to make another copy of the original image so that we can edit it. We'll store this copy in a [temporary](https://docs.unity3d.com/ScriptReference/RenderTexture.GetTemporary.html) `RenderTexture` called `rTex` that will get released at the end of the method.

The steps are basically the same for performing each of the three flip operations. We first allocate a temporary `RenderTexture` called `tempTex` to store the flipped image. We then call the `FlipImage` method with the appropriate function name. Next, we copy the flipped image to `rTex`. Finally, we release the resources allocated for `tempTex`. The steps for flipping the image across the diagonal axis is slightly different as we can't directly copy a flipped image with different dimensions back to `rTex`. Instead, we have to directly assign the currently active `RenderTexture` to `rTex`.

After we copy `tempTex` back to `rTex` we'll update the `Texture` for the `screen` with the flipped image and adjust the shape of the screen to fit the new dimensions.

![flip-script-update-method](./images/flip-script-update-method.png)



## Create Screen GameObject

Back in Unity, right-click an empty space in the `Hierarchy` tab and select `Quad` from the `3D Object` submenu. Name the new object `Screen`. The size will be updated automatically by the `Flip.cs` script.

![unity-create-screen-object](./images/unity-create-screen-object.png)

## Create ImageFlipper

Right-click an empty space in the `Hierarchy` tab and select `Create Empty` from the pop-up menu. Name the empty object `ImageFlipper`

![unity-create-image-cropper-object](./images/unity-create-image-cropper-object.png)

With the `ImageFlipper` selected, drag and drop the `Flip.cs` script into the `Inspector` tab.

![unity-inspector-attach-flip-script](./images/unity-inspector-attach-flip-script.png)

Drag and drop the `Screen` object from the `Hierarchy` tab as well as the `FlipShader` from the `Assets` folder onto their respective spots in the `Inspector` tab.

![unity-inspector-assign-parameters](./images/unity-inspector-assign-parameters.png)



## Test it Out

We'll need a test image to try out the `ImageFlipper`. You can use your own or download the one I used for this tutorial.

* [Test Image](https://drive.google.com/file/d/18_e6CpvsZcGuym66bGXsioAPEB0We8zV/view?usp=sharing)

 Drag and drop the test image into the `Assets` folder. Then drag it onto the `Screen` in the `Scene`. 

![unity-import-image](./images/unity-import-image.png)





Next, we need to set our Screen to use an `Unlit` shader. Otherwise it will be a bit dim. With the Screen object selected, open the `Shader` drop-down menu in the `Inspector` tab and select `Unlit`. 

![unity-inspector-tab-shader-drop-down](./images/unity-inspector-tab-shader-drop-down.png)



Select `Texture` from the `Unlit` submenu.

![unity-inspector-tab-unlit-texture](./images/unity-inspector-tab-unlit-texture.png)



Now we can click the Play button and toggle the different flip checkboxes to confirm our script is working properly. If you check the performance stats, you should see that there is a negligible performance hit from flipping the image even when performing all three operations at once.

### Default Image

![default-image](./images/default-image.png)

### Flip X-Axis

![flip-x-axis](./images/flip-x-axis.png)

### Flip Y-Axis

![flip-y-axis](./images/flip-y-axis.png)



### Flip Diagonal Axis

![flip-diagonal-axis](./images/flip-diagonal-axis.png)



### Flip X-Axis and Y-Axis

![flip-x-axis-and-y-axis](./images/flip-x-axis-and-y-axis.png)

### Flip X-Axis and Diagonal Axis

![flip-x-axis-and-diagonal-axis](./images/flip-x-axis-and-diagonal-axis.png)

### Flip Y-Axis and Diagonal Axis

![flip-y-axis-and-diagonal-axis](./images/flip-y-axis-and-diagonal-axis.png)

### Flip X-Axis, Y-Axis and Diagonal Axis

![flip-x-axis-y-axis-and-diagonal-axis](./images/flip-x-axis-y-axis-and-diagonal-axis.png)



## Conclusion

That is one approach to efficiently flip images on the GPU in Unity. As demonstrated above, the operations can be combined in different ways to rotate the image as well.



**Project Resources:** [GitHub Repository](https://github.com/cj-mills/Flip-Image-Compute-Shader)