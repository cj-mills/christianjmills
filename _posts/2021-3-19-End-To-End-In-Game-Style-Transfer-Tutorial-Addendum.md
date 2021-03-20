---
title: End-to-End In-Game Style Transfer Tutorial Addendum
layout: post
toc: false
comments: true
description: This post covers how to use a different style transfer model that is specialized for video.
categories: [style_transfer, pytorch, unity, tutorial]
hide: true
permalink: /:title/
search_exclude: false
---

### Previous: [Part 3](https://christianjmills.com/End-To-End-In-Game-Style-Transfer-Tutorial-3/)

* [Introduction](#introduction)
* [Train the Model](#train-the-model)
* [Modify the Unity Project](#modify-the-unity-project)
* [Test it Out](#test-it-out)
* [Conclusion](#conclusion)

## Introduction

When I was first planning out this tutorial series, I had intended to use a [different style transfer model](https://github.com/OndrejTexler/Few-Shot-Patch-Based-Training). This other model is designed to reduce flickering to better work with video. Unfortunately, I found I was unable to reliably get good results with that model in a Unity environment. I figured I might as well make a tutorial for anyone that wants to mess around that model since I already dumped so much time into it.

**Important:** This post assumes that you have already gone through the previous parts of this tutorial series. You can go to the start of this series by clicking the link below.

* [End-to-End In-Game Style Transfer Tutorial Pt.1](https://christianjmills.com/End-To-End-In-Game-Style-Transfer-Tutorial-1/)

Training this new model will require sample images from the target video game so checkout [Part 1.5](https://christianjmills.com/End-To-End-In-Game-Style-Transfer-Tutorial-1-5/) of this tutorial series if you haven't already. It shows how to use the Unity Recorder tool to capture in-game footage. We'll split the video into images to generate our training data.

We'll be using a modified version of the Google Colab notebook from [Part 2](https://christianjmills.com/End-To-End-In-Game-Style-Transfer-Tutorial-2/) of this series. The new model requires some examples of stylized images for training. We'll use the style transfer model from earlier in this series to generate the training samples. You can either use the `final.pth` checkpoint you made during [Part 2](https://christianjmills.com/End-To-End-In-Game-Style-Transfer-Tutorial-2/) or train a new one in the Colab notebook for this post.



## Train the Model

First, you need to get your own copy of the Colab Notebook. Open the notebook using the link below and save your own copy just like in [Part 2](https://christianjmills.com/End-To-End-In-Game-Style-Transfer-Tutorial-2/#open-google-colab-notebook).

* [Notebook Link](https://colab.research.google.com/drive/1511cxTph5bdfL9KLjn9AbQa0YI9IoPr5?usp=sharing)

### Continue in the Notebook

Follow the directions in the notebook to train your own video style transfer model. Return to this post once you have exported the trained model.



## Modify the Unity Project

Now we just need to updated our Unity project to support the new model. Unlike the training process, this part is pretty much identical to the other style transfer model.

### Update the `ComputeShader`

The only thing we need to add to the Unity project are some new image processing functions in the `ComputeShader` we made in [Part 3](https://christianjmills.com/End-To-End-In-Game-Style-Transfer-Tutorial-3/#create-compute-shader). This time, we need to remap the RGB values from `[0,1]` to `[-1,1]` instead of `[0,255]`. You can either swap out the code for the existing processing functions or make new ones like in the image below.

![unity-updated-compute-shader](..\images\end-to-end-in-game-style-transfer-tutorial\unity-updated-compute-shader.png)

If you make new functions, be sure to replace the function names in the `StyleTransfer.cs` script.

![unity-update-styleTransfer-script](..\images\end-to-end-in-game-style-transfer-tutorial\unity-update-styleTransfer-script.png)



### Replace the `modelAsset`

Download the ONNX file from your Google Drive just like in [Part 3](https://christianjmills.com/End-To-End-In-Game-Style-Transfer-Tutorial-3/#download-onnx-files) and drop it into your `Models` folder. You can also download the model I'll be using from the link below.

* [Mosaic Style Transfer Model](https://drive.google.com/file/d/1Q3RQSri3RURgfpjK2WEcQr8EBrkeoK9n/view?usp=sharing)

Now we just need to assign the ONNX file to the `modelAsset` variable in the `Inspector` tab. This model is much less efficient than the one use earlier in this series so we'll lower the `targetHeight` as well.

![unity-inspector-add-video-final](..\images\end-to-end-in-game-style-transfer-tutorial\unity-inspector-add-video-final.png)



## Test it Out

You'll probably notice that the in-game output for this model it less detailed than for the other model. It also doesn't completely get rid of the flickering. These are some of the reasons I ended up not using this model for the main tutorial.

 

## Conclusion

This model works great for videos where the scene doesn't change too drastically. Unfortunately, it isn't well suited for a dynamic, real-time, video game environment. The default model is way too large to get playable frame rates and it quickly loses detail when its size is reduced.