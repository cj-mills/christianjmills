---
title: End-to-End In-Game Style Transfer Tutorial Addendum
layout: post
toc: false
comments: true
description: This post covers how to use a different style transfer model meant for video instead of the model used in this tutorial series.
categories: [style_transfer, pytorch, unity, tutorial]
hide: true
permalink: /:title/
search_exclude: false
---

### Previous: [Part 1](https://christianjmills.com/End-To-End-In-Game-Style-Transfer-Tutorial-1/) [Part 1.5](https://christianjmills.com/End-To-End-In-Game-Style-Transfer-Tutorial-1-5/)

* [Introduction](#introduction)
* [Open Google Colab Notebook](#open-google-colab-notebook)
* [Continue in the Notebook](#continue-in-the-notebook)
* [Conclusion](#conclusion)

## Introduction

When I was first planning out this tutorial series, I had intended to use a [different style transfer model](https://github.com/OndrejTexler/Few-Shot-Patch-Based-Training). This other model was designed to generate more consistent output to better work with video. Unfortunately, as I started writing the tutorial, I found I was unable to reliably get good results with that model in a video game environment. Since I already dumped so much time into using this model, I figured I might as well make a tutorial for anyone that wants to mess around with it.

**Important:** This post assumes that you have already gone through the previous parts of this tutorial series. You can go to the start of this series by clicking the link below.

* [End-to-End In-Game Style Transfer Tutorial Pt.1](https://christianjmills.com/End-To-End-In-Game-Style-Transfer-Tutorial-1/)

Training this new model will require sample images from the target video game so checkout [Part 1.5](https://christianjmills.com/End-To-End-In-Game-Style-Transfer-Tutorial-1-5/) of this tutorial series if you haven't already. It shows how to use the Unity Recorder tool to capture in-game footage. We'll split the video into images to generate our training data.

We'll be using a modified version of the Google Colab notebook used in [Part 2](https://christianjmills.com/End-To-End-In-Game-Style-Transfer-Tutorial-2/) of this series. The new model requires some examples of stylized images for training. We'll use the style transfer model from earlier in this series to generate the training samples. You can either use one you trained during [Part 2](https://christianjmills.com/End-To-End-In-Game-Style-Transfer-Tutorial-2/) or train a new one in the Colab notebook for this post.



## Open Google Colab Notebook

First, you need to get your own copy of the Colab Notebook. Open the notebook using the link below and save your own copy just like in [Part 2](https://christianjmills.com/End-To-End-In-Game-Style-Transfer-Tutorial-2/#open-google-colab-notebook).

* [Notebook Link](https://colab.research.google.com/drive/1511cxTph5bdfL9KLjn9AbQa0YI9IoPr5?usp=sharing)

### Continue in the Notebook

Follow the directions in the notebook to train your video style transfer model. Return to this post once you have exported the trained model.



## Modify the Unity Project

Now we just need to updated our Unity project to support the new model.

### Update the `ComputeShader`

The only thing we need to add to the Unity project are some new image processing functions in the `ComputeShader` we made in [Part 3](https://christianjmills.com/End-To-End-In-Game-Style-Transfer-Tutorial-3/#create-compute-shader). This time, we need to remap the RGB values from `[0,1]` to `[-1,1]` instead of `[0,255]`. You can either swap out the code for the existing processing functions or make new ones like in the image below.

![unity-updated-compute-shader](..\images\end-to-end-in-game-style-transfer-tutorial\unity-updated-compute-shader.png)



If you make new functions, be sure to replace the function names in the `StyleTransfer.cs` script.



### Replace the `modelAsset`

Download the ONNX file from your Google Drive just like in [Part 3](https://christianjmills.com/End-To-End-In-Game-Style-Transfer-Tutorial-3/#download-onnx-files) and drop it into your `Models` folder.

Now we just need to assign the ONNX file to the `modelAsset` variable in the `Inspector` tab.

 

## Conclusion







[GitHub Repository](https://github.com/cj-mills/End-to-End-In-Game-Style-Transfer-Tutorial)