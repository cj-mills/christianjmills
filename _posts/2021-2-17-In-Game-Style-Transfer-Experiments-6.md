---
title: In-Game Style Transfer Experiments Pt.6
layout: post
toc: false
comments: true
description: Testing out an arbitrary style transfer model and a change in plans for the end-to-end style transfer tutorial.
categories: [style_transfer, log]
hide: True
permalink: /:title/
search_exclude: false
---

* [Introduction](#introduction)
* [Arbitrary Style Transfer](#arbitrary-style-transfer)
* [Change in Plans](#change-in-plans)
* [Conclusion](#conclusion)

## Introduction

Every time I think I'm ready to wrap up my style transfer project, I come across something that causes me to spend another week experimenting and troubleshooting. That's poor project management on my part and I've already spent more time on this topic than I intended. Therefore, I'm cutting myself off here and moving on to other topics. Unfortunately, that does mean making some changes to the end-to-end style transfer tutorial.

## Arbitrary Style Transfer

As mentioned in the [weekly recap](https://christianjmills.com/Weekly-Recap-3/), I came across an [implementation](https://colab.research.google.com/github/tensorflow/hub/blob/master/examples/colab/tf2_arbitrary_image_stylization.ipynb) of the same style transfer model that Unity used for its style transfer project. However, this one hadn't been watered down to run on a PS4 Pro. The results were impressive enough that I was tempted to use it for the end-to-end tutorial. Unfortunately, the training process is too long to use in the tutorial. Here are a couple examples using style images from Unity's sample project.

![arbitrary-style-transfer-3](G:\Projects\GitHub\christianjmills\images\in-game-style-transfer-experiments\part-6\arbitrary-style-transfer-3.jpg)

This method uses two deep learning models. One is based the same `fast_neural_style` model that I used in my first style transfer tutorial. It takes in an image and learns to generate a stylized output based on a target style image. The second model learns to predict values for part of the stylizing model that would produce a stylized image based on a target style. The regular `fast_neural_style` model would normally learn these values for a specific style image during training.

The models are trained on a large dataset of style images and content images to help them generalize. After training, the predictor model takes in an arbitrary style image and outputs the required values for the stylizing model. The stylizing model takes in those values along with a content image and outputs a stylized version. 

This method also has an added benefit of allowing the user to adjust the influence of the style image after training. The model in Unity's sample project seems to have the style influence set pretty low. I believe this is to minimize the flickering that is present when using the regular `fast_neural_style` model. As I've discovered, it's difficult to leverage the existing methods to maintain consistency between frames and still get playable framerates.

## Change in Plans

After completing the Google Colab notebook for the end-to-end style transfer tutorial, I began testing the code on a variety of styles as well as sample data from Unity. The goal was to identify areas where the models struggled. The good news is I successfully found areas where the models struggled. The bad news is that I probably won't be able to find a solution for those weaknesses in a reasonable amount of time.

The [video stylization model](https://github.com/OndrejTexler/Few-Shot-Patch-Based-Training) that I've been working with does a great job maintaining consistency between frames as long as the scene doesn't change too drastically. However, 3D video game environments tend to be dynamic. I've found that the video stylization model doesn't maintain frame consistency in when moving around in a 3D environment. 

The GitHub project does provide [methods](https://github.com/OndrejTexler/Few-Shot-Patch-Based-Training#temporal-consistency-optional) for improving temporal consistency. However, I don't know if these methods would be practical for stylizing a video game. It's also unclear if the model would still be able to generalize to different scenes with these methods. 

Since I'm not allowing myself to spend any more time experimenting I've decided to not use the video stylization model for the tutorial. Instead I'll be using a modified version of the `fast_neural_style` model. There will be more flickering than I'd like, but the training process will be much simpler.

## Conclusion

I'm disappointed that I wasn't able to resolve the flickering that comes with using a high style influence. Using a lower style influence feels like it defeats the whole point of style transfer. I'll probably come back to style transfer at some point.  For now it's time to move on to other projects.