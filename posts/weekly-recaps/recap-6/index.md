---
categories:
- recap
date: 2021-3-22
description: A summary of what I've been working on for the past week.
hide: false
search_exclude: false
title: Weekly Recap

aliases:
- /Weekly-Recap-6/
---

* [Introduction](#introduction)
* [End-to-End Style Transfer Tutorial Addendum](#end-to-end-style-transfer-tutorial)
* [Crop Image on GPU](#crop-image-on-gpu)
* [Flip Image with Compute Shaders](#flip-image-with-compute-shaders)
* [Links of the Week](#links-of-the-week)



## Introduction

Well I haven't got around to trying to make daily recaps. I actually didn't even remember that I was thinking about that until I start writing this post. I made a note on my white board so I don't forget tomorrow. On the plus side I managed to complete three tutorial posts over the weekend which I think is a record for me.

## [End-to-End Style Transfer Tutorial Addendum](../../end-to-end-in-game-style-transfer-tutorial/addendum/)

I completed the follow up post I mentioned in the last recap post covering how to use the [video style transfer model](https://github.com/OndrejTexler/Few-Shot-Patch-Based-Training) that I had been experimenting instead of the model used in the main tutorial. I don't recommend using the video model over the one from the tutorial but the instructions are there now for anyone interested.

## [Crop Image on GPU](../../crop-images-on-gpu-tutorial/)

I had worked out how to efficiently crop images in Unity with a GPU while working on my [PoseNet tutorial](../../barracuda-posenet-tutorial/part-1). I didn't end up using it as I decided to just squish the camera input into a square instead. However, this might not always be ideal so I decided to document how in case I need it in the future.

## [Flip Image with Compute Shaders](../../flip-image-compute-shader-tutorial/)

I also worked out how to flip images using Compute Shaders while working on my PoseNet tutorial. Again, I ended up not using it in the tutorial. I realized after the fact that I could just flip the output of the model instead. This is much less work than flipping the whole image. 

I had actually started to write a tutorial for this a while ago but ended up scrapping it. For some reason, Google still picked up the page and someone apparently tried to view it recently. I felt bad that there was nothing but a 404 page waiting for them so I ended up making this over weekend as well.



## Links of the Week

### Machine Learning

#### [Keypoint regression with heatmaps in fastai v2](https://elte.me/2021-03-10-keypoint-regression-fastai)

This post covers how to implement heatmap regression to perform human pose estimation with the fastai library.

#### [Towards the end of deep learning and the beginning of AGI](https://towardsdatascience.com/towards-the-end-of-deep-learning-and-the-beginning-of-agi-d214d222c4cb)

An interesting post exploring how recent neuroscience research may point the way towards defeating adversarial examples and achieving a more resilient, consistent and flexible form of artificial intelligence.

#### [A Downloadable Version of Google's C4 Dataset](https://github.com/allenai/allennlp/discussions/5056)

A colossal, cleaned version of Common Crawl's web crawl corpus.

#### [“Adam” and friends](https://amaarora.github.io/2021/03/13/optimizers.html)

This blog post covers how to re-implement Stochastic Gradient Descent, Momentum, RMSprop, and the Adam optimizer from scratch.

#### [Self Supervised Learning with Fastai](https://github.com/KeremTurgutlu/self_supervised)

This GitHub repository contains implementations of popular SOTA self-supervised learning algorithms as Fastai Callbacks.

### Unity

#### [Learn to Write Unity Compute Shaders](https://www.udemy.com/course/compute-shaders/)

This Udemy course teaches you how to write Unity Compute Shaders to create particle effects, flocking, fluid simulations, post processing image filters, and create a Physics engine.

### Blender

#### [VirtuCamera App](https://apps.apple.com/us/app/virtucamera-unlimited/id1461676842)

This app lets you use your iPhone/iPad to control the virtual camera in Blender, Autodesk, and Maya like an actual camera. You can explore the scene in real-time and record camera motions.

#### [Blender FREE Virtual Camera Setup on Android - Tutorial](https://www.blendernation.com/2021/03/17/blender-free-virtual-camera-setup-on-android-tutorial/)

A free setup for an Android-based virtual camera.

### Astronomy

#### [Milky Way, 12 years, 1250 hours of exposures and 125 x 22 degrees of sky](https://astroanarchy.blogspot.com/2021/03/gigapixel-mosaic-of-milky-way-1250.html?m=1)

An incredible gigapixel mosaic image of the Milky Way.







<!-- Cloudflare Web Analytics --><script defer src='https://static.cloudflareinsights.com/beacon.min.js' data-cf-beacon='{"token": "56b8d2f624604c4891327b3c0d9f6703"}'></script><!-- End Cloudflare Web Analytics -->