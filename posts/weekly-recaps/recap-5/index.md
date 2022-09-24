---
categories:
- recap
date: 2021-3-12
description: A summary of what I've been working on for the past few weeks.
hide: false
layout: post
search_exclude: false
title: Weekly Recap
toc: false

aliases:
- /Weekly-Recap-5/
---

* [Introduction](#introduction)
* [End-to-End Style Transfer Tutorial](#end-to-end-style-transfer-tutorial)
* [Next Project](#Next Project)
* [Links of the Last Few Weeks](#links-of-the-last-few-weeks)



## Introduction

Wow it is really easy to get off schedule with these weekly recaps. I let over three weeks go by without posting one. Maybe I should try doing daily recaps just so I don't fall out of the routine as easily. It's too tempting to put off these weekly posts when I'm in the middle of an actual project. 

## End-to-End Style Transfer Tutorial

I finally completed the end-to-end style transfer tutorial. I'm a bit irritated that I let myself dump so much time into experimenting with different style transfer models. I think it would have been better to invest that time explaining how others can conduct their own experiments. I might make a post examining how I could have better approached the project and avoided going so far over my time budget. Perhaps it would have helped to set an actual time budget.

I'm in the process of making a follow up post explaining how to use the [video style transfer model](https://github.com/OndrejTexler/Few-Shot-Patch-Based-Training) instead of the model used in the tutorial. I don't plan on trying to make a super optimized version of the model like in the tutorial. My previous experiments didn't yield great results when significantly reducing the size of the model. However, I dumped so much time into it that I figure I should at least make a post showing how someone else can tinker with it. I plan to have that finished next week.

## Next Project

Before moving on to a completely new project, I plan to make some updates and additions to the PoseNet tutorial series. I realized that there were some unnecessary steps that could be removed for a small performance gain. I also realized I completely forgot to explain how to use the more efficient MobileNet version of the model instead of the ResNet50 version. I didn't use the MobileNet version in the tutorial because it's less accurate. However, a reader expressed interest in performing inference with the C# Burst backend. The smaller model would be a much better choice in that instance.

After I update the PoseNet tutorial, I plan to work on getting a facial pose estimation model working with the Barracuda library. Hopefully, that will take less time than the style transfer tutorial. If the model proves incompatible with the current versions of Barracuda, I'll look into making a C++ plugin so I can use the PyTorch C++ frontend instead.



## Links of the Last Few Weeks

### Deep Learning

#### [PyTorch 1.8](https://pytorch.org/blog/pytorch-1.8-released/)

The latest version of PyTorch has been released and comes with a lot of new updates including beta support for AMD GPUs.

#### [PyTorch3D: A library for deep learning with 3D data](https://pytorch3d.org/)

![](https://raw.githubusercontent.com/facebookresearch/pytorch3d/master/.github/dolphin_deform.gif)

Last year, the FacebookAI research team introduced a deep learning library that aims to make working with 3D data a lot easier. I didn't learn about it until I decided to click on a video YouTube had been recommending for weeks. The library supports batching of inputs with different sizes. This is helpful since cropping a 3D mesh isn't as straight forward as cropping 2D images. The library also supports several common operations for 3D data as well as a differentiable rendering API. My mind instantly jumped to wondering what it would take to integrate this with Blender. I'll plan to make time soon so that I can work through the [available tutorials](https://github.com/facebookresearch/pytorch3d#tutorials).

#### [Neural Body: Implicit Neural Representations with Structured Latent Codes for Novel View Synthesis of Dynamic Humans](https://github.com/zju3dv/neuralbody)

![](https://camo.githubusercontent.com/b1b6429cb394905284abe3b365e6bec6233592e9fdfb46df61660b65b7f0a6b3/68747470733a2f2f7a6a753364762e6769746875622e696f2f6e657572616c626f64792f696d616765732f6d6f6e6f63756c61722e676966)

#### [Training models on Multiple GPUs using fastai](https://jarvislabs.ai/blogs/multiGPUs)

This blog post explores different approaches to training models using multiple GPUs.

#### [Introducing Noisy Imagenette](https://tmabraham.github.io/blog/noisy_imagenette)

This is a new version of the [Imagenette](https://github.com/fastai/imagenette) library with noisy labels.

#### [Taming Transformers for High-Resolution Image Synthesis](https://github.com/CompVis/taming-transformers)

This is an impressive new model that can be trained to perform a wide variety of image generation tasks. Check out the related [Two Minute Papers video](https://www.youtube.com/watch?v=o7dqGcLDf0A) to see what it can do.

#### [StyleGAN Components](https://isaac-flath.github.io/fastblog/computer%20vision/gan/2021/03/01/StyleGanComponents.html)

This blog post covers the key components of a StyleGAN model and uses them to build a basic version of the model.

#### [Multimodal Neurons in Artificial Neural Networks](https://openai.com/blog/multimodal-neurons/)

A recent post by OpenAI that explores how their Contrastive Language-Image Pre-Training (CLIP) model responds to the same concept whether presented literally, symbolically, or conceptually.



### Programming

#### [All of the python 3.9 standard library](https://gist.github.com/jph00/d5981f649a83a754946964cf22322cb2)

Organized and hyperlinked index to every module, function, and class in the Python standard library  

#### [Dev Simulator: A Coding RPG](https://simulator.dev/)

This is an upcoming RPG where players build a real full stack web app while playing through the storyline with 8-bit co-workers.

#### [Git scraping](https://simonwillison.net/2021/Mar/5/git-scraping/)

This five minute video demonstrates how to schedule web scrapers using GitHub Actions.

#### [Manim](https://github.com/ManimCommunity/manim)

An animation engine for creating precise animations in Python. This is a fork of the animation engine used in the videos on the 3Blue1Brown YouTube channel.



### Unreal Engine

#### [Building Simulation Applications with Unreal Engine](https://www.twitch.tv/videos/936853835)

#### [How to use Nvidia DLSS in Unreal Engine](https://www.twitch.tv/videos/945514664)

