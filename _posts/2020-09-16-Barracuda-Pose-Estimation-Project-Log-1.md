---
title: Barracuda Pose Estimation Project Log Pt.1
layout: post
toc: true
description: This is the first post documenting my progress implementing pose estimation using the Barracuda inference library in Unity.
categories: [unity,project,log]
hide: true
search_exclude: false
---

## Background

I've recently started playing around with the [Barracuda](https://docs.unity3d.com/Packages/com.unity.barracuda@1.1/manual/index.html) neural network inference library for the [Unity](https://unity.com/) game engine. What's cool about Barracuda is that it runs models locally within Unity. There's no remote server or special dependencies required to use an deep learning (DL) model in your application. With a few [constraints](https://docs.unity3d.com/Packages/com.unity.barracuda@1.1/manual/SupportedPlatforms.html), it should work on any platform supported by Unity. Running the model in Unity also allows you to combine it with all the other [features](https://unity.com/products/unity-platform) of a modern real-time development platform.

**Note:** If you're interested in learning how to get started with Barracuda, I plan to make a separate post on that in the near future. I will update this post with a link when it is ready.

While Barracuda was originally built for Unity's [ML-Agents](https://unity3d.com/machine-learning) toolkit, it has been gradually adding support for more network architectures. It's still early days, and the first verified release just came out in June. However, there's enough working to start playing around with it. I'm really excited about the potential for leveraging modern machine learning capabilities in interactive applications, so I dove in. I decided to start by making a fun project to get acquainted with the library. 

Since, I've finally gotten around to making a blog, I going to attempt to document my journey as well. If I can't force myself to write about something I'm this excited about, there really is no hope for me maintaining a blog. I actually started learning about the Barracuda library a couple months before creating this blog. This post will cover my journey up to this point.

## The project

One of the first ideas that came to mind for a project was mapping a user's movements to a virtual character in real-time. Fortunately, there are existing DL models that are well suited for this task. These models perform what is called pose estimation. Pose estimation is a technique where a model predicts the location of a person or object in an image or video. When tracking humans, a model is typically trained to predict the locations of key points on a person's body (e.g. joints, nose, eyes, etc.). You can learn more about pose estimation [here](https://www.fritz.ai/pose-estimation/).

I decided to start with an existing pretrained model so I could get to a proof of concept more quickly. However, I plan to train my own pose estimation models once I get far enough along with the rest of the application.

### Project Goals

The basic goal for this project is to use the key point locations predicted by the pose estimation model to control a virtual character. For example, when the user moves their arm, the virtual character's arm should move accordingly. That may seem straightforward enough, but (as often happens) it quickly got more complicated once I actually tried to spell out what I wanted. The requirements I've determined so far are listed below. 

#### Requirements:

* Scale the estimated pose that the model outputs to account for differences in size between the target character sprite/model and the size of the image being fed into the model. 
* Handle differences in body proportions between the user and the character. 
* Separate the estimated pose from the user's location in the camera frame.



## Current Progress

