---
categories:
- recap
date: 2021-4-25
description: A summary of what I've been working on for the past week.
hide: false
layout: post
search_exclude: false
title: Weekly Recap
toc: false

---

* [Introduction](#introduction)
* [Experimenting with AsyncGPUReadback](#experimenting-with-asyncgpureadback)
* [Pixel Effects with JavaScript and HTML Canvas](#pixel-effects-with-javascript-and-html-canvas)
* [Conclusion](#conclusion)
* [Links of the Week](#links-of-the-week)



## Introduction

Well I would like to say that I have had a super productive month. Unfortunately, I have hardly been able to work on anything at all. At least I'm done with finals now so, hopefully, I'll be able to make up for it. 



## Experimenting with AsyncGPUReadback

In preparation for updating the PoseNet tutorial, I took another shot at getting around the performance bottleneck of reading the model output from the GPU. This time, I tried using the [`AsyncGPUReadback`](https://docs.unity3d.com/ScriptReference/Rendering.AsyncGPUReadback.html) class. Keijiro Takahashi who works at Unity Technologies Japan had a [small demo project](https://github.com/keijiro/AsyncCaptureTest) on GitHub that I was able to learn from. As the name suggests, it allows you to read data from the GPU without blocking the main thread. I was successfully able to read model output back from the GPU using this method by first copying it to a `RenderTexture`. There was essentially no performance cost so I was pretty excited. Unfortunately, this approach does not seem feasible at least for the PoseNet model. The Barracuda library does not seem to support reading model output to a 3-dimensional `RenderTexture`. That means the model output needs to be [sliced up](https://docs.unity3d.com/Packages/com.unity.barracuda@1.0/api/Unity.Barracuda.ModelBuilder.html#Unity_Barracuda_ModelBuilder_StridedSlice_System_String_System_Object_System_Int32___System_Int32___System_Int32___) and read separately. This approach does not scale well and the PoseNet model actually has a lot of outputs. I did go through the effort creating separate slices for each key point predicted by the model. However, there was actually a significant performance hit when reading from the GPU so many times. It would probably work fine for a model with only 2-dimensional outputs or only a few 3-dimensional ones. I'll have to try it out with a simple image classifier.



## Pixel Effects with JavaScript and HTML Canvas

I've started thinking of ways to personalize my blog a bit. That lead me to this [tutorial](https://www.youtube.com/watch?v=UoTxOVEecbI) on [freeCodeCamp](https://www.youtube.com/channel/UC8butISFwT-Wl7EV0hUK0BQ). The tutorial shows how to create pixel rain effects and interactive particle effects from scratch with vanilla JavaScript. I don't have much experience with JavaScript so it was really fun learning about some of the more artistic applications. I recommend checking out the creator's personal [YouTube channel](https://www.youtube.com/channel/UCEqc149iR-ALYkGM6TG-7vQ).



## Conclusion

Well it's been just over a month since my first "daily recap" and I have not posted any since. That first week turned out to be rather poor timing, but I don't have any excuses for the rest of the month. I'm going to try starting fresh tomorrow.



## Links of the Week

### Unity

#### [FaceMeshBarracuda](https://github.com/keijiro/FaceMeshBarracuda)

Well it looks like [Keijiro Takahashi](https://github.com/keijiro) beat me to implementing the [Facemesh](https://google.github.io/mediapipe/solutions/face_mesh) model with the Barracuda library. Looking at his recent projects on GitHub, he appears to be working on implementing every mode from the [MediaPipe](https://google.github.io/mediapipe/) library. I guess I'll have to put a bit more effort into my future tutorials on those so that they add something new. 

### Deep Learning

#### [fastdebug](https://muellerzr.github.io/fastdebug/)

A helpful library that is meant to make life easier when dealing with Pytorch and fastai errors.

### Blender

#### [HyperBole! The Hyperball Hates You](https://www.youtube.com/watch?v=AqA71cWs1WA)

A master applying their craft.