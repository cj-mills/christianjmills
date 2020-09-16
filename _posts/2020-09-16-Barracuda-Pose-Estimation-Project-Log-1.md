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

I've recently started playing around with the [Barracuda](https://docs.unity3d.com/Packages/com.unity.barracuda@1.1/manual/index.html) neural network inference library for the [Unity](https://unity.com/) game engine. What's cool about Barracuda is that it runs models locally within Unity. There's no remote server or special dependencies required to use an ML model in your application. With a few [constraints](https://docs.unity3d.com/Packages/com.unity.barracuda@1.1/manual/SupportedPlatforms.html), it should work on any platform supported by Unity. Running the model in Unity also allows you to combine it with all the other [features](https://unity.com/products/unity-platform) of a modern real-time development platform.

**Note:** If you're interested in learning how to get started with Barracuda, I plan to make a separate post on that in the near future. I will update this post with a link when it is ready.

While Barracuda was originally built for Unity's [ML-Agents](https://unity3d.com/machine-learning) toolkit, it has been gradually adding support for more network architectures. It's still early days, and the first verified release just came out in June. However, there's enough working to start playing around with it. I'm really excited about the potential for leveraging modern machine learning capabilities in interactive applications, so I dove in. I decided to start by making a fun project to get acquainted with the library. 

Since, I've finally gotten around to making a blog, I going to attempt to document my journey as well. If I can't force myself to write about something I'm this excited about, there really is no hope for me maintaining a blog. I actually started learning about the Barracuda library a couple months before creating this blog. This post will cover my journey up to this point.

## The project

One of the first things that came to mind was mapping a user's movements to a virtual character in real-time.

