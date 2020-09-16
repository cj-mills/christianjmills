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

I've recently started playing around with the [Barracuda](https://docs.unity3d.com/Packages/com.unity.barracuda@1.1/manual/index.html) inference library for the [Unity](https://unity.com/) game engine. What's cool about Barracuda is that it runs models locally within Unity. There's no remote server or external framework required to use the ML model in your application. With a few [constraints](https://docs.unity3d.com/Packages/com.unity.barracuda@1.1/manual/SupportedPlatforms.html), it should work on any platform supported by Unity. In theory, that means you don't need to make separate versions of applications for different platforms. Additionally, since it is running in Unity, you can combine it with all the other [features](https://unity.com/products/unity-platform) of a modern real-time development platform.

**Note:** If you're interested in learning how to get started with Barracuda, I plan to make a separate post that in the near future. I will update this post with a link when it is ready.

While Barracuda was originally built for Unity's [ML-Agents](https://unity3d.com/machine-learning) toolkit, it has been gradually adding support for more network architectures. I'm really excited about the potential for leveraging modern machine learning capabilities in interactive applications. To get acquainted with the Barracuda library, I decided to start with a fun project. One of the first things that came to mind was mapping a user's movements to a virtual character in real-time.

