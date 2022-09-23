---
categories:
- recap
date: 2021-4-26
description: A summary of what I worked on today.
hide: false
layout: post
search_exclude: false
title: Daily Recap
toc: false

aliases:
- /Daily-Recap-2/
---

* [Barracuda Background Execution](#barracuda-background-execution)

  

## Barracuda Background Execution

The Barracuda library provides the option to execute models inside a coroutine. This is useful when running on hardware that is not fast enough to execute the model in a single frame. I have avoided using this approach for the PoseNet tutorial because I was concerned that the pose skeleton would become out of sync with the video or webcam feed. A reader confirmed that this occurred when they implemented scheduled execution on their own. The pose skeleton lagged slightly behind the video. This drawback is not necessarily an issue depending on the application so I decided to try it out so that I can include it in the updated tutorial. 

Unsurprisingly, the amount of lag depends on how demanding the model is to run. There was a negligible amount of lag when running either the ResNet or MobileNet versions of the model on the GPU. There was also a significant increase in frame rate. Executing the model in the background on the CPU results in a significant amount of lag. The ResNet model is basically unusable. Even with the MobileNet version, the input resolution still needs to lowered quite a bit for the pose skeleton to keep up with the video. 

If the target application only needs to classify the estimated pose, say for a yoga application, then it should be fine to lag behind the source video a bit. However, it can be problematic if the target application requires real-time motion tracking. If the application needs to be run on a mobile device, this might just be a compromise you need to make for now. This approach does still have the benefit of allowing the rest of the application to run smoothly while the model executes.

I believe a new backend is in the works for the Barracuda library that will make use of the Neural Processing Units (NPUs) that are in newer mobile devices. I don't know how far along the development is, but that should allow models to be executed much more efficiently on supported devices.