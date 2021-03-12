---
title: Weekly Recap
layout: post
toc: false
comments: true
description: A summary of what I've been working on for the past few weeks.
categories: [log]
hide: true
permalink: /:title/
search_exclude: false
---

* [Introduction](#introduction)
* [End-to-End Style Transfer Tutorial](#end-to-end-style-transfer-tutorial)
* [Next Project](#Next Project)
* [Links of the Last Few Weeks](#links-of-the-last-few-weeks)



## Introduction

Wow it is really easy to get off schedule with these weekly recaps. It's now been over three weeks since my last "weekly" recap. Maybe I should try doing daily recaps just so I don't fall out of the routine as easily. It's too tempting to put off these weekly posts when I'm in the middle of an actual project. 

## End-to-End Style Transfer Tutorial

I finally completed the end-to-end style transfer tutorial. I'm a bit irritated that I let myself dump so much time into experimenting with different style transfer models. I think it would have been better to invest that time explaining how others can conduct their own experiments. I might make a post examining how I could have better approached the project and avoided going so far over my time budget. Perhaps it would have helped to set an actual time budget.

I'm in the process of making a follow up post explaining how to use the [video style transfer model](https://github.com/OndrejTexler/Few-Shot-Patch-Based-Training) instead of the model used in the tutorial. I don't plan on trying to make a super optimized version of the model like in the tutorial. My previous experiments didn't yield great results when significantly reducing the size of the model. However, I dumped so much time into it that I figure I should at least make a post showing how someone else can tinker with it. I plan to have that finished by early next week.

## Next Project

Before moving on to a completely new project, I plan to make some updates and additions to the PoseNet tutorial series. I realized that there were some unnecessary steps that could be removed for a small performance gain. I also realized I completely forgot to explain how to use the more efficient MobileNet version of the model instead of the ResnNet50 version. I didn't use the MobileNet version in the tutorial because it's less accurate. However, a reader expressed interest in performing inference with the C# Burst backend. The smaller model would be a much better choice in that instance.

After I update the PoseNet tutorial, I plan to work on getting a facial pose estimation model working with the Barracuda library. Hopefully, that will take less time than the style transfer tutorial. If the model proves incompatible with the current versions of Barracuda, I'll look into making a C++ plugin so I can use the PyTorch C++ frontend instead.



## Links of the Last Few Weeks

### PyTorch3D



