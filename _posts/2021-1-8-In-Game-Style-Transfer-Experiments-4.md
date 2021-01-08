---
title: In-Game Style Transfer Experiments Pt.4
layout: post
toc: false
comments: true
description: Examining results from my initial attempts to optimize the few-shot video stylization model.
categories: [unity, log]
hide: true
permalink: /:title/
search_exclude: false
---

* [Introduction](#introduction)
* [](#)
* [Conclusion](#conclusion)

## Introduction

I followed up on the [results](https://christianjmills.com/In-Game-Style-Transfer-Experiments-3/#using-a-smaller-model) in the last post by testing how much I could shrink the [video stylization model](https://christianjmills.com/In-Game-Style-Transfer-Experiments-1/#video-stylization-model). I was initially skeptical given the video stylization model is twice the size of the fast_neural_style model. However, the model was easy to modify using the config files provided in the [GitHub repository](https://github.com/OndrejTexler/Few-Shot-Patch-Based-Training). I was able to shrink the model from 13MB to less than 1MB by modifying two lines.





## Performance Results



## Training Challenges





## Conclusion


