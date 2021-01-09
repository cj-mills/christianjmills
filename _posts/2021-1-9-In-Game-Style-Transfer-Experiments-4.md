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
* [Shrinking the Model](#shrinking-the-model)
* [Getting it Working in Unity](#getting-it-working-in-unity)
* [Performance Results](#performance-results)
* [Conclusion](#conclusion)

## Introduction

I followed up on the [results](https://christianjmills.com/In-Game-Style-Transfer-Experiments-3/#using-a-smaller-model) in the last post by testing how much I could shrink the [video stylization model](https://christianjmills.com/In-Game-Style-Transfer-Experiments-1/#video-stylization-model). I was initially skeptical since the video stylization model is twice the size of the `fast_neural_style` model. However, the model was easy to modify using the config files provided in the [GitHub repository](https://github.com/OndrejTexler/Few-Shot-Patch-Based-Training). The hard part turned out to be crafting an adequate training dataset.

## Shrinking the Model

I was able to shrink the model from 13MB to less than 1MB by modifying two lines. This required significantly reducing the number and size of layers in the model.

<img src="..\images\in-game-style-transfer-experiments\part-4\generator_combo.png" alt="generator_combo" style="zoom:50%;" />





## Getting it Working in Unity





## Performance Results



![few_shot_mosaic](..\images\in-game-style-transfer-experiments\part-4\few_shot_mosaic.png)



## Training Challenges



## Conclusion


