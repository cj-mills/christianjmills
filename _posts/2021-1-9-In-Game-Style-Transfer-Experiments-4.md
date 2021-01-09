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

I was able to shrink the model from 13MB to less than 1MB by modifying two lines in the config file. This involved significantly reducing the number and size of layers in the model.

<img src="..\images\in-game-style-transfer-experiments\part-4\generator_combo.png" alt="generator_combo" style="zoom:40%;" />

Fortunately, this didn't seem to have any significant impact on the quality of the output. The model did however need to be trained longer to achieve similar results.

## Getting it Working in Unity

It was a bit of a pain figuring out what preprocessing operations I needed to apply from the source code. It seems to boil down to normalizing the RGB color values to the range `[-1,1]`. The output seemed to look right in Unity so I stuck with that.

## Performance Results

The modified video stylization model has similar performance to 

![few_shot_mosaic](..\images\in-game-style-transfer-experiments\part-4\few_shot_mosaic_720x540.png)

![few_shot_mosaic_peformance](..\images\in-game-style-transfer-experiments\part-4\stats_720x540.gif)





## Training Challenges



## Conclusion


