---
categories:
- unity
- log
date: '2020-12-20'
description: Examining results from longer training sessions and Unity's implementation
  in Kinematica demo.
hide: false
layout: post
permalink: /:title/
search_exclude: false
title: In-Game Style Transfer Experiments Pt.2
toc: false

---

* [Introduction](#introduction)
* [Longer Training Session Results](#longer-training-session-results)
* [Unity's Implementation in Kinematica Demo](#unitys-implementation-in-kinematica-demo)
* [Conclusion](#conclusion)

## Introduction

To follow up on the last post, I let the `fast_neural_style` model train overnight. I also got Unity's style transfer implementation working in the Kinematica demo so I could directly compare it to other models. The results for both were a bit disappointing.

## Longer Training Session Results

The longer training session actually resulted in more noticeable flickering than the shorter training session. The flickering is still lower than the baseline at least. The model never seemed to settle on how to apply the style during training. The model might be able to get more consistent if I fiddled with the hyperparameters. It would take a long time to find the ideal values, so I'll hold off on that.

### COCO 2014 Training Images Dataset

![base_mosaic_3](./images/base_mosaic_3.gif)

### Kinematica Demo Image Dataset

![my_mosaic](./images/my_mosaic.gif)

### Longer Training Session

![my_mosaic_2](./images/my_mosaic_2.gif)



## Unity's Implementation in Kinematica Demo

The good news is there is basically no flickering and the frame rate is over 3x higher than the `fast_neural_style` model at around 80fps. For reference, the Kinematica demo runs at around 120fps with no style transfer. The bad news is the quality of style transfer isn't that great.

![](./videos/unity_style_kinematica_4.mp4)

There's a few things that probably contribute to this difference in quality. First, the model Unity chose is trained to handle different styles without additional training. It's trained on a wide variety of style images to help it generalize. This makes it more difficult to achieve the same level of quality for a specific style. Second, the team at Unity had to make some tradeoffs when optimizing the model's performance. They modified the model's architecture to reduce it's overall size. This likely had a negative impact on quality. Third, there seems to be some manually tuned parameters in their demo project. I have no idea how much impact these have or how to approach modifying them just yet. However, I'd be surprised if they didn't affect the quality of the stylized images.

As Unity stated in their blog post, improving the quality of style transfer while optimizing the model for real-time use is still an open research question. I think it might be easier to optimize models trained for specific styles rather than trying to use single models that support arbitrary styles. The specialized models don't take up much disk space and can benefit from the same optimization methods that the Unity team has already developed. 

## Conclusion

The video stylization model I tested in the last post seems the most promising for both quality and consistency between frames. The only roadblock is that it requires much more compute power. It might be worthwhile exploring methods to optimize this model. I'd like to try optimization techniques such as [quantization](https://pytorch.org/blog/introduction-to-quantization-on-pytorch/) and [pruning](https://pytorch.org/tutorials/intermediate/pruning_tutorial.html). Both approaches aim to reduce the size of the model and thus reduce hardware requirements. However, I don't know if the Barracuda library supports quantized models yet. I'm also curious how well the [temporal upsampling](https://blogs.unity3d.com/2020/11/25/real-time-style-transfer-in-unity-using-deep-neural-networks/) from Unity's implementation would work with the video stylization model. I've never used any of these optimization techniques before, so I don't know how long it would take to get them working.

