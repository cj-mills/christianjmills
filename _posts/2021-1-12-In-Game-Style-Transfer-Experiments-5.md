---
title: In-Game Style Transfer Experiments Pt.5
layout: post
toc: false
comments: true
description: I got the video stylization model to work properly in Unity and found some weaknesses in the fast neural style model.
categories: [unity, log]
hide: true
permalink: /:title/
search_exclude: false
---

* [Introduction](#introduction)
* [Many-Shot Learning](#many-shot-learning)
* [Difficult to Learn Styles](#difficult-to-learn-styles)
* [Conclusion](#conclusion)

## Introduction

I'm starting to see a trend where a seemingly difficult problem is easily resolved and something I didn't think was going to be a problem turns out to be difficult. 



## Many-Shot Learning

I was able to get the video stylization model to correctly stylize in Unity by adding a lot more training examples. The creators of the model only needed a few training examples to get great results for a specific scene. I decided to try using around 80 examples to see if that would help it generalize. It did.

![generator_mosaic_small_v6](..\images\in-game-style-transfer-experiments\part-5\generator_mosaic_small_v6.png)

This model did produce a bit more flickering than the models trained on fewer examples. It should be easy to reduce the flickering by tuning the quality and quantity of the training examples. Output from this model even be used as training examples for another model. Although, I actually prefer the color palette from the less accurate model.

![few_shot_mosaic_frame](..\images\in-game-style-transfer-experiments\part-5\few_shot_mosaic_frame.png)



## Difficult to Learn Styles





## Conclusion

It's annoying that the `fast_neural_style` model can't seem to capture certain types of styles. Fortunately, I can use any style transfer model I want to generate the training examples for the video stylization model. I'd prefer to have a style transfer model that can produce good results with a wide variety of styles. However, it might be necessary to use multiple models that handle to different styles. I want to develop a reliable workflow so I can write an end-to-end tutorial.