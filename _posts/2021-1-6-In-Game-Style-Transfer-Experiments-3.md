---
title: In-Game Style Transfer Experiments Pt.3
layout: post
toc: false
comments: true
description: Examining results from my initial attempts to optimize the fast neural style transfer model.
categories: [unity, log]
hide: true
permalink: /:title/
search_exclude: false
---

* [Introduction](#introduction)
* [Model Quantization](#model-quantization)
* [Network Pruning](#network-pruning)
* [Using a Smaller Model](#using-a-smaller-model)
* [Conclusion](#conclusion)

## Introduction

I finally got around to exploring different optimization methods to speed up the style transfer models in Unity. I started off with a couple post-training optimization techniques including quantization and pruning. I then tried training with smaller models. I should have started off with the smaller models.

## Model Quantization

I followed the example code in PyTorch's documentation to quantize the `fast_neural_style` model. The quantized model did seem to perform inference faster in Python. Unfortunately, there was no performance change after importing the quantized model to Unity.

Turns out it's not yet possible to export quantized models to ONNX for general use. According to this [forum post](https://discuss.pytorch.org/t/onnx-export-of-quantized-model/76884/8), quantized models can only be exported for Caffe2. While disappointing, this wasn't too surprising. Quantization is still in beta for PyTorch.

I then tried to quantize a regular ONNX model directly. This can be easily done using the Python [tool](https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/python/tools/quantization/README.md) provided with ONNX Runtime. I was able to reduce the size of the model from 6.5MB to 1.65MB by following the example code in the tool's documentation. As I [suspected](https://christianjmills.com/In-Game-Style-Transfer-Experiments-2/#conclusion) though, the quantized ONNX model uses operators that are not currently supported by Barracuda. I'd be surprised if supported wasn't added in the future, so I'll retry with new releases.

## Network Pruning

I went through the [pruning tutorial](https://pytorch.org/tutorials/intermediate/pruning_tutorial.html#remove-pruning-re-parametrization) provided by PyTorch and didn't encounter any issues applying the code to the fast_neural_style model. However, I didn't encounter any performance improvements. After some more [forum diving](https://discuss.pytorch.org/t/weight-pruning-on-bert/83429/2), I discovered that the pruning module is still an experimental feature. It's not currently meant as a means to improve inference speed. Also, it turns out that GPUs currently aren't optimized for the sparse networks that result from pruning. Apparently, this is [starting to change](https://timdettmers.com/2020/09/07/which-gpu-for-deep-learning/#Additional_Considerations_for_Ampere_RTX_30_Series) with the latest RTX 30 series cards from Nvidia.

## Using a Smaller Model

So both of the fancy optimization techniques I wanted to try are currently dead ends for my use case. Fortunately, I then thought to try the much simpler approach of just starting with a smaller model. This really should have been the first approach I tried. 

I kept the number and types of layers in the model the same but reduced the size of the many of the layers. This turned out to be quite effective. I was able to reduce the size of the model from 6.5MB to less than 600KB without any significant visual changes in the output.

### Resolution: 720 x 540



#### Original Model



![mosaic_original](..\images\in-game-style-transfer-experiments\part-3\mosaic_original.png)



![mosaic_stats](..\images\in-game-style-transfer-experiments\part-3\mosaic_stats.gif)



#### Smaller Model



![mosaic_small_v3](..\images\in-game-style-transfer-experiments\part-3\mosaic_small_v3.png)



![mosaic_small_v3_stats](..\images\in-game-style-transfer-experiments\part-3\mosaic_small_v3_stats.gif)



### Resolution: 1280 x 720



### Original Model



![mosaic_original_1280x720](..\images\in-game-style-transfer-experiments\part-3\mosaic_original_1280x720.png)



![mosaic_original_1280x720_stats](..\images\in-game-style-transfer-experiments\part-3\mosaic_original_1280x720_stats.gif)





#### Smaller Model



![mosaic_small_1280x720](..\images\in-game-style-transfer-experiments\part-3\mosaic_small_1280x720.png)



![mosaic_small_1280x720_stats](..\images\in-game-style-transfer-experiments\part-3\mosaic_small_1280x720_stats.gif)



## Conclusion



