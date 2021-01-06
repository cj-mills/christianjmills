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

I finally got around to exploring different optimization methods to speed up the style transfer models in Unity. I started off with post-training optimization techniques including quantization and pruning. I then tried training with smaller models. I should have started off with the smaller models.

## Model Quantization

I followed the example code in PyTorch's documentation to quantize the `fast_neural_style` model. The quantized model did seem to perform inference faster in Python. Unfortunately, there was no performance change after importing the quantized model to Unity. 

Turns out it's not yet possible to export quantized models to ONNX for general use. According to this [forum post](https://discuss.pytorch.org/t/onnx-export-of-quantized-model/76884/8), quantized models can only be exported for Caffe2. While disappointing, this wasn't too surprising. Quantization is still in beta for PyTorch.

I then tried to quantize a regular ONNX model directly. This can be easily done using the Python [tool](https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/python/tools/quantization/README.md) provided with ONNX Runtime. I was able to reduce the size of the model from 6.5MB to 1.65MB by following the example code in the tool's documentation. As I [suspected](https://christianjmills.com/In-Game-Style-Transfer-Experiments-2/#conclusion) though, the quantized ONNX model uses operators that are not currently supported by Barracuda. I'd be surprised if supported wasn't added in the future, so I'll retry with new releases.

## Network Pruning



## Using a Smaller Model



## Conclusion



