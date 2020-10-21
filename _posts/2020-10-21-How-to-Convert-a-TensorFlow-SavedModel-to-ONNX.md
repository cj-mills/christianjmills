---
title: How to Convert to TensorFlow model to ONNX with tf2onnx
layout: post
toc: true
description: An example of how to convert a TensorFlow SavedModel to ONNX.
categories: [tensorflow,onnx,tutorial]
hide: true
search_exclude: false
---

## Motivation

Currently, there is no officially supported method for exporting TensorFlow models to ONNX. The TensorFlow team has no plans to provide such support anytime soon. Fortunately, the python tool [tf2onnx](https://github.com/onnx/tensorflow-onnx) supports conversions for most types of models. This post coverts how to use tf2onnx to convert a PoseNet SavedModel to ONNX.



