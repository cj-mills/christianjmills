---
aliases:
- /log/project/unity/2020/09/21/Barracuda-Pose-Estimation-Project-Log-2
categories:
- unity
- log
date: '2020-09-21'
description: Trying to remove bottlenecks.
hide: false
layout: post
search_exclude: false
title: Barracuda Pose Estimation Project Log Pt. 2
toc: false

aliases:
- /Barracuda-Pose-Estimation-Project-Log-2/
---

### Update 7/31/2021: [Barracuda PoseNet Tutorial 2nd Edition](../../barracuda-posenet-tutorial-v2/part-1/)

I didn't make much actual progress yesterday. Although, it was fairly educational. I spent the day trying to create a PyTorch model that would help speed up the post processing steps for the pose estimation model. The model outputs a heatmap and offsets for each of the 17 key points. Each element in a heatmap contains a confidence level for whether the relevant key point is in that cell. The offsets contain x and y vectors that are used to refine the coarse location from the heatmap. The part that seems to create a bottleneck is extracting the index value for the heatmap element that contains the highest confidence estimate. Currently, I need to iterate through every element in each of the heatmaps. This isn't a huge issue when using a very small input image as the size of the heatmap is limited by input resolution. However, larger input images can yield more accurate predictions. Even when using 540p input images, the size of the heatmap increases quite a bit compared to a 300p image. 

Right now, post processing needs to be done on the main thread. Barracuda tensors either need to be manipulated on the GPU or on the main thread on the CPU. You need to download the data from the tensor to a float array if you want to perform the work in parallel on the CPU. Unfortunately, downloading the data from the GPU to the CPU seems to wipe out any performance gains from iterating over the data in parallel. That's why I decided to spend a day beating my head against a wall trying to do the post processing on the GPU. 

My plan was to simply use the argmax function available in PyTorch to extract the index with the highest confidence level for each of the 17 heatmaps. I thought this would be fine since the argmax operation is supported by ONNX. However, the dev team for the Barracuda library has not yet implemented the functionality to support the argmax operation yet. I confirmed this by double checking the current list of supported operations in the Barracuda documentation. Fortunately, the dev team is aware of the missing functionality and it's on there todo list. But, that doesn't really help me right now so I decided to spend the rest of the day trying to come up with a work around using a combination of supported operations. Basically nothing I tried really worked and the few things that did, didn't actually improve performance. 

The endeavor once again highlighted how much of a pain it is to access array slices in C#. Apparently, it gets slightly better in C# 8. Unfortunately, the only Unity version that supports C# 8 is the current beta build. I don't need yet another source of debugging rabbit holes right now so I'll wait to try that until it's out of beta. 

It's funny, the odd things that crop up when you make a model in one framework, export it to another, and then load it into Unity that then converts it to it's own internal representation. Barracuda apparently doesn't like tensors being squeezed when loading an ONNX model as it apparently introduces some ambiguity. There was also some compute shader related issue that kept cropping up every time I tried to use a PyTorch model to permute the axes in a tensor. I don't know if the error is because I did something wrong or if it's just a bug in part of the Barracuda library. It's probably a combination of both.

I also tried to create a PyTorch model that split the different heatmaps into separate outputs. The goal of this was to make it possible to leverage the argmax helper function in Unity. The helper function isn't super helpful for me since it needs to download the tensor data to the CPU to work. This is a problem since downloaded tensors get saved channel first. For example, a tensor containing an RGB image would be stored in an array pixel by pixel. In other words, every 3 elements contain the RGB values for a single pixel. For the heatmaps, this means that a sequence of 17 elements represent the confidence level for each key point for a single pixel rather than one entire heatmap after the other.

I'm getting close to the limit of how much time I'm willing to spend on this issue before moving on. It wouldn't be the end of the world if I can't do post processing on the GPU but it definitely caps the max framerate. I want to try a few other things, but I have a feeling that I'll need to wait until the Barracuda dev team implements support the the argmax operation. 



<!-- Cloudflare Web Analytics --><script defer src='https://static.cloudflareinsights.com/beacon.min.js' data-cf-beacon='{"token": "56b8d2f624604c4891327b3c0d9f6703"}'></script><!-- End Cloudflare Web Analytics -->