---
aliases:
- /log/project/unity/2020/09/22/Barracuda-Pose-Estimation-Project-Log-3
categories:
- unity
- project
- log
date: '2020-09-22'
description: I might be doing something wrong.
hide: false
layout: post
search_exclude: false
title: Barracuda Pose Estimation Project Log Pt. 3
toc: true

aliases:
- /Barracuda-Pose-Estimation-Project-Log-3/
---

### Update 7/31/2021: [Barracuda PoseNet Tutorial 2nd Edition](../../barracuda-posenet-tutorial-v2/part-1/)

## Post Processing Notes

Until I can use an ONNX model that contains an argmax operation in Unity, I don't think there's much to be gained from making a PyTorch model to handle post processing. If I can't reduce the size of the array that gets downloaded to the CPU, there isn't a consistent or noticeable improvement in performance.

I could try to make a compute shader that performs an argmax operation. I'm not sure what the best way would be to get the tensor data to the compute shader though. I also don't know what I would do to keep track of the index with the highest value when everything is done in parallel.

After some research, it seems that the best way to do this with a compute shader involves incrementally reducing the size of the `RenderTexture` by taking the max value of a group of neighboring pixels over and over. This seems like max pooling which could be done with a PyTorch model. I don't know what the best way would be to keep track of the original index of the max value though. It seems that Barracuda only has partial support for max pooling. The `MaxPool2d` layer in PyTorch appears to have the option for returning the max indices along with the output. I have a weird feeling that Barracuda won't support having that option enabled. Well, might as well give it a shot. If this doesn't work, I'm going to move on for now. This is really only a problem for extracting every last bit of processing power from the GPU.

As expected, Barracuda did not like having the `return_indices` option enabled for the max pooling layer. At best then, I would still probably need to download the tensor data to the CPU to determine index that contains the returned values. Oh well, time to move on for now.

## Key Point Location Scaling Notes


Depending on the model and input image resolution, the displayed key points seem to jump around a lot. Part of this is because the pose skeleton positions aren't being updated smoothly. However the gaps in between positions seem rather large. I'm guessing this is due to the size of the heatmaps. I would have thought that the offset vectors would decrease the perceived gap between positions. Cranking up the input resolution to 720p does make this behavior less obvious but it's still there. Lowering the input resolution to 256x256 makes the gaps rather absurd. It seems like the offset vectors are either wrong or they are not being scaled appropriately. Side note, there seems to be a cap in performance gained from lowering the input resolution. It seems that once you get to a point where the model can be executed instantaneously on the GPU, any post processing becomes a hard bottleneck.

Looking at the skeleton drawn when using lower resolutions. I think the problem is in my implementation for scaling the model output back up to the original resolution. The gaps between the estimated key points are massive.

So, removing the offset vectors from the key point skeleton revealed some insights. The lower resolution heatmaps definitely contribute to the large gaps in key point locations. I need to take a closer look at how I'm determining which offset vectors to use as they seem off. Using an input resolution of 720p, the same as the source video, without offset vectors looks pretty spot on. The pose skeleton becomes increasingly blocky and less accurate the more you lower the input resolution. I wonder if I can upscale the heatmaps? Using a 720p input resolution with a ResNet50 is fine for a high-end desktop graphics card, but probably not for less powerful devices. However, it does seem that the resolution of the heatmaps is the most important factor for visual accuracy. Using an input resolution that's higher than the source image doesn't appear to be useful.

It's definitely looking like a cut down ResNet50 model is still way better than a MobileNet. The MobileNet will require extra post processing to clean up the output from the heatmaps. Interestingly, MobileNet seems to do worse with higher resolution input images than lower resolutions.

Since my desktop can handle using a 720p input resolution, I'll stick with that for now and try to improve the heatmap accuracy later. Next, I'm going to focus on mapping the key point locations to a virtual character.











<!-- Cloudflare Web Analytics --><script defer src='https://static.cloudflareinsights.com/beacon.min.js' data-cf-beacon='{"token": "56b8d2f624604c4891327b3c0d9f6703"}'></script><!-- End Cloudflare Web Analytics -->