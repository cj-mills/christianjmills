---
title: In-Game Style Transfer Experiments Pt.1
layout: post
toc: false
comments: true
description: Trying out custom datasets, a new model, and Unity's style transfer project.
categories: [unity, log]
hide: false
permalink: /:title/
search_exclude: false
---

* [Introduction](#introduction)
* [Kinematica Image Dataset](#kinematica-image-dataset)
* [Video Stylization Model](#video-stylization-model)
* [Unity's Implementation: First Impressions](#unitys-implementation-first-impressions)
* [Conclusion](#conclusion)

## Introduction

I spent a bit of time this week messing around with different style transfer experiments. I wanted to see if training the [`fast_neural_style`](https://github.com/pytorch/examples/tree/master/fast_neural_style) model on images from the Kinematica demo would improve the output quality. I also got the model from the [`Interactive Video Stylization Using Few-Shot Patch-Based Training`](https://github.com/OndrejTexler/Few-Shot-Patch-Based-Training) project working in Unity. Lastly, I started exploring Unity's style transfer [project](https://github.com/UnityLabs/barracuda-style-transfer). Let's just say that calling my implementation basic was an understatement.

## Kinematica Image Dataset

I used [OBS](https://obsproject.com/) to record an approximately 13 minute video of the character running around the Kinematica demo. I then split the video into separate frames using [ffmpeg](https://ffmpeg.org/). I also created mirrored copies of the frames to double the size of my dataset. I ended up resizing the images to `640 x 480` to speed up training. My experiment resulted in noticeably less flickering when running the demo. Although, I still find the flickering in my results a bit distracting. I'm curious if I could further reduce flickering by training the model for longer. There are also some slight changes in color between the two models. However, the differences were mostly lost when creating the Gifs below.

### COCO 2014 Training Images Dataset

![base_mosaic_3](..\images\in-game-style-transfer-experiments\part-1\base_mosaic_3.gif)

### Kinematica Demo Image Dataset

![my_mosaic](..\images\in-game-style-transfer-experiments\part-1\my_mosaic.gif)

## Video Stylization Model

Training this style transfer model is a bit more involved than the one I've been using so far. First, this model doesn't learn from a source style image like the one below.

![mosaic](..\images\in-game-style-transfer-experiments\part-1\mosaic.jpg)

Instead, you need to provide a few stylized examples of images from your training dataset. I just used the `fast_neural_style` model to generate these examples. However, you need to put in a bit more work to get the best results. This involves creating masks for each image in the dataset like the one below.

![111_mask](..\images\in-game-style-transfer-experiments\part-1\111_mask.png)

You also need to generate noise for these masks as shown below.

![111_noise](..\images\in-game-style-transfer-experiments\part-1\111_noise.png)

You can technically just make masks and noise for the whole image rather than for specific parts. However, I didn't feel like doing that just yet. I wanted to see how the model ran in Unity first, so I used one of the sample training [datasets](https://drive.google.com/file/d/1EscSNFg4ILpB7dxr-zYw_UdOILLmDlRj/view) provided for the project. Specifically, I used the lynx dataset.

![lynx_000](..\images\in-game-style-transfer-experiments\part-1\lynx_000.jpg)

### Training Results

As you can see below, this model produces much less flickering than the `fast_neural_style` model. The next step was to see how well this transferred to Unity.

<center>
	<video style="width:auto;max-width:100%;height:auto;" controls loop>
		<source src="../videos/in-game-style-transfer-experiments\part-1\lynx_380p_cropped.mp4" type="video/mp4">
	</video>
</center>



### Unity Performance

The model did a surprisingly okay job stylizing the Kinematica demo despite having only trained on one hundred images of a lynx. Flickering is significantly reduced and it didn't even give me any headaches importing the ONNX file into Unity. The only catch was performance.

![few_shot_mosaic](..\images\in-game-style-transfer-experiments\part-1\few_shot_mosaic.gif)

On my desktop, the `fast_neural_style` model I've been using runs `720 x 540` at approximately 25fps. This new model runs the same resolution at about 9fps. You'd probably need to wait a few generations of GPUs before you could get playable frame rates with the model as is. It would take some insane optimization to make this viable for in-game style transfer. Fortunately, Unity has already figured out how to do some insane optimization for their style transfer project.

## Unity's Implementation: First Impressions

In short, I have some homework to do. I've only glanced through the code for Unity's project so far, but it's easy to see why they took so long to release it. They've put a lot of work into optimizing the performance of their model. With the default settings, I was consistently getting around 400fps. 

**Note:** Performance dropped slightly when recording for the Gifs below. Hence, the displayed fps is a bit lower.

![unity_style_transfer](..\images\in-game-style-transfer-experiments\part-1\unity_style_transfer.gif)

The actual scene is about as simple as it gets so I can't directly compare the lack of flickering just yet. I'll wait until I get this running in the Kinematica demo for that.

![unity_style_transfer_scene](..\images\in-game-style-transfer-experiments\part-1\unity_style_transfer_scene_2.jpg)

However, the performance numbers speak for themselves. The team at Unity did a fantastic job with optimization. What's more, their method for optimizing the performance looks like it should transfer to other models. I want to try applying their method to the PoseNet project as well as the style transfer models I've been working with. Although, the optimization process appears quite involved so I'll need to study it a bit more before attempting that.

## Conclusion

My experiments provided some useful insights in how I should move forward with future style transfer experiments. Training the models on images from the target game seems worthwhile to reduce flickering. I'll see if letting the model train overnight will further reduce flickering. 

The performance from the video stylization model was lower than I expected. I thought there might be some decrease in frame rate, but I did not expect it to drop by roughly two thirds. In hindsight, I guess it's not unreasonable. The video stylization model is double the size at 13MB versus 6.5MB for the `fast_neural_style` model.

I'm now even more grateful that Unity has released their example project. It applies a level of expertise that would have taken me a long time to figure out on my own. However, it also shows just how much work still remains in optimizing more sophisticated models for end-user devices. I'm curious if it's feasible to automate this optimization process.

