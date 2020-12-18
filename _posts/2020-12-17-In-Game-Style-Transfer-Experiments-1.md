---
title: In-Game Style Transfer Experiments Pt.1
layout: post
toc: false
comments: true
description: 
categories: [unity, log]
hide: true
permalink: /:title/
search_exclude: false
---



I spent a bit of time earlier this week messing around with different style transfer experiments. 



## Kinematica Image Dataset

I used OBS to record an approximately 13 minute video of the character running around the Kinematica demo. I then split the video into separate frames using ffmpeg. I also created mirrored copies of the frames to double my dataset. I ended up resizing the images to `640 x 480` to speed up training. My experiment resulted in noticeably less flickering when running the demo. Although, I still find the flickering in my results a bit distracting. There are also some slight changes in color between the two models. However, the differences were mostly lost when creating the Gifs below.

### Baseline

![base_mosaic_3](..\images\in-game-style-transfer-experiments\part-1\base_mosaic_3.gif)

### My Results

![my_mosaic](..\images\in-game-style-transfer-experiments\part-1\my_mosaic.gif)



## Video Stylization

Training this style transfer model is a bit more involved than the one I've been using so far. First, this model doesn't learn from a source style image like the one below.

![mosaic](..\images\in-game-style-transfer-experiments\part-1\mosaic.jpg)

Instead, you need to provide a few stylized examples of images from your training dataset. I just used the style transfer model I've been working with to generate these examples. However, you need to put in a bit more work to get the best results. You need to create image masks for each image in the dataset like the one below.

![111_mask](..\images\in-game-style-transfer-experiments\part-1\111_mask.png)

You also need to generate noise for these masks like the example below.

![111_noise](..\images\in-game-style-transfer-experiments\part-1\111_noise.png)

### Training Results

I wanted to see how the model ran in Unity before investing the time to make the masks and noise, so I used one of the sample training datasets provided with the project. As you can see below, the model produces much less flickering than the previous model.

![lynx_380p_cropped](..\images\in-game-style-transfer-experiments\part-1\lynx_380p_cropped.gif)

### Unity Performance

The model did a surprisingly okay job stylizing the Kinematica demo despite having only trained on one hundred images of a lynx. Flickering is also significantly reduced. It didn't even give me any headaches importing the ONNX file into Unity. The only catch was performance. The model that I've been using runs `720 x 540` at approximately 25fps. This new model runs the same resolution at about 9fps. You'd probably need to wait a couple generations of GPUs before you could get playable frame rates with the model as is.

![few_shot_mosaic](..\images\in-game-style-transfer-experiments\part-1\few_shot_mosaic.gif)



## Potential Solutions



## Unity's Implementation First Impressions

In short, I have some homework to do. 







![unity_style_transfer](..\images\in-game-style-transfer-experiments\part-1\unity_style_transfer.gif)