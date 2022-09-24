---
categories:
- recap
date: 2021-2-17
description: Revising the style transfer tutorial, working with the Blender Python
  API, and learning about freelancing.
hide: false
layout: post
search_exclude: false
title: Weekly Recap
toc: false

aliases:
- /Weekly-Recap-4/
---

* [End-to-End Style Transfer Tutorial](#end-to-end-style-transfer-tutorial)
* [Working With the Blender Python API](#working-with-the-blender-python-api)
* [Learning About Freelancing](#learning-about-freelancing)
* [Links of the Week](#links-of-the-week)

## End-to-End Style Transfer Tutorial

As I noted in my latest [experiment log](../../in-game-style-transfer-experiments/part-6/), I decided to cut the video style transfer model from the end-to-end tutorial. Instead I've been modifying the `fast_neural_style` model in an effort to improve output quality and increase inference speed. If nothing else, the framerate will be higher than in my [basic style transfer tutorial](../../basic-in-game-style-transfer-tutorial/).

## Working With the Blender Python API

I tried following a couple basic motion graphic tutorials for Blender using only the Python API. I thought they would be fun little exercises to get more familiar with the API. It took quite a bit longer this way as I had to look up basically every single step. Fortunately, it was easy enough to track down answers online. 

### [Shape-Key Motion Graphic Loading Icon in Blender 2.9 Eevee - Tutorial](https://www.youtube.com/watch?v=N3FZcFk-dZA&list=PLGKIkAXk1OeTti1rRVTJF_9_JCC3zY0bh&index=10)

![shape_key_mg](./images/shape_key_mg.gif)

### [Triangle Motion Graphic Animation - Blender 2.9 Eevee Tutorial](https://www.youtube.com/watch?v=xeH41Tz1zGI&list=PLGKIkAXk1OeTti1rRVTJF_9_JCC3zY0bh&index=21)

[![triangle_mg](./images/triangle_mg.gif)](https://www.youtube.com/watch?v=xeH41Tz1zGI&list=PLGKIkAXk1OeTti1rRVTJF_9_JCC3zY0bh&index=21)

Figuring out how to add shader nodes to materials was a bit of a pain. I didn't realize that the names for the nodes were different than what gets displayed in the UI. For example, an `Emission` node is actually called `ShaderNodeEmission`. Unlike basically everything else, the Python tooltips did not indicate that. I also learned that you need to manually link new shaders to the material. I'll probably make separate posts going through the code after I clean it up.

## Learning About Freelancing

I spent some time learning about what goes into running a freelance business. I watched the 3-hour guide provided by FreeCodeCamp ([link](https://www.youtube.com/watch?v=4TIvB8zDFio)). Turns out there is a lot that goes into running a freelance business if you want to do it properly. I recommend checking out the guide if your curious about freelancing. It's targeted towards web developers, but a lot of the information if relevant to freelancing in general.

## Links of the Week

### [ML Learner's Digest](http://learnersdigest.radekosmulski.com/)

Radek Osmulski knows a thing or two about learning machine learning. He learned out how to program and do deep learning using online resources and is now the AI Research Engineering Lead at the [Earth Species Project](https://www.earthspecies.org/). He recently started a newsletter that focuses on how to get better at machine learning. I also recommend checking out his Hacker Noon post, [Going From Not Being Able to Code to Deep Learning Hero](https://hackernoon.com/going-from-not-being-able-to-code-to-deep-learning-hero-2ou34fh).

### [MetaHumans](https://www.unrealengine.com/en-US/digital-humans)

The number of reasons for me to learn Unreal Engine keeps increasing. Unreal Engine showed off their new [MetaHuman Creator](https://www.youtube.com/watch?v=S3F1vZYpH8c) tool that should allow users to create photorealistic humans in a fraction of the time. They claim it can be done in a matter of minutes. The digital humans will be fully rigged and include hair and clothing. This could be huge for generating synthetic datasets involving humans. I'm curious if there will be an API for interacting with the tool to help automate the process of generating random humans. The tool isn't publicly available but there is currently a sample project available ([link](https://www.unrealengine.com/marketplace/en-US/learn/metahumans)). It includes to complete digital humans generated using MetaHuman Creator.

It seems like the tool will be cloud-streamed only which is a bit unfortunate. I'd prefer to have the option to run it locally when possible. However, you apparently get the source data in the form of a Maya file that includes meshes, skeleton, facial rig, animation controls, and materials.

You can check out a walkthrough of the controls for the facial rig included with MetaHumans on the Unreal Engine YouTube channel ([link](https://www.youtube.com/watch?v=GEpH3o44_58)).

### [Pytorch to fastai, Bridging the Gap](https://muellerzr.github.io/fastblog/2021/02/14/Pytorchtofastai.html)

A blog post by Zachary Mueller on understanding how to incorporate regular PyTorch code into a workflow with the fastai library.

### [Python Concurrency: The Tricky Bits](https://python.hamel.dev/concurrency/)

A blog post by Hamel Husain that explores threads, processes, and coroutines in Python. That reminds me, I need to see how Blender behaves with concurrency when using the Python API.

### [JarvisCloud](https://cloud.jarvislabs.ai/)

This is a 1-click GPU cloud platform, I saw recommended on Twitter. I supports PyTorch, Fastai, and TensorFlow and the user just needs to select the desired specs. The interface is about as simple as it can get the prices seem reasonable.





<!-- Cloudflare Web Analytics --><script defer src='https://static.cloudflareinsights.com/beacon.min.js' data-cf-beacon='{"token": "56b8d2f624604c4891327b3c0d9f6703"}'></script><!-- End Cloudflare Web Analytics -->