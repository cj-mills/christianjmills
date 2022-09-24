---
categories:
- ai
- 3d
- notes
date: 2021-12-9
description: My notes from Andrew Price's talk at Blender Conference 2018 on how A.I.
  will change the 3D industry.
hide: false
layout: post
search_exclude: false
comments:
  utterances:
    repo: cj-mills/christianjmills
title: Notes on How A.I. Will Change the 3D Industry
toc: false

aliases:
- /Notes-on-How-AI-Will-Change-The-3D-Industry/
---

* [Overview](#overview)
* [Assets are unreasonably expensive](#assets-are-unreasonably-expensive)
* [Machine Creep](#machine-creep)
* [Not in Presentation](#not-in-presentation)



## Overview

I went back and watched the [talk](https://www.youtube.com/watch?v=FlgLxSLsYWQ) Andrew Price ([Blender Guru](https://www.youtube.com/user/AndrewPPrice)) gave at Blender Conference 2018 on how A.I. will change the 3D industry. This time, I decided to take some notes.

**Question to consider:** What is ***not*** going to change in the next 10 years?



## Assets are unreasonably expensive

### Creating a building Asset:

- Modeling: 12 hours
- Texturing: 10 hours
- First Pass total: 22 hours
- Revisions: x2-3

### Problem: static workflows

### Solution: procedural workflows

- Practical Procedural Generation for Everyone (GDC 2017)
  
    [Practical Procedural Generation for Everyone](https://www.youtube.com/watch?v=WumyfLEa6bU)
    
- Procedural Modeling Example: Procedural Lake Village by Anastasia Opera
  
    [Procedural Lake Village](https://www.anastasiaopara.com/lakevillage)
    
    [Houdini Procedural Lake Houses Complete](https://anopara.gumroad.com/l/qaEZ)
    
- Procedural Texturing Example: Poliigon Substance Designer
  
    [poliigon](https://www.poliigon.com/)
    
    [poliigon generators](https://help.poliigon.com/en/articles/3175522-using-poliigon-generators)
    
- Procedural Texturing Example: Substance Painter
    - Bake → Smart Materials → Smart Masks
    
    [substance3d](https://www.substance3d.com/)
    
- Procedural Level Design: Houdini
  
    [Procedural World Generation of Ubisoft's Far Cry 5](https://www.youtube.com/watch?v=NfizT369g60)
    
    - Create an ecosystem
        - Set rules to define where certain trees and plants would live
        - Other factors
            - Occlusion
            - Flow
            - Slope
            - Curvature
            - Illumination
            - Altitude
            - Latitude
            - Longitude
            - Wind
        - Tools for customization like roads and buildings

## Machine Creep

**Traditional Software: input (e.g. photo) → action (filter) → output**

**Machine Learning: input (e.g. photo) → assess → appropriate action → compare → is it good? → output**

- Needs huge datasets and fast hardware

### Machine Learning Use Cases

- De-noising
- Super resolution
- Motion Capture
    - [Densepose](http://densepose.org/)
- Animation
    - Mode-adaptive Neural Networks for motion control

### Machine-Assisted Creativity

- Problem: experimentation takes up 50%-70% of time

- BicycleGAN
  
    ![https://github.com/junyanz/BicycleGAN/blob/master/imgs/day2night.gif?raw=true](https://github.com/junyanz/BicycleGAN/blob/master/imgs/day2night.gif?raw=true)
    
    - Model input: outline of an object in an image and the ground truth image
    - Model output: generate variations of image
    - [GitHub Repository](https://github.com/junyanz/BicycleGAN)
    - Toward Multi-modal Image-to-image translation
    
- DivCo: Diverse Conditional Image Synthesis via Contrastive Generative Adversarial Network
    - [GitHub Repository](https://github.com/ruiliu-ai/DivCo)
    
- Sketch to Image
    - pix2pix
      
        [Image-to-Image Demo - Affine Layer](https://affinelayer.com/pixsrv/index.html)
        
        - [GitHub Repository](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
    
- StyleGAN2
    - [GitHub Repository](https://github.com/lucidrains/stylegan2-pytorch)
    - [GitHub Repository](https://github.com/rosinality/stylegan2-pytorch)
    
- Progressive Growing of GANS (PGAN)
    - [GitHub Repository](https://github.com/nashory/pggan-pytorch)
    
    [PyTorch](https://pytorch.org/hub/facebookresearch_pytorch-gan-zoo_pgan/)
    
- Text to Image
    - [StackGAN V2](https://github.com/hanzhanggit/StackGAN-v2) (2017)
    - [text2image](https://github.com/wtliao/text2image) (April 2021)
    - [TediGAN](https://github.com/IIGROUP/TediGAN) (March 2021)
    - [DF-GAN](https://github.com/tobran/DF-GAN)
    
- Style Transfer
    - A Style-Aware Content Loss for Real-time HD Style Transfer
        - [Adaptive Style Transfer](https://github.com/CompVis/adaptive-style-transfer) (TensorFlow 2018)
        - [color-transform](https://github.com/Tonyhuiii/color-transform) (PyTorch 2019)
        



## Not in Presentation

### Related Works

- GANcraft: Unsupervised 3D Neural Rendering of Minecraft Worlds
  
    [imaginaire/projects/gancraft at master · NVlabs/imaginaire](https://github.com/NVlabs/imaginaire/tree/master/projects/gancraft)
    
    [Unsupervised 3D Neural Rendering of Minecraft Worlds](https://nvlabs.github.io/GANcraft/)
    
- NeRS: Neural Reflectance Surfaces for Sparse-View 3D Reconstruction in the Wild (October 2021)
    - [GitHub Repository](https://github.com/jasonyzhang/ners)
    
- PIFuHD: Multi-Level Pixel-Aligned Implicit Function for High-Resolution 3D Human Digitization (2020)
    - [GitHub Repository](https://github.com/facebookresearch/pifuhd)
      
        [Google Colaboratory](https://colab.research.google.com/drive/11z58bl3meSzo6kFqkahMa35G5jmh2Wgt)
        
    - 
    
- 3DStyleNet: Creating 3D Shapes with Geometric and Texture Style Variations
  
    ![https://nv-tlabs.github.io/3DStyleNet/assets/teaser.jpg](https://nv-tlabs.github.io/3DStyleNet/assets/teaser.jpg)
    
    
    
    [https://nv-tlabs.github.io/3DStyleNet/assets/animal-new.mp4](https://nv-tlabs.github.io/3DStyleNet/assets/animal-new.mp4)
    
    [3DStyleNet: Creating 3D Shapes with Geometric and Texture Style Variations](https://nv-tlabs.github.io/3DStyleNet/)
    
- DISN: Deep Implicit Surface Network for High-quality Single-view 3D Reconstruction
  
    ![https://github.com/laughtervv/DISN/raw/master/result.png](https://github.com/laughtervv/DISN/raw/master/result.png)
    
    - [GitHub Repository](https://github.com/laughtervv/DISN) (Tensorflow)
    - [GitHub Repository](https://github.com/cs674/pytorch-DISN) (PyTorch)
    
- Learning Linear Transformations for Fast Arbitrary Style Transfer
    - [GitHub Repository](https://github.com/sunshineatnoon/LinearStyleTransfer)
    
- Neural Cages for Detail-Preserving 3D Deformations
    - [GitHub Repository](https://github.com/yifita/deep_cage)
    
- Taming Transformers for High-Resolution Image Synthesis
    - [GitHub Repository](https://github.com/CompVis/taming-transformers)
    - [GitHub Repository](https://github.com/sunshineatnoon/taming-transformers)
    
    [[Overview] Taming Transformers for High-Resolution Image Synthesis](https://wandb.ai/ayush-thakur/taming-transformer/reports/-Overview-Taming-Transformers-for-High-Resolution-Image-Synthesis---Vmlldzo0NjEyMTY)
    
    [These Neural Networks Have Superpowers! 💪](https://www.youtube.com/watch?v=o7dqGcLDf0A)
    
- Rethinking Style Transfer: From Pixels to Parameterized Brushstrokes
    - [GitHub Repository](https://github.com/CompVis/brushstroke-parameterized-style-transfer) (TensorFlow)
    - [GitHub Repository](https://github.com/justanhduc/brushstroke-parameterized-style-transfer) (PyTorch)
    
- Network-to-Network Translation with Conditional Invertible Neural Networks
    - [GitHub Repository](https://github.com/CompVis/net2net)
    
- Artistic Style Transfer with Internal-external Learning and Contrastive Learning
    - [GitHub Repository](https://github.com/HalbertCH/IEContraAST)
    - Based on: [SANET](https://github.com/GlebSBrykin/SANET)
    
- Synthetic Silviculture: Multi-scale Modeling of Plant Ecosystems
  
    [Makowski.etal-2019-Synthetic-Silviculture.pdf](https://drive.google.com/file/d/1tbb0WPcxljqwWdKzM1CMXWwH5vWeID53/view?usp=drivesdk)
    
    [Makowski.etal-2019-Synthetic-SilvicultureSup.pdf](https://drive.google.com/file/d/13_G9p9rB3bLuZRr-pIEBo-AVsxHZwkGN/view?usp=drivesdk)
    
    [Synthetic Silviculture: Multi-scale Modeling of Plant Ecosystems](https://storage.googleapis.com/pirk.io/projects/synthetic_silviculture/index.html)
    
    [Simulating A Virtual World...For A Thousand Years! 🤯](https://www.youtube.com/watch?v=8YOpFsZsR9w)
    
- DualConvMesh-Net: Joint Geodesic and Euclidean Convolutions on 3D Meshes
    - [GitHub Repository](https://github.com/VisualComputingInstitute/dcm-net)
    

### Other Applications

[How A.I will affect the art industry](https://www.youtube.com/watch?v=0LK_MHLs__M)

Takeaway: AI will handle more and more of the tedious manual work that humans don't like doing (or is extremely time consuming)

This will reduce the cost of production, enabling more productions overall

- Rotoscoping
  
    [What is rotoscoping animation and how to do it ](https://www.adobe.com/creativecloud/video/discover/rotoscoping-animation.html)
    
    - segmentaion
    
- Retopology
  
    [What is Retopology? (A Complete Intro Guide For Beginners)](https://conceptartempire.com/retopology/)
    
    - Appearance-Driven Automatic 3D Model Simplification (2021)
        - [https://research.nvidia.com/publication/2021-04_Appearance-Driven-Automatic-3D](https://research.nvidia.com/publication/2021-04_Appearance-Driven-Automatic-3D)
        - [GitHub Repository](https://github.com/NVlabs/nvdiffmodeling)
    
- Human provides general outline/concept and a model fills in technical details
    - NVIDIA GauGAN2
      
        [NVIDIA Research's GauGAN AI Art Demo Responds to Words](https://blogs.nvidia.com/blog/2021/11/22/gaugan2-ai-art-demo/)
        
    - NVIDIA Canvas
      
        [NVIDIA Canvas : Harness The Power Of AI](https://www.nvidia.com/en-us/studio/canvas/)
    
- Facial animations



**References:**

* **Video:** [The Next Leap: How A.I. will change the 3D industry - Andrew Price](https://www.youtube.com/watch?v=FlgLxSLsYWQ)
* **Slides:** [Google Slides](https://docs.google.com/presentation/d/1nXwBdUEIbwtPyyMEu5nwJugd42sgU4YhunaUOSl2_tc/edit#slide=id.g44af636ac2_0_502)



<!-- Cloudflare Web Analytics --><script defer src='https://static.cloudflareinsights.com/beacon.min.js' data-cf-beacon='{"token": "56b8d2f624604c4891327b3c0d9f6703"}'></script><!-- End Cloudflare Web Analytics -->