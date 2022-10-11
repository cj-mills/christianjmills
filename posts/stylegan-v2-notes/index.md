---
aliases:
- /Notes-on-StyleGANv2/
categories:
- ai
- notes
date: '2021-12-16'
description: My notes on the overview of StyleGANv2 by Henry AI Labs.
hide: false
layout: post
search_exclude: false
title: Notes on StyleGANv2
toc: false

---



* [Overview](#overview)
* [Notable StyleGANv1 Characteristics](#notable-styleganv1-characteristics)
* [Notable StyleGANv2 Changes](#notable-styleganv2-changes)
* [StyleGANv1 Recap](#styleganv1-recap)
* [StyleGANv1 Artifacts](#styleganv1-artifacts)
* [Overview of StyleGANv2 Changes](#overview-of-styleganv2-changes)
* [Recap](#recap)



## Overview

I recently started learning more about generative deep learning models for some potential projects and decided to check out this [video](https://www.youtube.com/watch?v=u8qPvzk0AfY) by Henry AI Labs covering [StyleGANv2](https://arxiv.org/abs/1912.04958). Below are some notes I took while watching.



## Notable StyleGANv1 Characteristics

- Adaptive instance normalization
- A mapping network from the latent vector $z \ \epsilon \ Z$ into $w \ \epsilon \ W$
- Uses progressive GAN growing (starts with 4x4 input image and iteratively doubles dimensions)

## Notable StyleGANv2 Changes

- Restructures the use of adaptive instance normalization
- Gets away from progressive growing to get away from the artifacts introduced in v1
    - Water droplet effects
    - Fixed position of eyes and noses in generated faces
- Perceptual path-length normalization term in the loss function to improve on latent space interpolation
    - Latent space interpolation: the changes in the generated image when changing the latent vector $Z$
    - You want small changes in the latent vector to have small semantic perceptual changes in the generated image
    - The interpolation is so smooth in v2 that you can create an animated GIF
    - Can combine the vectors of two generated images and combine them to create an in-between image
- Introduces a deep deep fake detection algorithm to project the generated images back into latent space to try to see if you can contribute the generated image to the network that created it

## StyleGANv1 Recap

### Mapping Network

- Latent vector $Z$: a random vector that is passed to a network of eight fully connected layers that maps to the $w \ \epsilon \ W$ latent space
- $W$ latent code: used to control the features in the generative adversarial network using the adaptive instance normalization layers
    - The feature maps are normalized with the mean and variance parameters of the feature maps (is this channel wide or feature map wide)
    - The feature maps are then scaled using the $W$ parameters and shifted using the mean of the $W$ vector

### Uses progressive growing for the GAN

- Starts at a small model that generates 4x4 images and iteratively adds layers to increase the output resolution up to 1024x1024

### Perceptual path length quality loss metric

- measures how smooth the semantic change is to the output image when changing the latent vector $Z$

- Takes the baseline of the progressively growing GAN again with an FID score on the FFHQ dataset
- Introduce tuning of the bi-linear up and down sampling
- Add mapping and styles
- Remove traditional input: instead of using a latent vector $Z$, they start with a constant value
- Add noise inputs
- Mixing regularization

### StyleGANv1 Artifacts

#### Droplet Artifacts

- commonly produces shiny blobs that look somewhat like water splotches on old photographic prints
- often show up at the interface between hair and the background
- attributed to the way the adaptive instance normalization is structured
- Can be used to distinguish between a real and generated image
- starts to appear at the 64x64 resolution scale and persists all the way up to the final 1024x1024 resolution

#### Phase Artifacts

- Features like mouths, eyes, and noses are fixed in place across generated images
- Appear as images are scaled up and walk along the latent space
- attributed to the structure of the progressive growing and having intermediate scales and intermediate low resolution maps that have to be used  to produce images that fool the discriminator

## Overview of StyleGANv2 Changes

- Start with baseline StyleGAN
- Add weight demodulation
- Add lazy regularization
- Add path length regularization
- No growing, new Generator and Discriminator architecture
- Large networks

### Removing Normalization Artifacts

#### Adaptive Instance Normalization (StyleGANv1):

$AdaIN(x,y) = \sigma(y) \ \left(\frac{x - \mu(x)}{\sigma(x)}\right) + \mu(y)$
- Introduced in [Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization](https://arxiv.org/abs/1703.06868)
  
- Used in StyleGANv1 to have the latent vector $W$ influence the features of the generator model
    - The latent vector $W$ controls the scaling $\sigma(y)$ and shifting $\mu(y)$ parameters of the normalization of the intermediate features maps of the generator
    - They are separating the normalization of the feature maps
    

#### StyleGANv2 Changes

- Separate out the addition of the Gaussian noise $B$ with the adaptive instance normalization layer
    - Reasoning: they might have conflicting effects
    
- Switch from using adaptive instance normalization to weight demodulation layers
    - Scale the weight parameters by using $w^\prime_{ijk}=s_i \cdot w_{ijk}$ where $s_i$ is  from the adaptive instance normalization from the $W$ latent vector
    - Demodulate it to assume that the features have unit variance (dividing all values by the standard deviation?)
        - $$
          w^{\prime\prime}_{ijk} = w^{\prime}_{ijk} / {\sqrt{\sum_{i,k}{w^\prime_{ijk}}^2+\epsilon}}
          $$
        
    - Change the weight parameters of the 3x3 kernel size convolutional layer instead of having an intermediate modulation and normalizing layer
    - removing weight demodulation results in strange artifacts when interpolating between images
    
- Add perceptual path length regularization metric to the loss function for the generator
    - Make sure changes in the latent vector $Z$ result in proportional semantic changes in the output image
    
    - Small changes in the latent vector $Z$ should result in smooth changes in the output image
    
    - referenced a paper that found ImageNet-trained CNNs are biased towards texture and that increasing shape bias improves accuracy and robustness
        - Traditional metrics relied on using pretrained image classifiers that are biased towards texture rather than shape detection
          
            [ImageNet-trained CNNs are biased towards texture; increasing shape bias improves accuracy and robustness](https://arxiv.org/abs/1811.12231)
            
            [Texture vs Shape: The bias in CNNs](https://towardsdatascience.com/texture-vs-shape-the-bias-in-cnns-5ee423edf8db)
            
        - [Stylized Imgaenet](https://github.com/rgeirhos/Stylized-ImageNet)
        
        - [texture-vs-shape](https://github.com/rgeirhos/texture-vs-shape)
        
    - $$
        \mathbb{E}_{w,y  N(0,I)} \left(||J^{T}_{w}y||_{2}-a\right)^{2}
        $$

    - Jacobian matrix $J_{w} = \partial g(w)/\partial w$
    ​        - sort of seeing the partial derivatives of output with respect to small changes in the latent vector that produces the images
    ​        - Use the small changes and the Jacobian matrix and multiply it by a random image $Y$ and is randomly sampled at each iteration
    ​        

    - Lazy regularization: only perform regularization every 16 steps

- Get away from progressive growing of the GAN
    - progressive growing requires a lot of hyper parameter search for the $\alpha$ value used to perform the element wise sum for the upsampled image
        - complicates training
    - Inspired by MSG-GAN: Multi-Scale Gradient for Generative Adversarial Networks
        - Enforces the intermediate feature maps in the generator by generating images from one (e.g. 4x4, 8x8, 16x16)  and providing them as additional features to the discriminator
          
            [MSG-GAN: Multi-Scale Gradients for Generative Adversarial Networks](https://arxiv.org/abs/1903.06048)
            
        - [BMSG-GAN](https://github.com/akanimax/BMSG-GAN)
        - StyleGANv2 does not use the exact same technique as MSG-GAN
            - Instead of feeding intermediate features from the generator to the discriminator, they have more of a ResNet style architecture
                - Not a traditional skip connection: They flatten each intermediate feature map (e.g. 256x256, 512x512) to 3-channel RGB format and feed those into the skip connection
            - Allows the model to focus more on the larger feature maps
    - Deep fake detection algorithm
        - Projects generated images back into the latent space
        - Goal is to find the latent $W$ vector that produced the generated image
            - This allows the the generated image to be attributed to the generator model
            - The deep fake detection algorithm cannot find the latent vector would reproduce the real images
        - Note: Might not be a robust solution for an actual deepfake detector
    

## Recap

### StyleGANv2 Changes

- Restructured Adaptive Instance Normalization
- Replaced Progressive Growing with skip connections
- Perceptual path length (PPL) normalization
- PPL norm results in easier latent space projection (Deepfake Detection)

### Training Speed Gains (1024x1024 resolution)

- StyleGANv1 → 37 images per second
- V2 Config E → 61 images per second (40% faster)
- V2 Config F → 31 images per second (larger networks)
- V2 Config F → 9 days on 8 Tesla V100 GPUs for FFHQ dataset, 13 days for LSUN CAR dataset



| Configuration | Resolution | Total kimg | 1 GPU | GPU Memory |
| --- | --- | --- | --- | --- |
| config-f | 1024x1024 | 25000 | 69d 23h | 13.3 GB |
| config-f | 1024x1024 | 10000 | 27d 23h | 13.3 GB |
| config-e | 1024x1024 | 25000 | 35d 11h | 8.6 GB |
| config-e | 1024x1024 | 10000 | 14d 4h | 8.6 GB |
| config-f | 256x256 | 25000 | 32d 13h | 6.4 GB |
| config-f | 256x256 | 10000 | 13d 0h | 6.4 GB |



**References:**

* Henry AI Labs Video: [StyleGANv2 Explained!](https://www.youtube.com/watch?v=u8qPvzk0AfY)
* StyleGANv2 Paper: [Analyzing and Improving the Image Quality of StyleGAN](https://arxiv.org/abs/1912.04958)



<!-- Cloudflare Web Analytics --><script defer src='https://static.cloudflareinsights.com/beacon.min.js' data-cf-beacon='{"token": "56b8d2f624604c4891327b3c0d9f6703"}'></script><!-- End Cloudflare Web Analytics -->