---
categories:
- recap
date: 2021-4-27
description: A summary of what I worked on today.
hide: false
layout: post
search_exclude: false
title: Daily Recap
toc: false

---

* [Unity GPU Targeting](#unity-gpu-targeting)

  

## Unity GPU Targeting

I spent some time today trying to get Unity to use the integrated graphics on my CPU instead of my discrete graphics card. The goal was to see if the Barracuda library would perform inference on the integrated graphics, because of reasons. I was a bit surprised that the Unity Editor did not have a simple menu option to select a rendering device. In Blender, you can manually select which device is used to render scenes.

As one might imagine, most of the questions online asked how to get Unity to use a discrete graphics card instead of the integrated graphics. There seems to be a command line argument, `-force-device-index`, to make the Editor use a particular GPU device. However, it appears to be for macOS only and had no effect in Windows. I also could not find a setting in the Nvidia Control Panel to completely prevent Unity from using the graphics card.

The only way that I have been able to get the Barracuda library to execute models on the integrated graphics is to disable the Nvidia card in the Windows Device Manager. This method is not too much of a hassle as I did not have to restart the computer for it to take effect. 

I still find it a bit odd that their is no menu option in the Unity Editor. I wonder if Unreal Engine allows you to do this.

