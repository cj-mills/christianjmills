---
title: Create a Shape Key Motion Graphic with the Blender Python API
layout: post
toc: false
comments: true
description: This post covers how to create a simple shape-key motion graphic in Blender using the Python API.
categories: [blender, python, tutorial]
hide: true
permalink: /:title/
search_exclude: false
---

* [Introduction](#introduction)
* [Import Dependencies](import-dependencies)
* [Define Helper Functions](#define-helper-functions)
* [Create and Position a New Camera](#create-and-position-a-new-camera)
* [Create a Material With an Emission Shader](#create-a-material-with-an-emission-shader)
* [Create a New Plane With the Material](#create-a-new-plane-with-the-material)
* [Cut Out a Center Square From the Plane](#cut-out-a-center-square-from-the-plane)
* [Add Shape Keys to Deform the Plane](#add-shape-keys-to-deform-the-plane)
* [Add Keyframes to Animate the Plane](#add-keyframes-to-animate-the-plane)
* [Conclusion](#conclusion)

## Introduction

I decided to recreate this [short tutorial](https://www.youtube.com/watch?v=N3FZcFk-dZA&list=PLGKIkAXk1OeTti1rRVTJF_9_JCC3zY0bh&index=11) from YouTube to practice using the Blender Python API. This post goes through the code I came up with to replicate the tutorial.

![shape_key_mg](..\images\shape-key-motion-graphic-bpy\shape_key_mg.gif)



## Import Dependencies





![import-dependencies](..\images\shape-key-motion-graphic-bpy\import-dependencies.png)



## Define Helper Functions



![define-helper-functions](..\images\shape-key-motion-graphic-bpy\define-helper-functions.png)







## Create and Position a New Camera



![create-camera](..\images\shape-key-motion-graphic-bpy\create-camera.png)





## Create a Material With an Emission Shader



![create-emission-material](..\images\shape-key-motion-graphic-bpy\create-emission-material.png)



## Create a New Plane With the Material



![create-plane](..\images\shape-key-motion-graphic-bpy\create-plane.png)



## Cut Out a Center Square From the Plane



![cut-out-center-square](..\images\shape-key-motion-graphic-bpy\cut-out-center-square.png)



## Add Shape Keys to Deform the Plane



![add-shape-keys](..\images\shape-key-motion-graphic-bpy\add-shape-keys.png)



## Add Keyframes to Animate the Plane



![add-keyframes](..\images\shape-key-motion-graphic-bpy\add-keyframes.png)



## Conclusion

