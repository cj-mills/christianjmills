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

I decided to recreate this [short tutorial](https://www.youtube.com/watch?v=N3FZcFk-dZA&list=PLGKIkAXk1OeTti1rRVTJF_9_JCC3zY0bh&index=11) from YouTube to practice using the Blender Python API. This post goes through the code I came up with to replicate the tutorial with some small additions.

![shape_key_mg](..\images\shape-key-motion-graphic-bpy\shape_key_mg.gif)



## Import Dependencies

The only dependencies strictly required for this tutorial are `bpy` and `bmesh`. The `bpy` package is the base API for Blender and the `bmesh` module provides access to Blender's internal mesh editing API. I also added the `math` module from the Python Standard Library for one of my helper functions. 

![import-dependencies](..\images\shape-key-motion-graphic-bpy\import-dependencies.png)



## Define Helper Functions

I made some wrapper functions for the standard location, rotation, and scale transformations. These can be updated for individual objects with the following:

* `bpy.data.objects["object_name"].location`
* `bpy.data.objects["object_name"].rotation_euler`
* `bpy.data.objects["object_name"].scale`

I also made a couple functions to reset the scene. Specifically, the `reset_scene()` function sets the color management view transform to Standard, empties the default collection, and sets the background color. I run the `reset_scene()` method at the start of the script so that nothing gets duplicated. 

Finally, I made a function to easily add sequences of keyframes to a given object. The function uses the built-in `setattr()` method to set the desired value for the target object and uses the `object.keyframe_insert()` method to add the keyframe. 

![define-helper-functions](..\images\shape-key-motion-graphic-bpy\define-helper-functions.png)



## Create and Position a New Camera

After resetting the scene, the first thing I do is set up the camera. Cameras can be added using the `bpy.ops.object.camera_add()` method.

![create-camera](..\images\shape-key-motion-graphic-bpy\create-camera.png)





## Create a Material With an Emission Shader

I decided to add some color to the motion graphic so I needed to create a new material. It is recommended to check if the material exists before trying to create it. Since there's is no light I'll add an `Emission` shader. This requires enabling nodes for the material first. Next, I remove the default `Principled_BSDF` node as well as any `Emission` nodes from earlier runs. The `Emission` node needs to be linked to the first slot in the `Material Output` node.

![create-emission-material](..\images\shape-key-motion-graphic-bpy\create-emission-material.png)



## Create a New Plane With the Material

The main object of the above motion graphic is a plain. Plains can be added using the `bpy.ops.mesh.primitive_plane_add()` method. I then assign the previously created material to the plane.

![create-plane](..\images\shape-key-motion-graphic-bpy\create-plane.png)



## Cut Out a Center Square From the Plane

The next step is to cut the square whole in the plane like in the above Gif. This requires going into edit mode and modifying the mesh for the plane. Meshes can be edited using the `bmesh` module. We can make the square by adding a new inset to the plane and then deleting the new face that gets created as a result. The mesh then needs to be updated with these alterations.

![cut-out-center-square](..\images\shape-key-motion-graphic-bpy\cut-out-center-square.png)



## Add Shape Keys to Deform the Plane

To deform the plane, we need to access its vertices. We can do this in edit mode with the `bmesh` module as well. Unlike the tutorial video, I just set the positions for the inner vertices directly. It took some trial and error to determine the correct indices for the inner vertices.

![add-shape-keys](..\images\shape-key-motion-graphic-bpy\add-shape-keys.png)



## Add Keyframes to Animate the Plane

Before adding the keyframes, I set the render frame rate as well the start and end frames for the scene. The helper function I made makes it a lot easier to manage keyframes as it lets me organize the updates sequentially. All the values and target frames can be stored in lists. 

![add-keyframes](..\images\shape-key-motion-graphic-bpy\add-keyframes.png)



## Conclusion

