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
* [Import Dependencies](#import-dependencies)
* [Define Helper Functions](#define-helper-functions)
* [Set up Scene](#set-up-scene)
* [Create and Position Camera](#create-and-position-camera)
* [Create Material With Emission Shader](#create-material-with-emission-shader)
* [Create a Plane With the Material](#create-a-plane-with-the-material)
* [Cut Out Center From Plane](#cut-out-center-from-plane)
* [Add Shape Keys](#add-shape-keys)
* [Add Keyframes](#add-keyframes)
* [Conclusion](#conclusion)

## Introduction

I decided to recreate this [short tutorial](https://www.youtube.com/watch?v=N3FZcFk-dZA&list=PLGKIkAXk1OeTti1rRVTJF_9_JCC3zY0bh&index=11) from YouTube to practice using the Blender Python API. This post goes through the code I came up with to replicate the tutorial plus some small additions.

![shape_key_mg](..\images\shape-key-motion-graphic-bpy\shape_key_mg.gif)



## Import Dependencies

The only dependencies strictly required for this tutorial are `bpy` and `bmesh`. The `bpy` package is the base API for Blender and the `bmesh` module provides access to Blender's internal mesh editing API. I also added the `math` module from the Python Standard Library for one of my helper functions. 

![import-dependencies](..\images\shape-key-motion-graphic-bpy\import-dependencies.png)



## Define Helper Functions

I made some wrapper functions for the standard location, rotation, and scale transformations as well as getting the name of the active object.

You can get the name of the active object with `bpy.context.active_object.name` which gets a bit irritating to write out multiple times.

The three standard transformations can be accessed for individual objects with the following:

* `bpy.data.objects["object_name"].location`
* `bpy.data.objects["object_name"].rotation_euler`
* `bpy.data.objects["object_name"].scale`

I also made a function to empty the default collection so that nothing gets duplicated. Collections can be accessed with `bpy.data.collections["collection_name"]` or `bpy.data.collections[index]`.

Finally, I made a function to easily add sequences of keyframes to a given object. The function uses the built-in `setattr()` method to set the desired value for the target object and uses the `object.keyframe_insert()` method to add the keyframe. 

![define-helper-functions](..\images\shape-key-motion-graphic-bpy\define-helper-functions_2.png)



## Set up Scene

The first thing I do is set the Color Management property, View Transform, from the default value of Filmic to Standard. This setting can be accessed at `bpy.data.scenes["Scene"].view_settings.view_transform`.

Next, I set the background to the desired color. In my case, it's pure black. The background color is stored in `bpy.data.worlds['World'].node_tree.nodes["Background"].inputs[0].default_value`.

The last setup step is to clear any objects added from the last time the script was run with the `clear_collection()` function.

![set-up-scene](..\images\shape-key-motion-graphic-bpy\set-up-scene.png)



## Create and Position Camera

Cameras can be added using the `bpy.ops.object.camera_add()` method. I then positioned the camera using the wrapper functions I defined earlier.

![create-camera](..\images\shape-key-motion-graphic-bpy\create-camera.png)





## Create Material With Emission Shader

I decided to add some color to the motion graphic so I needed to create a new material. It is recommended to check if the material exists before trying to create it. This can be done in one line as shown below.

`material = bpy.data.materials.get(material_name) or bpy.data.materials.new(material_name)`

Since there's is no light I'll add an `Emission` shader. This requires enabling nodes for the material with `material.user_nodes = True`. 

Next, I remove the default `Principled_BSDF` node as well as any `Emission` nodes from earlier runs. Nodes can be removed using the `material.node_tree.nodes.remove()` method.

The `Emission` node needs to be linked to the first slot in the `Material Output` node. Nodes are linked using the `material.node_tree.links.new()` method.



![create-emission-material](..\images\shape-key-motion-graphic-bpy\create-emission-material.png)



## Create a Plane With the Material

The main object of the above motion graphic is a plain. Plains can be added using the `bpy.ops.mesh.primitive_plane_add()` method. 

I then assign the previously created material to the plane. Materials can be added to an object with `object.data.materials.append(material)`.

![create-plane](..\images\shape-key-motion-graphic-bpy\create-plane.png)



## Cut Out Center From Plane

The next step is to cut the square hole in the plane like in the above Gif. This requires modifying the mesh for the plane. 

Mesh data for the currently selected object is stored at `bpy.context.object.data`.

To edit the mesh, we need to get a BMesh representation. We first create an empty BMesh with `bm = bmesh.new()` and then fill it with the mesh using `bm.from_mesh(mesh)`.

We can make the square by adding a new inset to the plane using the `bmesh.ops.inset_individual()` method. Then, we delete the new face that gets created with `bmesh.ops.delete()`. 

The mesh then needs to be updated with these alterations using `bm.to_mesh(mesh)`. Finally, we need to free the BMesh representation we created with `bm.free()`.

![cut-out-center-square](..\images\shape-key-motion-graphic-bpy\cut-out-center-square_2.png)



## Add Shape Keys

To deform the plane, we need to access its vertices. We can do this in edit mode with the `bmesh` module as well. Unlike the tutorial video, I just set the positions for the inner vertices directly. It took some trial and error to determine the correct indices for the inner vertices.

![add-shape-keys](..\images\shape-key-motion-graphic-bpy\add-shape-key-1.png)





![add-shape-key-2](..\images\shape-key-motion-graphic-bpy\add-shape-key-2.png)





## Add Keyframes

Before adding the keyframes, I set the render frame rate as well the start and end frames for the scene. The helper function I made makes it a lot easier to manage keyframes as it lets me organize the updates sequentially. All the values and target frames can be stored in lists. 

![add-keyframes](..\images\shape-key-motion-graphic-bpy\add-keyframes.png)



## Conclusion

