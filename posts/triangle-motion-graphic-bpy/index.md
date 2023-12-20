---
title: Create a Triangle Motion Graphic with the Blender Python API
date: 3-27-2021
title-block-categories: true
hide: false
search_exclude: false
comments:
  utterances:
    repo: cj-mills/christianjmills
description: This post covers how to create a simple triangle motion graphic in Blender
  using the Python API.
categories: [blender, python, tutorial]

aliases:
- /Create-a-Triangle-Motion-Graphic-With-the-Blender-Python-API/

---


* [Introduction](#introduction)
* [Import Dependencies](#import-dependencies)
* [Define Helper Functions](#define-helper-functions)
* [Set up Scene](#set-up-scene)
* [Create and Position Camera](#create-and-position-camera)
* [Create Material With Emission Shader](#create-material-with-emission-shader)
* [Create a Cone With the Material](#create-a-cone-with-the-material)
* [Turn the Cone Into a Triangle](#turn-the-cone-into-a-triangle)
* [Duplicate the Triangle](#duplicate-the-triangle)
* [Create the Holdout Material](#create-holdout-material)
* [Add Keyframes](#add-keyframes)
* [Conclusion](#conclusion)



## Introduction

I decided to recreate this [short tutorial](https://www.youtube.com/watch?v=xeH41Tz1zGI&list=PLGKIkAXk1OeTti1rRVTJF_9_JCC3zY0bh&index=27) from YouTube to practice using the Blender Python API. This post goes through the code I came up with to replicate the tutorial.

![](./images/triangle-mg.gif){fig-align="center"}



## Import Dependencies

The only dependencies strictly required for this tutorial are `bpy` and `bmesh`. The `bpy` package is the base API for Blender and the `bmesh` module provides access to Blender's internal mesh editing API. I also used the `math` module from the Python Standard Library for one of my helper functions. 

```python
# The Blender Python API
import bpy
# Gives access to Blender's internal mesh editing API
import bmesh
# Provides access to mathematical functions
import math
```



## Define Helper Functions

I made some wrapper functions for the standard location, rotation, and scale transformations as well as getting the name of the active object.

You can get the name of the active object with `bpy.context.active_object.name`.

The three standard transformations can be accessed for individual objects with the following:

* `bpy.data.objects["object_name"].location`
* `bpy.data.objects["object_name"].rotation_euler`
* `bpy.data.objects["object_name"].scale`

I also made a function to empty the default collection so that nothing gets duplicated. Collections can be accessed with `bpy.data.collections["collection_name"]` or `bpy.data.collections[index]`.

Lastly, I made a function to easily add sequences of keyframes to a given object. The function uses the built-in `setattr()` method to set the desired value for the target object and uses the `object.keyframe_insert()` method to add the keyframe. 

```python
def get_name():
    """Get the name for the currently active object"""
    return bpy.context.active_object.name

def degToRadian(angle):
    """Convert angle from degrees to radians"""
    return angle*(math.pi/180)

def move_obj(name, coords):
    """Set object location to the specified coordinates"""
    bpy.data.objects[name].location = coords

def rotate_obj(name, angles):
    """Set object rotation to the specified angles"""
    rotation = [degToRadian(angle) for angle in angles]
    bpy.data.objects[name].rotation_euler = rotation

def scale_obj(name, scale):
    """Set object scale"""
    bpy.data.objects[name].scale = scale

def clear_collection(collection):
    """Remove everything from the specified collection"""
    for obj in collection.objects:
        bpy.data.objects.remove(obj)
        
def add_keyframe_sequence(obj, attribute, values, frames):
    """Add a sequence of keyframes for an object"""
    for v, f in zip(values, frames):
        setattr(obj, attribute, v)
        obj.keyframe_insert(data_path=attribute, frame=f)
```



## Set up Scene

The first thing I do is set the Color Management property, View Transform, from the default value of `Filmic` to `Standard`. This setting can be accessed at `bpy.data.scenes["Scene"].view_settings.view_transform`.

This tutorial requires transparency to be enabled. This can be done by setting `bpy.data.scenes['Scene'].render.film_transparent` to `True`.

Next, I set the background to the desired color. In my case, it's pure black. The background color is stored in `bpy.data.worlds['World'].node_tree.nodes["Background"].inputs[0].default_value`.

The last setup step is to clear any objects added from the last time the script was run with the `clear_collection()` function.

```python
"""Set up the scene"""
# Set View Transform to Standard
bpy.data.scenes["Scene"].view_settings.view_transform = "Standard"
# Enable transparency
bpy.data.scenes['Scene'].render.film_transparent = True
# Set the Background color to pure black
bpy.data.worlds['World'].node_tree.nodes["Background"].inputs[0].default_value = (0, 0, 0, 1)
# Clear Collection
clear_collection(bpy.data.collections[0])
```



## Create and Position Camera

Cameras can be added using the `bpy.ops.object.camera_add()` method. I then positioned the camera using the wrapper functions I defined earlier.

```python
"""Create and position a new camera"""
# Create a new camera
bpy.ops.object.camera_add()
# Get the name of the current object, the camera
name = get_name()
# Move the camera
move_obj(name, [0, -8, 0])
# Rotate the camera
rotate_obj(name, [90, 0, 0])
# Set camera to orthographic
bpy.context.active_object.data.type = "ORTHO"
```



## Create Material With Emission Shader

I decided to add some color to the motion graphic so I needed to create a new material. It is recommended to check if the material exists before trying to create it. This can be done in one line as shown below.

`material = bpy.data.materials.get(material_name) or bpy.data.materials.new(material_name)`

Since there's is no light, I'll add an `Emission` shader. This requires enabling nodes for the material with `material.use_nodes = True`. 

Next, I remove the default `Principled_BSDF` node as well as any `Emission` nodes from earlier runs. Nodes can be removed using the `material.node_tree.nodes.remove()` method.

The `Emission` node needs to be linked to the first slot in the `Material Output` node. Nodes are linked using the `material.node_tree.links.new()` method.

```python
"""Create a material with an Emission Shader"""
# Create a material named "Material" if it does not exist
mat_name = "Material"
mat = bpy.data.materials.get(mat_name) or bpy.data.materials.new(mat_name)

# Enable nodes for the material
mat.use_nodes = True
# Get a reference to the material's node tree
nodes = mat.node_tree.nodes

# Remove the 'Principled BSDF' node if there is one
if (nodes.get('Principled BSDF') is not None):
    nodes.remove(nodes.get('Principled BSDF'))
    
# Remove the 'Emission' node if there is one
if (nodes.get('Emission') is not None):
    nodes.remove(nodes.get('Emission'))

# Get a reference to the material's output node
mat_output = nodes.get('Material Output')
# Create a new Emission shader
emission = nodes.new('ShaderNodeEmission')
# Set the color for the Emission shader
emission.inputs['Color'].default_value = (0, 0.5, 1, 1)
# Link the Emission shader to the Surface value of the output node
mat.node_tree.links.new(mat_output.inputs[0], emission.outputs[0])
```



## Create a Cone With the Material

The motion graphic is made of two triangles with one being a duplicate of the other. The original triangle started off as a cone with `3` vertices. Cones can be added using the `bpy.ops.mesh.primitive_cone_add()` method. 

I then assign the previously created material to the cone. Materials can be added to an object with `object.data.materials.append(material)`.

```python
"""Create a cone with the Emission material"""
# Create a new cone with 3 vertices
bpy.ops.mesh.primitive_cone_add(vertices=3)

# Get the name of the new cone
name = get_name()
# Rotate the cone
rotate_obj(name, [90, 180, 0])
# Move cone to origin
move_obj(name, [0, 0, -0.25])
# Reduce the size of the cone
scale = 0.75
scale_obj(name, [scale]*3)

# Get a reference to the currently active objecct
cone = bpy.context.active_object
# Assign the material with the Emission shader to the cone
if cone.data.materials:
    cone.data.materials[0] = mat
else:
    cone.data.materials.append(mat)
```



## Turn the Cone Into a Triangle

The next step is to remove the tip of the cone. This requires modifying its mesh. Mesh data for the currently selected object is stored at `bpy.context.object.data`.

To edit the mesh, we need to get a BMesh representation. We first create an empty BMesh with `bm = bmesh.new()` and then fill it with the mesh using `bm.from_mesh(mesh)`.

We can delete vertices with the `bmesh.ops.delete()` and setting the `context` to `VERTS`. 

The mesh then needs to be updated with these alterations using `bm.to_mesh(mesh)`. We need to free the BMesh representation we created with `bm.free()`.

Finally, I reset the origin of the triangle with `bpy.ops.object.origin_set()`.

```python
"""Turn the cone into a triangle"""
# Get the mesh for the cone
mesh = bpy.context.object.data

# Get a BMesh representation from current mesh in edit mode
bm = bmesh.new()
bm.from_mesh(mesh)

# Get a list of vertices
verts = [v for v in bm.verts]
# Delete the middle face
bmesh.ops.delete(bm, geom=[verts[3]], context='VERTS')
# Update the mesh
bm.to_mesh(mesh)
# Free the Bmesh representation and prevent further access
bm.free()

# Set the origin to geometry
bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY")
```



## Duplicate the Triangle

We can make the second triangle with `bpy.ops.object.duplicate()`.

```python
"""Duplicate the triangle"""
# Duplicate the current object
bpy.ops.object.duplicate()
# Get the name of the current object, the triangle
name = get_name()
# Move the duplicate in front of the original
move_obj(name, [0, -0.05, -0.25])
```



## Create the Holdout Material

We need to add a `Holdout` material to the second triangle so we can see through anything behind it. The process is the same as adding the `Emission` shader.

```python
"""Create a new material that will make objects behind it transparent"""
# Create a material named "X-ray" if it does not exist
mat_name = "X-ray"
mat = bpy.data.materials.get(mat_name) or bpy.data.materials.new(mat_name)

# Enable nodes for the material
mat.use_nodes = True
# Get a reference to the material's node tree
nodes = mat.node_tree.nodes

# Remove the 'Principled BSDF' node if there is one
if (nodes.get('Principled BSDF') is not None):
    nodes.remove(nodes.get('Principled BSDF'))
    
# Remove the 'Holdout' node if there is one
if (nodes.get('Holdout') is not None):
    nodes.remove(nodes.get('Holdout'))

# Get a reference to the material's output node
mat_output = nodes.get('Material Output')
# Create a new Holdout shader
holdout = nodes.new('ShaderNodeHoldout')
# Link the Holdout shader to the Surface value of the output node
mat.node_tree.links.new(mat_output.inputs[0], holdout.outputs[0])

# Assign the material with the Holdout shader to the currently active object
bpy.context.active_object.data.materials[0] = mat
```



## Add Keyframes

Before adding the keyframes, I set the render frame rate as well the start and end frames for the scene. The frame rate is stored at `bpy.context.scene.render.fps`.

The start and end frames are stored in `bpy.data.scenes['Scene'].frame_start` and `bpy.data.scenes['Scene'].frame_end` respectively. 

```python
"""Set up for animation"""
# Set the render frame rate to 60
bpy.context.scene.render.fps = 60

# Set the start frame to 0
bpy.data.scenes['Scene'].frame_start = 0
# Set the end frame to 250
bpy.data.scenes['Scene'].frame_end = 250
# Set the current frame to 0
bpy.data.scenes['Scene'].frame_current = 0
```



### X-ray Triangle

We only need to animate the rotation and scale for the x-ray triangle.

```python
"""Add keyframes to animate the X-ray triangle"""
# Get the name of the current object
xray_triangle = bpy.context.active_object
# Set values for keyframes
values = [[degToRadian(angle) for angle in [90, 180, 0]],
          [degToRadian(angle) for angle in [90, 145, 0]],
          [degToRadian(angle) for angle in [90, 90, 0]],
          [degToRadian(angle) for angle in [90, 180, 0]]]
# Set the frames for keyframes
frames = [20, 70, 120, 250]
# Add keyframes for the rotation of the xray_triangle
add_keyframe_sequence(xray_triangle, 'rotation_euler', values, frames)

# Set values for keyframes
values = [[scale]*3, [0.5]*3, [0]*3, [scale]*3]
# Set the frames for keyframes
frames = [10, 60, 100, 250]
# Add keyframes for the scale of the xray_triangle
add_keyframe_sequence(xray_triangle, 'scale', values, frames)
```



## Conclusion

This tutorial did not require learning any new parts of the API after the last tutorial I [replicated](../shape-key-motion-graphic-bpy/). I guess in that sense, it was a waste of time. However, I still enjoyed working on it and I like the resulting motion graphic.



**Tutorial Resources:** [GitHub Repository](https://github.com/cj-mills/Triangle-Motion-Graphic-Blender-API)



