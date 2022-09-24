---
categories:
- unity
- project
- log
date: '2020-09-16'
description: The journey so far.
hide: false
layout: post
search_exclude: false
title: Barracuda Pose Estimation Project Log Pt. 1
toc: false

aliases:
- /Barracuda-Pose-Estimation-Project-Log-1/
---

### Update 3/29/2021: [Barracuda PoseNet Tutorial](../../barracuda-posenet-tutorial/part-1/)

### Update 7/31/2021: [Barracuda PoseNet Tutorial 2nd Edition](../../barracuda-posenet-tutorial-v2/part-1/)

* [Background](#background)
* [The Project](#the-project)
* [Current Progress](#current-progress)
* [Next Steps](#next-steps)
* [Conclusion](#conclusion)



## Background

I've been learning how to get pose estimation working in Unity using their new inference library called [Barracuda](https://docs.unity3d.com/Packages/com.unity.barracuda@1.0/manual/index.html). I think the name might be a reference to Nvidia's CUDA library. Apparently, barracuda are really fast swimmers so maybe inference speed is a priority for the library as well. I have difficulty coming up with names for things so I'm always looking for new methods. 

Anyways, I've made some decent progress and have actually managed to get a basic proof of concept working. I figured now would be a good time to start a blog and actually track my progress. I've never been good about keeping notes when working on projects. Hopefully, this project will provide sufficient motivation for changing that and get me to write more often (or at all really).

I've already noticed a conflict between the desire to actually work on the project and writing about it. Maybe, I should try writing posts as I work on projects in the future. Well, I guess I should keep that in mind for later. For this post, I just want to outline the project and what I've accomplished so far.

## The project

I'm really interested in the potential applications for mapping a user's movements to a virtual character in real-time. Traditionally, this required specialized hardware like motion capture suits. Fortunately, there are deep learning models that now let you accomplish the same thing with a regular webcam and sufficient computing power. These models perform what is called pose estimation. 

Pose estimation is a technique where a model predicts the location of a person or object in an image or video. When tracking humans, a model is typically trained to predict the locations of key points on a person's body (e.g. joints, nose, eyes, etc.). You can learn more about pose estimation [here](https://www.fritz.ai/pose-estimation/). I want to use these types of models inside a Unity application so I can combine the with all other [features](https://unity.com/products/unity-platform) included in a modern real-time 2D/3D development platform.

### Project Goals

My goals for this project are evolving as I discover more of what's possible. As a result, I don't really a have a "definition of done". I'm just seeing where it takes me for now. However, the basic goal for this project is to use the key point locations predicted by a pose estimation model to control a virtual character. For example, when the user moves their arm, the virtual character's arm should move accordingly. Below are some more specific requirements that I'm currently working towards.

#### Requirements:

* Scale the estimated pose that the model outputs to account for differences in size between the target character sprite/model and the size of the image being fed into the model. 
* Handle differences in body proportions between the user and the character. 
* Separate the estimated pose from the user's location in the camera frame.
* Smooth out any choppiness in character movement due to updates in the pose predictions made by the model.
* Keep the frame rate high enough that the application isn't jarring for the user.



## Current Progress

### Proof of Concept

As mentioned previously, I have actually gotten a basic example working in Unity. I've also learned how to leverage compute shaders to perform the preprocessing steps on the GPU. 

### Some Weak Points

Figuring out how to process inputs and outputs for neural networks efficiently inside Unity has been the most irritating part of the project so far. It's given me a new appreciation for all the great data science resources available in the Python ecosystem. Basic things like getting slices of arrays and matrices are such a pain in C# compared to Python. It really highlights the need to identify and learn how to leverage the strengths and weaknesses of the tools your working with. Compute shaders are definitely one of the more important strengths to leverage in Unity.

### Some Strong Points

Compute shaders are awesome for doing the same thing to every element in a data structure on the GPU. They're actually what the Barracuda library uses. There is a bit of a learning process for figuring out how to actually get the data you want to the shader, access the data within the shader, and how to get the output back. It took a lot of googling to figure out but it's worth the effort. 

So far, I've managed to get my whole preprocessing pipeline to run on the GPU. I can load an input image onto the GPU, crop it, resize it, normalize the values, and execute the model all without sending data back and forth between the CPU and GPU. Aside from being way faster, this also has the benefit of completely freeing up the CPU to do other things.

### Current Challenges

I still need to figure out if I can get the post processing done on the GPU. That's my last remaining bottleneck for fully utilizing my GPU during runtime. It also highlights just how much overhead there is when transferring data from the GPU back to the CPU. When testing different methods for preprocessing, I discovered that the simple act of downloading a tensor from the GPU to the CPU took longer than it did to actually run the model. 

Unfortunately, the Barracuda library does not make it intuitive for extracting information from a model's output. This is an area that could really use improvement. My current method for processing the output in Unity is nested for loops. Part of the reason is that tensors can't be accessed outside of the main thread. If I want to leverage parallel processing, I need to download the tensor data to an array on the CPU. The overhead from downloading the data seems to wipe out any performance benefits. 

Since I haven't figured out how to efficiently access slices of arrays, I need iterate through the whole output tensor to find the most likely locations for key points in the input image. That's not a big deal if the output tensor is small, but quickly becomes a problem when using larger input images. Unfortunately, the model architecture I'm using seems to require large input images to get more accurate pose estimations.

### Potential Roadblocks

Something else this project has highlighted is how much of a pain it still is to get machine learning models from the training environment to arbitrary production applications. Unity requires your model to either be in the native Barracuda or ONNX format. I decided to start with a pretrained TensorFlow model while I get the hang of using Barracuda. That meant I needed to convert the model to ONNX before I could begin working with it in Unity. 

Unfortunately, TensorFlow does not contain any built-in methods for exporting trained models to ONNX. They seem to prefer that you stay within their ecosystem. However, it also doesn't contain a complete set of methods for converting between different TensorFlow libraries. For example, they provide a method to convert standard TensorFlow models to either TensorFlow Lite or Tensorflow.js format but not the other way around. This created a bit of a road block, since TensorFlow only provides tflite and TFJS versions of the pretrained PoseNet model I'm using. That combined with the lack of built in support for exporting to ONNX means that there aren't any officially supported ways to get an arbitrary TensorFlow model into Unity. That's a bit inconvenient since I wanted to start with a pretrained model so that I wouldn't need to spend a bunch of time training my own model before even getting into Unity.

### Potential Solutions

I eventually found a third-party library to convert a pretrained TFJS model to the standard TensorFlow SavedModel format. I then had to use another third-party library to convert the converted TensorFlow model into ONNX. It might seem like these third-party libraries completely resolve the missing functionality in TensorFlow. Unfortunately, this method requires that all third-party libraries used, implement support for whatever neural network layers are used in the model you want to convert. The PoseNet architecture is fully supported, but a lot of the the newer pretrained models released for TensorFlow contain new types of layers that are not yet supported by these libraries. Even if these libraries did support these new layer types, Unity would likely still need to implement support for them in Barracuda. This all introduced a bunch work that I had to get done before I could even begin making a proof of concept in Unity.

While I was looking for solutions to the challenges in the previous paragraph, I came across some methods that others developed for manually converting models from one library to another. It involves using a neural network analysis tool to construct a JSON file that contains the network topology and then using that to construct the same topology using the target library. You then need to iterate through the trained model to get the weights for each layer and assign them to the appropriate layer in the new model. 

I haven't tried out this method yet, but I'll be sure to make a post describing how that goes. It definitely isn't ideal, and it probably requires that all the layer types supported in the source library be implemented in the target library. For me, the source library would likely be a TensorFlow library and the target library would be PyTorch. PyTorch includes built in support for exporting models to ONNX so that should help streamline that part of the process. Even if I can't get the trained weights to work in PyTorch, I should at least be able to get a model that I can then train on my own without having to work out the network topology from some research paper. I plan to do this after I get further along inside Unity anyways.

## Next Steps

I want to see if I can get the post processing steps done on the GPU. I'd still need to get the processed output to the CPU to actually do anything with it. However, there would be much less data that would need to be transferred. That should hopefully minimize any overhead.

The next step is to figure out how to actually map the predicted key point locations to a character model/sprite. Before that, I'll need to learn more about character animation in Unity. The pose estimation model I'm currently using only supports predictions in 2D space. Therefore, I'll start with 2D characters to keep things simple.

Once I figure out how to map the model outputs to the virtual character, I'll see if I can improve the model's accuracy. The pretrained PoseNet models provided by TensorFlow work well enough but I doubt they're as good as they could be. I also plan on training the models using custom datasets to see if I can improve the model's performance. I've been learning about how you can make synthetic datasets using tools such as Blender and I'm really curious to see if I can use Blender's built in automation capabilities to make a high quality dataset for pose estimation as well as other computer vision applications.



## Conclusion

That's all for now. I'm feel like I'm forgetting to mention a bunch of stuff. It's definitely not ideal to write about something weeks after the fact. Fortunately, I plan to make separate posts that go into further detail anyways. Hopefully those will fill in any blanks. I'll probably update this post with images and other stuff as well. That way it won't be just a wall of text. 



