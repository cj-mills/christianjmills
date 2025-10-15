---
title: "Code Walkthrough: Unity Barracuda Inference Base Package"
date: 2023-5-5
image: /images/empty.gif
hide: false
search_exclude: false
categories: [unity, walkthrough]
description: "Walk through the code for the Unity Barracuda Inference Base package, which provides a foundation for performing inference with the Barracuda inference library."

twitter-card:
  creator: "@cdotjdotmills"
  site: "@cdotjdotmills"
  image: /images/default-preview-image-black.png
open-graph:
  image: /images/default-preview-image-black.png
---





* [Introduction](#introduction)
* [Package Overview](#package-overview)
* [Code Explanation](#code-explanation)
* [Conclusion](#conclusion)




## Introduction

The [Barracuda Inference Base](https://github.com/cj-mills/unity-barracuda-inference-base) package provides a foundation for performing inference with the Barracuda  inference library. It includes a flexible base class to extend with  task-specific packages. [Barracuda](https://docs.unity3d.com/Packages/com.unity.barracuda@3.0/manual/index.html) is a lightweight cross-platform [neural network](https://en.wikipedia.org/wiki/Neural_network) inference library for Unity. 



I use the Barracuda inference library in multiple tutorials. This package makes that shared functionality more modular and reusable, allowing me to streamline my tutorial content. Here are some demo videos from projects that extend this package.



**Image Classification**

![](./videos/barracuda-inference-image-classification-demo.mp4){fig-align="center"}



**Object Detection**

![](./videos/barracuda-inference-yolox-demo.mp4){fig-align="center"}



**Pose Estimation**

![](./videos/barracuda-inference-posenet-demo.mp4){fig-align="center"}



In this post, I'll walk through the package code, providing a solid understanding of its components and their roles.




## Package Overview

The package contains two C# scripts.

1. `BarracudaModelRunner.cs`: This script provides an abstract class for running Barracuda neural network models in Unity.
2. `AddCustomDefineSymbol.cs`: An Editor script that automatically adds a custom scripting define symbol to the project after the package installs.



## Code Explanation

In this section, we will delve deeper into the Barracuda Inference Base package by examining the purpose and functionality of each C# script.



### `BarracudaModelRunner.cs`

This script provides an abstract class for running Barracuda neural network models in Unity. This class serves as a base class for others that perform inference using Barracuda to inherit from.



The complete code is available on GitHub at the link below.

- [BarracudaModelRunner.cs](https://github.com/cj-mills/unity-barracuda-inference-base/blob/main/Runtime/Scripts/BarracudaModelRunner.cs)



#### Model Assets

These are serialized fields for the neural network model, channel order preference, and the execution backend.

```c#
[Header("Model Assets")]
[Tooltip("The neural network model")]
[SerializeField] protected NNModel model;
[Tooltip("Option to order tensor data channels first (EXPERIMENTAL)")]
[SerializeField] private bool useNCHW = true;
[Tooltip("Execution backend for the model")]
[SerializeField] protected WorkerFactory.Type workerType = WorkerFactory.Type.Auto;
```



#### `ModelBuilder` and `IWorker`

These are Instances of `ModelBuilder` and `IWorker` for creating and executing models.

```c#
protected ModelBuilder modelBuilder;
protected IWorker engine;
```



#### `Start`

The `Start()` method runs when the script is enabled. It loads and prepares the model and initializes the engine.

```c#
protected virtual void Start()
{
    LoadAndPrepareModel();
    InitializeEngine();
}
```



#### `LoadAndPrepareModel`

This method loads and prepares the model for execution. Child classes can override this method to apply custom modifications to the model.

```c#
/// <summary>
/// Load and prepare the model for execution.
/// Override this method to apply custom modifications to the model.
/// </summary>
protected virtual void LoadAndPrepareModel()
{
    Model runtimeModel = ModelLoader.Load(model);
    modelBuilder = new ModelBuilder(runtimeModel);
}
```



#### `InitializeWorker`

This method initializes the worker for executing the model with the specified backend and channel order.

```c#
/// <summary>
/// Initialize the worker for executing the model with the specified backend and channel order.
/// </summary>
/// <param name="model">The target model representation.</param>
/// <param name="workerType">The target compute backend.</param>
/// <param name="useNCHW">The channel order for the compute backend (default is true).</param>
/// <returns>An initialized worker instance.</returns>
protected IWorker InitializeWorker(Model model, WorkerFactory.Type workerType, bool useNCHW = true)
{
    // Validate worker type
    workerType = WorkerFactory.ValidateType(workerType);

    // Set channel order if required
    if (useNCHW)
    {
        ComputeInfo.channelsOrder = ComputeInfo.ChannelsOrder.NCHW;
    }

    // Create and return the worker instance
    return WorkerFactory.CreateWorker(workerType, model);
}
```



#### `InitializeEngine`

This method initializes the inference engine by creating a worker instance.

```c#
/// <summary>
/// Initialize the inference engine.
/// </summary>
protected virtual void InitializeEngine()
{
    engine = WorkerFactory.CreateWorker(workerType, modelBuilder.model);
    engine = InitializeWorker(modelBuilder.model, workerType, useNCHW);
}
```



#### `ExecuteModel`

These overloaded methods execute the model with the provided input Tensor(s). Child classes can override them to implement custom input and output processing.

```c#
/// <summary>
/// Execute the model with the given input Tensor.
/// Override this method to implement custom input and output processing.
/// </summary>
/// <param name="input">The input Tensor for the model.</param>
public virtual void ExecuteModel(Tensor input)
{
    engine.Execute(input);
}

/// <summary>
/// Execute the model with the given input Tensor.
/// Override this method to implement custom input and output processing.
/// </summary>
/// <param name="input">The input Tensor for the model.</param>
public virtual void ExecuteModel(IDictionary<string, Tensor> inputs)
{
    engine.Execute(inputs);
}
```



#### `OnDisable`

This method runs when the component is disabled. It cleans up resources by disposing of the engine.

```c#
/// <summary>
/// Clean up resources when the component is disabled.
/// </summary>
protected virtual void OnDisable()
{
    engine.Dispose();
}
```






---



### `AddCustomDefineSymbol.cs`

This Editor script contains a class that adds a custom define symbol  to the project. We can use this custom symbol to prevent code that  relies on this package from executing unless the Barracuda Inference Base package is present. The complete code is available on GitHub at the link below.

* [AddCustomDefineSymbol.cs](https://github.com/cj-mills/unity-barracuda-inference-base/blob/main/Editor/AddCustomDefineSymbol.cs)

```c#
using UnityEditor;
using UnityEngine;

namespace CJM.BarracudaInference
{
    public class DependencyDefineSymbolAdder
    {
        private const string CustomDefineSymbol = "CJM_BARRACUDA_INFERENCE";

        [InitializeOnLoadMethod]
        public static void AddCustomDefineSymbol()
        {
            // Get the currently selected build target group
            var buildTargetGroup = EditorUserBuildSettings.selectedBuildTargetGroup;
            // Retrieve the current scripting define symbols for the selected build target group
            var defines = PlayerSettings.GetScriptingDefineSymbolsForGroup(buildTargetGroup);

            // Check if the CustomDefineSymbol is already present in the defines string
            if (!defines.Contains(CustomDefineSymbol))
            {
                // Append the CustomDefineSymbol to the defines string, separated by a semicolon
                defines += $";{CustomDefineSymbol}";
                // Set the updated defines string as the new scripting define symbols for the selected build target group
                PlayerSettings.SetScriptingDefineSymbolsForGroup(buildTargetGroup, defines);
                // Log a message in the Unity console to inform the user that the custom define symbol has been added
                Debug.Log($"Added custom define symbol '{CustomDefineSymbol}' to the project.");
            }
        }
    }
}
```









## Conclusion

This post provided an in-depth walkthrough of the code for the Barracuda Inference Base package. The package provides a foundation for performing inference with the Barracuda inference library with a flexible base class to extend with task-specific packages.

You can continue to explore the package by going to its GitHub repository linked below, where you will also find instructions for installing it using the Unity Package Manager.

- GitHub Repository: [unity-barracuda-inference-base](https://github.com/cj-mills/unity-barracuda-inference-base)

You can find the code for the demo projects shown in the videos at the beginning of this post linked below.

- [Barracuda Image Classification Demo](https://github.com/cj-mills/barracuda-image-classification-demo): A simple Unity project demonstrating how to perform image classification with the `barracuda-inference-image-classification` package.
- [Barracuda Inference YOLOX Demo](https://github.com/cj-mills/barracuda-inference-yolox-demo): A simple Unity project demonstrating how to perform object detection with the `barracuda-inference-yolox` package.
- [Barracuda Inference PoseNet Demo](https://github.com/cj-mills/barracuda-inference-posenet-demo): A simple Unity project demonstrating how to perform 2D human pose estimation with the `barracuda-inference-posenet` package.









{{< include /_about-author-cta.qmd >}}
