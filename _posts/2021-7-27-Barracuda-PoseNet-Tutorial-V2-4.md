---
title: Barracuda PoseNet Tutorial 2nd Edition Pt. 4
layout: post
toc: false
comments: true
description: This post covers how to initialize, modify, and execute the PoseNet models.
categories: [unity,barracuda,tutorial]
hide: true
permalink: /:title/
search_exclude: false
---

* [Overview](#overview)
* [Update `PoseEstimator` Script](#update-poseestimator-script)
* [Assign Model Assets](#assign-model-assets)
* [Test it Out](#test-it-out)
* [Summary](#summary)



## Overview



## Update `PoseEstimator` Script





### Add Public Variables



```c#
[Tooltip("The MobileNet model asset file to use when performing inference")]
public NNModel mobileNetModelAsset;

[Tooltip("The ResNet50 model asset file to use when performing inference")]
public NNModel resnetModelAsset;

[Tooltip("The backend to use when performing inference")]
public WorkerFactory.Type workerType = WorkerFactory.Type.Auto;
```



### Add Private Variables



```c#
/// <summary>
/// 
/// </summary>
private struct Engine
{
    public WorkerFactory.Type workerType;
    public IWorker worker;
    public ModelType modelType;

    public Engine(WorkerFactory.Type workerType, Model model, ModelType modelType)
    {
        this.workerType = workerType;
        worker = WorkerFactory.CreateWorker(workerType, model);
        this.modelType = modelType;
    }
}

// The interface used to execute the neural network
private Engine engine;


// The name for the heatmap layer in the model asset
private string heatmapLayer;

// The name for the offsets layer in the model asset
private string offsetsLayer;

// The name for the forwards displacement layer in the model asset
private string displacementFWDLayer;

// The name for the backwards displacement layer in the model asset
private string displacementBWDLayer;

// The name for the Sigmoid layer that returns the heatmap predictions
private string predictionLayer = "heatmap_predictions";
```



### Create `InitializeBarracuda` Method



```c#
/// <summary>
/// Updates the output layer names based on the selected model architecture
/// and initializes the Barracuda inference engine witht the selected model.
/// </summary>
private void InitializeBarracuda()
{
    // The compiled model used for performing inference
    Model m_RunTimeModel;

    if (modelType == ModelType.MobileNet)
    {
        preProcessFunction = Utils.PreprocessMobileNet;
        // Compile the model asset into an object oriented representation
        m_RunTimeModel = ModelLoader.Load(mobileNetModelAsset);
        displacementFWDLayer = m_RunTimeModel.outputs[2];
        displacementBWDLayer = m_RunTimeModel.outputs[3];
    }
    else
    {
        preProcessFunction = Utils.PreprocessResNet;
        // Compile the model asset into an object oriented representation
        m_RunTimeModel = ModelLoader.Load(resnetModelAsset);
        displacementFWDLayer = m_RunTimeModel.outputs[3];
        displacementBWDLayer = m_RunTimeModel.outputs[2];
    }

    heatmapLayer = m_RunTimeModel.outputs[0];
    offsetsLayer = m_RunTimeModel.outputs[1];

    // Create a model builder to modify the m_RunTimeModel
    ModelBuilder modelBuilder = new ModelBuilder(m_RunTimeModel);

    // Add a new Sigmoid layer that takes the output of the heatmap layer
    modelBuilder.Sigmoid(predictionLayer, heatmapLayer);

    // Validate if backend is supported, otherwise use fallback type.
    workerType = WorkerFactory.ValidateType(workerType);

    // Create a worker that will execute the model with the selected backend
    engine = new Engine(workerType, modelBuilder.model, modelType);
}
```



### Modify `Start` Method



```c#
// Initialize the Barracuda inference engine based on the selected model
InitializeBarracuda();
```



#### Final Code

```c#
// Start is called before the first frame update
void Start()
{
    if (useWebcam)
    {
        // Limit application framerate to the target webcam framerate
        Application.targetFrameRate = webcamFPS;

        // Create a new WebCamTexture
        webcamTexture = new WebCamTexture(webcamDims.x, webcamDims.y, webcamFPS);

        // Start the Camera
        webcamTexture.Play();

        // Deactivate the Video Player
        videoScreen.GetComponent<VideoPlayer>().enabled = false;

        // Update the videoDims.y
        videoDims.y = webcamTexture.height;
        // Update the videoDims.x
        videoDims.x = webcamTexture.width;
    }
    else
    {
        // Update the videoDims.y
        videoDims.y = (int)videoScreen.GetComponent<VideoPlayer>().height;
        // Update the videoDims.x
        videoDims.x = (int)videoScreen.GetComponent<VideoPlayer>().width;
    }

    // Create a new videoTexture using the current video dimensions
    videoTexture = RenderTexture.GetTemporary(videoDims.x, videoDims.y, 24, RenderTextureFormat.ARGBHalf);

    // Initialize the videoScreen
    InitializeVideoScreen(videoDims.x, videoDims.y, useWebcam);

    // Adjust the camera based on the source video dimensions
    InitializeCamera();

    // Adjust the input dimensions to maintain the source aspect ratio
    aspectRatioScale = (float)videoTexture.width / videoTexture.height;
    targetDims.x = (int)(imageDims.y * aspectRatioScale);
    imageDims.x = targetDims.x;

    // Initialize the RenderTexture that will store the processed input image
    rTex = RenderTexture.GetTemporary(imageDims.x, imageDims.y, 24, RenderTextureFormat.ARGBHalf);

    // Initialize the Barracuda inference engine based on the selected model
    InitializeBarracuda();
}
```





### Modify `Update` Method



```c#
// Execute neural network with the provided input
engine.worker.Execute(input);
// Release GPU resources allocated for the Tensor
input.Dispose();
```



#### Final Code

```c#
// Update is called once per frame
void Update()
{
    // Copy webcamTexture to videoTexture if using webcam
    if (useWebcam) Graphics.Blit(webcamTexture, videoTexture);

    // Prevent the input dimensions from going too low for the model
    imageDims.x = Mathf.Max(imageDims.x, 64);
    imageDims.y = Mathf.Max(imageDims.y, 64);

    // Update the input dimensions while maintaining the source aspect ratio
    if (imageDims.x != targetDims.x)
    {
        aspectRatioScale = (float)videoTexture.height / videoTexture.width;
        targetDims.y = (int)(imageDims.x * aspectRatioScale);
        imageDims.y = targetDims.y;
        targetDims.x = imageDims.x;
    }
    if (imageDims.y != targetDims.y)
    {
        aspectRatioScale = (float)videoTexture.width / videoTexture.height;
        targetDims.x = (int)(imageDims.y * aspectRatioScale);
        imageDims.x = targetDims.x;
        targetDims.y = imageDims.y;
    }

    // Update the rTex dimensions to the new input dimensions
    if (imageDims.x != rTex.width || imageDims.y != rTex.height)
    {
        RenderTexture.ReleaseTemporary(rTex);
        // Assign a temporary RenderTexture with the new dimensions
        rTex = RenderTexture.GetTemporary(imageDims.x, imageDims.y, 24, rTex.format);
    }

    // Copy the src RenderTexture to the new rTex RenderTexture
    Graphics.Blit(videoTexture, rTex);


    if (modelType == ModelType.MobileNet)
    {
        preProcessFunction = Utils.PreprocessMobileNet;
    }
    else
    {
        preProcessFunction = Utils.PreprocessResNet;
    }

    // Prepare the input image to be fed to the selected model
    ProcessImage(rTex);

    // 
    if (engine.modelType != modelType || engine.workerType != workerType)
    {
        engine.worker.Dispose();
        InitializeBarracuda();
    }

    // Execute neural network with the provided input
    engine.worker.Execute(input);
    // Release GPU resources allocated for the Tensor
    input.Dispose();
}
```



### Define `OnDisable` Method



```c#
// OnDisable is called when the MonoBehavior becomes disabled or inactive
private void OnDisable()
{
    // Release the resources allocated for the inference engine
    engine.worker.Dispose();
}
```





## Assign Model Assets



![inspector-tab-assign-model-assets](..\images\barracuda-posenet-tutorial-v2\part-4\inspector-tab-assign-model-assets.png)



## Test it Out





### ResNet50



#### GPU Preprocessing and GPU Inference

![resnet-compute-usegpu](..\images\barracuda-posenet-tutorial-v2\part-4\resnet-compute-usegpu.png)





#### CPU Preprocessing and GPU Inference

![resnet-compute-usecpu](..\images\barracuda-posenet-tutorial-v2\part-4\resnet-compute-usecpu.png)







#### GPU Preprocessing and CPU Inference

![resnet-burst](..\images\barracuda-posenet-tutorial-v2\part-4\resnet-burst.png)





### MobileNet



#### GPU Preprocessing and GPU Inference

![mobilenet-compute-usegpu](..\images\barracuda-posenet-tutorial-v2\part-4\mobilenet-compute-usegpu.png)





#### GPU Preprocessing and CPU Inference

![mobilenet-burst-usegpu](..\images\barracuda-posenet-tutorial-v2\part-4\mobilenet-burst-usegpu.png)







## Summary

_.



**Previous:** [Part 3](https://christianjmills.com/Barracuda-PoseNet-Tutorial-V2-3/)

**Project Resources:** [GitHub Repository - Version 1](https://github.com/cj-mills/Barracuda-PoseNet-Tutorial)

