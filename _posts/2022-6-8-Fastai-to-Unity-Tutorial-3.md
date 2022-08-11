---

title: Fastai to Unity Beginner Tutorial Pt. 3
layout: post
toc: false
comments: true
description: Part 3 covers building a Unity project to run in a web browser and hosting it using GitHub Pages.
categories: [fastai, unity, barracuda]
hide: false
permalink: /:title/
search_exclude: false
---



* [Overview](#overview)
* [Create GitHub Pages Repository](#create-github-pages-repository)
* [Build WebGL Application](#build-webgl-application)
* [Test Live Demo](#test-live-demo)
* [Summary](#summary)





## Overview

[Part 1](https://christianjmills.com/Fastai-to-Unity-Tutorial-1/) covered training an image classification model using the fastai library and exporting it to ONNX. [Part 2](https://christianjmills.com/Fastai-to-Unity-Tutorial-2/) covered implementing a trained image classification model in a Unity project using the Barracuda library. In this post, we'll build a Unity project as a shareable web demo and host it for free using GitHub Pages. The image classifier will execute locally in the user's web browser. 

WebGL builds use Barracuda's Pixel Shader backend, which is not nearly as performant as the Compute Shader backend. Therefore, I recommend using WebGL for sharing prototypes and target operating systems for final projects.

**Pixel Shader Backend: ResNet18**

![unity-webcam-pixel-shader-fps-resnet18](../images/fastai-to-unity-tutorial/part-3/unity-webcam-pixel-shader-fps-resnet18.png)

**Pixel Shader Backend: ResNet34**

![unity-webcam-pixel-shader-fps-resnet34](../images/fastai-to-unity-tutorial/part-3/unity-webcam-pixel-shader-fps-resnet34.png)



**Compute Shader Backend: ResNet18**

![unity-webcam-compute-shader-fps-resnet18](../images/fastai-to-unity-tutorial/part-3/unity-webcam-compute-shader-fps-resnet18.png)

**Compute Shader Backend: ResNet34**

![unity-webcam-compute-shader-fps-resnet34](../images/fastai-to-unity-tutorial/part-3/unity-webcam-compute-shader-fps-resnet34.png)



**Compute Shader Backend with asynchronous GPU readback: ResNet18**

![unity-webcam-compute-shader-async-fps-resnet18](../images/fastai-to-unity-tutorial/part-3/unity-webcam-compute-shader-async-fps-resnet18.png)

**Compute Shader Backend with asynchronous GPU readback: ResNet34**

![unity-webcam-compute-shader-async-fps-resnet34](../images/fastai-to-unity-tutorial/part-3/unity-webcam-compute-shader-async-fps-resnet34.png)








## Create GitHub Pages Repository

We first need to create a [new GitHub repository](https://github.com/new) to store the WebGL build. We can do this on GitHub or locally using Git, GitHub Desktop, or another tool. 



![github-desktop-create-new-repository](../images/fastai-to-unity-tutorial/part-3/github-desktop-create-new-repository.png)



Open the Settings tab for the new repository on GitHub.



![github-new-repository](../images/fastai-to-unity-tutorial/part-3/github-new-repository.png)



Open the Pages section and select the main branch as the source for GitHub Pages.



![github-pages-select-main-branch](../images/fastai-to-unity-tutorial/part-3/github-pages-select-main-branch.png)



Click the Save button to start the automated build process.



![github-pages-click-save](../images/fastai-to-unity-tutorial/part-3/github-pages-click-save.png)



GitHub will provide a URL for accessing the web demo once it finishes building.



![github-pages-get-url](../images/fastai-to-unity-tutorial/part-3/github-pages-get-url.png)



We can check the GitHub Pages build progress under the Actions tab for the repository.



![github-pages-check-build-progress](../images/fastai-to-unity-tutorial/part-3/github-pages-check-build-progress.png)



The web page will be accessible once the "pages build and deployment" workflow completes. Although, we don't have any web pages at the moment.



![github-pages-build-complete](../images/fastai-to-unity-tutorial/part-3/github-pages-build-complete.png)







## Build WebGL Application

In the Unity project, select `File → Build Settings...` in the top menu bar to open the Build Settings window.

![unity-open-build-settings](../images/fastai-to-unity-tutorial/part-3/unity-open-build-settings.png)



Select `WebGL` from the list of platforms and click Switch Platform.



![unity-build-settings-switch-to-webgl](../images/fastai-to-unity-tutorial/part-3/unity-build-settings-switch-to-webgl.png)



Unity enables compression by default for WebGL builds, which GitHub Pages does not support. We can disable compression in the Player Settings. Click the Player Settings button in the bottom-left corner of the Build Settings window.



![unity-build-settings-open-player-settings](../images/fastai-to-unity-tutorial/part-3/unity-build-settings-open-player-settings.png)



Select `Disabled` from the Compression Format dropdown menu and close the Project Settings window.



![unity-player-settings-disable-webgl-compression](../images/fastai-to-unity-tutorial/part-3/unity-player-settings-disable-webgl-compression.png)



We can test the WebGL build locally by clicking Build and Run in the Build Settings window.



![unity-build-settings-build-and-run](../images/fastai-to-unity-tutorial/part-3/unity-build-settings-build-and-run.png)



Unity will prompt us to select a folder to store the build files.



![unity-select-build-folder](../images/fastai-to-unity-tutorial/part-3/unity-select-build-folder.png)



Navigate to the local folder for the GitHub Pages repository and click Select Folder to start the build process.



![unity-build-select-github-pages-repo-folder](../images/fastai-to-unity-tutorial/part-3/unity-build-select-github-pages-repo-folder.png)



Once the build completes, Unity will launch the demo in the default web browser. Unity caps the framerate to the platform's default [target framerate](https://docs.unity3d.com/ScriptReference/Application-targetFrameRate.html) by default. On my Windows 10 desktop, that is 60fps.



![unity-webgl-build-local-test](../images/fastai-to-unity-tutorial/part-3/unity-webgl-build-local-test.png)



If we examine the repository folder, we can see a new `Build` folder, a `StreamingAssets` folder, a `TemplateData` folder, and an `index.html` file.



![github-pages-repo-folder-after-webgl-build](../images/fastai-to-unity-tutorial/part-3/github-pages-repo-folder-after-webgl-build.png)



We can push the local changes to GitHub, which will automatically trigger the "pages build and deployment" workflow.



![github-pages-check-webgl-build-progress](../images/fastai-to-unity-tutorial/part-3/github-pages-check-webgl-build-progress.png)







## Test Live Demo

We can test the web demo at the URL provided by GitHub once the build workflow completes.

![github-pages-webgl-demo](../images/fastai-to-unity-tutorial/part-3/github-pages-webgl-demo.png)








## Summary

This post covered how to build a Unity project as a shareable web demo and host it using GitHub Pages.





**Previous:** [Fastai to Unity Tutorial Pt. 2](https://christianjmills.com/Fastai-to-Unity-Tutorial-2/)

**Follow Up:** [How to Create a LibTorch Plugin for Unity on Windows Pt.1](https://christianjmills.com/Fastai-to-LibTorch-to-Unity-Tutorial-Windows-1/)

**Follow Up:** [How to Create an OpenVINO Plugin for Unity on Windows Pt. 1](https://christianjmills.com/Fastai-to-OpenVINO-to-Unity-Tutorial-Windows-1/)

**Intermediate Tutorial:** [End-to-End Object Detection for Unity With IceVision and OpenVINO Pt. 1](https://christianjmills.com/IceVision-to-OpenVINO-to-Unity-Tutorial-1/)



**Project Resources:** [GitHub Repository](https://github.com/cj-mills/fastai-to-unity-tutorial)







