---
title: In-Browser Hand Gesture Recognition for Unity with Fastai and TensorFlow.js Pt. 3
date: 2022-10-6
image: /images/empty.gif
title-block-categories: true
layout: post
toc: false
hide: false
search_exclude: false
comments:
  utterances:
    repo: cj-mills/christianjmills
description: Build a Unity project as a [WebGL](https://www.khronos.org/webgl/wiki/Getting_Started) application and host it using [GitHub Pages](https://pages.github.com/).
categories: [fastai, unity, tensorflow, webgl]

---

* [Overview](#overview)
* [Create GitHub Pages Repository](#create-github-pages-repository)
* [Add Unity Build Files](#add-unity-build-files)
* [Test Live Demo](#test-live-demo)
* [Summary](#summary)





## Overview

[Part 2](../part-2/) covered implementing a hand gesture classifier in a Unity project as a [jslib plugin](https://docs.unity3d.com/Manual/webgl-interactingwithbrowserscripting.html). In this post, we'll host the Unity project as a shareable web demo using [GitHub Pages](https://pages.github.com/). The demo will run locally in the user's web browser.




## Create GitHub Pages Repository

We first need to create a [new GitHub repository](https://github.com/new) to store the WebGL build. We can do this on GitHub or locally using Git, GitHub Desktop, or another tool. 



![](./images/github-desktop-create-new-repository.png){fig-align="center"}



Open the Settings tab for the new repository on GitHub.



![](./images/github-new-repository.png){fig-align="center"}



Open the `Pages` section. Switch the source branch to `main` and click the `Save` button to start the automated build process.



![](./images/github-pages-click-save.png){fig-align="center"}





We can check the GitHub Pages build progress under the Actions tab for the repository.



![](./images/github-pages-check-build-progress.png){fig-align="center"}





The web page will be accessible once the `pages build and deployment` workflow completes. Although, we don't have any web pages at the moment.



![](./images/github-pages-build-complete.png){fig-align="center"}



GitHub will provide a URL for accessing the web demo in the GitHub Pages section once it finishes building.



![](./images/github-pages-get-url.png){fig-align="center"}







## Add Unity Build Files

Now let's add the WebGL build from the previous post. Open the `Build` folder from [part 2.](../part-2/#test-in-browser)



![](./images/open-build-folder.png){fig-align="center"}



Copy the folder content to the repository for the GitHub Pages demo.

![](./images/copy-webgl-build-to-demo-repo.png){fig-align="center"}



We can push the local changes to GitHub, which will automatically trigger the `pages build and deployment` workflow.



![](./images/github-pages-check-webgl-build-progress.png){fig-align="center"}







## Test Live Demo

We can test the web demo at the URL provided by GitHub once the build workflow completes.

![](./images/github-pages-webgl-demo.png){fig-align="center"}








## Summary

In this tutorial series, we trained a hand gesture classifier with the fastai library, created a TensorFlow.js plugin for Unity to perform inference in a web browser, built the Unity project as a shareable web demo, and hosted it for free using GitHub Pages. 

Now that we have a general approach for using TensorFlow.js in Unity applications, we could extend the work in this tutorial by instead using one of the several [pretrained models](https://www.tensorflow.org/js/models) TensorFlow.js provides.





**Previous:** [In-Browser Hand Gesture Recognition for Unity with Fastai and TensorFlow.js Pt. 2](../part-2/)

**Project Resources:** [GitHub Repository](https://github.com/cj-mills/tensorflow-js-unity-tutorial)







<!-- Cloudflare Web Analytics --><script defer src='https://static.cloudflareinsights.com/beacon.min.js' data-cf-beacon='{"token": "56b8d2f624604c4891327b3c0d9f6703"}'></script><!-- End Cloudflare Web Analytics -->