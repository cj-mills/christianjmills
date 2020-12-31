---
title: Using PyTorch with CUDA on WSL2
layout: post
toc: false
comments: true
description: This post covers my experience getting PyTorch to run with CUDA on WSL2.
categories: [tutorial]
hide: true
permalink: /:title/
search_exclude: true
---

* [Introduction](#introduction)
* [Installation Preparations](#installation-preparations)
* [Set Up Ubuntu](#set-up-ubuntu)
* [Conclusion](#conclusion)

## Introduction

I spent a couple days figuring out how to get my Linux environment for deep learning working on Microsoft’s Windows Subsystem for Linux (WSL). The process was a bit of a hassle. While the official guides are adequate for getting set up, I needed a few extra steps to actually train a model. This post summarizes my experience making it work.

### What is WSL

WSL is a compatibility layer that let's you run Linux environments directly on Windows. You can run Linux command-line tools and applications, invoke Windows applications from the Linux command-line, and access Windows drives through the Linux file system. The most recent version, WSL2, uses a real Linux kernel. This provides support for more applications such as Docker. More importantly for my purposes, it also enables GPU accelerated applications.

### Motivation

I've been dual-booting Windows and Linux for a while now. I prefer Linux for coding and training models while Windows can be more convenient for things like gaming and school work. This setup didn't have any drawbacks for me until I started working with the Barracuda library for Unity. Unity is installed on Windows but my environment for training deep learning models is on Linux. This is inconvenient when I want to test out a newly trained model in Unity. I decided to try WSL2 in the hopes that it would remove the need to switch between operating systems.



## Installation Preparations

The [install process](https://docs.microsoft.com/en-us/windows/wsl/install-win10) for most WSL2 use cases is straightforward. You just need to enable a few features and install your preferred Linux distribution from the Microsoft Store. However, the process for enabling CUDA support is a bit more involved.

### Install Windows Insider Build

CUDA applications are only supported in WSL2 on Windows build versions 20145 or higher. These are currently only accessible through the [Dev Channel](https://blogs.windows.com/windows-insider/2020/06/15/introducing-windows-insider-channels/) for the [Windows Insider Program](https://insider.windows.com/en-us/getting-started#register). Microsoft requires you to enable Full telemetry collection to install Insider builds for Windows. This was annoying since the first thing I do when installing Windows is disable every accessible telemetry setting. Fortunately, you can disable everything again once you've installed an Insider build.

### Install Nvidia's Preview Driver

Nvidia provides a preview Windows display driver for their graphics cards that enables CUDA on WSL2. This Windows driver includes both the regular driver components for Windows and WSL. We don't install display drivers on the Linux distribution itself.

* [Nvidia Drivers for CUDA on WSL](https://developer.nvidia.com/cuda/wsl/download)

## Install WSL

You can [install](https://docs.microsoft.com/en-us/windows/wsl/install-win10#simplified-installation-for-windows-insiders) WSL with one line in the command window if you install a preview build first. I did it backwards so I had to use the slightly longer [manual installation](https://docs.microsoft.com/en-us/windows/wsl/install-win10#manual-installation-steps). I went with [Ubuntu 20.04](https://www.microsoft.com/store/productId/9N6SVWS3RX71) for my distribution since that's what I currently have installed on my desktop. 



## Set Up Ubuntu

#### Update Ubuntu

The Ubuntu distribution needs to be updated after it gets installed.

```bash
sudo apt update
sudo apt upgrade
```



### Install CUDA Toolkit

The next step was to install the CUDA toolkit. Nvidia lists `WSL-Ubuntu` as a separate distribution. However, I ended up using the standard `Ubuntu` option. You can view the selected instructions I followed by clicking the link below.

* [Instructions](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=2004&target_type=deblocal)



### Install Anaconda

#### Download and Run the Install Script

```bash
cd ~
wget https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh
chmod +x Anaconda3-2020.11-Linux-x86_64.sh
./Anaconda3-2020.11-Linux-x86_64.sh
```

#### Restart Ubuntu

```bash
wsl.exe --shutdown Ubuntu
```

#### Confirm Python Works

```bash
(base) innom-dt@INNOM-DT:~$ python
Python 3.8.5 (default, Sep  4 2020, 07:30:14)
[GCC 7.3.0] :: Anaconda, Inc. on linux
Type "help", "copyright", "credits" or "license" for more information.
>>>
```



### Install Fastai Library

```bash
conda install -c fastai -c pytorch -c anaconda fastai gh anaconda
```

#### Confirm CUDA Works

```bash
(base) innom-dt@INNOM-DT:~$ python
Python 3.8.5 (default, Sep  4 2020, 07:30:14)
[GCC 7.3.0] :: Anaconda, Inc. on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import torch
>>> torch.cuda.is_available()
True
>>>
```



## Additional Steps





## Conclusion

