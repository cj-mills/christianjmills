---
title: Using PyTorch with CUDA on WSL2 (2020)
date: '2020-12-31'
image: /images/empty.gif
title-block-categories: true
hide: false
search_exclude: false
comments:
  utterances:
    repo: cj-mills/christianjmills
description: This post covers my experience getting PyTorch to run with CUDA on WSL2.
categories: [log, tutorial, pytorch, wsl2]

aliases:
- /Using-PyTorch-with-CUDA-on-WSL2/
- /posts/pytorch-cuda--wsl2/

---

* [Introduction](#introduction)
* [Installing WSL](#installing-wsl)
* [Setting Up Ubuntu](#setting-up-ubuntu)
* [The Headaches](#the-headaches)
* [Conclusion](#conclusion)

## Introduction

I spent a couple days figuring out how to train deep learning models on Microsoft’s Windows Subsystem for Linux (WSL). The process was a bit of a hassle. While the official installation guides are adequate, there were some headaches that came up during regular use. This post summarizes my experience making it work.

### What is WSL

WSL is a compatibility layer that let's you run Linux environments directly on Windows. You can run Linux command-line tools and applications, invoke Windows applications from the Linux command-line, and access Windows drives through the Linux file system. The most recent version, WSL2, uses a real Linux kernel. This provides support for more applications such as Docker. More importantly for my purposes, it also enables GPU accelerated applications.

### Motivation

I've been dual-booting Windows and Linux for a while now. I prefer Linux for coding and training models while Windows is supported by more applications. This setup didn't have any drawbacks for me until I started working with the Barracuda library for Unity. Unity is installed on Windows but my environment for training deep learning models is on Linux. This is inconvenient when I want to test out a newly trained model in Unity. I decided to try WSL2 in the hopes that it would remove the need to switch between operating systems.

## Installing WSL

The [install process](https://docs.microsoft.com/en-us/windows/wsl/install-win10) for most WSL2 use cases is straightforward. You just need to enable a few features and install your preferred Linux distribution from the Microsoft Store. However, the process for enabling CUDA support is a bit more involved.

### Install Windows Insider Build

CUDA applications are only supported in WSL2 on Windows build versions 20145 or higher. These are currently only accessible through the [Dev Channel](https://blogs.windows.com/windows-insider/2020/06/15/introducing-windows-insider-channels/) for the [Windows Insider Program](https://insider.windows.com/en-us/getting-started#register). I confirmed it does not work with the latest public release. Microsoft requires you to enable Full telemetry collection to install Insider builds for Windows. This was annoying since the first thing I do when installing Windows is disable every accessible telemetry setting. Fortunately, I only needed to temporarily enable a couple of the settings to install an Insider build.

### Install Nvidia's Preview Driver

Nvidia provides a preview Windows display driver for their graphics cards that enables CUDA on WSL2. This Windows driver includes both the regular driver components for Windows and WSL. We're not supposed to install display drivers on the Linux distribution itself.

* [Nvidia Drivers for CUDA on WSL](https://developer.nvidia.com/cuda/wsl/download)

### Install WSL

You can [install](https://docs.microsoft.com/en-us/windows/wsl/install-win10#simplified-installation-for-windows-insiders) WSL with one line in the command window if you install a preview build first. I did it backwards so I had to use the slightly longer [manual installation](https://docs.microsoft.com/en-us/windows/wsl/install-win10#manual-installation-steps). I went with [Ubuntu 20.04](https://www.microsoft.com/store/productId/9N6SVWS3RX71) for my distribution since that's what I currently have installed on my desktop. 

## Setting Up Ubuntu

The set up process was basically the same as regular Ubuntu with the exception of no display drivers.

#### Update Ubuntu

As usual, I first checked for any updates. There were quite a few.

```bash
sudo apt update
sudo apt upgrade
```

### Install CUDA Toolkit

The next step was to install the CUDA toolkit. Nvidia lists `WSL-Ubuntu` as a separate distribution. I don't know what makes it functionally different than the regular `Ubuntu` distribution. Both worked and performed the same for me when training models. You can view the instructions I followed for both by clicking the links below.

* [Ubuntu](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=2004&target_type=deblocal)
* [WSL-Ubuntu](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&target_distro=WSLUbuntu&target_version=20&target_type=deblocal)

### Install Anaconda

I like to use Anaconda, so I downloaded the latest available release to the home directory and installed it like normal.

```bash
cd ~
wget https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh
chmod +x Anaconda3-2020.11-Linux-x86_64.sh
./Anaconda3-2020.11-Linux-x86_64.sh
```

I had to restart bash to use the new python interpreter like normal as well.

```bash
exec bash
```

After that, the interactive python interpreter started without issue.

```bash
Python 3.8.5 (default, Sep  4 2020, 07:30:14)
[GCC 7.3.0] :: Anaconda, Inc. on linux
Type "help", "copyright", "credits" or "license" for more information.
>>>
```

### Install Fastai Library

I installed the fastai library which is built on top of PyTorch to test whether I could access the GPU. The installation went smoothly.

```bash
conda install -c fastai -c pytorch -c anaconda fastai gh anaconda
```

I was able to confirm that PyTorch could access the GPU using the `torch.cuda.is_available()` method.

```bash
Python 3.8.5 (default, Sep  4 2020, 07:30:14)
[GCC 7.3.0] :: Anaconda, Inc. on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import torch
>>> torch.cuda.is_available()
True
>>>
```

I opened up a jupyter notebook and trained a ResNet50 model to confirm that the GPU was actually being used. The Task Manager in Windows accurately displays the available GPU memory and temperature but not GPU usage for WSL applications. The `nvidia-smi` command doesn't work yet in WSL either. I believe Nvidia is planning on adding that functionality in a future release. However, the `nvidia-smi.exe` command does accurately show GPU usage.

## The Headaches

Everything seemed to be working as I'd hoped. However, I started encountering some issues the more I used WSL.

### Memory Usage

By default, WSL distributions will take up as much system memory as is available and not release it. This problem is compounded since Windows already takes up a decent chuck of memory. This seems to be something Microsoft is still [working on](https://github.com/microsoft/WSL/issues/4166). However, you can limit the amount of memory WSL can access. The [workaround](https://github.com/microsoft/WSL/issues/4166#issuecomment-526725261) involves creating a `.wslconfig` file and adding it to you Windows user folder (e.g. `C:\Users\Username`). You can see the contents for an example config file below.

```
[wsl2]
memory=6GB
```

GPU memory usage doesn't suffer from this problem, so it wasn't too big of an issue for me.

### File Permissions

This is where things started to get more inconvenient for my use case. The way in which WSL handles permissions for files in attached drives isn't readily apparent for new users. I didn't have any problem accessing the previously mentioned jupyter notebook or the image dataset I used to train the model. However, I couldn't access the images in a different dataset when training a different model. 

I tried adding the necessary permissions in Ubuntu but that didn't work. I even tried copying the dataset to the Ubuntu home directory. I ended up finding a solution on [Stack Exchange](https://superuser.com/a/1392722). It involves adding another config file, this time to Ubuntu. I needed to create a `wsl.conf` file in the `/etc/` directory. This one enables metadata for the files so that changes in permission actually work.

```bash
[automount]
enabled = true
root = /mnt/
options = "metadata,umask=22,fmask=11"
```

I had to restart my computer after creating the file for it to take effect. You can learn more about `wsl.conf` files and the settings in the above example at the links below.

* [Automatically Configuring WSL](https://devblogs.microsoft.com/commandline/automatically-configuring-wsl/)
* [Chmod/Chown WSL Improvements](Chmod/Chown WSL Improvements)
* [File Permissions for WSL](https://docs.microsoft.com/en-us/windows/wsl/file-permissions)

### Disk Space

This is the one that killed the whole endeavor for me. I deleted the copy of the dataset I made in the Ubuntu home directory after I was able to access the original. I noticed that my disk usage didn't decrease after I deleted the 48GB of images. This is also a [known](https://github.com/microsoft/WSL/issues/4699) problem with WSL. There is another [workaround](https://github.com/microsoft/WSL/issues/4699#issuecomment-635673427) where you can manually release unused disk space that involves the following steps.

1. Open PowerShell as an Administrator.
2. Navigate to the folder containing the virtual hard drive file for your distribution. 
3. Shutdown WSL.
4.  Run [`optimize-vhd`](https://docs.microsoft.com/en-us/powershell/module/hyper-v/optimize-vhd?view=win10-ps) for the virtual hard drive.

```powershell
cd C:\Users\UserName_Here\AppData\Local\Packages\CanonicalGroupLimited.Ubuntu20.04onWindows_79rhkp1fndgsc\LocalState
wsl --shutdown
optimize-vhd -Path .\ext4.vhdx -Mode full
```

You currently need to do this every time you want to reclaim disk space from WSL. By this point, any convenience I'd gain over a dual-boot setup had been wiped out.

## Conclusion

I'm excited about the future of WSL. Having such tight integration between Windows and Linux has a lot of potential. Unfortunately, it's not at a point where I'd feel comfortable switching over from a dual-boot setup. I'm hoping that the issues I encountered will get resolved in 2021. I'll give it another shot when CUDA support comes out of preview.



