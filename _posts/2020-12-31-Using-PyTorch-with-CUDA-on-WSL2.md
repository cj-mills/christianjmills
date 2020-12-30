---
title: Using PyTorch with CUDA on WSL2
layout: post
toc: false
comments: true
description: This post covers my experience getting PyTorch to run with CUDA on WSL2.
categories: [tutorial]
hide: true
permalink: /:title/
search_exclude: false
---

* [Introduction](#introduction)
* [](#)
* [Conclusion](#conclusion)

## Introduction

I've spent the past couple days figuring out how to get my Linux environment for training deep learning models working on Microsoft's Windows Subsystem for Linux (WSL). The process was a bit of a hassle. While the official guides are adequate for getting set up, I needed a few extra steps to actually train a model. 



The process was involved enough that it feels worth the effort to consolidate the steps I took in a blog post.



### What is WSL



### Motivation

I've been dual-booting Windows and Linux for a while now. I prefer Linux for coding and training models while Windows can be more convenient for things like gaming and school work. This setup didn't have any drawbacks for me until I started working with the Barracuda library for Unity. Unity is installed on Windows but my environment for training deep learning models is on Linux. This is inconvenient when I want to test out a newly trained model in Unity. I decided to try Microsoft's Windows Subsystem for Linux (WSL) in the hopes that it would remove the need to switch between operating systems.



However, Microsoft has been taking steps to make Windows more appealing for software development. One of the those steps is the Windows Subsystem for Linux (WSL). WSL lets you run a Linux environment on Windows without the overhead of traditional virtual machines.

I didn't have a use case for WSL until I started working with the Barracuda library for Unity. I have Unity installed on Windows but my environment for training models is in Linux. So far, I've been 

Nvidia Driver for CUDA on WSL

* [download](https://developer.nvidia.com/cuda/wsl/download)



### Select Target Platform

![select_target_platform](..\images\enable-cuda-on-wsl2\select_target_platform.png)

### Download Installer

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.2.0/local_installers/cuda-repo-ubuntu2004-11-2-local_11.2.0-460.27.04-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2004-11-2-local_11.2.0-460.27.04-1_amd64.deb
sudo apt-key add /var/cuda-repo-ubuntu2004-11-2-local/7fa2af80.pub
sudo apt-get update
sudo apt-get -y install cuda
```



### Install Anaconda

```bash
cd ~
wget https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh
chmod +x Anaconda3-2020.11-Linux-x86_64.sh
./Anaconda3-2020.11-Linux-x86_64.sh
```

```bash
Last updated September 28, 2020


Do you accept the license terms? [yes|no]
[no] >>> yes
```



```bash
Anaconda3 will now be installed into this location:
/home/innom-dt/anaconda3

  - Press ENTER to confirm the location
  - Press CTRL-C to abort the installation
  - Or specify a different location below

[/home/innom-dt/anaconda3] >>>
```



```bash
Preparing transaction: done
Executing transaction: done
installation finished.
Do you wish the installer to initialize Anaconda3
by running conda init? [yes|no]
[no] >>> yes
```



```bash
wsl.exe --shutdown Ubuntu
```



```bash
(base) innom-dt@INNOM-DT:~$ python
Python 3.8.5 (default, Sep  4 2020, 07:30:14)
[GCC 7.3.0] :: Anaconda, Inc. on linux
Type "help", "copyright", "credits" or "license" for more information.
>>>
```





```bash
conda install -c fastai -c pytorch -c anaconda fastai gh anaconda
```



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



## Conclusion

