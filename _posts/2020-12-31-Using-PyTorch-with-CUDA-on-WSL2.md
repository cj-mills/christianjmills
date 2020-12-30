---
title: Using PyTorch with CUDA on WSL2
layout: post
toc: false
comments: true
description: This post covers getting PyTorch to run with CUDA on WSL2.
categories: [tutorial]
hide: true
permalink: /:title/
search_exclude: false
---

* [Introduction](#introduction)
* 
* [Conclusion](#conclusion)

## Introduction



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

