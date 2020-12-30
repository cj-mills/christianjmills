---
title: Enable CUDA on WSL2
layout: post
toc: false
comments: true
description: This post covers the steps I took to get CUDA working on WSL2.
categories: [tutorial]
hide: true
permalink: /:title/
search_exclude: false
---

* [Introduction](#introduction)
* 
* [Conclusion](#conclusion)

## Introduction





```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.2.0/local_installers/cuda-repo-ubuntu2004-11-2-local_11.2.0-460.27.04-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2004-11-2-local_11.2.0-460.27.04-1_amd64.deb
sudo apt-key add /var/cuda-repo-ubuntu2004-11-2-local/7fa2af80.pub
sudo apt-get update
sudo apt-get -y install cuda
```



## Conclusion

