---
title: Daily Recap
layout: post
toc: false
comments: true
description: A summary of what I worked on today.
categories: [recap]
hide: false
permalink: /:title/
search_exclude: false
---

* [Unity GPU Targeting](#unity-gpu-targeting)

  

## Unity GPU Targeting

After a bit more testing, it turns out that the `-force-device-index` command line argument does work on Windows, but only for built games. The command line argument works even when there is no display cable plugged into the secondary GPU.

Something that I never really considered before is that it does not appear to be possible to switch which GPU an application is running on while the application is running. Every option I have found for manually selecting a GPU requires restarting the application to take effect.

