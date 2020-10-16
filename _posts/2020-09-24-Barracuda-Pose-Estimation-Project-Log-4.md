---
title: Barracuda Pose Estimation Project Log Pt.4
layout: post
toc: false
description: I'm an idiot.
categories: [unity,project,log]
hide: false
search_exclude: false
---

## I'm an Idiot

Well I feel like an idiot. It turns out I've had the x and y offset vectors swapped this whole time. That explains why the gap problem with the heatmaps seemed so much worse when using lower resolutions. The offsets had to be bigger but they were going in the wrong directions. I can't believe I didn't check the values for those sooner to make sure they made sense with the current input image. Also, this completely resolved the issues with the MobileNet models mentioned in the previous post. They work great now. Well, lesson learned I guess. On the bright side, the project works way better now! Annoyingly, now that I've finally fixed this issue the post processing bottleneck is all the more noticeable.



Before:

![pose_estimation_swapped_offsets](\images\pose_estimation_swapped_offsets.gif)

After:

![pose_estimation_corrected_offsets](\images\pose_estimation_corrected_offsets.gif)
