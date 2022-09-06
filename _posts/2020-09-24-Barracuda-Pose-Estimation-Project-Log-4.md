---
title: Barracuda Pose Estimation Project Log Pt. 4
layout: post
toc: false
comments: true
description: I'm an idiot.
categories: [unity,project,log]
hide: false
search_exclude: false
permalink: /:title/
---

### Update 7/31/2021: [Barracuda PoseNet Tutorial 2nd Edition](https://christianjmills.com/Barracuda-PoseNet-Tutorial-V2-1/)

## I Fixed the Offset Vectors...

Well I feel like an idiot. It turns out I've had the x and y offset vectors swapped this whole time. That explains why the gap problem with the heatmaps seemed so much worse when using lower resolutions. The offsets had to be bigger but they were going in the wrong directions. 

I finally stumbled across my mistake by pausing the project and clicking through frame by frame to see if the corresponding x and y values made sense. I can't believe I didn't check the values for those sooner. I'm guessing that I was just glad it was working at all at the time and moved on to other things. Well, lesson learned I guess. 

On the bright side, the pose skeleton looks way better now! Comparing the pose skeletons with and without the offsets swapped makes it really apparent in hind sight. Also, this completely resolved the issues with the MobileNet models mentioned in the previous post. Their performance is still noticeably worse than the ResNet model but that's expected. 

Annoyingly, now that I've finally fixed this issue the post processing bottleneck is all the more noticeable. The MobileNet model is much more efficient but the current bottleneck means there isn't much difference in framerate.



### Before:

![pose_estimation_swapped_offsets](../images/barracuda-pose-estimation-project-log/part-4/pose_estimation_swapped_offsets.gif)

### After:

![pose_estimation_corrected_offsets](../images/barracuda-pose-estimation-project-log/part-4/pose_estimation_corrected_offsets.gif)





