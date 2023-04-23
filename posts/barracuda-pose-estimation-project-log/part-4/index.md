---
categories:
- unity
- log
date: '2020-09-24'
description: I'm an idiot.
hide: false
image: ./images/pose_estimation_corrected_offsets.gif
search_exclude: false
title: Barracuda Pose Estimation Project Log Pt. 4

aliases:
- /Barracuda-Pose-Estimation-Project-Log-4/
- /log/project/unity/2020/09/24/Barracuda-Pose-Estimation-Project-Log-4
---

### Update 7/31/2021: [Barracuda PoseNet Tutorial 2nd Edition](../../barracuda-posenet-tutorial-v2/part-1/)

## I Fixed the Offset Vectors...

Well I feel like an idiot. It turns out I've had the x and y offset vectors swapped this whole time. That explains why the gap problem with the heatmaps seemed so much worse when using lower resolutions. The offsets had to be bigger but they were going in the wrong directions. 

I finally stumbled across my mistake by pausing the project and clicking through frame by frame to see if the corresponding x and y values made sense. I can't believe I didn't check the values for those sooner. I'm guessing that I was just glad it was working at all at the time and moved on to other things. Well, lesson learned I guess. 

On the bright side, the pose skeleton looks way better now! Comparing the pose skeletons with and without the offsets swapped makes it really apparent in hind sight. Also, this completely resolved the issues with the MobileNet models mentioned in the previous post. Their performance is still noticeably worse than the ResNet model but that's expected. 

Annoyingly, now that I've finally fixed this issue the post processing bottleneck is all the more noticeable. The MobileNet model is much more efficient but the current bottleneck means there isn't much difference in framerate.



### Before:
![](./videos/pose_estimation_swapped_offsets.mp4){fig-align="center"}


### After:

![](./images/pose_estimation_corrected_offsets.gif){fig-align="center"}








<!-- Cloudflare Web Analytics --><script defer src='https://static.cloudflareinsights.com/beacon.min.js' data-cf-beacon='{"token": "56b8d2f624604c4891327b3c0d9f6703"}'></script><!-- End Cloudflare Web Analytics -->