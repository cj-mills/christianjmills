---
categories:
- recap
date: 2021-4-29
description: A summary of what I worked on today.
hide: false
layout: post
search_exclude: false
title: Daily Recap
toc: false

aliases:
- /Daily-Recap-5/
---

* [Targeted In-Game Style Transfer](#targeted-in-game-style-transfer)

  

## Targeted In-Game Style Transfer

I spent some time testing out a method for stylizing only specific GameObjects in Unity. My approach was to use a second camera along with layers and culling masks. It actually worked pretty well, with no real performance cost. One concern I have with my current approach is that it assumes the outline of the targeted GameObjects will be the same in the stylized output. This might not always be the case.

![targeted_stylization](./images/targeted_stylization.jpg)





<!-- Cloudflare Web Analytics --><script defer src='https://static.cloudflareinsights.com/beacon.min.js' data-cf-beacon='{"token": "56b8d2f624604c4891327b3c0d9f6703"}'></script><!-- End Cloudflare Web Analytics -->