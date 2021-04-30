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

* [Targeted In-Game Style Transfer](#targeted-in-game-style-transfer)

  

## Targeted In-Game Style Transfer

I spent some time testing out a method for stylizing only specific GameObjects in Unity. My approach was to use a second camera along with layers and culling masks. It actually worked pretty well, with no real performance cost. Once concern I have with my current approach is that it assumes the outline of the targeted GameObjects will be the same in the stylized output. This might not always be the case.

![targeted_stylization](..\images\daily_recaps\recap-5\targeted_stylization.jpg)

