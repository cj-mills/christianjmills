---
categories:
- game-dev
- notes
date: '2021-12-29'
description: My notes from Squirrel Eiserloh's presentation on 1D nonlinear transformations
  for game development.
hide: false
search_exclude: false
title: Notes on 1D Nonlinear Transformations for Games

aliases:
- /Notes-on-1D-Nonlinear-Transformations-for-Games/
---

* [Overview](#overview)
* [Motivations](#motivations)
* [Implicit and Parametric Equations](#implicit-and-parametric-equations)
* [Parametric Manipulations](#parametric-manipulations)
* [Parametric Opportunities](#parametric-opportunities)
* [The Big Idea](#the-big-idea)
* [The Two Most Important Number Ranges](#the-two-most-important-number-ranges)
* [Normalized Non-Linear Functions](#normalized-non-linear-functions)
* [Range Mapping](#range-mapping)
* [Related Material](#related-material)



## Overview

Here are some notes I took while watching Squirrel Eiserloh's [presentation](https://www.youtube.com/watch?v=mr5xkf6zSzk) covering how 1D nonlinear transformations can be used by game programmers.



## Motivations

- [Juice it or lose it talk](https://www.youtube.com/watch?v=Fy0aCDmgnxg): Makes the case for thinking about is something linear or non-linear, is it mechanical or organic
- [The art of screenshake](https://www.youtube.com/watch?v=SkgkIXZ_13Y):

## Implicit versus Parametric Equations

- Implicit equations are rules:
    - Equation for a circle: $x^{2} + Y^{2} = 25$
        - A point is either on the circle or not
- Parametric functions
    - Yield an output for an input value
        - $P_{x} = 5 \cdot cos(2 \pi \cdot t)$
        - $P_{y} = 5 \cdot sin(2 \pi \cdot t)$
    - $P(t) = ?$
    - $P(t) = (t, t \cdot cos(t), t*sin(t))$
        - $(x, y, z)$
        - Generates a spiral that increases in radius along the x axis
    - Anything you can express in terms of a single float as input
    - A common float input is “time”
    

## Parametric Manipulations

- Do NOT mess with the interpolation itself (e.g. color, position, AI disposition, etc.)
- Instead just mess the parameter

## Parametric Opportunities

- Anytime you have a single float to change
- Anytime you can express something in terms of a single float
- Pretty much whenever you use time

## The Big Idea

- You can make any parametric equation more interesting without modifying the function itself, without knowing anything about the function

## The Two Most Important Number Ranges

- $[0,1]$
    - Useful for fractions
        - % shadow
        - % luminance
        - % falloff
        - % complete
        - % damage
        - % experience
        - % cost
        - % penalty
        - % fog
        - % AI aggression
        - % chance to hit
        - % chance to drop loot
        - % time to complete
        - Fuzzy Logic
        - Most anything parametric
- $[-1,1]$
    - Useful for deviations
        - noise
        - perturbation
        - terrain and map generation
        - variation
        - distribution
        - sinusoidal
        - AI response curves
    

## Normalized Non-Linear Functions

- $[0,1]$ 
- Functions for which:
    - $P(0) = 0$
    - $P(1) = 1$
    - $P(t) \ != t$
- Examples
    - Position over time
    - Scale over time
    - Alpha over time
    - Color over time
    - Strength over time
    - Aggression over time
- Also called
    - easing functions
    - filter functions
    - lerping functions
    - tweening functions

## Range Mapping

- can be applied during middle of range-mapping

```csharp
out RangeMap(in, inStart, inEnd, outStart, outEnd)
{
	// Puts in [0, inEnd - inStart]
	out = in - inStart;
	// Puts in [0,1]
	out /= (inEnd - inStart);
	// in [0,1]
	out = ApplySomeEasingFunction(out);
	// Puts in [0, outRange]
	out *= (outEnd - outStart);
	// Puts in [outStart, outEnd]
	return out + outStart
}
```

### SmoothStart

- $SmoothStartN(t) = t^{n}$
- Larger exponents result in steeper curve
- Will always start and end at the same time, regardless of exponent value
- Technique
    - exponentiating
    

### SmoothStop

- $SmoothStopN(t) = 1 - (1 - t)^{n}$
- Larger exponents results in longer braking period at the end
- Techniques
    - exponentiating
    - flipping
    

### $Mix(a, b, weightB, t)= a + weightB(b-a)$

- $Mix(SmoothStart2, SmoothStop2, blend, t)$
- $SmoothStart2.2 = Mix(SmoothStart2, SmoothStart3, 0.2);$
    - Way faster than using the `pow()` function

### Crossfade

- Like Mix, but use t itself as the mix weight
- Also called SmoothStep

### Scale

- $Scale(Function, t) = t \cdot Function(t)$

### ReverseScale

- $ReverseScale(Function, t) = (1-t) \cdot Function(t)$

$Arch2(t) = Scale(Flip(t)) = t \cdot (1-t)$

$SmoothStartArch3(t) = Scale(Arch2, t) = t^{2}(1-t)$

$SmoothStopArch3(t) = ReverseScale(Arch2, t) = t(1-t)^{2}$

 

$SmoothStepArch3(t) = ReverseScale(Scale(Arch2, t), t)$

$BellCurve6(t) = SmoothStop3(t) \cdot SmoothStart3(t)$

## Related Material

[Juice it or lose it - a talk by Martin Jonasson & Petri Purho](https://www.youtube.com/watch?v=Fy0aCDmgnxg)

[The Art of Screenshake - Jan Willem Nijman - Vlambeer](https://www.youtube.com/watch?v=SkgkIXZ_13Y)

   

**References:**

* [Math for Game Programmers: Fast and Funky 1D Nonlinear Transformations](https://www.youtube.com/watch?v=mr5xkf6zSzk)


<!-- Cloudflare Web Analytics --><script defer src='https://static.cloudflareinsights.com/beacon.min.js' data-cf-beacon='{"token": "56b8d2f624604c4891327b3c0d9f6703"}'></script><!-- End Cloudflare Web Analytics -->