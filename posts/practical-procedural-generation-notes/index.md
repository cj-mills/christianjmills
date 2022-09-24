---
aliases:
- /Notes-on-Practical-Procedural-Generation/
categories:
- procedural-generation
- notes
date: '2021-12-29'
description: My notes from Kate Compton's talk on practical techniques for procedural
  generation.
hide: false
layout: post
search_exclude: false
title: Notes on Practical Procedural Generation
toc: false

---

* [Overview](#overview)
* [Examples](#examples)
* [Steps](#steps)
* [The IKEA Catalog of Generativity](#the-ikea-catalog-of-generativity)
* [Making use of Generativity](#making-use-of-generativity)
* [Further Reading](#further-reading)



## Overview

Here are some notes I took while watching Kate Compton's [talk](https://www.youtube.com/watch?v=WumyfLEa6bU) covering practical procedural generation techniques.



## Examples

[Minecraft Official Site](https://www.minecraft.net/en-us)

[No Man's Sky](https://www.nomanssky.com/)

[Bay 12 Games: Dwarf Fortress](http://www.bay12games.com/dwarves/)

[These Monsters](https://procedural-generation.tumblr.com/post/145212706991/these-monsters-strangethink-released-a-new-project)

[Cameron's Yavalath Page](http://cambolbro.com/games/yavalath/)

- Made a system that could generate game rules
- Made a player that could play arbitrary games
- Had virtual players play thousands of games, until he found a game that was pretty well balanced

[PANORAMICAL on Steam](https://store.steampowered.com/app/284260/PANORAMICAL/)

[It is as if you were playing chess](http://www.pippinbarr.com/games/itisasifyouwereplayingchess/)

[Fitzwilliam Darcy's Dance Challenge](https://squinky.itch.io/darcy)

[The Treachery of Sanctuary - CHRIS MILK](http://milk.co/treachery)

[Kinematics Dress](https://n-e-r-v-o-u-s.com/projects/sets/kinematics-dress/)

[V&A Design a Wig](https://www.vam.ac.uk/designawig/)

[Toca Hair Salon - The Power of Play - Toca Boca](https://tocaboca.com/app/toca-hair-salon/)

- Lots of generative content uses extremely sophisticated and brilliant AI and fails anyway
- Some of the best generative content is simple
- The hardest part of procedural content is design

## Steps

1. Understand the design space
2. Enumerate your constraints
3. Understand the process
4. Pick a generative method
5. Iterate and be flexible
    - A lot of great generative projects are things that were tried because it is a stupid idea

What are you making?

- Be specific
    - Level generator
    - Character creator
    - Abstract art generator
    - cocktail recipe generator
    - game title generator
    - conversational character
    - poetry generator
    - twitterbot

Making an artist-in-a-box

- teaching an algorithm to make art like an artist
- Find and expert (or read their writing)
    - How do they think through a problem?
    - Example Question: “If you are designing a creature, what do you do?”
    - Example Answer: They start by drawing a bean shape as a base for the creature, and hangs a mouth on it.

Additive and Subtractive Methods

- Build up a space of good stuff
- (optional) Remove bad stuff
- Vocab:
    - Possibility space
    - Expressive range

## The IKEA Catalog of Generativity

- A catalog of generative methods and why you might chose each

### Additive Methods

#### Tiles

- Works well for
    - Something you can break into (equal-sized) regions
    - where tile-to-tile placement don’t need to be constrained
        - Can use WaveFunctionCollapse when placement needs to be constrained
    - but you can still get emergence from the placement of tiles
    - one of the oldest forms

#### Grammars

- Recursively make things from other things
- [Tracery](http://tracery.io/) and other templating systems (for text)
- L-Systems (for geometry)
- Replacement grammars
  
    [Level design as model transformation - Proceedings of the 2nd International Workshop on Procedural Content Generation in Games](https://dl.acm.org/doi/10.1145/2000919.2000921)

#### Distribution

- put down a bunch of stuff
- can use random numbers (actual randomness does not look good)
- real distributions are hierarchical and clustered, but also maintain spacing
- Barnacling: when you have a large object in your world, there should be medium sized objects around it and smaller objects around those
- Footing: When two things intersect, there should be an awareness of them intersecting
    - Example: If you stick tree in the ground, there will be dirt piled up around it
- Greebling: **cosmetic detailing** added to the surface of an larger object that makes it appear more complex or technologically advanced
- Options
    - start with a grid, and offset a bit
        - (less obvious with a hex grid)
    - Use a voronoi diagram with easing
    - Do it properly with a [Halton Sequence](https://en.wikipedia.org/wiki/Halton_sequence)

#### Parametric

- An array of floats representing settings, “morph handles”
- modellable as points in an N-dimensional cube
- Any position is a valid artifact
- You can do genetic algorithms
    - or use directed walks through the space
    - or “regionize” the space

#### Interpretive

- Start with an input
    - Run an algorithm to process data into some other data
- You have a simple structure
    - some distribution of points, a skeleton, a connectivity map, a curve or path and want to make it more complex
- Examples:
    - Noise (Perlin/simplex)
    - Voronoi/Delaunay
    - Constructive Solid Geometry Extrusion, revolution
    - Metaballs
    - Fractals, mathematical models of impossible shapes
    - (Hypernom, Miegakure)
        - low control, high weirdness, not suitable for most games

#### Simulations

- Particle trails
    - simulate particle path responding to forces
- draw directly
- OR record path and use for extrusions or distributions (Photoshop brushes)
- Goes great with user input (Leapmotion, Kinect)
- Cellular automata
- Agent-based simulations
- Physics simulation

### Subtractive Methods

#### Saving Seeds

- Seeded random numbers
    - Same seed, same random generation
        - Make sure nothing is framerate or input dependent
- Whitelist a catalog of **known good** content
    - It’s faster to verify questionable content than to build a testing function

#### Generate and test

- If you can write an algorithm to judge “quality”
    - Throwaway vs ranking/prioritization
        - Use ranking/prioritization
    - Test for brokenness/connectivity
- Beware of false functions
    - beware the “fun equation”

#### Computationally exploring the possibility space

- Also called “search”
    - Brute force search
        - “Find the tallest creature that the tool can make”
        - “Make a level that has these properties”
    - Hill-climbing
        - Genetic algorithms
        - Works best with parametric methods

#### Constraint-solving

- You can describe a possibility space and constraints, just find the valid parameters.
- Inverse Kinematics-solving
- Answer set solving
    - [Potassco Clingo](https://github.com/potassco/clingo)
    - DO NOT WRITE YOUR OWN SOLVER
- Brute force
    - pay attention to exponential growth



## Making use of Generativity

- You can generate many things
- They are all mathematically unique
- But they aren’t perceived as unique
- Is this a problem?
    - Do not boast about really big numbers

### Different kinds of generative content

- Background
    - In-fill (don’t be empty)
- Perceptual differentiation
- Perceptual uniqueness
- Characterful
    - Test: Would you write a fanfic for this generated item?

### Ownership: MSG for PCG

- Allow users to name content
- Showing off content with their name attached, to a large audience
    - The “victoriain explorers club” model
- promote players
- Let players take credit for your generativity
    - creators, curators, retellers

### Data Structures: Make your life easier

- A/B test generators
- Release new generative content safely
- Create editors and run user-made generators safely
- Visualize your generators

## Further Reading

* [Encyclopedia of
  Generativity](http://www.galaxykate.com/zines/EncyclopediaOfGenerativity-KateCompton.pdf)

* [So you want to build a generator...](https://galaxykate0.tumblr.com/post/139774965871/so-you-want-to-build-a-generator)

* [ICCC'21: Int. Conference on Computational Creativity](https://computationalcreativity.net/iccc21/)

* [My Liner Notes for Spore - Chris Hecker's Website](http://www.chrishecker.com/My_Liner_Notes_for_Spore)

* [A Brief History of Spore](http://www.levitylab.com/blog/2011/02/brief-history-of-spore/)

* [Danesh](http://www.danesh.procjam.com/): A tool to help  people explore, explain and experiment with procedural generators



**References:**

* [Practical Procedural Generation for Everyone](https://www.youtube.com/watch?v=WumyfLEa6bU)

