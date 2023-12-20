---
title: Notes on Procedural Map Generation Techniques
date: 2021-12-9
title-block-categories: true
hide: false
search_exclude: false
comments:
  utterances:
    repo: cj-mills/christianjmills
description: My notes on Herbert Wolverson's talk on procedural map generation techniques
  from the 2020 virtual Roguelike Celebration.
categories: [game-dev, procedural-generation, notes]

aliases:
- /Notes-on-Procedural-Map-Generation-Techniques/

---




* [Overview](#overview)
* [Influential Games](#influential-games)
* [Simple Room-Placement](#simple-room-placement)
* [Binary Space Partition Rooms](#binary-space-partition-rooms)
* [Cellular Automata](#cellular-automata)
* [Drunkard's Walk](#drunkards-walk)
* [Diffusion Limited Aggregation](#diffusion-limited-aggregation)
* [DLA with a Central Attractor](#dla-with-a-central-attractor)
* [Voronoi Diagrams](#voronoi-diagrams)
* [Perlin and Simplex Noise](#perlin-and-simplex-noise)
* [You can use more than one technique](#you-can-use-more-than-one-technique)
* [Removing Unreachable Areas](#removing-unreachable-areas)
* [The Hot Path](#the-hot-path)
* [Telling a Story](#telling-a-story)



## Overview

My notes on Herbert Wolverson's [talk](https://www.youtube.com/watch?v=TlLIOgWYVpI) on procedural map generation techniques from the 2020 virtual Roguelike Celebration.



## Influential Games

### Rogue (1980)

- One of the first uses of procedural generation
- Generates up to 9 rooms and connects them randomly
- Used procedural generation because they needed to keep the game small
- Different map every time the game is started
- Effectively infinite replay

### Dwarf Fortress (2006 - Present)

- Probably crammed the most procedural generation into one game
- Procedurally Generates:
    - Massive overworld with sweeping mountain ranges, forests, volcanoes, demon-infested fortresses
    - Civilizations that either like or hate each other
        - Can drill down to a single person and their procedurally generated backstory
    - Mid-scale
        - Can zoom into any particular block on the map to find it is beautifully rendered and still matches the overall shape of the overworld
        - Trees gain and lose foliage depending on their type and biome
            - Their type spawns the appropriate biome

**Takeaway:** The randomness does not define the above games. The randomness is fed into an algorithm the generates something that approximates what you want to get, but ensures that it is different every time

## Simple Room-Placement

![](https://github.com/thebracket/roguelike-celebration-2020/blob/master/gifs/01-RoomCorridors.gif?raw=true){fig-align="center"}


1. Start with a solid map (random rectangle)
2. Fill the map with walls.
3. Randomly pick a room location.
    1. If the map location is not already occupied by another room, add the room
4. Keep picking rooms.
5. Join the rooms you kept with corridors.
    1. Example: Using a simple dog leg algorithm that randomly switches between being either vertical first or horizontal first.
    

## Binary Space Partition Rooms

![](https://github.com/thebracket/roguelike-celebration-2020/blob/master/gifs/02-Bsp.gif?raw=true){fig-align="center"}

- Similar results to random room placement, better spaced out.
    - Used in Nethack
1. Divide map into two. Randomly decide whether to divide vertically or horizontally. 
2. Divide area into two. 
3. Repeat. 
4. Use divided space for room.
- Add a gutter of one tile around to avoid rooms joining together (unless desired)

## Cellular Automata

![](https://github.com/thebracket/roguelike-celebration-2020/blob/master/gifs/03-cellular.gif?raw=true){fig-align="center"}

- Evolve order from chaos.
- Popularized in Conway's Game of Life.
1. Make a random map.
2. Make a copy of it.
3. Apply cell life rules to each tile. 
    1. Iterate every tile that isn't on the edge and count the number of neighbors, including diagonals.
        1. If there are no neighbors, then it becomes a wall
        2. If there is one to four neighbors, it becomes empty
        3. If there are five or more neighbors, it becomes a wall
        4. Tweak rules to suit specific game
4. Repeat.
- Simple
- Fast
- Deterministic (same random seed generates the same results)

## Drunkard's Walk

![](https://github.com/thebracket/roguelike-celebration-2020/blob/master/gifs/04-drunkard.gif?raw=true){fig-align="center"}

- Find Umber Hulk. Insert beer.
- Place Hulk randomly on solid map. See what he smashes
- Hulks stop when they leave the map, or pass out after n steps.
1. Start with a solid map
2. Random walk through map
3. Tiles get removed based on walking path
4. Pick maximum number of walking steps
5. Repeat.
- Guarantees the map will be contiguous
- Tends to generate maps that look like it was carved out by water.
    - Ideal for creating limestone caverns and similar.
    

## Diffusion Limited Aggregation

![](https://github.com/thebracket/roguelike-celebration-2020/blob/master/gifs/05-dla-inward.gif?raw=true){fig-align="center"}

- [Explanation](http://www.roguebasin.com/index.php/Diffusion-limited_aggregation) 
- Start with a targeted seed.
- Randomly - or not - fire particles at it.
- Dig out the last edge the particle hit.
1. Start by digging out a small target seed
2. Pick a random point anywhere on the map
3. Pick a random direction
4. Shoot a particle
    1. Keep shooting until you hit something
    2. If you hit a target area, carve out the last solid area you passed through
- Tends to give you a very winding open map
- Guaranteed to be contiguous
- Lots of ways to tweak the algorithm to make things more interesting

## DLA with a Central Attractor

![](https://github.com/thebracket/roguelike-celebration-2020/blob/master/gifs/06-dla-attractor.gif?raw=true){fig-align="center"}

- More likely to always hit the target
- Randomly spawn your starting point and then shoot the particle directly at the middle of the map
- Helps ensure your get an open space in the middle
    - Ideal, for example, to put a dragon with his hoard
- More interesting pattern around the edges of the map
- Can also apply symmetry down the vertical
    - Use sparingly

## Voronoi Diagrams

![](./images/1024px-Euclidean_Voronoi_diagram.png){fig-align="center"}

- Randomly (or deliberately) placed seeds.

- Each tile joins the closest seed.

- Vary distance heuristic for different effects.

- Iterate every point on the map and it joins the area belonging to the closest seed.
    - Example Algorithms:
        - Delauney triangulations
        - Brute force
    
- Can customize the result using a different distance algorithm to determine which group every tile joins
    - Pythagorean distance
    - Manhattan distance
    
- Find the edges, place walls there and wind up with an alien cell structure

- Can be used to determine spawning placement/behavior based on cell location

- Can be used for effective city generation
    - Apocalypse Taxi
        - ![](https://img.itch.zone/aW1hZ2UvMzIxNDkxLzE1ODg3MjYuanBn/original/Mtk75O.jpg){fig-align="center"}
        - Uses the edges of the generated cells to determine where the roads went
        - Randomly populated the content of each cell with something like "heavy industrial city", "light industrial city", etc.
    
    
    [Apocalypse Taxi](https://thebracket.itch.io/apocalypse-taxi)
    
- Can be combine with other techniques

## Perlin and Simplex Noise

![](https://github.com/thebracket/roguelike-celebration-2020/blob/master/gifs/11-noise-overworld.gif?raw=true){fig-align="center"}

- Basically a bunch of gradients combined together with a few variables
- Can generate it in either two or three dimensions
- X/Y Value: gives a number in the range $[-1,1]$
- Smoothly moving either up or down
- Continuous
- Octaves: number of gradients being mixed in.
- Gain: how long the various gradients last
- Lacunarity: adds in randomness
- Frequency: how frequently each of the various octaves peaks
- Commonly used to make an overworld/terrain map
- Problem: The gradients are kind of dull
    - Can be addressed by adding a second layer of noise that is more "bumpy"
        - Interpolate between smooth and bumpy gradients as you zoom in and out
- Easy to implement
- Can also be used to generate realistic looking clouds, particles, wood grain

## You can use more than one technique

- Can help generate maps that tell a story
- Example: Use BSP to generate a more structured part of the map leads into a more chaotic section generated using cellular automata
- Example: Use DLA for erosion
    - Take map and then use DLA to fire particles at it to blast parts of the map away
    - Map becomes more organic-looking while keeping its basic structure
- Example: Mix procedurally generated content with human-made prefabs

## Dijkstra Maps

![](./images/Dijk_basic.png){fig-align="center"}

1. [Explanation](http://www.roguebasin.com/index.php/The_Incredible_Power_of_Dijkstra_Maps)
1. Start with 1 or more starting points.
2. Rest of the map " sentinel" value - unreachable
3. Set points adjacent to start to 1.
4. Points adjacent to those 2.
    1. Keep going until whole map walked

## Removing Unreachable Areas

- Cellular automata can give you chunks of the map that you can't get to.
1. Find Central Start
2. Run Dijkstra
3. Cull tiles without a valid distance.
    1. Or hide it for underground levels

**Finding a Starting Point**

- Find a desired starting point
- Find closest open tile for actual start.

**Finding an Endpoint**

- Use distance to target
- Use Dijkstra to find farthest point

## The Hot Path

- Path-find from start to end
- Dijkstra Map with the path as starting points.
- $<n$ distance is "hot path"
- Can use A* algorithm
- Can be used to minimize branching in game map by culling irrelevant parts of the map outside the hot path.
- Or "bonus" content to reward exploration of the hot path.

## Telling a Story

- Rooms are ordered.
- Story progression is in order, but RNG is retained
- Maybe room 5 has a locked door, meaning the key must be in rooms 1-4.

Takeaway: Guide the randomness and use algorithms to check the randomness.



**References:**

* [Herbert Wolverson - Procedural Map Generation Techniques](https://www.youtube.com/watch?v=TlLIOgWYVpI&list=WL&index=112)

* Source Code for Talk: [GitHub Repository](https://github.com/thebracket/roguelike-celebration-2020)

* Online Book: [Roguelike Tutorial - In Rust](https://bfnightly.bracketproductions.com/chapter23-prefix.html)

  



