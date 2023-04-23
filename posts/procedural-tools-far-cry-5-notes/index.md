---
categories:
- procedural-generation
- game-dev
- notes
date: 2021-12-9
description: My notes from Etienne Carrier's overview of the procedural tools developed
  to create Far Cry 5.
hide: false
search_exclude: false
title: Notes on the Procedural Tools Used to Make Far Cry 5

aliases:
- /Notes-on-the-Procedural-Tools-Used-to-Make-Far-Cry-5/
---

* [Overview](#overview)
* [Introduction](#introduction)
* [Goal of the pipeline](#goal-of-the-pipeline)
* [Available procedural tools](#available-procedural-tools)
* [User point of view](#user-point-of-view)
* [Pipeline](#pipeline)
* [Cliffs tool in detail](#cliffs-tool-in-detail)
* [Biome tool in detail](#biome-tool-in-detail)
* [Conclusion](#conclusion)



## Overview

I recently started learning about tools and techniques used for procedural generation in game development. I came across a [presentation from 2018](https://www.youtube.com/watch?v=NfizT369g60) given by Etienne Carrier, a technical artist at Ubisoft. He provides an overview the tools he developed for the production of Far Cry 5. Below are some notes I took while watching.



## Introduction

- How to maintain quality of features like forests through multiple terrain iterations without manually updating them.

## Goal of the pipeline

- Macro management tool to fill up the world with natural looking content
- Maintain content consistency with the terrain topology
- Automatable
- Deterministic (same result given the same input)
- User friendly

## Available procedural tools

- Freshwater tool: generates lakes, rivers, streams, and waterfalls
- Fences and power-lines
- Cliff generation on steep terrain surfaces
- Biome tool: generates vegetation throughout the world
- Fog density map generation (2D map based on topology, forests, freshwater, etc)
- Worldmap terrain

## User point of view

1. Terrain Terraforming Pass
2. Freshwater: Artist lays down fresh water network using curves and splines
3. Cliffs: based on terrain slope
4. Vegetation: Artist uses biome painter → color density = forest density?
    1. Tool naturally distributes grass and other vegetation
    2. Reacts to water proximity
    3. Avoids adding vegetation on cliff erosion lines
    4. Altitude affects forest density
5. Points of Interest: Artist can manually tweak results and specific locations
    1. Example: lay down road splines
    2. Example: clear out spots for bases
    3. Assets like houes, sheds, tools, etc. still need to be manually placed
        1. Might require tweaking of local biome 
    4. Add fences using splines
    5. Add power-line networks using splines
        1. Transformer boxes automatically added where required
        2. Refresh biome to account for new power-lines
6. Terrain can be adjusted at any time and is non-destructive

## Pipeline

1. Dunia2 (FarCry Game Engine/Editor) Inputs → Houdini
2. Houdini Outputs → Dunia2
3. Inputs via Python Scripts:
    - World information
    - File Paths
    - Terrain Sectors
    - Splines and Shapes
4. Inputs from Disk:
    - Height maps (.raw)
    - Biome painter (.png)
    - 2D terrain masks (.png)
    - Houdini Geometry (.geo or .bgo)
5. Outputs saved (data is saved temporarily as buffers on disk):
    - Entity point cloud
        - Exported with object id
        - Could be anything that has a position in editor
            - Vegetation assets
            - Rocks
            - Collectibles
            - Decals
            - VFX
            - Prefabs
    - Terrain texture layers
    - Terrain height map layers
    - 2D terrain data
    - Geometry
    - Terrain Logic zones
6. Tools Interconnectivity
    - Each tool will output necessary masks to affect the next ones
    - Cooking order is important if one tool requires input from a previous one
        - freshwater → roads → fences & power-lines → cliffs → biomes → fog → world map

## Cliffs tool in detail

### Tool input

- slope terrain data → slope threshold (is it too steep to walk on?) → cliffs input
- prepare geometry by remeshing to get uniform mesh triangles

### Stratification

[stratification](https://www.britannica.com/science/stratification-geology)

- Visible horizontal lines formed by the accumulation of sedimentary rock and soil
- Slice input geometry into strata chunks
    - Each strata has a random thickness
    - Assign strata id to each slice
    - Control stata angle with RGB painter
    - Split noise (on low res mesh) to split mesh into two groups
        - run stratification tool on both groups with different seed values to break up strata lines
    - Extrude and displace strata
    - Reduce mesh triangle count
    - Exported geometry is divided per sector

### Cliffs are shaded in-game (same texture as terrain underneath)

### Erosion

- Run a flow simulation
    - Points scattered on cliff surfaces that will flow down the slope to create an erosion effect
- Use erosion data to scatter crumbled rocks on the erosion surfaces
    - export as point cloud
- use noise to mix two different cliff textures

### Vegetation growing surfaces

- Scatter vegetation on viable cliff surfaces
    - Clear above or not using raycast

### Exported data

- Cliffs geometry
- Entities point cloud
- Terrain Texture IDs
- 2D Cliffs color
- 2D Cliffs mask

## Biome tool in detail

### Input

- Generate terrain from height map

### Terrain abiotic data

- Physical features of the land that are generated from terrain topology
    - Occlusion
    - Flow
    - Slope
    - Curvature
    - Illumination
    - Altitude
    - Latitude
    - Longitude
    - Wind
- Importing 2D data
    - Biome painter data
    - Procedurally generated data
        - Freshwater masks
        - Roads masks
        - Fences mask
        - Power lines mask
        - Cliffs mask

### Processing main biomes

- Biome and sub-biome
    - Main biome (e.g. Mountain) large scale
        - sub biome: mountain grass
        - sub biome: mountain forest
    - Main biome processes power line clearings
    - Sub-biomes recipes
        - node-based
        - mountain forest
            - ingredient: aspen undergrowth
            - ingredient: dead conifer
            - ingredient: hemlock
        - mountain grass
        - intermountain prarie
    - Viability
        - Each species is fighting for ground to grow and thrive
        - viability is defined by setting up favored terrain attributes for each species
        - species that accumulate the most viability will win over others
        - Factors
            - occlusion terrain data
            - flow map
            - viability radius
                - Example: a tree with a lower viability will be discarded if it is withing the radius of a tree with a higher viability
            - priority radius
                - Useful for populating bushes, etc. under larger trees
                - filtering will process priority first
                - if priority is equal, the viability will be used instead

### Combine terrain data

- Mix terrain abiotic data to achieve specific distribution patterns
    - Occlusion
    - Altitude
    - Flow map
    - Noise (e.g. perlin noise)
    - Exclusion masks (generated from fresh water maps, power lines, etc.)
    - Used as viability for species

### Sizes

- multiple sizes for same species
- driven by viability
    - asset size value linked to viability value
- small and young trees more likely to spawn at edge of forest
- taller trees more likely near the center
- Altitude
- Sizes variation
    - several assets of the same size
    - probability control on each variation
    - forest canopy
        - ecological succession
        - Overstory
        - Midstory
        - Woody understory
        - Herbaceous understory
    - Age parameter
        - sine-distance field generated from viability data
        - ramp for profile shape
    - Density
        - ramp from size, age, or viability
        - slope aspect effect on density
        - illumination (how much light does the vegetation receive)

### Entities Color

- per instance color ramp variation

### Rotation

- orient on terrain slope
    - e.g. grass leaning towards water
    - pre-bended tree trunks from growing on a slope
- grassland oriented on wind vector map
- percent angle of terrain
- rotation jitter

### Terrain elements affected by vegetation in biomes

- Terrain deformation
    - terrain elevation around trunks
    - height map layer
    - need to blend assets with terrain
        - generate matching terrain texture underneath asset
- Terrain textures
    - pine needles
    - acorns
    - dead branches
    - shadows affect how light other vegetation receives
- Terrain data output
- Terrain color
    - terrain texture tint

### Exported data

- reuse data on following species (e.g. viability map)
- species age output
- terrain height map
- entities point cloud
- terrain texture IDs
- terrain color
- forest mask

## Conclusion

### Lessons Learned

- procedural tools can generate a lot of data
- gives a lot of control over performance, game-play, and art
- design elegant tools that open up possibilities
- keep things simple
- listen to users
    - might prefer more manual controls instead of automation
- be flexible
- balance between control and automation

 


**References:**

* [Procedural World Generation of Ubisoft’s Far Cry 5](https://www.youtube.com/watch?v=NfizT369g60)





<!-- Cloudflare Web Analytics --><script defer src='https://static.cloudflareinsights.com/beacon.min.js' data-cf-beacon='{"token": "56b8d2f624604c4891327b3c0d9f6703"}'></script><!-- End Cloudflare Web Analytics -->