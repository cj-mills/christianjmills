---
aliases:
- /Notes-on-WaveFunctionCollapse-for-3D/
categories:
- procedural-generation
- notes
date: '2021-12-28'
description: My notes from Martin Donald's video on using the WaveFunctionCollapse
  algorithm for 3D modules.
hide: false
layout: post
search_exclude: false
title: Notes on WaveFunctionCollapse for 3D
toc: false

---

* [Overview](#overview)
* [Sudoku](#sudoku)
* [WaveFunctionCollapse Algorithm](#wavefunctioncollapse-algorithm)



## Overview

Here are some notes I took while watching Martin Donald's [video](https://www.youtube.com/watch?v=2SuvO4Gi7uY) providing an overview of the WaveFunctionCollapse algorithm as well as implementation considerations when using it with 3D modules.



## [Sudoku](https://en.wikipedia.org/wiki/Sudoku)

- Solitaire game played on a 9x9 grid
    - The 81 cells are grouped into 3x3 boxes
- Goal: Fill each space with a single number in range $[1,9]$
- Constraints
    - Every row, column and box must contain all numbers 1-9
    - No number may ever appear twice in the same row, column or box
- Each cell in an empty board could potentially any number in the range $[1,9]$
    - Each cell is in a superposition (occupying all nine possible states at once)
- Typically, there are a few cells already with numbers
    - Their superpositions are already collapsed to a single possibility
    - This reduces the number of possible values for the other cells in the relevant row, column, and square
        - Example: If a cell contains the number 5, no other cell in that row, column, or square can be a 5
    - After reducing the possible values for each cell in the board based on the initial cell values, select a cell with the lowest number of remaining possible values (i.e. lowest entropy)
    - Randomly select one of the remaining possible values for that cell
        - This again reduces the number of possible values for the other cells in the relevant row, column, and square
    - Eventually each cell will only contain one possible value

## WaveFunctionCollapse Algorithm

- A procedural solver that takes a procedural solver that takes a grid of cells, each occupying a super position containing all possible states for that cell
- Each tile (potential state) comes with its own set of adjacency rules
    - Only certain tiles can be above it
    - Only certain tiles can be below it
    - Only certain tiles can be left of it
    - Only certain tiles can be right of it
- The algorithm looks for the cell with the lowest number of possible tiles
    - It will randomly pick a cell with the lowest number, if there is more than one
- The algorithm then randomly picks one the possible tiles for that cell
- The algorithm then updates the list of possible tiles for the surrounding cells based on the selected tile
- The algorithm repeats until each cell contains only one possible tile

### Adjacency Constraints

- Tells the algorithm which tiles or modules can slot together for each side (e.g. 4 for 2D square tiles and 6 for 3D cube modules)
- Can have the model determine the adjacency constraints by looking at a hand crafted output
    - The model breaks the example down into tiles and keeps track of which tiles are placed next to each other and considers those combinations valid
- Can use a socket system
    - Mark each side of a tile or module with an identifier (e.g. a number)
    - Tiles and modules can only slot together if the connecting sides both have the same identifier value
    - Could increase granularity by having  three identifiers for each side, so that the outer edges and middle of each side are considered rather than the side as a whole

### 3D Modules

- Each cube module needs to have six lists of valid neighbors, one for each side of the cube.
- Whenever a cell is collapsed to one potential module, we remove modules from neighboring cells that are incompatible that are not present in the list of valid neighbors for the selected module
- To create the lists of valid neighbors, label the side of each module with socket identifiers
    - Loop over each module in the set of available modules
    - Store the positions of each vertex the sits along the edges of each of the six boundaries
    - Store and label the boundaries
    - This will build a dictionary of socket identifiers and side profiles
    - Add a tag to indicate symmetrical sockets
        - Check for symmetry by mirroring the vertex positions of each socket and check if it is still the same
        - If a socket is not symmetrical, store the mirrored and unmirrored versions as two different sockets,
        - Indicate mirrored socket with a specific tag
    - For top and bottom boundaries, store four rotated versions of each socket, indicating the which rotation index with a tag
        - Vertical sockets will be considered valid if they have the same socket name and rotation index

**Prototypes**

- The metadata for modules
    - The associated 3D mesh object
    - The rotation value
    - The six lists of valid neighbors
- Allows us to get around needing to export four different meshes for each module
    - Create four prototypes that reference the same mesh with different rotation index
- Store prototypes in a JSON file and load in game engine as a dictionary

**Building Prototypes**

- Create four prototype entries for each module, one for each rotation
    - Will start with the mesh name, rotation index, and list of socket identifiers
    
    ```json
    "proto_0" = {
    	mesh: "myMesh.obj",
    	rotation: 0,
    	sockets: [
    		posX: "0",
    		negX: "1s",
    		posY: "1s"
    		negY: "0f"
    		posZ: "-1"
    		negZ: "v0_0"
    	]
    }
    ```
    
- Compare each prototype six times, one for each side of the module cube
    - For each side check if the connecting socket identifiers are valid
        - Check for special conditions for symmetrical, asymmetrical, and vertrical
    - Add relevant prototypes to the neighbor list for the valid side for the current prototype
    
    ```json
    "proto_0" = {
    	mesh: "myMesh.obj",
    	rotation: 0,
    	sockets: [
    		posX: "0"
    		negX: "1s"
    		posY: "1s"
    		negY: "0f"
    		posZ: "-1"
    		negZ: "v0_0"
    	],
    	neighbor_list = [
    		posX: [..],
    		negX: [..],
    		posY: [..],
    		negY: [..],
    		posZ: [..],
    		negZ: [..]
    	]
    }
    ```
    
- Note: A module might not contain vertices along every face
    - Store the socket identifier as -1
    - Add a blank prototype with no mesh reference where all socket identifiers are set to -1 to represent empty space
      
        ```python
        "p-1" = {
        	"mesh_name: "-1",
        	"mesh_rotation": 0,
        	"posX": "-1f",
        	"negX": "-1f",
        	"posY": "-1f",
        	"negY": "-1f",
        	"posZ": "-1f",
        	"negZ": "-1f",
        	"constrain_to": "-1"
        	"constrain_from": "-1",
        	"weight": 1,
        	"valid_neighbors": [...]
        }
        ```




### WFC Demos on Itch:

[Wave Function Collapse - Mixed Initiative Demo](https://bolddunkley.itch.io/wfc-mixed)

[Wave Function Collapse - Simple Tiled Model](https://bolddunkley.itch.io/wave-function-collapse)

   

**References:**

* [Superpositions, Sudoku, the Wave Function Collapse algorithm.](https://www.youtube.com/watch?v=2SuvO4Gi7uY)
* [Infinite procedurally generated city with the Wave Function Collapse algorithm](https://marian42.de/article/wfc/)
* [Wave - by Oskar Stålberg](https://oskarstalberg.com/game/wave/wave.html)
* [WaveFunctionCollapse](https://github.com/mxgmn/WaveFunctionCollapse)
* [The Wavefunction Collapse Algorithm explained very clearly](https://robertheaton.com/2018/12/17/wavefunction-collapse-algorithm/)
* [Unity WaveFunctionCollapse](https://github.com/selfsame/unity-wave-function-collapse/)





<!-- Cloudflare Web Analytics --><script defer src='https://static.cloudflareinsights.com/beacon.min.js' data-cf-beacon='{"token": "56b8d2f624604c4891327b3c0d9f6703"}'></script><!-- End Cloudflare Web Analytics -->