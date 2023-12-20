---
categories:
- ai
- notes
date: 2021-12-9
description: My notes from an overview of the Learning Generative Models of 3D Structures
  paper.
hide: false
search_exclude: false
title: Notes on Learning Generative Models of 3D Structures

aliases:
- /Notes-on-Learning-Generative-Models-of-3D-Structures/
---

* [Overview](#overview)
* [Motivation](#motivation)
* [Generative models](#generative-models)
* [Structure-Aware Representations](#structure-aware-representations)
* [Application](#application)



## Overview

I wanted to get an idea of where the research is at for using deep learning models to generate 3D models for applications in procedural generation tools and creating synthetic datasets. I came across a [video](https://www.youtube.com/watch?v=dDTU43UpCe0) going over the 2020 paper, [Learning Generative Models of 3D Structures](https://par.nsf.gov/servlets/purl/10155956). Below are some notes I took while watching.



## Motivation

- 3D Graphics are now critical to many industries
- Huge cost in data capture and human labeling leads to lack of training data

## Generative models

- generative: 
  $$
  P(X) \ vs \ discriminative: P(Y|X)
  $$
  
- Instead of learning to predict some attribute Y given an input X, the generative model learns the entire input distribution, enabling them to sample objects directly from X

- Can be useful in simulating real-world environments and synthetically generating training data

## Structure-Aware Representations

- Scope: learned generative models of structured 3D content

### Learned:

- Determined with data ↔ By hand or rules

### Structured:

- 3D shapes and scenes that are decomposed into sub-structures ↔ a monolithic chunk of geometry

![](./images/structured-3d-chair-example.png){fig-align="center"}

### Structure-Aware

- Express 3D shapes and scenes using abstractions that allow manipulation of their high-level structure
- represent the geometry of the atomic structural elements
- represent the structural patterns

### Structure-Aware Representations

- Representations of Part/Object Geometry
    - Voxel Grid
    - Point Cloud
    - Implicit Surface
        - A function that determines whether a point is inside or outside a surface
    - Triangle Mesh
- Representations of Structure
    - Segmented geometry
        - Links a label to each part of the entity's geometry
    - Part sets
        - an unordered set of atoms (pieces)
    - Relationship graphs
        - With edges between different parts of a scene or object
    - Hierarchies (trees)
    - Hierarchical Graphs
        - Combine relationship graphs and hierarchies
    - Deterministic Programs
        - Can be made to output any of the above representations
        - Beneficial for making patterns clear
        - Allows editing by users

## Methodologies

![](./images/methodologies.png){fig-align="center"}

### Program synthesis

- Constrain-based program synthesis
    - Used when only a few training examples are available
    - Tries to find the minimum cost program while satisfying some constraints

### Classical Probabilistic Models

- Probabilistic graphical models
    - Input Type:
        - Small dataset, not large enough to train a deep learning model
        - Fixed structure
    - Examples:
        - Factor graph
        - Bayesian network
        - Markov random field
- Probabilistic grammars
    - Input Type:
        - Small dataset, not large enough to train a deep learning model
        - Dynamic, tree-like structure
    - Examples:
        - Context-free grammar (CFG)
            - Used in natural language processing
            - a start symbol
            - a set of terminals and non-terminals
            - a set of rules that map a non-terminal to another layout
            - generates a tree where the leaf nodes are terminals
        - Probabilistic CFG (PCFG)
            - Adds a probability of each rule

### Deep Generative Models

- Input Type:
    - Big dataset
- Autoregressive models
    - Input Type:
        - Not globally-coherent
    - Iteratively consumes it's output from one iteration as input for the next iteration
      
        ![](./images/autoregressive-model-example.png){fig-align="center"}
        
    - Weakness:
        - If one step drifts from the training data, it can cause subsequent output to diverge further
- Deep latent variable models
    - Input Type:
        - Globally-coherent
    - Variational AutoEncoders (VAE)
    - Generative Adversarial Networks (GAN)
    - Code Idea:
        - Sample over a low dimensional latent space in a trained generator that maps latent vectors to actual 3D shapes which are hard to sample.
        - Use a global latent variable to control the generation
        - Trained with a reconstruction loss between the input and generated output
        - Often perform better than autoregressive models in terms of global coherence

### Structure Type

- Recurrent Neural Network
    - Data represented as a linear chain
- Recursive Neural Network RvNN
    - Data represented as a tree
- Graph Convolutional Network
    - Data represented as a graph

- Neural Program Synthesis


​                        
## Application

- Synthesize a plausible program that recreates an existing piece of 3D content
- Recover shape-generating programs from an existing 3D shape
- Learning Shape Abstractions by Assembling Volumetric Primitives (2017)
    - Learned to reconstruct 3D shapes with simple geometric primitives
    - Decompose shapes into primitives and used chamfer distance as a loss function
    - https://github.com/shubhtuls/volumetricPrimitives
    
    [Learning Shape Abstractions](https://shubhtuls.github.io/volumetricPrimitives/)
    
    [Learning Shape Abstractions by Assembling Volumetric Primitives](https://arxiv.org/abs/1612.00404)
    
- Learning to Infer and Execute 3D Shape Programs (2019)
    - Model can output a 3D shape program consisting of loops and other high level structures
    
    
    
    ![](./images/infer-and-execute-3d-shape.jpeg){fig-align="center"}
    
    - https://github.com/HobbitLong/shape2prog
    
    [Learning to Infer and Execute 3D Shape Programs](https://arxiv.org/abs/1901.02875)
    
- Superquadrics Revisited: Learning 3D Shape Parsing beyond Cuboids
    - https://github.com/paschalidoud/superquadric_parsing
- Perform visual program induction directly from 2D images
    - Liu et al. 2019
                - Other Applications:
- Part-based shape synthesis
- Indoor scene synthesis



 

**References:**

* [CSC2547 Learning Generative Models of 3D Structures](https://www.youtube.com/watch?v=dDTU43UpCe0)
* [Learning Generative Models of 3D Structures (2020)](https://par.nsf.gov/servlets/purl/10155956) (PDF)




