---
categories:
- ai
- game-dev
- notes
date: 2021-12-9
description: "In this presentation from 2017, Ben Berman explores the use of machine learning for generating game content, focusing on techniques that produce results that appear handcrafted rather than procedurally generated."
hide: false
search_exclude: false
title: "Notes on *Machine Learning and Level Generation*"

aliases:
- /Notes-on-Machine-Learning-and-Level-Generation/

twitter-card:
  creator: "@cdotjdotmills"
  site: "@cdotjdotmills"
  image: /images/default-preview-image-black.png
open-graph:
  image: /images/default-preview-image-black.png

---



* [Introduction](#introduction)
* [What is Machine Learning?](#what-is-machine-learning)
* [Machine Learning Landscape](#machine-learning-landscape)
* [Machine Learning at Roguelike Celebration](#machine-learning-at-roguelike-celebration)
* [Machine Learning in Mainstream Games](#machine-learning-in-mainstream-games)
* [Research vs. Commercial Use](#research-vs.-commercial-use)
* [Influential Demos](#influential-demos)
* [Major Techniques](#major-techniques)
* [Demos: Exploring the Techniques](#demos-exploring-the-techniques)
* [Using Machine Learning for Level Generation: A Practical Guide](#using-machine-learning-for-level-generation-a-practical-guide)
* [Modeling a Roguelike for Machine Learning](#modeling-a-roguelike-for-machine-learning)
* [Data Collection](#data-collection)
* [Libraries and Tools](#libraries-and-tools)
* [Challenges and Solutions with LSTM Models](#challenges-and-solutions-with-lstm-models)
* [The Future: Functional Game Content](#the-future-functional-game-content)
* [Conclusion](#conclusion)
* [Related Work](#related-work)



::: {.callout-tip title="Source Material"}

* [Ben Berman - Machine Learning and Level Generation](https://www.youtube.com/watch?v=Z6lHExfem6U)

:::



## Introduction 

- **Level generation using machine learning** is a growing area of interest in game development, offering potential for "magical fantasy problem-solving."
- This talk aims to provide a comprehensive overview of the current state of machine learning in level generation, including:
  - How it's currently used.
  - How to get started.
  - Relevant research.
  - Speaker's personal experiences in both commercial and research settings.

## What is Machine Learning?

- **Machine Learning**: Put simply, it's a computer using data to learn.
- **Key Tradeoff**: Achieving good results in machine learning requires either:
  - **Lots of Data**: With minimal guidance for the computer. 
  - **Little Data**: With explicit instructions on what's important and how to learn.
- This tradeoff characterizes the field broadly, not just in content generation.

## Machine Learning Landscape

- **Spectrum of Techniques**: Berman visually represents the tradeoff (lots of data vs. explicit instructions) on a 2x2 matrix.

  ![Machine_Learning_and_Level_Generation_Diagram_1.png](./images/Machine_Learning_and_Level_Generation_Diagram_1.png){fig-align="center"}

- **Examples**:

  - **Upper Left (Lots of Data, Explicit Instructions):**
    - Ad tech (highly effective).
    - High-energy physics.
  - **Lower Left (Little Data, Explicit Instructions):**
    - Natural language processing (NLP), like n-grams for generating grammatically correct text.
  - **Upper Right (Lots of Data, Minimal Instructions):**
    - Recurrent neural networks for content generation.
  - **Lower Right (Little Data, Minimal Instructions):**
    - Strong AI (currently doesn't exist).

## Machine Learning at Roguelike Celebration

- Several examples of machine learning techniques were present at Roguelike Celebration: 
  - **Caves of Qud**: Uses a Monte Carlo Markov Chain method (MCMC) with "local similarity" (and other) constraints to generate game levels based on a mix of examples and heuristics.
  - **Botnik**: Uses NLP (n-grams) and has experimented with recurrent neural networks.
  - **Computational Flaneur**: Generates poetry using recurrent neural networks.
  - **Max Kreminski**: Works in a department focused on content generation in video games, utilizes various techniques in his projects.
  - **Darius Kazemi**:  Primarily uses NLP (ConceptNet), which occupies a powerful niche.
  - **Ian Holmes**: Uses heuristics and cellular automata for level, visual, and text generation.

## Machine Learning in Mainstream Games

- **Commercial Games**:  Tend to avoid content generation but utilize machine learning for other purposes.
  - **League of Legends**: Uses unsupervised machine learning for client analysis.
  - **Clash Royale**: Matchmaking likely employs sophisticated machine learning.
  - **Ad Tech & Monetization**: Machine learning is extensively used in these areas, with some techniques dating back to 1987.

## Research vs. Commercial Use

- **Research is significantly ahead of commercial games** in terms of machine learning for content generation.
  - Don't expect widespread adoption in AAA games yet.
  - Experimental and indie games are leading the way.
- **Key Academic Researchers**:
  - **Michael Mateas (UCSC)**: Focuses on level generation.
  - **Julian Togelius (NYU)**: Conducts research on various aspects of procedural content generation.
  - **Dan Ritchie (Brown University)**: Specializes in graphics generation, particularly relevant for roguelikes.

## Influential Demos

- Several demos have spurred interest in machine learning for content generation:
  - **"The Unreasonable Effectiveness of Recurrent Neural Networks" by Andrej Karpathy**: Showcased character-based recurrent neural networks, generating impressive results.
  - **Wave Function Collapse**: Demonstrates texture synthesis, used by Caves of Qud.
  - **"Phase-Functioned Neural Networks for Character Control"**: Showcases animation generated by neural networks, trained on motion capture data.
  - **Deep Dream**: Neural network hallucinations, sparked early interest in the field.

## Major Techniques

- **Texture Synthesis**: Treats levels as images, learning to create new images based on examples.
- **Recurrent Neural Networks**: Treats levels as sequences explored over time, assuming the future should resemble the past.
- **Monte Carlo Markov Chain (MCMC) Design**: Allows for specifying desired level properties (e.g., symmetry, good layout) and iteratively generates levels until they meet those criteria.
  - Heuristic level generation in many roguelikes can be considered a special case of MCMC. 

## Demos: Exploring the Techniques

- **Wave Function Collapse (Texture Synthesis)**:
  - [EPC2018 - Oskar Stalberg - Wave Function Collapse in Bad North](https://www.youtube.com/watch?v=0bcZb-SsnrA)
  - Shown in a 3D city generation context, allowing for user edits and generating plausible geometry based on constraints and examples.
  - Also showcased in a 2D Super Mario level generation context.
- **Character-Based Recurrent Neural Networks**:
  - Generates playable Super Mario levels based on text representations, without explicit knowledge of game rules.
    - [Super Mario as a String: Platformer Level Generation Via LSTMs](https://arxiv.org/abs/1603.00930)
    - [Building Mario Levels with Machine Learning](https://www.youtube.com/watch?v=U-CDQtIJ8eg)
- **Neural Network Animated Character**:
  - Demonstrates realistic character animation generated by a neural network trained on motion capture data, controlled by user input.
    - GDC 2018 Character Control with Neural Networks and Machine Learning
      - **GDC Talk:** [Character Control with Neural Networks and Machine Learning](https://www.youtube.com/watch?v=o-QLSjSSyVk)
      - **Presentation slides:** [Slides (PDF)](https://ubm-twvideo01.s3.amazonaws.com/o1/vault/gdc2018/presentations/Holden_Daniel_CharacterControlWith.pdf)
    - **Two Minute Papers (2017):** [Real-Time Character Control With Phase-Functioned Neural Networks](https://www.youtube.com/watch?v=wlndIQHtiFw)
- **Dan Ritchie's Spaceship Generator (MCMC)**: 
  - Generates detailed spaceship designs based on user-provided block layouts.
  - **PhD Thesis:** [Probabilistic Programming for Procedural Modeling and Design](https://dritchie.github.io/pdf/thesis.pdf)
- **MCMC Room Layout**:
  - Arranges objects in a room using interior design principles.

## Using Machine Learning for Level Generation: A Practical Guide

- **Objective**: Focus on creating levels that look handcrafted, not computer-generated.
- **Reject Difficulty as the Primary Parameter**: 
  - Roguelikes are already challenging, focus on other aspects of level design.
  - Explore levels that are fun to watch even with zero difficulty, like auto-playing Mario levels.
- **Embrace Symmetry and Pacing**: 
  - Handcrafted levels often exhibit symmetry and regular features.
  - Examples like Binding of Isaac demonstrate effective use of symmetry.
- **Avoid Overly Precise Numbers**: 
  - Precise numbers can make levels feel artificial and calculated.
  - Use small, easily-reasoned-about numbers, like in Hearthstone.

## Modeling a Roguelike for Machine Learning

- **Think of it as a 1D Platformer**: 
  - Focus on left/right movement decisions, simplifying the problem and opening up more machine learning tools.
  - This approach has been successful for generating Mario levels.
  - 2D levels can be represented as multiple 1D levels.

## Data Collection

- **Rip Data from Existing Games**: 
  - Don't necessarily require massive datasets like Google or Facebook.
  - Data can be extracted from various games through modding communities and tools.
    - Rip level data from existing Unity games using **DevXUnity Pro**
    - Colossal amount of content unlocked from the Google Play store
    - Always Google for mods for a game with rich data
    - Data: Successfully Extracted
      - Rolling Sky (Unity)
      - Wayward Souls (Apportable)
      - Hotline Miami, Hotline Miami 2 (GameMaker)
      - Botanicula (Adobe AIR)
      - Geometry Dash (Cocos 2D)
  - Even raw level data (e.g., hexadecimal) can be used for training.

  

## Libraries and Tools

- **TensorFlow/PyTorch**: Recommended as a flexible, well-supported, and widely used library.
- **LSTM Models**: 
  - Specifically, Long Short-Term Memory (LSTM) models are highlighted for their success in character-based recurrent neural networks.

## Challenges and Solutions with LSTM Models

- **Buggy Implementations**: 
  - Be wary of bugs in online code examples.
  - Example: Incorrect array duplication in character RNN implementations.
- **Dividing and Chunking Levels**: 
  - Consider how to break down the level into chunks for the model to process.
  - **Snaking**: Processing tiles in a snake-like pattern can improve performance, potentially by aiding the model in recognizing patterns and counting.
  - Compare different chunking methods to find what works best for your specific game.
- **Encoding Data**: 
  - **Categorical Encoding**: Representing tiles based on their properties (e.g., passable, interactable, reward) can significantly improve performance.
    - **[Solution: Semantically-Balanced Hierarchical Encoding](https://youtu.be/Z6lHExfem6U?t=1460)**. Categorize each tile into a binary string of "traits" (e.g. piece of text), where each trait has real semantic meaning.
      - Train an RNN on each binary "view" of your tiles (like a black and white image), then train feed forward networks on for important traits.
        - $$
          x_1 = f(x_i \ \epsilon \ traits : i \neq 1)
          $$
      - Some experiments saw a 20x improvement in loss Training was immensely faster.
      - Definitely sensitive to your choice of traits.
      - Note: models do not work with rarity very well
        - Reward tiles are so rare that you'd need a huge corpus just to learn where they appear
      - Tend to implement convolutions to reinterpret the data that helps the model learn
  - **Addressing Rarity**: Use techniques like convolutions to handle rare elements effectively, preventing them from being lost as noise.
  - Categorical encoding can enable training across multiple games.
- **Noise Reduction**: 
  - **Denoising**: Treat levels like images and apply noise reduction techniques, similar to Photoshop.
    - Non-local SSIM (Structural Similarity Index) denoiser works really well for some level textures
    - SSIM increases (higher is better) as noisy chunks move towards border, while MSE (Mean Squared Error) stays the same
  - **Symmetry-Based Denoising**:  Combine noise reduction with symmetry enforcement to achieve more cohesive level architecture.
- **Performance Issues**: 
  - Level generation with machine learning can be computationally expensive.
  - **Focus on Quality over Optimization**:  Prioritize getting good results over extreme performance optimizations, especially when working with limited data.
  - Consider using GPU acceleration (e.g., custom TensorFlow build for Macs).

## The Future: Functional Game Content

- The next frontier is generating **functional game content**, such as monster behaviors and game rules, not just level layouts.
- Example: Generating functional playing cards for Hearthstone-like games (treating cards as monsters with behaviors).
  - **[Spellsource](https://www.playspellsource.com/):**
    - A free, open-source fully-hosted multiplayer card game engine for experimentation
    - A feature-complete implementation of the Hearthstone ruleset with all 1,312 cards

- This will require different techniques and approaches compared to level generation.

## Conclusion

- **Mission**: Strive to create content that feels handcrafted, not computer-generated.
- **Focus on Functional Game Content**: This is the next big challenge and opportunity in applying machine learning to game development.
- Berman invites further discussion and collaboration on these topics. 



## Related Work

- **AI4Animation: Deep Learning for Character Control**
    - https://github.com/sebastianstarke/AI4Animation
    - [These AI-Driven Characters Dribble Like Mad! 🏀](https://www.youtube.com/watch?v=pBkFAIUmWu0)
- GDC 2018 Deep Learning: Beyond the Hype (by SEED/EA)
    - **Presentation:** [Deep Learning: Beyond the Hype](https://www.youtube.com/watch?v=yA-lJy52Ais)
- Learning Generative Models of 3D Structures (2020)
    - [Paper (PDF)](https://par.nsf.gov/servlets/purl/10155956)
    - [CSC2547 Learning Generative Models of 3D Structures](https://www.youtube.com/watch?v=dDTU43UpCe0)            


​    

 








{{< include /_about-author-cta.qmd >}}
