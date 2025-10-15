---
categories:
- ai
- 3d
- procedural-generation
- notes
date: 2021-12-9
description: "In this talk at Blender Conference 2018, Andrew Price explores the potential impact of AI and automation on the 3D industry."
hide: false
search_exclude: false
comments:
  utterances:
    repo: cj-mills/christianjmills
title: "Notes on *The Next Leap: How A.I. will change the 3D industry*"


twitter-card:
  creator: "@cdotjdotmills"
  site: "@cdotjdotmills"
  image: /images/default-preview-image-black.png
open-graph:
  image: /images/default-preview-image-black.png


aliases:
- /Notes-on-How-AI-Will-Change-The-3D-Industry/
---




* [Introduction](#introduction)
* [Jeff Bezos’ Principle of Change](#jeff-bezos-principle-of-change)
* [The Rising Costs of Game Development](#the-rising-costs-of-game-development)
* [Asset Creation as a Major Cost Driver](#asset-creation-as-a-major-cost-driver)
* [Leap 1: Procedural Workflows](#leap-1-procedural-workflows)  
* [Leap 2: Machine Learning Creep](#leap-2-machine-learning-creep)  
* [Leap 3: Machine-Assisted Creativity](#leap-3-machine-assisted-creativity)  
* [Expected Changes in the Next 5 Years](#expected-changes-in-the-next-5-years)
* [Addressing Concerns about Job Displacement](#addressing-concerns-about-job-displacement)
* [Identifying At-Risk and Safe Jobs](#identifying-at-risk-and-safe-jobs)
* [The Future of the 3D Industry](#the-future-of-the-3d-industry)
* [Closing Remarks](#closing-remarks)
* [Not in Presentation](#not-in-presentation)





::: {.callout-tip title="Presentation Materials"}

* **Video:** [The Next Leap: How A.I. will change the 3D industry - Andrew Price](https://www.youtube.com/watch?v=FlgLxSLsYWQ)
* **Slides:** [Google Slides](https://docs.google.com/presentation/d/1nXwBdUEIbwtPyyMEu5nwJugd42sgU4YhunaUOSl2_tc/edit#slide=id.g44af636ac2_0_502)

:::





## Introduction

- **Show of hands:** Andrew asks who makes money in the 3D industry (almost everyone).
- **Andrew's perspective:** Loves 3D work and the "childlike wonder" of bringing ideas to life. 
- **Concern:** Automation and AI potentially replacing 3D artists, despite initial belief that art couldn't be replicated by computers.
- **Examples of AI in art:**
    - Machine learning for Thanos' facial animations.
    - Algorithm applying styles of famous paintings to photos.
- **Central question:**  Will 3D artists be replaced by AI, similar to how 2D Disney animators were replaced by 3D animators?
- **Presentation focus:** How AI and automation might change the 3D industry.
- **Clarification:**  Using terms like AI and machine learning broadly, regardless of technical distinctions, as the end result is software doing artistic tasks.



## Jeff Bezos' Principle of Change 

- **Jeff Bezos' insight:** Focus on what won't change in the future, as these are the core desirables.
    - Example: Amazon customers will always want lower prices and faster delivery.
- **Application to 3D:** Any technology making things **better, faster, or cheaper** will inevitably become standard in the 3D industry.
    - Studio executives will adopt cost-saving technologies.



## The Rising Costs of Game Development

- **Raph Koster's analysis:** Game development costs increase 10x every 10 years (25% annually).
    - Based on plotting game costs from 1985 to present.
    - Logarithmic scale makes the increase appear deceptively small.
- **Projected costs:** Average AAA game might cost $200 million by 2020, exceeding feature film budgets.
- **Mobile gaming:** Initially cheap, but costs are rising due to market saturation.
- **Key takeaway:** Game development costs are unsustainable and need to be reduced.



## Asset Creation as a Major Cost Driver

- **Asset costs:**  A significant portion of game development costs are attributed to creating assets.
- **Example: Modeling a building**
    - Modeling: 12 hours.
    - Texturing: 10 hours.
    - Total: 22 hours (assuming 100% productivity, which is unrealistic).
    - Revision multiple: 2-4x due to narrative changes, design iterations, etc.
    - Cost at $60/hour: $3,900 per building.
- **Games like The Division:**  Illustrate the cumulative cost of numerous detailed assets.
- **Inefficiency of current workflow:** Static, one-to-one input-output ratio leads to repeated work.



## Leap 1: Procedural Workflows

- **Proceduralism:** Shifting from manual asset creation to defining parameters and letting software generate variations.
    - [Practical Procedural Generation for Everyone](https://www.youtube.com/watch?v=WumyfLEa6bU)

- **Benefits:**
    - **Cost reduction:**  Create multiple assets with less manual labor.
    - **Creative exploration:** Forces artists to understand the underlying principles of good design and can generate unexpected ideas.
- **Anastasia Opara's example:** Created a procedural lake village in Houdini.
    - [Procedural Lake Village](https://www.anastasiaopara.com/lakevillage)
    - [Houdini Procedural Lake Houses Complete](https://anopara.gumroad.com/l/qaEZ)

- **Polygon's experience:**
    - Initially used camera-captured textures.
    - Switched to **Substance Designer** for procedural textures.
        - [substance3d](https://www.substance3d.com/)
    - Benefits: 
        - Easier to create variations.
        - Ability to create textures for difficult-to-capture materials (e.g., marble, wood).
        - Significant cost savings.
    - [poliigon](https://www.poliigon.com/)
    - [poliigon generators](https://help.poliigon.com/en/articles/3175522-using-poliigon-generators)
- **Industry trend:** Game studios are increasingly hiring Substance Designer artists.

### Procedural Texturing with Substance Painter

- **Substance Painter:** Algorithmic texturing software that complements Substance Designer.
    - Bakes maps and applies smart materials from Substance Designer.
    - Adds grunge and other details procedurally.
- **Industry standard:** Leading texturing software, saving studios significant time and money.
- **Procedural advantage:** Textures can auto-update when models are changed, if the pipeline is set up correctly.

### Procedural Level Design

- **Example: Far Cry 5's ecosystem approach**
    - [Procedural World Generation of Ubisoft's Far Cry 5](https://www.youtube.com/watch?v=NfizT369g60)
    - Defined rules for tree and plant placement based on factors like forest density, proximity to water, altitude.
    - Created an automatically updating environment that adapted to changes in the game world.
    - Included tools for easily adding roads and buildings.
- **The future of level design:** Procedural generation of environments, reducing manual placement and updating of assets.

### Summary of Procedural Workflows

- **Prediction:** Procedural modeling, materials, texturing, and world building will become the standard workflow.
- **Current status:** Materials and texturing are already widely adopted.
- **Next steps:** Modeling and world building are likely to see increased adoption.
- **Houdini:** Currently a strong tool for procedural workflows.
- **Hope for Blender:** Andrew expresses desire for Blender to incorporate more procedural capabilities.



## Leap 2: Machine Learning Creep

- **Traditional software:**  Linear input-action-output workflow, predictable but labor-intensive.
- **Machine learning:**  Involves iterative learning and improvement through comparison with training data.
    - Input is assessed, actions are applied, and the output is compared to a dataset. 
    - The process repeats until a satisfactory output is achieved.
- **Benefits:**  Often produces superior results compared to traditional software.
- **Requirements:** 
    - Large datasets for training.
    - Fast hardware for processing.
- **Past hype vs. reality:**  Initial excitement about machine learning five years ago didn't fully materialize due to limitations in data and hardware.
- **Current state:** Reaching a tipping point with more data and faster hardware, leading to consumer-level machine learning applications.

### Machine Learning in Denoising

- **Denoising:** Removing noise (grain) from images or videos.
- **Machine learning denoisers:**  Significantly outperform traditional denoisers.
- **NVIDIA's denoiser:**  Used in RTX graphics cards for real-time ray tracing.
    - Renders one sample per frame and applies denoising in real-time.
- **Blender's Cycles denoiser:**  Not based on machine learning, making it less effective in certain situations.
- **Disney and Pixar's denoiser:**  Addresses frame flicker issues and aims to provide artist-friendly tools.
- **NVIDIA's dominance:**  Holds numerous patents related to denoising, recognizing its potential across rendering and camera technology.

### Machine Learning in Up-Resing

- **Up-resing:** Increasing the resolution of an image.
- **AI Gigapixel (Topaz Labs):**  Consumer-level software demonstrating the power of machine learning in up-resing.
- **Andrew's test:** 
    - Rendered a kitchen scene at 50% resolution.
    - Up-resed to 200% using AI Gigapixel.
    - Compared to a 100% resolution render.
- **Results:**  The up-resed image was comparable in quality to the native 100% render, highlighting potential time and resource savings.

### Other Applications of Machine Learning

- **Motion capture:** 
    - [Densepose](http://densepose.org/)
    - Replacing mocap suits and dedicated studios with algorithms that analyze raw video footage.
    - Ability to estimate occluded body parts with impressive accuracy.
    - Example:  Translating dog motion capture data to game characters with seamless transitions and minimal foot sliding.
- **Prediction:**  Machine learning will become increasingly integrated into various software, automating tasks and enhancing workflows.
- **Examples:**  Photoshop, Premiere, Autodesk products, and potentially Blender.
- **Industry perspective:**  Silicon Valley companies are actively investing in machine learning.
- **Quote from Thanos facial animation team:**  "If you're not using machine learning in your software, you're doing it wrong."



## Leap 3: Machine-Assisted Creativity

- **Initial skepticism:**  Andrew initially believed that computers couldn't replicate human creativity.
- **Intent vs. assistance:**  While computers struggle with intent, they can be powerful tools for assisting creativity.

### Exploring Ideas with Machine Learning

- **Andrew's kitchen scene example:**  Illustrates the time-consuming process of iterating on designs and exploring different options.
    - Trying various lighting setups, adding or removing objects, experimenting with compositions.
    - This exploration phase can consume 50-70% of production time.
- **The potential of automated idea generation:**  Software that could quickly generate variations without requiring manual modeling and rendering would be invaluable.

    - BicycleGAN

        ![](https://github.com/junyanz/BicycleGAN/blob/master/imgs/day2night.gif?raw=true){fig-align="center"}

        - Model input: outline of an object in an image and the ground truth image
        - Model output: generate variations of image
        - [GitHub Repository](https://github.com/junyanz/BicycleGAN)
        - Toward Multi-modal Image-to-image translation

    - DivCo: Diverse Conditional Image Synthesis via Contrastive Generative Adversarial Network

        - [GitHub Repository](https://github.com/ruiliu-ai/DivCo)

    - Sketch to Image

        - pix2pix

          [Image-to-Image Demo - Affine Layer](https://affinelayer.com/pixsrv/index.html)

          - [GitHub Repository](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)

    - StyleGAN2

        - [GitHub Repository](https://github.com/lucidrains/stylegan2-pytorch)
        - [GitHub Repository](https://github.com/rosinality/stylegan2-pytorch)

    - Progressive Growing of GANS (PGAN)

        - [GitHub Repository](https://github.com/nashory/pggan-pytorch)
        - [PyTorch](https://pytorch.org/hub/facebookresearch_pytorch-gan-zoo_pgan/)

    - Text to Image

        - [StackGAN V2](https://github.com/hanzhanggit/StackGAN-v2) (2017)
        - [text2image](https://github.com/wtliao/text2image) (April 2021)
        - [TediGAN](https://github.com/IIGROUP/TediGAN) (March 2021)
        - [DF-GAN](https://github.com/tobran/DF-GAN)


### Examples of Machine-Assisted Creativity

- **Generating building facades and shoe designs:**  Paper demonstrating the ability to generate diverse design ideas based on simple outlines.
    - Given a starting point, the algorithm generates a range of unique designs.
- **Character and environment variations:**  Applying similar techniques to explore different character outfits, environments, and scenery.
- **Online cat drawing tool:**  Web-based application that "finishes" simple drawings, showcasing the potential for concept art generation.
    - Based on a different paper than the previous examples, but demonstrates similar principles.
- **Generating imaginary celebrities and bedrooms:**  Algorithm trained on images can create realistic-looking faces and environments that don't exist in reality.
    - Potential applications for generating unique NPCs in games or exploring environmental design ideas.

### Generating Images from Text Descriptions

- **Text-to-image generation:**  Describing an object in text and having the software generate a corresponding image.
- **Example:**  "This bird is red and brown in color with a stubby beak." 
- **How it works:**
    - The algorithm is trained to recognize features associated with specific descriptions.
    - It creates a basic shape based on the description.
    - A second pass adds details to the shape.
- **Andrew's reaction:**  Describes it as "the closest thing to sorcery."
- **Potential implications:**  Revolutionizing creative brainstorming and concept development.

### Style Transfer

- **Style transfer:**  Applying the artistic style of one image to another.
  - A Style-Aware Content Loss for Real-time HD Style Transfer
    - [Adaptive Style Transfer](https://github.com/CompVis/adaptive-style-transfer) (TensorFlow 2018)
    - [color-transform](https://github.com/Tonyhuiii/color-transform) (PyTorch 2019)

- **Example:**  Transferring the style of Claude Monet to a photograph.
- **Effectiveness:**  Foolability: 39% of art historians thought that style transfer outputs were real paintings. 
- **Prediction:**  Artists will increasingly use machine learning to explore new ideas and styles.



## Expected Changes in the Next 5 Years

- **Procedural workflows:**  Becoming standard across modeling, materials, texturing, and level design.
- **Machine learning integration:**  Gradually incorporated into existing software to automate technical tasks.
- **Creative assistance:**  Machines will play a larger role in generating ideas and exploring variations.



## Addressing Concerns about Job Displacement

- **Historical parallel:  Kasparov vs. Deep Blue (1997)**
    - Initial fear that chess would become obsolete after a computer defeated a human champion.
- **Advanced chess:**  Kasparov's concept of human-machine collaboration in chess.
    - Humans leverage computer analysis but retain decision-making power.
- **Outcomes:** 
    - The best chess players today are human-machine teams.
    - The number of grandmasters has doubled since Deep Blue's victory.
- **Lessons for the 3D industry:** 
    - AI and automation are likely to enhance artists' capabilities rather than replace them entirely.
    - Human intent and artistic vision remain crucial.



## Identifying At-Risk and Safe Jobs

- **At-risk jobs:**  Labor-intensive, narrow-skilled, and repetitive tasks.
    - Examples: Mocap cleanup, rotoscoping, retopo, mesh cleanup.
    - These tasks are often outsourced and easily automated.
- **Safe jobs:**  Involve critical thinking, wide-ranging skills, and niche expertise.
    - Examples: Art direction, project management, generalists, programmers, freelancers.
- **Key takeaway:**  Undesirable, grunt work is most likely to be automated, while jobs requiring creativity and adaptability are more secure.



## The Future of the 3D Industry

- **Positive outlook:**  The 3D industry is experiencing rapid growth across various sectors.
- **Projected growth:**
    - 3D rendering and visualization: 25.5% compound annual growth until 2025.
    - Potential for the industry to double in size by 2022 and quadruple by 2025.
- **Impact of VR:**  Further growth potential not even fully factored into these projections.
- **Conclusion:**
    - While some job displacement may occur, the overall industry is expanding, creating new opportunities.
    - AI and automation are likely to lead to a net increase in the number of 3D-related jobs.



## Closing Remarks

- **Andrew acknowledges the audience's concerns.**
- **Reiterates that AI and automation are tools to enhance creativity, not eliminate artists.**
- **Expresses enthusiasm for the future of the 3D industry.** 



## Not in Presentation

### Related Works

- GANcraft: Unsupervised 3D Neural Rendering of Minecraft Worlds
  [imaginaire/projects/gancraft at master · NVlabs/imaginaire](https://github.com/NVlabs/imaginaire/tree/master/projects/gancraft)
  [Unsupervised 3D Neural Rendering of Minecraft Worlds](https://nvlabs.github.io/GANcraft/)
- NeRS: Neural Reflectance Surfaces for Sparse-View 3D Reconstruction in the Wild (October 2021)
  - [GitHub Repository](https://github.com/jasonyzhang/ners)
- PIFuHD: Multi-Level Pixel-Aligned Implicit Function for High-Resolution 3D Human Digitization (2020)
  - [GitHub Repository](https://github.com/facebookresearch/pifuhd)
    [Google Colaboratory](https://colab.research.google.com/drive/11z58bl3meSzo6kFqkahMa35G5jmh2Wgt)
- 3DStyleNet: Creating 3D Shapes with Geometric and Texture Style Variations
  ![](https://nv-tlabs.github.io/3DStyleNet/assets/teaser.jpg){fig-align="center"}
  [https://nv-tlabs.github.io/3DStyleNet/assets/animal-new.mp4](https://nv-tlabs.github.io/3DStyleNet/assets/animal-new.mp4)
  [3DStyleNet: Creating 3D Shapes with Geometric and Texture Style Variations](https://nv-tlabs.github.io/3DStyleNet/)
- DISN: Deep Implicit Surface Network for High-quality Single-view 3D Reconstruction
  ![](https://github.com/laughtervv/DISN/raw/master/result.png){fig-align="center"}
  - [GitHub Repository](https://github.com/laughtervv/DISN) (Tensorflow)
  - [GitHub Repository](https://github.com/cs674/pytorch-DISN) (PyTorch)
- Learning Linear Transformations for Fast Arbitrary Style Transfer
  - [GitHub Repository](https://github.com/sunshineatnoon/LinearStyleTransfer)
- Neural Cages for Detail-Preserving 3D Deformations
  - [GitHub Repository](https://github.com/yifita/deep_cage)
- Taming Transformers for High-Resolution Image Synthesis
  - [GitHub Repository](https://github.com/CompVis/taming-transformers)
  - [GitHub Repository](https://github.com/sunshineatnoon/taming-transformers)
  [[Overview] Taming Transformers for High-Resolution Image Synthesis](https://wandb.ai/ayush-thakur/taming-transformer/reports/-Overview-Taming-Transformers-for-High-Resolution-Image-Synthesis---Vmlldzo0NjEyMTY)
  [These Neural Networks Have Superpowers! 💪](https://www.youtube.com/watch?v=o7dqGcLDf0A)
- Rethinking Style Transfer: From Pixels to Parameterized Brushstrokes
  - [GitHub Repository](https://github.com/CompVis/brushstroke-parameterized-style-transfer) (TensorFlow)
  - [GitHub Repository](https://github.com/justanhduc/brushstroke-parameterized-style-transfer) (PyTorch)
- Network-to-Network Translation with Conditional Invertible Neural Networks
  - [GitHub Repository](https://github.com/CompVis/net2net)
- Artistic Style Transfer with Internal-external Learning and Contrastive Learning
  - [GitHub Repository](https://github.com/HalbertCH/IEContraAST)
  - Based on: [SANET](https://github.com/GlebSBrykin/SANET)
- Synthetic Silviculture: Multi-scale Modeling of Plant Ecosystems
  [Makowski.etal-2019-Synthetic-Silviculture.pdf](https://drive.google.com/file/d/1tbb0WPcxljqwWdKzM1CMXWwH5vWeID53/view?usp=drivesdk)
  [Makowski.etal-2019-Synthetic-SilvicultureSup.pdf](https://drive.google.com/file/d/13_G9p9rB3bLuZRr-pIEBo-AVsxHZwkGN/view?usp=drivesdk)
  [Synthetic Silviculture: Multi-scale Modeling of Plant Ecosystems](https://storage.googleapis.com/pirk.io/projects/synthetic_silviculture/index.html)
  [Simulating A Virtual World...For A Thousand Years! 🤯](https://www.youtube.com/watch?v=8YOpFsZsR9w)
- DualConvMesh-Net: Joint Geodesic and Euclidean Convolutions on 3D Meshes
  - [GitHub Repository](https://github.com/VisualComputingInstitute/dcm-net)

### Other Applications

[How A.I will affect the art industry](https://www.youtube.com/watch?v=0LK_MHLs__M)

Takeaway: AI will handle more and more of the tedious manual work that humans don't like doing (or is extremely time consuming)

This will reduce the cost of production, enabling more productions overall

- Rotoscoping
  [What is rotoscoping animation and how to do it ](https://www.adobe.com/creativecloud/video/discover/rotoscoping-animation.html)
  - segmentaion
- Retopology
  [What is Retopology? (A Complete Intro Guide For Beginners)](https://conceptartempire.com/retopology/)
  - Appearance-Driven Automatic 3D Model Simplification (2021)
    - [https://research.nvidia.com/publication/2021-04_Appearance-Driven-Automatic-3D](https://research.nvidia.com/publication/2021-04_Appearance-Driven-Automatic-3D)
    - [GitHub Repository](https://github.com/NVlabs/nvdiffmodeling)
- Human provides general outline/concept and a model fills in technical details
  - NVIDIA GauGAN2
    [NVIDIA Research's GauGAN AI Art Demo Responds to Words](https://blogs.nvidia.com/blog/2021/11/22/gaugan2-ai-art-demo/)
  - NVIDIA Canvas
    [NVIDIA Canvas : Harness The Power Of AI](https://www.nvidia.com/en-us/studio/canvas/)
- Facial animations










{{< include /_about-author-cta.qmd >}}
