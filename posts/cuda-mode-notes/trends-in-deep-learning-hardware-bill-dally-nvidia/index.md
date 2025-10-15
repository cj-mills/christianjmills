---
title: "Notes on *Trends in Deep Learning Hardware: Bill Dally (NVIDIA)*"
date: 2024-9-11
image: /images/empty.gif
hide: false
search_exclude: false
categories: [notes, cuda]
description: "In this lecture, Bill Dally discusses the historical progress of deep learning, driven by hardware advancements, especially GPUs, and explores future directions focusing on improving performance and efficiency through techniques like optimized number representation, sparsity, and specialized hardware accelerators."

twitter-card:
  creator: "@cdotjdotmills"
  site: "@cdotjdotmills"
  image: /images/default-preview-image-black.png
open-graph:
  image: /images/default-preview-image-black.png
---



::: {.callout-tip}
## This post is part of the following series:
* [**GPU MODE Lecture Notes**](/series/notes/cuda-mode-notes.html): My notes from the **GPU MODE** reading group lectures run by **Andreas Kopf** and **Mark Saroufim**.
:::





* [Motivation and History](#motivation-and-history)
* [GPU Performance Improvements: Huang’s Law](#gpu-performance-improvements-huangs-law)
* [Complex Instructions and Their Importance](#complex-instructions-and-their-importance)
* [NVIDIA Hopper GPU: Current State (2023)](#nvidia-hopper-gpu-current-state-2023)
* [Scaling with Multiple GPUs](#scaling-with-multiple-gpus)
* [The Importance of Software](#the-importance-of-software)
* [Future Directions](#future-directions)
* [Number Representation: Choosing the Right System](#number-representation-choosing-the-right-system)
* [Logarithmic Number Systems](#logarithmic-number-systems)
* [Optimal Clipping](#optimal-clipping)
* [Scaling Granularity](#scaling-granularity)
* [Sparsity](#sparsity)
* [Accelerators vs. GPUs](#accelerators-vs.-gpus)
* [Magnetic BERT Accelerator](#magnetic-bert-accelerator)
* [Conclusion](#conclusion)
* [Q&A Session](#qa-session)





::: {.callout-tip title="Source Material"}

* **YouTube:** [Trends in Deep Learning Hardware: Bill Dally (NVIDIA)](https://www.youtube.com/watch?v=kLiwvnr4L80)

:::





## Motivation and History

* **Deep learning** can be viewed as a process of distilling data into value.
  * Starting with massive datasets (e.g., 10<sup>12</sup> tokens, 10<sup>9</sup> images).
  * Aiming to extract valuable insights, applications, and functionalities.
* **Deep learning models** require a training process analogous to education:
  * **Undergraduate School (General Training):**
    * Input: Trillion-token dataset (now exceeding 10 trillion).
    * Process: Extensive training on GPUs (e.g., via AWS).
    * Output: A broadly trained model with a general understanding of the data.
  * **Graduate School (Specialization):**
    * **Pre-tuning:** Continued training on domain-specific data (e.g., Chip Nemo trained on 24 billion tokens of NVIDIA hardware design documents).
    * **Fine-tuning with Human Feedback:** Model generates responses, humans grade them, and feedback refines the model for better user experience.
  * **Inference (Real-World Application):**
    * The bulk of deep learning computation occurs in the inference stage.
    * Trained models are used for extended periods, performing inference on new data (e.g., answering queries, generating content).
    * **Retrieval Augmented Generation:** 
      * Recent approach to enhance inference accuracy and prevent "hallucinations."
      * Queries a database (e.g., Chip Nemo's document database) before running the LLM.
      * Relevant documents are fed into the transformer's input window along with the original query.
* **Deep learning's revolution** was enabled by hardware:
  * **Algorithms (Fuel):** Mostly established by the 1980s (e.g., deep neural networks, convolutional networks, stochastic gradient descent, backpropagation).
  * **Data (Air):** Large labeled datasets emerged in the early 2000s (e.g., Pascal, ImageNet).
  * **Computing Power (Spark):** Sufficient compute to train large models on large datasets in reasonable time was the catalyst (e.g., AlexNet trained on ImageNet in two weeks on a pair of Fermi GPUs).
* **Progress in deep learning** is gated by available compute power:
  * Training time demands have increased dramatically:
    * AlexNet (2012): 10<sup>-2</sup> petaflop days.
    * ResNet (2016): ~1 petaflop day.
    * BERT (2018): ~10 petaflop days.
    * GPT-4 (2023, estimated): ~10<sup>6</sup> petaflop days.
  * This represents a 10<sup>8</sup> increase in compute demand over a decade.
  * Improvements have come from:
    * **Increased individual GPU performance (~1000x).**
    * **Scaling up GPU numbers and training time (~10<sup>6</sup>x).**



## GPU Performance Improvements: Huang's Law

* **Huang's Law:** Deep learning inference performance on NVIDIA GPUs has doubled annually for the last decade.
  * Kepler generation: ~4 int8 TOPS (Tera Operations Per Second) on single-chip inference.
  * Hopper generation: ~4000 int8 TOPS.
  * This represents a ~1000x increase over 10 years.
* **Key contributors to performance gains:**
  * **Smaller numbers (biggest gain):**
    * Shifting from FP32 (used in Kepler for scientific computing and graphics) to int8 for inference.
    * Reduced data size (4x) and quadratic reduction in multiply operations (16x).
    * Note: Google TPU V1's efficiency advantage over Kepler stemmed primarily from using int8 vs. Kepler's FP32.
  * **Complex instructions:**
    * GPUs have a simplified pipeline (no branch prediction, out-of-order execution), but instruction execution still has significant overhead (~20x the cost of arithmetic within the instruction).
    * **Complex instructions amortize this overhead by performing more work per instruction.**
    * Examples:
      * **FMA (Fuse Multiply Add) in Kepler.**
      * **DP4 (4-element Dot Product) in Pascal.**
      * **HMMA (Half-precision Matrix Multiply Accumulate) in Volta.**
      * **IMMA (Integer Matrix Multiply Accumulate) in Turing.**
      * These instructions offer efficiency comparable to hardwired accelerators (e.g., TPUs).
  * **Process technology:**
    * Four generations of process technology advancements (28nm to 5nm) contributed ~2.5x improvement.
    * **Most gains have come from architectural improvements, not process technology.**
  * **Sparsity:**
    * Exploiting sparsity (currently 2:1 on weights only) yields performance improvements.
    * Future potential lies in exploiting higher levels of sparsity and sparsity in activations.
  * **Algorithm improvements:**
    * More efficient deep learning models have also contributed significantly to performance gains (estimated ~1000x).
    * Example: GoogleNet's efficiency improvements over VGGNet in the ImageNet competition.



## Complex Instructions and Their Importance

| Operation | Energy** | Overhead* |
| --------- | -------- | --------- |
| HFMA      | 1.5pJ    | 2000%     |
| HDP4A     | 6.0pJ    | 500%      |
| HMMA      | 110pJ    | 22%       |
| IMMA      | 160pJ    | 16%       |

* Even simplified GPU pipelines have an overhead factor of ~20 for instruction execution compared to the cost of arithmetic operations within the instruction.
* Complex CPUs have even higher overhead (~1000x for FP16 operations).
* **Complex instructions are crucial for amortizing instruction overhead:**
  * **FMA:** Two arithmetic operations, overhead dominates energy consumption.
  * **DP4:** Eight operations (4 multiplies, 4 adds), overhead reduced to ~5x.
  * **Tensor Cores (HMMA, IMMA, QMMA):** Matrix multiply instructions perform hundreds of operations per instruction, significantly reducing overhead to ~15-20%.
    * **HMMA (Volta):** Two FP16 4x4 matrices, matrix multiply (64 multiplies), accumulate into an FP32 4x4 matrix (128 total operations).
    * **IMMA (Turing):** Two int8 8x8 matrices, matrix multiply and accumulate.
    * **QMMA (Hopper):** Quarter-precision (FP8) matrix multiply accumulate.
* **Complex instructions make programmable GPUs as efficient as hardwired accelerators for deep learning while retaining programmability advantages.**



## NVIDIA Hopper GPU: Current State (2023)

* **Hopper H100:**
  * 1 petaflop TensorFloat32.
  * 1-2 petaflops FP16/BFloat16 (dense/sparse).
  * 2-4 petaflops FP8/int8 (dense/sparse).
  * 3 TB/s memory bandwidth.
  * 96 GB HBM3 memory.
  * 18 NVLink ports (900 GB/s off-chip bandwidth).
  * 700 watts power consumption.
  * 9 teraOPS/watt (int8/FP8).
  * Includes dynamic programming instructions for bioinformatics.
* **Note:** Export restrictions to China may be counterproductive, potentially driving Chinese developers to Huawei's hardware.



## Scaling with Multiple GPUs

* **Model Parallelism:** 
  * Necessary because large models (e.g., GPT-4 with 1.2 trillion parameters) don't fit on a single GPU.
  * **Tensor Parallelism:** Dividing individual matrices (e.g., into column strips) and distributing operations across multiple GPUs.
  * **Pipeline Parallelism:** Assigning different network layers to different GPUs, forwarding results sequentially. Earlier layers start processing the next batch of training data while later layers finish the current batch.
* **Data Parallelism:**
  * Running multiple copies of the model.
  * Splitting a batch of training data across model copies.
  * Each copy trains on its portion, then weight updates are exchanged to ensure all copies have the same weights for the next iteration.
* **Hardware for Multi-GPU Scaling:**
  * **HGX Server:** 8 H100 GPUs, 4 NV switches, 32 petaflops compute, 11 kilowatts power, 900 GB/s bandwidth.
  * **NV Switch Pizza Box:** Connects multiple HGX servers with active optical cables.
  * **DGX SuperPOD:** Large-scale system comprised of multiple interconnected HGX servers.
    * **Pre-configured software for rapid deployment.**
    * **Network collectives (all-reduce) on NVLink and InfiniBand for efficient data parallel training.**



## The Importance of Software

* "Anybody can build a matrix multiplier, but software makes it useful."
* **NVIDIA's Deep Learning Software History:**
  * Started in 2010 with cuDNN, developed in collaboration with Andrew Ng (Stanford).
  * Has evolved into a comprehensive software stack:
    * **AI Stack:** CUDA, cuDNN, TensorRT, etc.
    * **HPC Stack:** Libraries and tools for high-performance computing.
    * **Graphics Stack (Omniverse):** Platform for 3D design collaboration and simulation.
    * **Vertical Applications:** Clara (medical), Modulus (physics), Drive (autonomous vehicles), etc.
  * Represents tens of thousands of person-years of development effort.
* **MLPerf Benchmarks:** Demonstrate the impact of software on performance.
  * NVIDIA GPUs consistently lead in these benchmarks, showcasing the strength of the software ecosystem.
  * **Significant performance gains are achieved through software optimizations even on existing hardware (e.g., Ampere's performance increased 2.5x since its initial release).**



## Future Directions

* **The challenge:** Meeting the ever-increasing compute demand for deep learning (10<sup>8</sup> increase in 10 years).
* **Energy breakdown in deep learning inference:**
  * Math (Datapath and MAC): 47%.
  * Memories (accumulation buffer, input buffer, weight buffer, accumulation collector): 47%.
  * Data movement: 6%.
* **Strategies for future improvements:**
  * **Number Representation:**
    * **Use the cheapest representation that maintains sufficient accuracy.**
    * **Optimal scaling:** Adjust the dynamic range of the number system to minimize error (e.g., minimize mean squared error).
    * **Logarithmic numbers:** Offer better accuracy than integers or floats, especially for small values. Efficient addition methods exist.
    * **Sparsity:** Exploit sparsity in both weights and activations, explore lower density sparsity patterns.
  * **Data Movement:**
    * **Better tiling:** Optimize loop scheduling to minimize data movement and maximize reuse.
  * **Circuits:**
    * **More efficient memories:** 
      * **Write-once, read-many optimizations (e.g., bit line per cell).** This reduces read energy significantly.
    * **Better communication circuits:** Reduce energy consumption in on-chip data transfer (e.g., using lower voltage signaling).
    * **3D memory:** Stack DRAM directly on top of the GPU for higher bandwidth and lower energy (long-term goal with significant technical challenges).



## Number Representation: Choosing the Right System

* **Evaluating a number system:**
  * **Accuracy:** Maximum error introduced when converting a real number to the number system's representation (due to rounding).
  * **Dynamic Range:** The range of numbers that can be represented.
  * **Cost:** 
    * Number of bits (affects storage and data movement).
    * Cost of arithmetic operations (e.g., multiply-accumulate).
* **Comparison of number systems:**
  * **Integer:** Poor accuracy (error is independent of value), worst-case error of 33%.
  * **Floating Point:** Better accuracy than integer, error scales with value but in blocks.
  * **Logarithmic:** Even better accuracy, error scales continuously with value.
    * Example: Log 4.3 (8-bit representation) offers 50% higher worst-case accuracy than FP8 (E4M3) with the same dynamic range.
  * **Symbolic:** Optimal for a fixed number of representable values (e.g., codebook), but lookup and arithmetic costs can be high.
  * **Spiking:** Extremely inefficient in terms of energy consumption due to high toggling activity.
  * **Analog:** Advantages in individual operations are negated by the need for digital conversion for storage and movement.



## Logarithmic Number Systems

* **Principle:** Similar to slide rules, using logarithmic scales to turn multiplication into addition.
* **Advantages:**
  * **Cheap multiplications:** Simple addition operation.
  * **High accuracy, especially for small values:** Error scales continuously with value.
* **Challenges:**
  * **Additions are traditionally expensive:** Requires table lookups to convert between logarithmic and linear representations.
* **Efficient Addition in Logarithmic Systems:**
  * **Factor out table lookups:** Perform lookups once per tensor (e.g., 10,000 elements) instead of per element.
  * **Sort elements based on fractional exponent part (EF).**
  * **Add integer parts (EI) together efficiently.**
  * **Perform a single lookup (or use hardwired constants) for each EF value.**
  * **Multiply partial sums by the looked-up values.**
  * **Convert the final sum back to logarithmic form.**



## Optimal Clipping

* **Goal:** Minimize error by optimally centering the representable range of a number system on the distribution of values to be represented (e.g., weights or activations).

* **Traditional Scaling:** 
  * Scale the number system to represent the minimum and maximum values without clipping.
  * No clipping noise, but high quantization noise due to large spacing between representable values.
  
* **Optimal Clipping:**
  
  * Introduce clipping noise by saturating values outside a chosen range.
  * **Reduces quantization noise significantly.**
  * **Minimizes the overall error (e.g., mean squared error) by balancing clipping noise and quantization noise.**
  
* **Finding the optimal clipping point:**
  $$
  s_{n+1} = \frac{\mathbb{E}\left[|X| \cdot \mathbb{1}_{\{|X| > s_n\}}\right]}{\frac{4^{-B}}{3} \mathbb{E}\left[\mathbb{1}_{\{|X| \leq s_n\}}\right] + \mathbb{E}\left[\mathbb{1}_{\{|X| > s_n\}}\right]}
  $$
  
  
  * An iterative equation approximates the integral that minimizes mean squared error.
  * A few iterations provide a good approximation of the optimal clipping range.
  
* **Benefits:**
  
  * **Can be worth more than a bit of precision in terms of mean squared error.**
  * **Significantly improves accuracy, especially for lower-precision representations.**
  
* **Note:** Clipping is typically done post-training, but training the clipping factor along with the model's weights could potentially yield further improvements.



## Scaling Granularity

* **Layer-wise scaling:** Initially used for both forward and backward propagation (separate scale factors).
* **Finer granularity scaling (e.g., vector-wise):** Improves accuracy by scaling smaller groups of elements independently.
  * **Example:** In ConvNets, scaling each 32-element vector in the channel dimension independently.
* **Hardware support:** Requires additional multipliers to apply activation and weight scale factors (SW and SA) after the MAC operation.
* **Benefits:** Tighter scaling for smaller groups of numbers leads to significantly reduced error, equivalent to gaining a couple of bits of precision.



## Sparsity

* **Pruning neural networks:** Removing less important weights (e.g., based on magnitude or sensitivity) can significantly reduce the number of computations without significant accuracy loss.
  * Multi-layer perceptrons (MLPs): Can prune up to 90% of weights.
  * Convolutional neural networks (ConvNets): Can prune up to 60-70% of weights.
* **Challenges in implementing sparsity efficiently:**
  * **Irregularity of sparsity patterns:** Leads to high overhead for bookkeeping and data shuffling.
  * **Difficulty in parallelizing sparse computations efficiently.**
* **Structured Sparsity (Ampere and Hopper):**
  * **Enforces a regular sparsity pattern (e.g., no more than 2 out of every 4 weights can be non-zero).**
  * **Dense training followed by structured pruning and retraining with a mask.**
  * **Compression by storing only non-zero weights and metadata indicating their positions.**
  * **Benefits:** Predictable sparsity pattern enables efficient parallel computations, achieving ~2x speedup.
* **Future directions:** Extending structured sparsity to activations and exploring other regular sparsity patterns.



## Accelerators vs. GPUs

* **NVIDIA's accelerator projects:** EIE, IRIS, SCNN, multi-chip module.
* **How accelerators achieve performance:**
  * **Special data types and operators:** Similar to GPUs, using specialized data types (e.g., FP8) and complex instructions (e.g., matrix multiply).
  * **Massive parallelism:** Exploiting parallelism to perform many operations concurrently.
  * **Optimized memory:** Minimizing main memory accesses through hierarchical scratchpads and data reuse.
  * **Algorithm-architecture co-design:** Adapting algorithms to the accelerator's architecture for optimal performance (e.g., bioinformatics accelerator with efficient alignment engine).
  * **Reduced dramatized overhead:** Simplifying control and data movement compared to CPUs.
* **Amortizing overhead:** Crucial for efficiency.
  * Example: Simple ARM out-of-order core:
    * CPU instruction overhead: 250 picojoules.
    * 16-bit integer add: 32 femtojoules.
    * Overhead dominates energy consumption (~99.99%).
* **Memory access costs:**
  * Local 8kB memory: 5 picojoules/word.
  * On-chip (hundreds of MB): 50 picojoules/word (45 picojoules for communication).
  * Off-chip LPDDR/HBM: 640 picojoules/word (32-bit).



## Magnetic BERT Accelerator

* **Design:**
  * Optimized for LLMs (BERT and BERT-large).
  * Incorporates optimal clipping and vector-level scaling (32-element vectors).
  * Achieves int4 (4-bit) inference with negligible accuracy loss.
* **Performance:**
  * **95.6 teraOPS/watt for int8 operations (~10x more efficient than Hopper).**
  * Demonstrates the potential for further efficiency improvements in future GPU designs.



## Conclusion

* Deep learning's progress is heavily reliant on hardware advancements.
* GPUs have provided significant performance gains (~1000x in 10 years) through architectural innovations (smaller numbers, complex instructions, sparsity).
* Scaling with multiple GPUs and longer training times has enabled the training of increasingly large models.
* **Future directions focus on:**
  * Number representation (logarithmic numbers, optimal clipping, scaling granularity).
  * Sparsity (exploiting higher levels and activation sparsity).
  * Memory and circuit optimizations (efficient memories, communication circuits, 3D memory).
  * Algorithm-architecture co-design.
* **Accelerators like Magnetic BERT demonstrate the potential for further efficiency gains, paving the way for future GPU architectures.**





## Q&A Session

#### Question 1: Network Size Optimization and Pruning Techniques

* **Question:** What techniques are used for optimizing network size (pruning)?
* **Answer:**
    * **Neural Architecture Search:** A team at NVIDIA uses neural architecture search to find optimal models based on various parameters (number of layers, channels per layer, layer sizes, etc.) given constraints on accuracy, execution time, or power.
    * **Pruning Techniques:**
        * **Magnitude-based pruning (most common):**
            * Scan weights in a layer and prune those with the smallest magnitudes based on a target density.
            * Essentially histograms the layer and sets weights below a threshold to zero.
        * **Sensitivity-based pruning (more complex but potentially more effective):**
            * Considers both the weight value and the sensitivity of the weight's connections to the output.
            * Prunes weights with the least sensitivity, taking into account both the weight value and its impact on the output.

#### Question 2: Energy Savings Breakdown for Complex Instructions

* **Question:** Regarding the energy savings from complex instructions, could you elaborate on the breakdown between fetch/decode savings and operand loading savings?
* **Answer:** While the exact breakdown isn't readily available, Dally believes that the majority of energy savings come from reduced fetch and decode operations. However, more detailed analysis would be required to provide specific numbers.

#### Question 3: Systolic Array Architectures vs. NVIDIA's Approach

* **Question:** Does NVIDIA use systolic arrays for matrix multiplications, like Google's TPU?
* **Answer:**
    * NVIDIA uses smaller matrix multiplies (4x4 for FP16/BFloat16) compared to Google's TPUs, which often use larger sizes (e.g., 128x128).
    * Smaller matrix sizes help avoid fragmentation issues that can arise when the natural matrix size isn't a power of two. Large matrix multipliers can lead to significant rounding-up inefficiencies.
    * While NVIDIA's approach isn't systolic, the efficiency of the core matrix multiply operation is comparable. The energy cost is dominated by the math units themselves.
    * NVIDIA feeds matrix multipliers from register files, potentially incurring slightly higher shuffling overhead for smaller matrices compared to systolic arrays.
    * However, even this overhead is likely minimal (estimated around 10%) and comparable to the data movement costs in systolic arrays. Google's TPUs still require data movement to feed the systolic array, and they also have control overhead.

#### Question 4: Hardware/Software Implementation of Clipping

* **Question:** Is the clipping technique implemented in hardware, software, or programmable hardware like FPGAs?
* **Answer:**
    * Clipping is primarily implemented in software.
    * NVIDIA GPUs already have the capability to use scale factors for adjusting the dynamic range of number representations.
    * Clipping simply involves choosing a larger scale factor, causing some numbers to saturate to the maximum representable value.
    * The only hardware requirement is the presence of multipliers for applying activation and weight scale factors (SW and SA), which are already present in GPUs that support scaling.
    * The clipping itself and the granularity of scaling are software-controlled, allowing flexibility in implementation. 








{{< include /_about-author-cta.qmd >}}
