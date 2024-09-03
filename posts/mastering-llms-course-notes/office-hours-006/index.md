---
title: "Office Hours 6: Johno Whitaker"
date: 2024-7-11
image: /images/empty.gif
hide: false
search_exclude: false
categories: [notes, llms]
description: "This Q&A session covers a wide range of topics related to LLMs, including practical tips for training and optimization, insights into the current research landscape, and thoughts on future trends."

twitter-card:
  creator: "@cdotjdotmills"
  site: "@cdotjdotmills"
  image: ../social-media/cover.jpg
open-graph:
  image: ../social-media/cover.jpg
---



::: {.callout-tip}
## This post is part of the following series:
* [**Mastering LLMs Course Notes**](/series/notes/mastering-llms-course-notes.html): My notes from the course **Mastering LLMs: A Conference For Developers & Data Scientists** by **Hamel Husain** and **Dan Becker**.
:::







## Key takeaways

- **Striking a balance between GPU utilization and cache:** While maximizing GPU utilization seems ideal, leaving some headroom for caching can actually improve performance.
- **The economics of cloud GPU pricing:** Faster, more expensive GPUs often end up being cost-effective due to reduced training time.
- **Prioritizing alternative customization methods before fine-tuning:** Consider techniques like prompt engineering and context injection before resorting to fine-tuning.
- **The importance of quick iteration cycles in research:** Start with smaller models and datasets to test ideas rapidly before scaling up.
- **The value of understanding the tools and digging deeper:** While off-the-shelf libraries are convenient, having a deeper understanding of the underlying code can be crucial for research and debugging.
- **The potential of alternative hardware and algorithms:** While GPUs and deep learning dominate the current landscape, there is hope for innovation with new programming paradigms and architectures.
- **Focusing on practical skills and real-world applications:** Building projects and gaining hands-on experience are invaluable, even if specific technologies become obsolete.
- **The evolving role of LLMs in research:** LLMs are becoming increasingly useful for tasks like paper retrieval and summarization, with the potential for even greater impact in the future.
- **The importance of exploring diverse research directions:** While chasing the latest trends can be tempting, exploring less crowded research areas can lead to novel and impactful contributions.



## GPU Utilization and Batch Size

- Increasing batch size generally improves performance, especially when memory bandwidth is a bottleneck.
- However, pushing GPU memory utilization too close to 100% can hinder performance by limiting space for caching.
- It's crucial to leave some headroom for pre-caching layers and avoiding out-of-memory errors during evaluation or unexpected events.
- Find the largest comfortable batch size with reasonable memory utilization.



## Balancing Compute and IO Costs

- While using more GPUs increases cost per hour, it also significantly reduces training time.
- Cloud GPU pricing often reflects this trade-off, making faster GPUs cost-effective overall.
- Consider the time cost of experimentation when choosing between more or fewer GPUs.
- For multi-node setups, prioritize data parallelism across nodes and leverage memory-saving techniques within each node to minimize communication overhead.



## Hyperparameter Tuning

- Fine-tuning is not always necessary, as many models perform well with default settings or prompt engineering.
- Focus on finding a model and basic configuration that learns effectively before extensively optimizing hyperparameters.
- Consider the specific requirements of the application; stylistic formatting may benefit from fine-tuning, while direct knowledge integration might be better served by context injection.



## Fine-tuning vs. Alternative Customization Methods

- Fine-tuning can be useful for customizing model behavior, but it's not the only or always the best approach.
- Explore alternative methods like prompt engineering, context injection, and retrieval-augmented generation.
- Consider the trade-offs between fine-tuning efficiency, context length limitations, and the ability to incorporate external knowledge.



## TPUs and Other Accelerators

- TPUs offer high-speed memory access and interconnects, making them suitable for large-scale training.
- The principles of memory management and optimization still apply, but the specific considerations might differ.
- As dedicated hardware evolves, the lines between GPUs and other accelerators are blurring, with a focus on faster interconnects and larger memory pools.



## Optimizing Memory Usage for LLaMa Models

- Quantization (e.g., QLoRA) can significantly reduce memory footprint without major performance degradation.
- Consider using 4-bit quantization for base weights to free up memory.
- Adjust batch size and gradient accumulation steps to accommodate longer sequences.
- Be mindful of the memory overhead associated with larger vocabularies and embedding layers.



## Understanding Sequence Length and Memory Usage

- Sequence length during training often represents a maximum value, and actual memory usage depends on the length of individual samples in a batch.
- Padding to the maximum sequence length can waste compute resources.
- Consider techniques like packing short sequences together or prioritizing longer sequences to optimize memory utilization.
- Be aware of how truncating sequences during training might impact the model's ability to learn effectively.



## Tips for Quick Iteration Cycles

- Start with smaller models and datasets to test code and ideas quickly.
- Gradually increase model size and data complexity as needed.
- Prioritize being able to evaluate results quickly, ideally within seconds or minutes.
- Develop a workflow that allows for rapid testing and debugging without long waiting times.



## Choosing the Right Tools for the Job

- Off-the-shelf libraries like Axolotl and HuggingFace Trainer are convenient for standard training tasks.
- For research and deeper understanding, consider using simpler training loops and libraries that provide more transparency and control.
- Strive for a balance between ease of use and the ability to inspect and debug the underlying code.



## CPU Offloading and GPU Memory Management

- While CPU offloading exists, it's often slow due to the speed difference between CPU and GPU RAM.
- Explicitly manage CPU offloading rather than relying on automatic mechanisms.
- Consider CPU offloading when dealing with very long sequences that exceed GPU memory capacity, even with quantization.



## Exploring CPU Offloading for Long Sequences

- CPU offloading might become more attractive as longer context lengths become increasingly important.
- By storing weights on the CPU and transferring them to the GPU as needed, larger batches and longer sequences can be processed.
- This approach trades off increased transfer time for the ability to handle more data within the GPU memory constraints.



## Curating Information and Staying Up-to-Date

- Develop a system for filtering and prioritizing information from various sources, such as colleagues, Twitter, and research papers.
- Focus on areas of personal interest and relevance to current projects.
- Actively seek out information that can be applied to real-world problems and experiments.



## The Importance and Joy of Teaching

- Teaching is a rewarding way to learn, solidify understanding, and contribute to the community.
- Sharing knowledge through blog posts, tutorials, and presentations can benefit both the teacher and the audience.
- Engaging with questions and feedback from learners can spark new ideas and research directions.



## The Hardware Lottery and Future of AI

- The dominance of GPUs and deep learning might be partly due to historical coincidence rather than inherent superiority.
- The significant investment in GPU-optimized hardware and software creates inertia.
- Breaking free from this paradigm requires making alternative approaches more accessible and efficient.
- Innovations in GPU programming, new hardware architectures, and a willingness to explore unconventional ideas offer hope for a more diverse AI landscape.



## Balancing Research and Practical Skills

- Pursuing purely novel research and building practical applications are distinct but valuable pursuits.
- For practical impact, focus on solving current problems, learning industry-standard tools, and gaining experience with real-world data.
- For research breakthroughs, explore less crowded areas, challenge assumptions, and develop a deep understanding of the fundamentals.



## Choosing Projects for Skill Development

- Engaging in projects, even if they become less relevant with time, provides valuable learning experiences and demonstrates technical skills.
- Focus on projects that allow you to learn transferable skills like data processing, model evaluation, and problem-solving.
- Stay adaptable and continuously update your skillset as the field evolves.



## LLMs Impacting Research

- LLMs are already proving useful for tasks like paper retrieval, summarization, and knowledge extraction.
- Tools like undermind.ai showcase the potential of LLMs for improving research workflows.
  - **[undermind.ai](https://undermind.ai/home/):** During each search, undermind.ai examines results in stages, and uses language  models to make key decisions, such as recognizing crucial information and adapting the search strategy.

- Future research directions include exploring more sophisticated applications of LLMs for tasks like knowledge synthesis, hypothesis generation, and experimental design.
  - **[sakana.ai](https://sakana.ai/):** On a quest to create a new kind of foundation model based on nature-inspired intelligence.
    - **GitHub Repository:** [Evolutionary Optimization of Model Merging Recipes](https://github.com/SakanaAI/evolutionary-model-merge)
    - **Blog Post:** [Can LLMs invent better ways to train LLMs?](https://sakana.ai/llm-squared/):




## Rapid-Fire Q&A Highlights

### Research Interests and Coding Style

* **Current Focus:** While still interested in GANs and diffusion models, Johno's primary focus has shifted to LLMs. 
* **Coding Style:**
  * Uses tools like Copilot for boilerplate code.
  * Emphasizes clear, tutorial-like code for better Copilot integration and beginner comprehension.
  * Values explicitness over extreme code compression, especially when teaching.
  * Adopts a mix of notebooks (VS Code, Cursor, Jupyter Classic) and scripts depending on the project.
    * **[Cursor](https://www.cursor.com/):** The AI code editor
  * Leverages keyboard shortcuts and learns from colleagues like Jeremy Howard for efficiency.

### Quantization and Long Context

* **Quantization Overhead:**  While quantization methods like QLoRA reduce memory footprint, they introduce computational overhead due to decompression, sometimes impacting performance.
* **Long Context Challenges:**  Initially, QLoRA faced memory efficiency issues with long context lengths compared to LoRA, potentially due to gradient checkpointing implementations. 
  * This has been addressed to some extent. 

* **Unexpected Behavior:**  Practical implementations often reveal unexpected behavior compared to theoretical calculations. 
  * Bugs, precision errors, and data duplication can arise.


### 1.58-bit LLMs and Hybrid Approaches

* **1-bit and 1.58-bit LLMs:**  Microsoft's research explored extreme weight compression using 1-bit (-1, 0, 1) representation, achieving workable but diminished performance. 
  * **Paper:** [The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits](https://arxiv.org/abs/2402.17764)

* **Hybrid Approach Potential:** Whitaker believes combining extreme quantization with techniques like LoRA adapters could offer a sweet spot between memory efficiency, speed, and accuracy.
  * Base weights in 1-bit or 2-bit.
  * LoRA adapters for fine-tuning and accuracy recovery.
  * Hardware-efficient kernels for low-bit operations.
* **Mobius Labs:** Highlighted for their work on efficient kernels and proof-of-concept implementations of hybrid approaches. 
  * **Homepage:** [Mobius Labs](https://www.mobiuslabs.com/): Multimodal AI for the world's scale.
  * **Blog Post:** [Half-Quadratic Quantization of Large Machine Learning Models](https://mobiusml.github.io/hqq_blog/)


### Alternative Architectures

* **Exploring Alternatives:** While transformers dominate, alternative architectures like KAN, state-space models (SSMs), and recurrent models offer benefits in specific areas.
  * **State-Space Models:**
    * **Paper:** [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)
    * **GitHub Repository:** [mamba](https://github.com/state-spaces/mamba)
  * **KAN: Kolmogorov Arnold Networks:**
    * **Paper: KAN:** [Kolmogorov-Arnold Networks](https://arxiv.org/abs/2404.19756)
    * **GitHub Repository:** [pykan](https://github.com/KindXiaoming/pykan)
* **SSMs and Long Sequences:**  SSMs show promise for tasks involving long sequences, such as text-to-speech and DNA analysis. 
* **Practical Implications:** The emergence of viable alternatives provides practitioners with more tools. The core focus remains on performance and benchmark results. 
