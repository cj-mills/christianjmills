---
title: "Office Hours 2: Q&A Session with Zach Mueller"
date: 2024-6-11
image: /images/empty.gif
hide: false
search_exclude: false
categories: [notes, llms]
description: "This Q&A session covers various aspects of LLM fine-tuning, including tools, techniques, data sets, and hardware considerations."

twitter-card:
  creator: "@cdotjdotmills"
  site: "@cdotjdotmills"
  image: /images/default-preview-image-black.png
open-graph:
  image: /images/default-preview-image-black.png
---



::: {.callout-tip}
## This post is part of the following series:
* [**Mastering LLMs Course Notes**](/series/notes/mastering-llms-course-notes.html): My notes from the course **Mastering LLMs: A Conference For Developers & Data Scientists** by **Hamel Husain** and **Dan Becker**.
:::



## Key Takeaways

* **Hands-on experience is crucial for learning LLM fine-tuning:** Experimenting with code and models is more valuable than solely reading about it.
* **Community engagement is essential for feedback and learning:** Platforms like Twitter and Discord provide valuable spaces to connect with experts and peers.
* **Choosing the right data set is crucial for effective fine-tuning:** Synthetic data sets and those that evolve over time offer unique advantages.
* **Hardware plays a significant role in LLM training and inference:** NVIDIA GPUs remain dominant.
* **Model size should be determined by inference constraints and desired quality:** Smaller models often provide a good balance between performance and cost.



## 1. Axolotl vs. HF AutoTrain

* **Axolotl** and **HF AutoTrain** address different aspects of LLM training.
* **Axolotl** focuses on high-level, rapid model training for text-based tasks.
* **HF AutoTrain** offers a more agnostic approach, allowing for training various models with custom data.
* **Key difference:** Axolotl prioritizes speed and ease of use, while HF AutoTrain provides greater flexibility.



## 2. Learning Journey for LLM Engineers

* **Practical experience is paramount:** Start by experimenting with code and building models.
* **Active community engagement is crucial:** Seek feedback, ask questions, and share your learnings.
* **Focus on practical projects:** Choose small, manageable tasks to gain hands-on experience.
* **Iterate and learn from mistakes:** Analyze results, identify areas for improvement, and continuously refine your approach.



## 3. Finding Feedback and Community

* **Engage with experts on platforms like Twitter and Discord:**
* **Be proactive and demonstrate effort:** Share your work, ask specific questions, and show that you've attempted to solve the problem.
* **Contribute to the community:** Share your learnings, participate in discussions, and help others.



## 4. Public Data Sets for LLM Fine-Tuning

* **Hugging Face Data Sets:** Offers a wide variety of data sets suitable for LLM fine-tuning.
* **StarCoder 2 Self-Instruct Data Set:** Based on code from GitHub, provides benchmarks and a transparent pipeline.
* **Instruction Tuning Data Sets:** Help understand the principles of fine-tuning and prepare for more complex tasks.
  * [Extended Guide: Instruction-tune Llama 2](https://www.philschmid.de/instruction-tune-llama-2)

* **Synthetic Data Sets:** Offer control over the data generation process and enable testing for overfitting.



## 5. Accelerate, Torch Compile, and Distributed Training

* **FSDP (Fully Sharded Data Parallelism):** Essential for training large models by distributing data and model parameters across multiple GPUs.
* **DeepSpeed:** Offers more configuration options than FSDP, allowing for fine-grained control over offloading and device placement.
* **Torch Compile:** Primarily an inference-time optimization, but PyTorch aims to integrate it into training workflows.
* **Recommendation:** Use FSDP for models that fit in memory across all GPUs; consider DeepSpeed for scenarios requiring offloading.



## 6. Inference Precision and Hardware

* **BF16 (BFloat16):** Offers a good balance between performance and accuracy for training and inference.
* **FP16 (Half Precision):** Can be slower than BF16, especially on hardware optimized for BF16.
* **Recommendation:** Train models in BF16 to ensure compatibility with a wider range of inference hardware.



## 7. Downsides of FSDP

* **All-or-nothing approach:** FSDP distributes the entire model across GPUs, which can be limiting if the model doesn't fit in memory.
* **Lack of fine-grained control:** Unlike DeepSpeed, FSDP doesn't allow for selectively offloading specific layers to the CPU.



## 8. NVLink and GPU Performance

* **NVLink:** Provides high-bandwidth communication between GPUs, improving performance in multi-GPU setups.
* **Impact of NVLink absence:**  Debated, with some reporting significant performance degradation and others claiming minimal impact.
* **Consumer cards and throttling:** Consumer-grade GPUs might have driver-level limitations compared to their professional counterparts.
* **Recommendation:** Consider the RTX A4000 or RTX A4500 over RTX 4090s if budget allows:
  * **NVIDIA RTX A4500 (24GB)** 
    * [Product Page](https://www.nvidia.com/en-us/design-visualization/rtx-a4500/)
    * [Purchase Page](https://store.nvidia.com/en-us/nvidia-rtx/products/nvidia-rtx-4500-ada-generation/?nvid=em-VCNRTX4500ADA-PB)

  * **NVIDIA RTX A4000 (20GB)** 
    * [Product Page](https://www.nvidia.com/en-us/design-visualization/rtx-a4000/)
    * [Purchase Page](https://store.nvidia.com/en-us/nvidia-rtx/products/nvidia-rtx-4000-ada-generation/index.html?nvid=em-VCNRTX4000ADA-PB)




## 9. Fine-Tuning vs. Frontier Models

* **Fine-tuning can achieve comparable or even surpass the performance of frontier models:** Community-driven fine-tuning efforts like Teknium's models demonstrate this potential.
* **Data access remains a challenge:** Closed-source models often benefit from significantly larger and proprietary data sets.



## 10. Ensuring Prompting and Tokenization Consistency

* **Hugging Face Pipelines:** Provide a reliable way to load and use fine-tuned models for inference.
* **Chat Templating:** Hugging Face's chat templates offer a standardized approach to prompting, but they might not be directly compatible with all tools.
* **Thorough Testing:** Always test inference with the same tokenization and prompting procedures used during training.



## 11. Running Inference on an 8 Billion Parameter Model with a 24GB GPU

* **Quantization:** Techniques like AWQ (AutoAWQ) can reduce model size and memory footprint, enabling inference on less powerful hardware.
* **Offloading:** Offloading parts of the model to the CPU can enable inference on limited VRAM, but it comes with a performance trade-off.



## 12. Training Models in 8-Bit Precision

* **Instability:** Training in 8-bit precision (INT8 or FP8) can lead to instability and convergence issues.
* **Experimental Support:** While frameworks like PyTorch are adding support for 8-bit training, it remains experimental.
* **BF16 with FP8:** Some hardware platforms utilize a combination of BF16 and FP8 for training, potentially offering a middle ground.



## 13. Limitations of Accelerate

* **Accelerate as a wrapper:** Accelerate primarily acts as a wrapper around existing distributed training frameworks, so its failures often stem from underlying issues.
* **Timeout issues:** Occasional timeout problems have been observed, but the root cause remains unclear.



## 14. Relevance of Chinchilla Scaling Laws

* **Still relevant for optimal resource allocation:** Chinchilla scaling laws provide guidance on balancing parameters and data size for a given compute budget.
* **Don't guarantee the best model:**  Continuously training a model until convergence often yields the best results, regardless of scaling laws.
* **Under-trained models and fine-tuning:**  Models trained with fewer steps than suggested by scaling laws might be more amenable to fine-tuning.



## 15. Relevance of TensorFlow for LLM Fine-Tuning

* **PyTorch dominance:** PyTorch has become the dominant framework for LLM research and development.
* **TensorFlow's role:** TensorFlow, particularly Keras, still serves as a backend in some LLM frameworks, but its popularity has diminished.



## 16. Training on Apple Silicon

* **Inference:** Apple Silicon performs well for LLM inference tasks.
* **Training:** Training support is improving but remains behind NVIDIA GPUs in terms of maturity and performance.
* **Hardware limitations:** Apple's system-on-a-chip architecture and lack of dedicated server-grade GPUs pose challenges for large-scale training.



## 17. Serving Multiple LoRAs with Accelerate Inference

* **VLLM:** Supports loading and switching between multiple LoRAs during inference.
* **Hot-swapping:** VLLM allows for selecting different LoRAs on a per-request basis, enabling dynamic model customization.



## 18. Mixture of LoRAs

* **Concept:** Training multiple LoRAs specializing in different tasks and using a router to dynamically select the most appropriate LoRA for a given input.
* **Kraken Model:** An example of a model that utilizes dynamic model routing with multiple expert models.
  * [VAGOsolutions/Kraken-LoRA](https://huggingface.co/VAGOsolutions/Kraken-LoRA)




## 19. Choosing a Fine-Tuning Project

* **Personal interest and relevance:** Select projects that align with your interests and current work.
* **Replicating existing work:** Recreating existing projects is a valuable learning experience.
* **Data availability:** Choose projects with readily available or easily obtainable data sets.
* **Document your process:** Keep track of your experiments, results, and lessons learned.



## 20. Constraints and Sweet Spots in Fine-Tuning

* **Budget and hardware:** Determine the available compute resources and select a model size accordingly.
* **Inference time and cost:** Prioritize inference efficiency, as it significantly impacts real-world deployment costs.
* **Iteration speed:** Smaller models allow for faster experimentation and iteration cycles.
* **Quality requirements:** Balance model size with the desired performance level for the specific task.
  * 7 to 8 billion parameter models are often the sweet spot in real-world projects.




## 21. Fine-Tuning on Phi-3

* **Limited real-world performance:** Despite its size, Phi-3 has not demonstrated competitive performance in practical applications.
* **Data and training methodology:** Potential issues with the training data or methodology might contribute to its shortcomings.





{{< include /_about-author-cta.qmd >}}
