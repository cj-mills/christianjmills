---
title: "Office Hours 1: Axolotl Q&A with Wing Lian"
date: 2024-6-11
image: /images/empty.gif
hide: false
search_exclude: false
categories: [notes, llms]
description: "This Q&A session covered various topics, including template-free prompt construction, data type selection for HuggingFace datasets, DPO and RLHF, understanding chat templates and dataset types, ensuring consistent tokenization, multimodal fine-tuning, and future directions for Axolotl."

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





## 1. Template-Free Prompt Construction in Axolotl

* **Purpose:** Offers flexibility in defining custom chat roles and formats.
* **Format:** Uses simple input-output pairs with labels (true for model output, false for user input).
* **Advantages:**
  * Easier to understand for some users.
  * Translates well to platforms with existing input-output data.
* **Disadvantages:**
  * Less flexible for changing chat templates later.
  * Requires more manual string handling during inference.
* **Recommendation:** Use only if existing chat templates are insufficient.
* **Documentation:** [https://openaccess-ai-collective.github.io/axolotl/docs/input_output.html](https://openaccess-ai-collective.github.io/axolotl/docs/input_output.html)
* ::: {.callout-note title="Example:" collapse=true}
  
  ```json
  {
      "segments": [
          {
              "label": true,
              "text": "<s>Hello\n"
          },
          {
              "label": true,
              "text": "hi there!. "
          },
          {
              "label": false,
              "text": "goodbye "
          },
          {
              "label": true,
              "text": "farewell</s>"
          }
      ]
  }
  ```
  
  :::







## 2. How to Decide Data Type for Datasets on HuggingFace?

* **Dataset Types:**
  * **Alpaca:** Instruction-based, with separate fields for instruction, input, and output.
  * **ShareGPT:** Conversation-based, with variations in field names (e.g., human/user, GPT/assistant, value/content). Axolotl's `type: sharegpt` handles most variations.
  * **DPO:** Includes chosen and rejected responses for preference learning. Variations exist in field names and prompt construction.
  * **KTO:** Similar to DPO but without explicit preference pairs.
  * **User Defined:** Allows custom formatting defined in the YAML file.
* **Recommendation:** Choose the type that best matches the dataset structure and fine-tuning objective.
* **Examples:**
  * **[argilla/dpo-mix-7k](https://huggingface.co/datasets/argilla/dpo-mix-7k):** Chosen, rejected fields.
  * **[Intel/orca_dpo_pairs](https://huggingface.co/datasets/Intel/orca_dpo_pairs):** Question, chosen, rejected fields.
  * **[argilla/kto-mix-15k](https://huggingface.co/datasets/argilla/kto-mix-15k):** Prompt, completion, label, rating.



## 3. DPO and RLHF

* **DPO (Direct Preference Optimization):**
  * Trains a model to maximize the probability of choosing preferred responses over rejected ones.
  * Simpler to implement than RLHF but limited to single-turn preference learning.
* **RLHF (Reinforcement Learning from Human Feedback):**
  * Uses reinforcement learning to optimize a model based on human feedback.
  * More complex but potentially leads to higher quality alignment and multi-turn capabilities.
* **Future Direction:** Both DPO and RLHF are important, with a potential shift towards more robust RL-based methods.



## 4. Difference Between Chat Template and Datasets Type Parameters

* **Chat Template:** Defines the specific format of the chat conversation (e.g., LLAMA3, Mistral, ChatML). It gets added to the tokenizer config.
* **Datasets Type:** Specifies how Axolotl should parse and structure the input dataset (e.g., share_gpt, alpaca, dpo).
* **Interaction:**
  * Setting a chat template can automatically set the output format for certain dataset types.
  * The `chat_template` parameter in the YAML file overrides any default settings.



## 5. No Ops for Validation

* **Currently, Axolotl lacks built-in validation checks for potential issues like:**
  * Rounding errors when saving models in different precision formats (e.g., float32 vs. bfloat16).
  * Tokenization discrepancies between training and inference.
* **Recommendation:**
  * Carefully manage model precision during saving.
  * Implement custom checks to compare tokenization between training and inference pipelines.



## 6. Trust No One for Tokenization

* **Key Takeaway:** Always verify the actual tokens being fed into the model, as string handling and YAML parsing can introduce subtle errors.
* **Example:** YAML can remove trailing spaces in non-quoted strings, potentially affecting tokenization.
* **Recommendation:** Implement rigorous checks to ensure consistent tokenization between training and inference.



## 7. Ensuring Consistent Tokenization

* **Challenge:** Tokenization differences can arise from:
  * Separate vs. concatenated tokenization of input and output strings.
  * Special token handling in different chat templates.
* **Recommendations:**
  * Use the `chat_template` parameter in Axolotl to enforce consistent formatting.
  * Implement tests to compare tokenization between fine-tuning and inference setups.
  * Consider introducing minor tokenization variations during training as a form of data augmentation.



## 8. Tokenizer Configs from Training to Inference

* **Importance:** Consistent tokenizer configurations are crucial for seamless transition from training to inference.
* **Axolotl's Approach:** Setting the `chat_template` parameter in the YAML file updates the tokenizer config, which is then used by inference engines.
* **Challenge:** Not all inference engines may fully support or utilize the chat template information.
* **Recommendation:** Verify that the chosen inference engine correctly interprets and applies the tokenizer config, including the chat template.



## 9. Multimodal Fine-tuning

* **Current Status:** Axolotl lacks native support for multimodal datasets and models.
* **Challenges:**
  * Handling image data and integrating it with text data.
  * Adapting to evolving approaches for multimodal tokenization and model architectures.
* **Future Direction:**
  * Implementing dataset handling for images and other modalities.
  * Potentially supporting both Lava-like approaches and native multimodal models.
* **Call for Contributions:**  Help is needed in developing and implementing multimodal capabilities.



## 10. Is RLHF Still a Common Fine-tuning Technique?

* **Answer:** Yes, RLHF and other preference-based tuning methods (like DPO) are becoming increasingly common.
* **Reasoning:**
  * Supervised fine-tuning has limitations in achieving high-quality alignment.
  * RLHF and DPO enable learning from human preferences, leading to better model behavior.
* **Future Trend:** Expect to see wider adoption of both RL-based and non-RL preference optimization techniques.



## 11. DPO Limitations and RL Advantages

* **DPO Limitation:** Primarily designed for single-turn preference learning.
* **RL Advantages:**
  * Supports multi-turn conversations and intermediate rewards.
  * Can lead to better alignment and more nuanced model behavior.
* **Trade-offs:**
  * RLHF is more complex and data-intensive than DPO.
  * DPO is simpler to implement and doesn't require a separate reward model.



## 12. Sample Files for PaliGemma and Phi-3

* **Phi-3:**
  * Should work with existing Phi-2 configurations by swapping the baseline model.
  * May require setting `trust_remote_code: true` in the YAML file.
* **PaliGemma:**
  * No specific examples available yet.
  * LLM fine-tuning might be possible, but full support requires multimodal dataset handling.



## 13. Conversational Datasets vs. QA Pairs

* **Assumption:** Conversational datasets are always more effective for fine-tuning.
* **Clarification:** The choice depends on the specific use case and desired model behavior.
* **Recommendations:**
  * **QA Pairs:** Suitable for single-turn interactions or when mimicking a retrieval-based system.
  * **Conversational Datasets:** Beneficial for training models to engage in multi-turn dialogue.
* **Instruction Tuning:** Recommended for gaining intuition about conversational datasets and fine-tuning.



## 14. Training Datasets for Completion Models

* **Dataset Characteristics:** Typically similar to pre-training datasets, often with a single "text" field.
* **Examples:**
  * Story generation datasets.
  * Any dataset focused on text completion or continuation.



## 15. Prompt Template for LLAMA3 and LLAMA Index

* **Goal:** Fine-tune LLAMA3 for use with LLAMA Index, which uses an OpenAI-like message abstraction.
* **Recommendation:**
  * Choose a chat template that aligns with the message-based format (e.g., ChatML).
  * Avoid instruction-based templates as they might not be suitable for multi-turn interactions.



## 16. Future Directions of Axolotl

* **Areas for Contribution:**
  * Join the [Axolotl Discord server]() and contribute to discussions.
  * Explore the GitHub repository for [open issues](https://github.com/OpenAccess-AI-Collective/axolotl/issues) and feature requests.
  * Developing new features and improving existing ones.
* **Ongoing Development:**
  * Building a turnkey platform for simplified fine-tuning and deployment (similar to Modal).
  * Integrating DPO, PPO, and enhanced dataset pipelines.
  * Creating a user-friendly CLI and cloud integration.



## 17. VRAM Estimation

* **Need:** A tool for accurate VRAM estimation based on Axolotl configurations.
* **Challenges:**
  * Complexities introduced by techniques like FSDP and DeepSpeed.
  * Variations in VRAM usage based on batch sizes and model parallelism.
* **Potential Approach:** Leverage existing LLM math estimations and account for the impact of distributed training techniques.



## 18. Vibe Checks During Training

* **Goal:** Evaluate model performance and "vibes" during training.
* **Options:**
  * **Periodic Checkpointing:** Pause training, run inference on the checkpoint, and resume.
  * **Dedicated Evaluation:** Use a separate process to run inference on an eval dataset and log the results.
  * **Callbacks:** Implement callbacks to trigger inference on demand during training.
* **Challenges:**
  * VRAM limitations might make it difficult to run inference alongside training.
  * Ensuring consistent tokenization and prompt handling between training and evaluation.



## 19. Familiarizing with Prompt Templates

* **Recommendation:** Use the `axolotl.cli.preprocess` command with the `debug` flag to visualize how Axolotl processes and tokenizes prompts.
* **Output:** Displays the tokenized prompt with color-coding to distinguish between input, output, and masked tokens.



## 20. Axolotl vs. Unsloth

* **Unsloth:**
  * Specialized for Lora fine-tuning.
  * Offers memory optimizations but might be limited in GPU scalability.
* **Axolotl:**
  * Provides a more comprehensive framework for fine-tuning, including prompt management and dataset handling.
  * Focuses on performance optimizations like sample packing.
* **Recommendation:** Choose the tool that best aligns with your specific needs and priorities.



## 21. Quick and Dirty Fine-tuning

* **Recommendation:**
  * Start with a small "tiny llama" example for faster iteration.
  * Use Gradio inference for quick model evaluation.



## 22. Function Calling Fine-Tunes

* **Dataset Example:** Glade datasets from the Noose team.
* **Configuration:** Might require specific role handling and a compatible version of the ShareGPT dataset type.



## 23. Visualizing Tokenization in Batches

* **Challenge:** Axolotl's sample packing happens at runtime, making it difficult to visualize tokenization in batches during pre-processing.
* **Potential Approach:** Modify the Transformers or LLAMA model code to print or log input IDs during the forward pass.





{{< include /_about-author-cta.qmd >}}
