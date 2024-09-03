---
title: "Conference Talk 6: Train Almost Any LLM Model Using 🤗 autotrain"
date: 2024-7-12
image: /images/empty.gif
hide: false
search_exclude: false
categories: [notes, llms]
description: "In this talk, **Abhishek Thakur**, who leads AutoTrain at 🤗, shows how to use 🤗 AutoTrain to train/fine-tune LLMs without having to write any code."

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





* [Introduction to AutoTrain](#introduction-to-autotrain)
* [Getting Started with AutoTrain](#getting-started-with-autotrain)
* [Fine-tuning LLMs with AutoTrain](#fine-tuning-llms-with-autotrain)
* [Training Your Model](#training-your-model)
* [Config Files and Advanced Options](#config-files-and-advanced-options)
* [Additional Features and Considerations](#additional-features-and-considerations)
* [Q&A Session](#qa-session)





## Introduction to AutoTrain

* **Homepage:** [https://huggingface.co/autotrain](https://huggingface.co/autotrain)
* **Documentation:** [https://huggingface.co/docs/autotrain/index](https://huggingface.co/docs/autotrain/index)
* **GitHub Repository:** [autotrain-advanced](https://github.com/huggingface/autotrain-advanced)
* Simplifies model training and fine-tuning for users with varying levels of expertise, from beginners to experienced data scientists.
* **Supported Tasks:**
  * **NLP:** Token classification, text classification, LLM tasks (e.g., SFT, R4, DPO, reward tuning), sentence transformer fine-tuning, etc.
  * **Computer Vision:** Image classification, Object Detection
  * **Tabular Data:** Classification, Regression
* Leverages the Hugging Face ecosystem, including transformers, datasets, diffusers, and Accelerate, ensuring compatibility with the latest models and tools.



## Getting Started with AutoTrain

* **Create a new project:**
  * **Link:** [`Create new project`](https://huggingface.co/login?next=%2Fspaces%2Fautotrain-projects%2Fautotrain-advanced%3Fduplicate%3Dtrue)
  * Optionally specify an organization and attach hardware (local or Hugging Face spaces).
  * Choose the desired task (e.g., LLM SFT).

* **User-Friendly Interface:**
  * Select a task.
  * Upload your data or use a dataset from the Hugging Face Hub.
  * Configure parameters or use default settings.
  * Monitor training progress and logs.
* **Documentation:** [Creating a New AutoTrain Space](https://huggingface.co/docs/autotrain/quickstart_spaces#creating-a-new-autotrain-space)



## Fine-tuning LLMs with AutoTrain

### Supervised Fine-tuning (SFT) and Generic Fine-tuning

* **Documentation:**  [Supervised Fine-tuning Trainer](https://huggingface.co/docs/trl/main/en/sft_trainer)
* Both trainers are similar, but SFT uses the TRL library's SFT trainer.
  * **Documentation:** [TRL - Transformer Reinforcement Learning](https://huggingface.co/docs/trl/main/en/index)

* Requires a "text" column in your dataset (can be mapped from a different column name).
* Supports chat template formatting (chatML, Sapphire, tokenizer's template).
* **Example datasets:**
  * [Salesforce/wikitext](Salesforce/wikitext): plain text format
  * Chat format with "content" and "role" fields (requires chat template).

### Reward Modeling

* **Documentation:**  [Reward Modeling](https://huggingface.co/docs/trl/main/en/reward_trainer)
* Trains a custom reward model for sequence classification.
* Dataset requires "chosen" and "rejected" text columns.



### DPO and ORPO

* **DPO - Direct Preference Optimization**
  * **Documentation:**  [DPO Trainer](https://huggingface.co/docs/trl/main/en/dpo_trainer)

* **ORPO - Odds Ratio Preference Optimization**
  * **Documentation:**  [ORPO Trainer](https://huggingface.co/docs/trl/main/en/orpo_trainer)

* ORPO is recommended over DPO as it requires less memory and compute.
* Dataset requires "prompt," "chosen," and "rejected" columns (all conversations).
* Supports chat templates.



## Training Your Model

### Data Format

* Use CSV or JSON Lines (JSONL) format 
  * JSONL preferred for readability and ease of use.

* **Format examples:**
  * Alpaca dataset: Single "text" field with formatted text (no chat template needed).
  * Chat format: Requires chat template or offline conversion to plain text.

### Training Locally

* **Documentation:** [Quickstart](https://huggingface.co/docs/autotrain/quickstart)

* Set Hugging Face token: 

  * ```sh
    export HF_TOKEN=<your_token>
    ```

* Run the AutoTrain app: `autotrain app`

  * ```sh
    autotrain app --port 8080 --host 127.0.0.1
    ```

* Alternatively, use config files or CLI commands.

  * ```sh
    autotrain --config <path_to_config_file>
    ```

    

#### Local Installation

::: {.panel-tabset}

## PIP

```sh
pip install autotrain-advanced
```

## Conda

```sh
conda create -n autotrain python=3.10
conda activate autotrain
pip install autotrain-advanced
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install -c "nvidia/label/cuda-12.1.0" cuda-nvcc
```

:::



### Training on Other Platforms

* **[Jarvis Labs](https://jarvislabs.ai/):** Provides AutoTrain templates for easy setup and training.
* **[DGX Cloud](https://www.nvidia.com/en-us/data-center/dgx-cloud/):** Rent high-performance GPUs for training large models.
* **[Google Colab](https://colab.research.google.com/):** Run AutoTrain directly in Colab using provided notebooks and UI.



## Config Files and Advanced Options

* **Documentation:**  [AutoTrain Configs](https://huggingface.co/docs/autotrain/config)
* Config files offer more flexibility and control over training parameters.
* Define task, base model, data paths, column mapping, hyperparameters, logging, and more.
* Access example config files in the AutoTrain GitHub repository.
  * **GitHub Repository:** [autotrain-advanced/configs](https://github.com/huggingface/autotrain-advanced/tree/main/configs)




## Additional Features and Considerations

* AutoTrain automatically handles multi-GPU training using DeepSpeed or distributed data parallel.
* QLORA is supported on DeepSpeed for efficient training.
* Sentence Transformer fine-tuning is available for tasks like improving RAG models.



## Q&A Session

* **Logging:** Supports Weights & Biases (W&B) logging when using config files.
* **Mixed Precision:** Supports BF16 and FP16, but not FP8.
* **Parameter Compatibility:** AutoTrain ensures parameter compatibility based on the chosen base model.
* **Hyperparameter Optimization:** Not currently supported for LLMs due to long training times.
* **CPU Training:** Possible, but may come with performance limitations.
* **Custom Chat Templates:** Can be added by modifying the `tokenizer_config.json` file of a cloned model.
* **Synthetic Data Generation:** Not currently supported, but users can generate their own.



