---
title: "Conference Talk 13: When to Fine-Tune with Paige Bailey"
date: 2024-7-25
image: /images/empty.gif
hide: false
search_exclude: false
categories: [notes, llms]
description: "In this talk, **Paige Bailey**, Generative AI Developer Relations lead at Google, discusses Google's AI landscape with a focus on Gemini models and their applications."

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





* [Google AI Landscape and Gemini](#google-ai-landscape-and-gemini)
* [Understanding Context Windows](#understanding-context-windows)
* [Fine-tuning vs. Prompting vs. Retrieval](#fine-tuning-vs.-prompting-vs.-retrieval)
* [Prompting Strategies and Examples](#prompting-strategies-and-examples)
* [Retrieval Augmented Generation](#retrieval-augmented-generation)
* [Fine-tuning Considerations and Gemma](#retrieval-augmented-generation)





## Google AI Landscape and Gemini

* **Vertex AI:**

  * **[Vertex AI](https://cloud.google.com/vertex-ai?hl=en):** Collection of APIs, compute infrastructure, model deployment tools available through Google Cloud, geared towards enterprise use.  Comparable to Azure Open AI services.
  * **[Gemini Developer API](https://ai.google.dev/) (through [AI Studio](https://ai.google.dev/aistudio)):** Easier path for rapid prototyping and personal projects.  Comparable to OpenAI APIs. 

* **Gemini Flash Fine-Tuning:**

  * **[Gemini 1.5 Flash](https://deepmind.google/technologies/gemini/flash/):** Google's most performant, efficient, and cost-effective model, boasting a 1 million token context window (and growing).
  * Supports fine-tuning and is part of an early tester program. 

* **Gemini Nano & Gemma:**

  * **[Gemini Nano](https://deepmind.google/technologies/gemini/nano/):** Brief mention of its planned integration into Chrome and Pixel/Android devices (details deferred).
  * **[Gemma](https://ai.google.dev/gemma):**  
    * Open-source versions of Gemini, available on Hugging Face, Kaggle, and Ollama, making local experimentation easy.
    * Kaggle hosts checkpoints, code samples, and runnable notebooks. 

  

### Generative AI and Google

* Google's history in machine learning: TensorFlow, transformer models (BERT, AlphaFold, AlphaStar, AlphaGo, T5), and now Gemini.
* Generative AI extends beyond text and code, mentioning:
  * **[Imagen 2](https://deepmind.google/technologies/imagen-2/):** Detailed image generation.
  * **[Chirp](https://cloud.google.com/speech-to-text/v2/docs/chirp-model):** Speech-to-text with multilingual capabilities and a small model footprint.
* **Gemini:**  Google's flagship model (currently on version 1.5) 



### Gemini Model Features

* **Multimodal Understanding:**  Processes images, audio, text, code, video, and more simultaneously.
* **State-of-the-art Performance:**  Excels across various tasks, though reliant on academic benchmarks (discussed later).
* **Embedded Reasoning:**  Strong capabilities in chain-of-thought and step-by-step reasoning.
* **Scalable Deployment:**  Optimized for both large-scale (Google products) and small-scale (edge devices) use cases.
* **Efficiency and Privacy:** Focus on cost-effective token analysis, reduced inference compute, and on-device processing for privacy preservation.
* **Model Options:**
  * **[Gemini 1.5 Pro](https://blog.google/technology/ai/google-gemini-next-generation-model-february-2024/):** High-performance, efficient model.
  * **Gemini Nano:** Ultra-small model for edge deployments.
  * **Gemma:** Open-sourced models (2B and 7B parameters)
* Key considerations for integration: user experience, performance, and cost trade-offs.
* **Available Options:**
  * **Gemini 1.5 Flash:** Fast, 1 million token context window.
  * **Gemini 1.5 Pro:** 2 million token context window
* **Gemini Flash for Code:**
  * Performs well for code generation and structured outputs like JSON out-of-the-box.
  * Fine-tuning and using code examples in the context window further enhance results.
  * Applicable to code generation, translation, debugging, code review, etc.





## Understanding Context Windows

* **Importance of Context Window Size:**  
  * Historically limited to 2,000-8,000 tokens, hindering model capability.
  * Current models: GPT-4 Turbo (128,000+), Claude (2,000), Gemini (2 million).
* **Impact of Larger Context Windows:**
  * Can handle massive amounts of data (emails, texts, videos, codebases, research papers).
  * Reduces the need for fine-tuning, as more information can be provided at inference time.
  * Allows for more complex and nuanced outputs. 





## Fine-tuning vs. Prompting vs. Retrieval 

### Common Questions & Trade-offs

* Key decision points when working with large language models.
* **Considerations:**
  * **Prompt Design:**  Simple, cost-effective, but may require detailed prompts.
  * **Fine-Tuning:**  
    * Increasingly difficult to justify due to maintenance overhead and rapid release of new open-source models.
    * Recommended only when other options fail or for on-premise/local data requirements.
* **Recommendations:**
  * **Start with Closed-Source APIs:** Rapid iteration, prove product-market fit, focus on UX. 
  * **Hire ML Team When Necessary:** If highly specialized fine-tuning becomes essential. 

### Model Evaluation & Its Importance

* **Limitations of Academic Benchmarks:**
  * **Example:**  [HumanEval](https://huggingface.co/datasets/openai/openai_humaneval)
    * Often misinterpreted as involving human evaluation (it doesn't).
    * Tests a narrow scope of Python function completion with simplistic tasks.
    * Not representative of real-world software engineering or other programming languages.
* **[HumanEval X](https://huggingface.co/datasets/THUDM/humaneval-x):** Created to address some limitations of HumanEval, but still has limitations.
* **Key Takeaways:**
  * Carefully consider the relevance and limitations of evaluation metrics.
  * Prioritize custom evaluations tailored to your specific use case and business needs. 





## Prompting Strategies and Examples

### Power of Prompting & Video Understanding

* **Detailed Example:**
  * Using Gemini in AI Studio to analyze a 44-minute video. 
  * Asking the model to find a specific event (paper removed from a pocket), identify information on the paper, and provide the timestamp.
  * Demonstrates the ability to understand and extract information from lengthy video content, potentially revolutionizing video analysis workflows.
* **Implications:**
  * Transforms how we interact with video content, making it searchable and analyzable at scale.
  * Also applicable to large text documents (PDFs with images, graphs, code) for summarization, analysis, and research. 

* **Prefix Caching:** 
  * Optimizes API calls for repeated analysis of the same codebase or repository.
  * Improves latency and grounds responses within a consistent context.


### AI Studio Overview & Examples

* **Key Features:**
  * Adjust stop sequences, top-k configurations, and temperature.
  * Toggle between Gemini models (Pro, Flash, etc.).
  * Access prompt gallery, cookbook, and getting started resources.
  * View past prompts and outputs. 
* **Examples:**
  * Scraping GitHub issues and Stack Overflow questions for analysis.
  * Converting COBOL code to Java with specific instructions and architecture preferences.
* **Key Takeaway:**  With detailed instructions, models can achieve impressive results, much like a skilled contractor team. 



## Retrieval Augmented Generation

### Retrieval in Google Products

* **[gemini.google.com](https://gemini.google.com/app) (formerly Bard):**
  * Example:  Querying for information about the San Francisco Ferry Building and requesting recommendations. 
  * Results are grounded in Google Search, with an option to view source citations and confidence levels.
* **Personalized Retrieval:**  The concept can be extended to internal corporate data and codebases.



## Fine-tuning Considerations and Gemma

* **Fine-Tuning:**
  * Should be approached with caution and a clear understanding of the maintenance commitment. 
  * Consider the rapid evolution of open-source models. 
* **Gemma Family:**
  * Solid starting point for open-source fine-tuning.
  * Available in 2B and 7B parameter sizes, with both instruction-tuned and non-instruction-tuned variants.
  * **[CodeGemma](https://huggingface.co/blog/codegemma):**  For code-related tasks.
  * **RecurrentGemma:**  For sequential data.
    * **Paper:** [RecurrentGemma: Moving Past Transformers for Efficient Open Language Models](https://arxiv.org/abs/2404.07839)
    * **HuggingFace Hub:** [google/recurrentgemma-2b-it](google/recurrentgemma-2b-it)
  * **PaliGemma:** Open-vision language model.
    * **Paper:** [PaliGemma: A versatile 3B VLM for transfer](https://arxiv.org/abs/2407.07726)
    * **Blog Post:** [PaliGemma – Google's Cutting-Edge Open Vision Language Model](https://huggingface.co/blog/paligemma) 
* **Resources:**
  * [Model Garden on Vertex AI](https://cloud.google.com/model-garden?hl=en)
  * [HuggingFace Hub](https://huggingface.co/google)
* **Deployment:** Easy one-click deployment to Google Cloud.
* **Model Builders:**  Provides automatic comparisons and prompt management. 



