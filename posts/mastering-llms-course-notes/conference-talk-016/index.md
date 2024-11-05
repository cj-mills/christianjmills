---
title: "Conference Talk 16: A Deep Dive on LLM Evaluation"
date: 2024-8-29
image: /images/empty.gif
hide: false
search_exclude: false
categories: [notes, llms]
description: "In this talk, **Hailey Schoelkopf** from **Eleuther AI** provides an overview of the challenges in LLM evaluation, exploring different measurement techniques, highlighting reproducibility issues, and advocating for best practices like sharing evaluation code and using task-specific downstream evaluations."

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



::: {.callout-tip title="Presentation Slides"}

* **Slides:** [A Deep Dive on LM Evaluation](https://docs.google.com/presentation/d/1qTaDYqLCgxkUaTfxQkN1it4tx6_jixwv9ZtsbqQgE4U/)

:::





## Introduction

* **Speaker:** Hailey Schoelkopf, Research Scientist at Eleuther AI
* **Topic:** Deep dive into the challenges and best practices of Large Language Model (LLM) evaluation.

### About the Speaker

* **Hailey Schoelkopf:** 
  * Research Scientist at Eleuther AI.
  * Maintainer of the **LM Evaluation Harness**, a widely used open-source library for evaluating LLMs.

### About Eleuther AI: 

* **Website:** [https://eleuther.ai/](https://eleuther.ai/)
* **Project Page:** [Evaluating LLMs](https://www.eleuther.ai/projects/large-language-model-evaluation)
* **Non-profit research lab** known for:
  * Releasing open-source LLMs like GPT-J and GPT-NeoX-20B.
  * Research on: 
    * Model Interpretability 
    * Datasets
    * Distributed Training
    * LLM Evaluation
  * Building and maintaining tools for the open-source AI ecosystem.

### LM Evaluation Harness

* **GitHub Repository:** [https://github.com/EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
* **Purpose:**
  * Originally created to reproduce and track the evaluations from the GPT-3 paper.
  * Evolved into a comprehensive library for evaluating LLMs.
* **Usage:**
  *  Widely used by researchers and practitioners.
  *  Powers the backend for the OpenLLM Leaderboard.



## Challenges in LLM Evaluation

#### 1. Scoring Difficulties

* **Core Issue:** Reliably evaluating the correctness of LLM responses in natural language.
* **Challenges:**
  * **Subjectivity of Language:** What constitutes a "correct" response can be subjective and context-dependent.
  * **Hallucination:**  LLMs can generate plausible-sounding but incorrect information, making it challenging to determine accuracy based on surface-level analysis.
  * **Lack of Standardized Metrics:** Absence of universally agreed-upon metrics for evaluating LLM performance across different tasks and domains. 

#### 2. Reproducibility Issues

* **Importance:**  Ensuring that evaluation results are consistent and replicable.
* **Challenges:**
  * **Sensitivity to Implementation Details:**  LLM performance can vary significantly based on seemingly minor differences in implementation, such as tokenization, prompt formatting, and hyperparameters.
  * **Lack of Transparency:**  Limited sharing of evaluation code and detailed methodologies makes it difficult for others to reproduce results.
  * **Data Set Variability:**  Differences in data set composition and quality can lead to inconsistent evaluations.





## Common LLM Evaluation Methods

#### 1. Log Likelihoods: Assessing the Probability of Expected Outputs

* **Background:**
  * LLMs output a probability distribution over vocabulary for each possible next token.
  * This distribution represents the model's confidence in different words following the given input.
  
* **How It Works:**
  * **Input:** A prompt (X) and a potential output (Y). 
  
  * **Process:** Calculate the probability of the model generating Y given X. This involves summing the log probabilities of each token in Y, conditioned on the preceding tokens in X and Y.
  
    * $$
      \log P(y|x) = \sum_{i=0}^{m-1} \log p(y_i | x, y_0, \ldots, y_{i-1}) = \sum_{i=0}^{m-1} l(n+i, y_{i})
      $$
  
      
    
    * where $\log p(y_i | x, y_0, \ldots, y_{i-1})$ is the log probability of the $i$-th target token conditioned on the full input $x$ and the preceding target tokens. (and where $x, y_0, \ldots, y_{i-1}$ denotes conditioning on only $x$.).
    
  * **Example:**  For the prompt "The cow jumped over the," calculate the probability of the model generating "moon" versus other words.
  
* **Use Case: Multiple Choice Question Answering**
  
  * **Advantages:** 
    * Computationally cheaper than generation-based evaluation.
    * Avoids issues with parsing errors in generated text.
    * Suitable for evaluating smaller LLMs or those in early training stages.
  * **Disadvantages:**
    * Limited real-world applicability compared to open-ended generation.
    * Doesn't assess a model's ability to formulate its own answers.
    * Cannot evaluate chain-of-thought reasoning.
  
* **Challenges with Log Likelihoods and Perplexity:**
  * **Tokenizer Sensitivity:**  Metrics are affected by the specific tokenizer used, making comparisons between models with different tokenizers difficult.
    * **Solution:** Implement normalization techniques to account for tokenizer variations.
  * **Limited Information:**  Log likelihoods only consider the probability of a given output, not its overall quality, coherence, or factual accuracy.


#### 2.  Perplexity: Measuring How Well a Model Fits a Data Distribution

* **Concept:** Quantifies how well a language model predicts a given text, indicating its familiarity with the data distribution.
* **Calculation:** Based on the average per-token log probability of the text, with lower perplexity indicating a better fit to the data. 

  * $$
    \text{PPL} = \exp \left( -\frac{1}{\sum_{j=1}^{|D|} N_j} \sum_{j=1}^{|D|} \sum_{i=1}^{N_j} \log P(y_{ji} | y_{j1}, \ldots, y_{ji-1}) \right)
    $$

    

* **Use Case:** Evaluating a model's understanding of a specific text corpus (e.g., Wikipedia).
* **Limitations:** 
  *  **Domain Specificity:** Perplexity on one dataset (e.g., Wikipedia) may not generalize to other domains or tasks. 
  *  **Limited Insight into Downstream Performance:** A low perplexity doesn't guarantee good performance in real-world applications like chatbots or question answering. 

#### 3.  Text Generation: Evaluating Real-World Output but Facing Scoring Challenges

* **Importance:** Crucial for assessing LLMs in tasks involving text generation (e.g., chatbots, story writing).
* **Challenges:**
  * **Scoring Free-Form Text:** Determining the correctness and quality of generated text is difficult.
    *  Simple heuristics (e.g., keyword matching) are unreliable and prone to gaming.
    *  Human evaluation is expensive and time-consuming.
    *  LLM-based judges introduce their own biases and limitations.
  * **Sensitivity to Prompt Details:** Minor variations in prompts (e.g., trailing whitespace) can drastically impact results, hindering reproducibility.
    * **Example:**  In code generation, a trailing tab in the prompt can create syntax errors for models that generate code with specific formatting, leading to artificially lower performance scores.



## The Need for Reproducibility and Best Practices

* **Reproducibility is Crucial:**  Ensuring that evaluation results can be independently verified is essential for:
  *  **Fair Model Comparisons:** Accurately assessing the relative performance of different LLMs.
  *  **Meaningful Progress Tracking:** Tracking improvements in model development and evaluation methods.
* **Challenges to Reproducibility:**
  *  Lack of standardized evaluation practices and metrics.
  *  Incomplete reporting of evaluation details (e.g., prompts, code, evaluation settings). 
* **Best Practices for Reproducible LLM Evaluation:**
  * **Share Evaluation Code:** Publicly release code used for evaluation to allow for scrutiny and replication of results.
  * **Detailed Reporting:** Provide comprehensive information about evaluation procedures, including:
    * Specific prompts and instructions given to models.
    * Data preprocessing steps and evaluation datasets used. 
    * Evaluation metrics and their calculation.
  * **Use Standardized Evaluation Frameworks:** Leverage libraries like the `lm-evaluation-harness` or other tools (Helm, OpenCompass) to promote consistency and reduce implementation discrepancies.
    * **`lm-evaluation-harness`:** [GitHub Repository](https://github.com/EleutherAI/lm-evaluation-harness)
    * **`helm`:** [GitHub Repository](https://github.com/stanford-crfm/helm)
    * **`opencompass`:** [GitHub Repository](https://github.com/open-compass/opencompass)
* **Distinction Between Model Evals and Downstream Evals:**
  * **Model Evals (e.g., MMLU benchmark):** Measure general language understanding and capabilities across diverse tasks.
  * **Downstream Evals:** Focus on performance in a specific application or domain (e.g., chatbot for customer support).
  * Prioritize Downstream Evals whenever possible for production settings, as they directly reflect real-world performance needs. 
  * **OpenLLM Leaderboard:** [https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard) 
  * **MMLU Benchmark:** [https://paperswithcode.com/dataset/mmlu](https://paperswithcode.com/dataset/mmlu)
  * **HellaSwag Benchmark:** [https://paperswithcode.com/dataset/hellaswag](https://paperswithcode.com/dataset/hellaswag)
  * **ARC Benchmark:** [https://paperswithcode.com/dataset/arc](https://paperswithcode.com/dataset/arc)
    * ARC focuses on generalization and multi-step reasoning, making it more challenging than benchmarks that rely heavily on memorization.





## Conclusion 

* **Implementation Details Matter:** LLMs are highly sensitive to minor variations in evaluation procedures.
* **Transparency and Standardization are Key:** Sharing code, detailed reporting, and using standardized frameworks are crucial for reproducible LLM evaluation.
* **Prioritize Downstream Evaluations:**  Focus on evaluations that directly measure performance in your specific application context. 
* **For Further Exploration:** 
  * **Paper:** [*Lessons from the Trenches on Reproducible Evaluation of Language Models*](https://arxiv.org/abs/2405.14782) (Eleuther AI)
  * **Library:** [`lm-evaluation-harness`](https://github.com/EleutherAI/lm-evaluation-harness)  (Eleuther AI)




###  Q&A Highlights: 

*  **Dataset Quality:**  Errors or biases in benchmark datasets can significantly affect evaluation results and limit the usefulness of benchmarks. 
*  **Overfitting to Evaluations:** Repeatedly optimizing for a specific benchmark can lead to overfitting, where models excel on the benchmark but fail to generalize to other tasks or data.
*  **Measurement Validity:** It's essential to ensure that evaluation metrics accurately measure the desired aspects of LLM performance (e.g., factual accuracy, reasoning, coherence).
* **LLMs as Judges:**
  * **Benefits:** LLMs can potentially automate the evaluation of tasks requiring nuanced understanding and reasoning, which are difficult to assess with simple heuristics.
  * **Considerations:** 
    *  **Judge Model Selection:** Carefully choose an LLM judge that possesses the necessary capabilities for the task being evaluated. 
    *  **Judge Model Limitations:** Be aware of the judge model's own biases and limitations, as these can influence the evaluation outcomes.
* **Reliable Multiple-Choice Answers Without Additional Text:**
  * **Structured Generation:** Use techniques that constrain the model's output to specific formats. 
  * **System Prompts:** Provide clear instructions to the model to only output the answer. 
  * **Log Likelihoods:** Rely on log likelihood-based multiple-choice evaluations if structured generation isn't possible.




{{< include /_about-author-cta.qmd >}}
