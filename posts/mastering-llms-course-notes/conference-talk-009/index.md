---
title: "Conference Talk 9: Why Fine-Tuning is Dead"
date: 2024-7-19
image: /images/empty.gif
hide: false
search_exclude: false
categories: [notes, llms]
description: "In this talk, **Emmanuel Ameisen** from Anthropic argues that fine-tuning LLMs is often less effective and efficient than focusing on fundamentals like data quality, prompting, and Retrieval Augmentation Generation (RAG)."

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





* [Trends in Machine Learning](#trends-in-machine-learning)
* [Performance Observations: Fine-tuning vs. RAG](#performance-observations-fine-tuning-vs.-rag)
* [The Moving Target: Fine-tuning and Frontier Models](#the-moving-target-fine-tuning-and-frontier-models)
* [The Difficulty of Fine-tuning: Prioritizing Fundamentals](#the-difficulty-of-fine-tuning-prioritizing-fundamentals)
* [Extrapolating Trends: Context Size, Price, and Latency](#extrapolating-trends-context-size-price-and-latency)
* [Conclusion](#conclusion)
* [Q&A Session](#qa-session)







## Trends in Machine Learning

* **Focus on fundamentals:** Throughout ML history, focusing on "boring" fundamentals like data quality and SQL queries often yielded better results than chasing the latest "cool" technology.
  * **2009:** Data analysis and SQL were more impactful than training ML models.
  * **2012-2014:** XGBoost outperformed deep learning for many tasks.
  * **2015:** Data cleaning and error correction were more effective than inventing new loss functions.
  * **2023:** Prompting and RAG often outperform fine-tuning.
* **Fine-tuning's uncertain future:** While fine-tuning became popular with the rise of pre-trained models like ResNet and BERT, LLMs might be shifting the paradigm again towards prompt-based and RAG-augmented approaches.



### Q&A on Trends and Embedding Models

* **Fine-tuning's future relevance:** Fine-tuning might still become more valuable in the future, just like deep learning did after an initial period of limited practical use. 
* **Embedding models:** While LLMs are currently the focus of improvement, embedding models might also see advancements that reduce the need for fine-tuning. However, domain-specific applications might still require fine-tuning or hybrid approaches like combining keyword search with embedding search.
* **Domain-specific ranking and retrieval:** RAG, combined with techniques like query expansion driven by LLMs, might be able to address the need for domain-specific ranking and retrieval without fine-tuning embedding models.
* **Benchmarking prompting vs. fine-tuning:**  Limited data exists comparing prompting and RAG to fine-tuning, but existing research suggests RAG often outperforms fine-tuning.



## Performance Observations: Fine-tuning vs. RAG

* **RAG often outperforms fine-tuning:**  Several studies indicate that RAG, even without fine-tuning, often achieves comparable or better performance than fine-tuning alone, especially for larger models and knowledge-based tasks.
  * **Forum Post:** [Fine-tuning vs Context-Injection (RAG)](https://community.openai.com/t/fine-tuning-vs-context-injection-rag/550286/1)

* **Prioritization over "versus":** While combining RAG and fine-tuning can yield incremental improvements, prioritizing RAG is crucial due to its efficiency and effectiveness.
  * **Paper:** [Fine Tuning vs. Retrieval Augmented Generation for Less Popular Knowledge](https://arxiv.org/abs/2403.01432)
  * **Paper:** [Fine-Tuning or Retrieval? Comparing Knowledge Injection in LLMs](https://arxiv.org/abs/2312.05934)

* **Fine-tuning for style, not knowledge:**  Fine-tuning might be less suitable for incorporating domain knowledge into a model compared to RAG, which directly provides relevant context.
  * **Paper:** [RAG vs Fine-tuning: Pipelines, Tradeoffs, and a Case Study on Agriculture](https://arxiv.org/abs/2401.08406)

* **The evolving definition of "knowledge":**  Distinguishing between "knowledge" and "style" in LLMs is complex and changes with each model generation. What previously required fine-tuning (e.g., writing style) might be achievable through prompting in newer models.



### Q&A on Fine-tuning, RAG, and Knowledge

* **Fine-tuning effectiveness and model size:** Fine-tuning might be more beneficial for smaller models compared to larger ones, which are already more capable of learning from context.
* **Domain-specific knowledge and RAG:** Even for smaller models, RAG remains crucial for tasks involving domain-specific knowledge.
* **Evaluating fine-tuning success:**  The choice of evaluation metric significantly impacts the perceived effectiveness of fine-tuning. For tasks like style adherence, fine-tuning might appear more beneficial than RAG.
* **The blurry line between style and content:**  The distinction between "style" and "content" can be ambiguous, making it difficult to definitively determine when fine-tuning is beneficial.



### Audience Questions and Examples of Fine-tuning Success

* **Complex knowledge bases and fine-tuning:** When dealing with large, curated knowledge bases, it's crucial to evaluate whether the next generation of LLMs, combined with RAG, might be sufficient without fine-tuning.
* **Adding knowledge via prompting and RAG:**  In many cases, adding knowledge to the model can be achieved through prompting, RAG, or a combination of both, eliminating the need for fine-tuning.
* **Fine-tuning for multilingual models:** Fine-tuning might be beneficial for improving the performance of multilingual models on languages with limited training data, as it leverages the model's existing understanding of language mapping.
* **Fine-tuning for code generation:** While fine-tuning can be used to adapt code generation models to specific styles and conventions, RAG remains highly effective for providing codebase context.
* **Contextual learning vs. fine-tuning:**  LLMs are demonstrating impressive abilities to learn from context, potentially replacing the need for fine-tuning in scenarios where sufficient context can be provided.



## The Moving Target: Fine-tuning and Frontier Models

* **Rapid LLM advancements:** The rapid pace of LLM development makes fine-tuning a moving target, as newer models often surpass the performance of previously fine-tuned models.
* **Bloomberg GPT example:** Bloomberg GPT, a large language model pre-trained on financial data, initially outperformed existing models on financial tasks. However, its performance was subsequently surpassed by newer models like GPT-4.
  * **Paper:** [BloombergGPT: A Large Language Model for Finance](https://arxiv.org/abs/2303.17564)
  * **Paper:** [Are ChatGPT and GPT-4 General-Purpose Solvers for Financial Text Analytics? A Study on Several Typical Tasks](https://arxiv.org/abs/2305.05862)

* **The cost of keeping up:** Continuously fine-tuning on new model releases can be prohibitively expensive, especially for large datasets. Prompt-based and RAG-based pipelines offer more flexibility and cost-effectiveness.
* **Fine-tuning effectiveness and model scale:** Fine-tuning might become less effective as models grow larger and more capable.



## The Difficulty of Fine-tuning: Prioritizing Fundamentals

* **The 80/20 rule of ML:**  Similar to traditional ML, most effort in LLM development should be dedicated to data work (80%), followed by engineering (18%), debugging (2%), and architecture research (0%).
* **Fine-tuning as a last resort:** Fine-tuning should only be considered after thoroughly addressing fundamentals like data quality, evaluation, prompting, and RAG.
* **Hierarchy of needs:** Prioritize building a solid ML system with robust evaluation, prompting, and RAG before attempting to fine-tune.
  * **Book:** [Building Machine Learning Powered Applications: Going from Idea to Product](https://www.mlpowered.com/book/)
  * **Continuous Integration**
    * Model Backtesting
    * Model Evaluation
    * Experimentation Framework
  * **Application Logic**
    - Input Validation → Filtering Logic → Model Code → Output Validation → Displaying Logic
  * **Monitoring**
    - Monitoring Input Distribution
    - Monitoring Latency
    - Monitoring Output Distribution



## Extrapolating Trends: Context Size, Price, and Latency

* **Decreasing costs and increasing capabilities:** LLM costs are rapidly decreasing, while context windows and processing speeds are increasing.
* **The impact of future trends:** If these trends continue, providing sufficient context to highly capable LLMs might become more efficient than fine-tuning for many use cases.
* **Context window limitations:** Fine-tuning might still be necessary for applications requiring context exceeding the limits of available LLMs. However, techniques like prefix caching could mitigate this need.



## Conclusion

- **Finetuning is:**
  - expensive and complex
  - has become less valuable
  - often underperforms simpler approaches

- **Models are continuously becoming:**
  - cheaper
  - smarter
  - faster
  - longer context

- **Always start with:**
  - prompting
  - making a train/test set
  - RAG

- **Treat finetuning as a niche/last resort solution**
  - like cloud vs on prem





## Q&A Session

### Question #1

* **Context window size and computational cost:**  Passing large amounts of context through an LLM for every request can be computationally expensive. However, increasing LLM efficiency and advancements like prefix caching could mitigate this cost.
* **Fine-tuning complexity:**  Fine-tuning increasingly complex and larger LLMs might become more challenging, potentially outweighing the benefits compared to context-based approaches.

### Question #2

* **Dynamic few-shot learning:**  Dynamically selecting and providing relevant few-shot examples from a database is a powerful technique for improving LLM performance without fine-tuning.
* **Iterative prompt and example improvement:**  Invest time in iteratively refining prompts and curating effective few-shot examples before considering fine-tuning.









{{< include /_about-author-cta.qmd >}}
