---
title: "Office Hours 5: LangChain/LangSmith"
date: 2024-7-6
image: /images/empty.gif
hide: false
search_exclude: false
categories: [notes, llms]
description: "This Q&A session on LangChain/LangSmith covers topics like product differentiation, features, use cases, agent workflows, data set creation, and full-stack development for ML engineers."

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





## LangSmith's Position in the Observability Market

* **Question:** How does LangSmith differentiate itself from other observability tools in the market?
* **Answer:**
  * **LLM Application Focus:**  LangSmith is specifically designed for LLM applications, offering specialized features like message and document visualization for debugging.
    * **Guide:** [Log retriever traces](https://docs.smith.langchain.com/how_to_guides/tracing/log_retriever_trace)
  * **Chains of LLM Calls:**  It emphasizes visualizing and analyzing entire chains of LLM calls and retrieval steps, which is crucial for complex applications.
    * **Tutorial:** [Add observability to your LLM application](https://docs.smith.langchain.com/tutorials/Developers/observability)
  * **Human-in-the-Loop Features:** LangSmith prioritizes human interaction with features like:
    * Data visualization
    * Annotation queues for collaboration with subject matter experts
    * Side-by-side comparisons for evaluating improvements
    * Alignment of evaluators with human preferences
    * **How-to guides:** [Human feedback](https://docs.smith.langchain.com/how_to_guides/human_feedback)
  * **Pairwise Evaluation:**  LangSmith enables pairwise evaluation of models, leading to more stable results.
    * **Guide:** [Run pairwise evaluations](https://docs.smith.langchain.com/how_to_guides/evaluation/evaluate_pairwise)
  * **Strong Support and Openness:** LangSmith is praised for its excellent support, responsive team, and open APIs that allow integration with other tools. 



## LangSmith's Support for Human Annotation and Action Items

* **Question:**  What support does LangSmith offer for human annotation, annotation queues, and taking action on user feedback? 
* **Answer:**
  * **Annotation Queues:**  
    * Data can be sent to annotation queues programmatically (e.g., based on user thumbs down) or manually.
    * Annotators can provide feedback, edit outputs, and add corrected examples to datasets.
    * **Guide:** [Use annotation queues](https://docs.smith.langchain.com/how_to_guides/human_feedback/annotation_queues)
  * **Datasets for Improvement:**  Corrected examples in datasets can be used for testing and future model improvement. 
    * **Concept:** [Evaluation](https://docs.smith.langchain.com/concepts/evaluation#datasets-and-examples)
  * **Few-Shot Learning:** LangSmith aims to be a platform for gathering few-shot example datasets, which can be used for personalization by pulling down the most similar examples during runtime.



## Understanding the LangChain "Lang" Namespace

* **Question:** What's the difference between Langchain, Langsmith, Langgraph, Langflow, and Langserve? 
* **Answer:**
  * **[LangChain](https://www.langchain.com/):** The foundational open-source package for building LLM apps, offering a runtime, abstractions, integrations, and off-the-shelf chains.
  * **[LangFlow](https://www.langflow.org/) (Not Langchain Company):** A low-code/no-code UI built on top of LangChain.
  * **[LangServe](https://python.langchain.com/v0.2/docs/langserve/):**  Simplifies deploying LangChain applications by exposing them as [FastAPI](https://fastapi.tiangolo.com/) endpoints.
  * **[LangGraph](https://langchain-ai.github.io/langgraph/):** An extension of LangChain specifically designed for building and managing highly controllable agent-based workflows.
  * **[LangSmith](https://www.langchain.com/langsmith):** A standalone observability and testing tool for LLM apps, usable with or without LangChain.



## When to Use LangChain vs. LangGraph

* **Question:** When would you choose LangChain, and when is LangGraph the better option?
* **Answer:**
  * **LangChain:** Ideal for beginners and for rapidly prototyping simple LLM applications with single LLM calls.
  * **LangGraph:**  Suited for advanced teams building complex, agentic workflows that require:
    * Cyclical agent execution
    * Fine-grained control
    * Built-in persistence
    * Streaming and background modes



## Popularity of TypeScript vs. Python in LLM Tools

* **Question:** How does the usage of TypeScript APIs compare to Python APIs in LangChain and related tools? 
* **Answer:**
  * **Python Dominates:** Python remains more popular overall, possibly due to:
    * A larger community focused on LLM application prototyping.
    * Stronger ecosystem for data engineering tasks related to retrieval.
  * **TypeScript for Generative UI:** TypeScript is gaining traction, especially for applications involving generative UI, which is more challenging to implement in Python. 



## Generative UI Explained

* **Question:** What is generative UI, and how does it work?
* **Answer:**
  * **Beyond Simple Chat:** Generative UI enables LLMs to return more than text; they send UI components to create richer interfaces. 
  * **Example:**  Instead of a list of weather data, an LLM might return a dynamic graph component with zoom and interaction capabilities.
  * **Vercel AI SDK Integration:**  LangChain now integrates with Vercel's AI SDK for easier development of generative UI experiences.
    * **[Vercel AI SDK](https://sdk.vercel.ai/docs/introduction):** TypeScript toolkit designed to help developers build AI-powered  applications with React, Next.js, Vue, Svelte, Node.js, and more.



## Defining "Agentic" in the Context of LLMs 

* **Question:** What does "agentic" mean in the context of LLMs, and is it a significant distinction?
* **Answer:**
  * **LLM in Control:** An agentic system is one where the LLM controls the application's control flow and decision-making process.
  * **More Than Function Calling:** While related to function calling, agentic systems go further by enabling LLMs to loop, adapt, and make dynamic decisions about the next steps.
  * **Implications for Development:** This distinction introduces new challenges and considerations in UX design, observability, and testing.



## LangChain/LangSmith Features for Agentic Workflows

* **Question:**  What features in LangChain/LangSmith specifically aid in developing and managing agentic workflows?
* **Answer:**
  * **LangGraph's Strengths:**
    * **Controllability:** LangGraph's low-level design provides a high degree of control, which is essential for managing complex agents.
    * **Persistence and Human-in-the-Loop:** Built-in persistence and easy access to execution history enable checkpointing, resuming from specific states, and human intervention when needed.
  * **LangSmith's Role:** While not agent-specific, LangSmith's observability features are particularly valuable for debugging and understanding complex, agentic applications. 



## Multiple LLM Collaboration in Practice

* **Question:** Is the idea of using multiple LLMs with different strengths in a single application realistic? 
* **Answer:**
  * **Planning and Execution:** A common pattern involves a powerful LLM (e.g., GPT-4) for high-level planning and decision-making, while specialized or more cost-effective models (e.g., specialized code generation models) handle specific tasks.



## Building Evaluation Sets with LangSmith

* **Question:** What's the most effective way to use LangSmith for creating evaluation sets?
* **Answer:**
  1. **Manual Seeding:** Begin with a small set (5-10) of manually crafted examples.
  2. **Production Feedback Loop:**  Integrate with production logs to capture real-user interactions and identify edge cases. 
  3. **Iterative Refinement:**  
     * Manually add challenging or interesting cases to the dataset.
     * Encourage user feedback and incorporate relevant examples.
     * Consider synthetic data generation to expand the dataset, but prioritize human review and labeling.
* **Concepts:** [Evaluation](https://docs.smith.langchain.com/concepts/evaluation)
* **How-to guides:** [Evaluation](https://docs.smith.langchain.com/how_to_guides/evaluation)
* **Tutorial:** [Evaluate your LLM application](https://docs.smith.langchain.com/tutorials/Developers/evaluation)



## Recommended Stack for Full-Stack ML Engineers

* **Question:**  What's a good technology stack for Python-centric ML engineers who want to build and ship full-stack applications? 
* **Answer:**
  * **Python-First Options:**
    * **[Streamlit](https://streamlit.io/)/[Gradio](https://www.gradio.app/):** Excellent for rapid prototyping and simpler applications.
    * **[LangChain Templates](https://python.langchain.com/v0.2/docs/templates/):** Explore and adapt existing LangChain repositories with Python backends.
    * **[LangServe](https://github.com/langchain-ai/langserve):** Easily deploy LangChain apps.
  * **Long-Term Goal:** Aim to become proficient in 2-3 languages (Python, JavaScript/TypeScript, SQL) for greater flexibility and control over the entire application stack.
  * **Tips:**
    * Leverage LLMs (like ChatGPT) to assist with JavaScript/TypeScript code generation and understanding.
    * Don't shy away from forking and modifying existing repositories to learn and adapt. 





{{< include /_about-author-cta.qmd >}}
