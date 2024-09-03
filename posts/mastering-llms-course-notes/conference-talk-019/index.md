---
title: "Conference Talk 19: Fine Tuning LLMs for Function Calling"
date: 2024-8-30
image: /images/empty.gif
hide: false
search_exclude: false
categories: [notes, llms]
description: "In this talk, **Pawell Garbacki** from **Fireworks.ai** covers the process and best practices of finetuning an LLM for function/tool use."

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





### Introduction

* **Pawell**, from **[Fireworks AI](https://fireworks.ai/)**, discusses fine-tuning LLMs for function calling, covering key decisions, challenges, and solutions.
* **Documentation:** [Using function-calling](https://docs.fireworks.ai/guides/function-calling)
* **Documentation:** [Fine-tuning models](https://docs.fireworks.ai/fine-tuning/fine-tuning-models)
* **Documentation:** [Using grammar mode](https://docs.fireworks.ai/structured-responses/structured-output-grammar-based)



### Understanding Function/Tool Calling

* **Definition:** Giving LLMs the ability to interact with the external world.
* **Use Cases:**
  * **Accessing real-time or unavailable information:** E.g., retrieving current stock prices.
  * **Orchestrating multi-agent systems:** LLMs can access and utilize multiple tools to assist users.



### Key Decisions in Fine-Tuning for Function Calling

#### 1.  Objective Selection:

* **Impact:** The objective significantly impacts data preparation, training data volume, fine-tuning complexity, and model usage.
* **Recommendation:** Choose the simplest objective that meets the use case requirements.
* **Common Objectives:**
  * **Single-Turn Forced Call (Routing Use Case):**
    * User provides a single instruction.
    * Model maps the instruction to one of several pre-defined functions and its parameters.
    * **Forced Call:** The model is constrained to respond with a function call, not natural language.
    * **Example:** User requests the current stock price of Nvidia, the model identifies the appropriate function and its parameters.
      * ```text
        User: What is the stock price of Nvidia?
        
        Assistant: {
          "name": "get_stock_price",
          "arguments": {"ticker": "NVDA"}
        }
        ```
  * **Parallel Function Calling:**
    * Similar to single-turn forced call, but the model can call multiple independent functions in parallel.
    * **Example:** Retrieving stock prices for multiple companies simultaneously.
      * ```text
        User: What is the stock price of Nvidia and Apple?
        
        Assistant: [
          {
            "name": "get_stock_price",
            "arguments": {"ticker": "NVDA"}
          },
          {
            "name": "get_stock_price",
            "arguments": {"ticker": "AAPL"}
          }
        ]
        ```
  * **Nested Function Calls:**
    * Model calls functions sequentially, with the output of one function feeding into the next.
    * **Example:** Retrieving stock prices and then using those prices to generate a plot.
      * ```text
        User: Plot the stock price of Nvidia and Apple over the last two weeks
        
        Assistant: [
          {
            "name": "get_stock_price",
            "arguments": {"ticker": "NVDA", "start_time": "2 weeks ago"}
          },
          {
            "name": "get_stock_price",
            "arguments": {"ticker": "AAPL", "start_time": "2 weeks ago"}
          }
        ]
        
        Tool: {"NVDA": [120, 121, …],"AAPL": [192, 190, …]}
        
        Assistant: {
          "name": "plot",
          "arguments": {"NVDA": [120, 121, …],"AAPL": [192, 190, …]}
        }
        ```
    * **Implementation:**
      * **User Role:** Client interacting with the model.
      * **Assistant Role:** The LLM generating function calls.
      * **Tool Role:** Client-side component that executes function calls and returns results to the model. 
  * **Multi-Turn Chat with Optional Function Calling:**
    * Most complex objective, combining natural language conversation with optional function calls.
    * **Example:**  User asks for news, model fetches trending news, summarizes them, and engages in further conversation.
      * ```text
        User: What's in the news today?
        
        Assistant: {"name": "trending_news"}
        
        Tool: {
          "headlines": [
            "Nvidia market cap surpasses Apple", …
          ]
        }
        
        Assistant: Nvidia is now more valuable than Apple
        
        User: What is Nvidia stock price?
        
        Assistant: {
          "name": "get_stock_price",
          "arguments": {"ticker": "NVDA"}
        }
        ```

#### 2. Function Call Token:

* **Purpose:** 
  * Indicate to the client when the model is switching to function call mode.
  * Enable efficient parsing of model responses, especially in mixed natural language and function call outputs.
* **Implementation:** Introduce a special token to prefix function calls.
  * ```text
    Assistant: <function_call_token>{
      "name": "get_stock_price",
      "arguments": {"ticker": "NVDA"}
    }
    ```
* **Benefits:**
  * Easier parsing of model responses.
  * Improved **streaming generation** by enabling the client to wait for the entire function call signature before processing.
  * Facilitates **constraint generation**, ensuring the model adheres to predefined function schemas.

#### 3. Syntax for Function Calling:

* **Options:**
  * **Python Syntax:** Generate function calls using Python function call signature syntax.
  * **JSON Schema:** Generate JSON structures describing the function name and parameters.
* **Trade-offs:**
  * **Python Syntax:**
    * **Advantages:** Easier for LLMs to generate due to extensive training on Python code.
    * **Disadvantages:** Less natural for representing complex, nested parameter structures within a single-line invocation.
  * **JSON Schema:**
    * **Advantages:** Better suited for complex, nested parameter types; easier to enforce schema with constraint generation; compatible with OpenAI APIs.
    * **Disadvantages:** Potentially more challenging for LLMs to generate compared to Python syntax.

#### 4. Preserving Existing Model Capabilities:

* **Challenge:** Fine-tuning for function calling can inadvertently degrade pre-existing instruction following and general language capabilities.
* **Recommendations:**
  * **Fine-tune on Instruction-Tuned Models:** Use the "Instruct" version of the base model instead of the "Base" version when mixing general chat with function calling.
    * Using the "Base" version is fine for forced function calling.
  * **Reduce Training Data:** Minimize the amount of training data to reduce the risk of overwriting existing capabilities.
  * **High-Quality Data:** Use a smaller volume of carefully curated, high-quality training data.

#### 5. Full-Weight Tuning vs. LoRA Tuning:

* **Recommendation:** LoRA tuning is generally sufficient and preferable for function calling, particularly in low-data regimes.
* **Advantages of LoRA:**
  * Fewer parameters to converge, leading to faster training and better results with limited data.
  * Faster iteration cycles, enabling more experimentation.
  * Lower hosting and experimentation costs, especially with efficient LoRA serving solutions like Fireworks AI's platform.

#### 6. Constraint Generation:

* **Purpose:** Reduce hallucinations in model-generated function calls by leveraging the known schema of available functions.
* **Implementation:**
  * Provide the model with the schema of the functions (e.g., function name, parameter names and types).
  * Use a constraint generation mechanism (like a [context-free grammar](https://web.stanford.edu/class/archive/cs/cs103/cs103.1164/lectures/18/Small18.pdf)) to guide the model's output and enforce adherence to the schema.
* **Benefits:**
  * **Reduced Hallucinations:**  Significantly minimizes or even eliminates hallucinations in function call outputs.
  * **Faster Generation:** Enables short-circuiting generation by autocompleting predictable tokens based on the grammar, improving inference speed. 
* **Fireworks AI:** Offers constraint generation support for function calling, requiring users to provide the function schemas.



### General Recommendations and Considerations

* **Work Smart:** Utilize existing open-source function-calling models whenever possible, as they are often sufficient for many use cases.
* **Fine-Tuning Effort:** Be prepared for an iterative and potentially time-consuming process when fine-tuning for complex function-calling objectives.

### Fireworks AI's Fire Function Models

* **Playground:** [Firefunction V2](https://fireworks.ai/models/fireworks/firefunction-v2)
* **Blog Post:** [Firefunction-v2: Function calling capability on par with GPT4o at 2.5x the speed and 10% of the cost](https://fireworks.ai/blog/firefunction-v2-launch-post)
* **Fire Function V2:**
  * Based on LLaMa 3 70B (Instruct variant).
  * Outperforms GPT-4 on the Gorilla benchmark. 
  * Designed to approximate GPT-4's conversational capabilities mixed with function calling.
  * Addresses limitations of existing datasets by leveraging:
    * Naturally occurring function-calling conversations.
    * Open-source multi-agent system data (e.g., AutoGPT).
    * Synthetic datasets with complex instructions and system prompts. 
* **Benchmark Comparison:**

  |                                    | Firefunction v2 | Gpt-4o   |
  | ---------------------------------- | --------------- | -------- |
  | Gorilla simple                     | **0.94**        | 0.88     |
  | Gorilla multiple_function          | 0.91            | 0.91     |
  | Gorilla parallel_function          | 0.89            | 0.89     |
  | Gorilla parallel_multiple_function | 0.79            | 0.72     |
  | Nexus parallel                     | 0.53            | 0.47     |
  | Mtbench                            | 0.84            | **0.93** |



### Challenges in Fine-tuning for Function Calling

- **Data Scarcity:** Unlike general language modeling, readily available datasets for function calling are limited.
  - Existing datasets often focus on specific use cases (e.g., GPT-4 conversations or a limited number of functions).
  - **Solution:** Invest in building custom datasets.
- **Data Set Design:** 
  - **Define Data Categories:** Consider types of function calls (parallel, nested), number of turns, and number of functions supported.
    - **Parallel function calling:** Multiple functions are called simultaneously.
    - **Nested function calling:**  Functions are called within other functions.
    - **Turn-based conversations:** Single-turn or multi-turn interactions (e.g., exceeding 10 turns).
    - **Number of functions supported:**  Fine-tuning for a small set of functions (e.g., 5) is different from tuning for a larger set (e.g., 50).
  - **Objective Alignment:** Ensure the dataset represents the model's intended use cases and boundary conditions.
  - **Leverage Existing Resources:** Explore open-source datasets (e.g., Glaive) and multi-agent systems (e.g., Autogen) for inspiration and data.
    - [**Autogen**](https://github.com/microsoft/autogen), a multi-agent system, can be a good data source, especially for scenarios with multiple agents and complex prompts.
- **Complex System Prompts:** Real-world applications often require intricate instructions for function selection, which are difficult to find in existing datasets.
  - **Solution:** Invest in generating synthetic datasets with complex instructions. 
- **Security Concerns:** Allowing arbitrary function calls raises security risks, especially with functions that modify data.
  - **Mitigation:**
    - Focus on read-only functions.
    - Include precise instructions in system prompts. 
  - **Ongoing Research:** This area requires further exploration as function calling and multi-agent systems become more prevalent.



### Prompt Templates for Fine-tuning

- **System prompts** provide context and instructions to the model.
- **General Guidelines:**
  - **Preserve Instruct Model Capabilities:** When fine-tuning on top of existing instruct models, maintain the prompt format to retain existing capabilities.
  - **Clear Role Prefixes:** Use distinct prefixes for different roles (e.g., system, user, assistant, tool) in multi-turn conversations.
- **Message Format:**
  - **Parsability:** Ensure the format allows easy parsing of function calls by the client.
  - **Mixed Output Handling:** Use special tokens to delineate between natural language and function call sections in assistant responses.



### Successful Fine-tuning Examples

- **GPT-4 Limitations:** Fine-tuning can overcome limitations in existing models, such as character limits in function descriptions.
- **Complex Instructions:** Fine-tuning is particularly effective for scenarios with complex instructions on when to call specific functions, even with relatively simple functions.



### Function Calling Data Sets and Evaluation

- **Datasets:**
  - [**Glaive**](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2)**:** High-quality but limited coverage of use cases.
  - [**Gorilla**](https://gorilla.cs.berkeley.edu/blogs/7_open_functions_v2.html)**:** Simple functions, Python syntax focus.
  - [**Nexus Raven**](https://github.com/nexusflowai/NexusRaven-V2)**:** More complex parameters, Python focus.
- **Evaluation:**
  - **[Gorilla Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html):** Useful for initial assessment, but consider its limitations (e.g., focus on Python syntax).
  - **[Nexus Benchmarks](https://huggingface.co/Nexusflow):** More challenging than Gorilla.
  - **Empty Bench:**  Evaluates general instruction following without function calling.
  - **Evaluation Challenges:**
    * **Real-World Use Case Mismatch:** Benchmarks may not fully capture the complexities of real-world scenarios, such as those requiring precise system prompting and multi-turn conversations.
- **Recommendations:**
  * **Benchmark Selection:** Start with publicly available benchmarks to get a general idea of the model's capabilities.
  * **Real-World Testing:**  It's essential to test and evaluate models on the specific use cases they are intended for.
  * **Model Selection:**  Don't rely solely on benchmark scores; try out the top-performing models on your own data and use case to determine the best fit.



### Base Models for Fine-tuning

- **Llama 3 & Llama 3.1:** Strong general-purpose models.
  - **FireFunction V1:** Based on Mistral.
  - **FireFunction V2:** Based on Llama, showed significant improvement.
- **Coding Models:** Consider coding-focused models (e.g., Llama 2 Python code generation model) for single-turn, Python-based function calling.
- **[Qwen Models](https://huggingface.co/Qwen):**  Show promise but require further exploration.
- **Phi (Microsoft):** Smaller models that perform well for their size and can potentially run without a GPU.
- **Model Selection:**  Consider the specific objective (e.g., forced function calling, Python syntax) when choosing a base model. 



### Memory Retention in Long Chains of Calls

- **Longer Context Models:** Opt for models with larger context windows (e.g., beyond Llama 3's 8K context) for extended conversations.
  - [Llama 3.1](https://llama.meta.com/) has 128K content window
- **Dataset Representation:** Include sufficient long conversation examples in the training data.
- **Intelligent Conversation Pruning:**
  - Develop algorithms to selectively retain the most relevant messages from previous turns when the conversation exceeds the context window.
  - Explore semantic matching techniques to identify relevant past messages.
- **Whiteboard Approach:**
  - Have the model summarize the key aspects of the conversation at the end of each turn.
  - Pass only the summary to the model in subsequent turns, effectively resetting the context while retaining essential information.



### Multi-Agent Systems and Function Calling

* **Function as Agent:** A function can be considered an agent within a multi-agent system, interacting with other agents (potentially other functions or models) to complete tasks.
* **Orchestration:**  Multi-agent frameworks like Autogen provide tools for defining agents, extracting function schemas, routing messages, and executing function calls based on model responses.
* **Agent Team Creation:**
  * **Identify Strengths and Weaknesses:** Analyze individual models to understand their capabilities and limitations.
  * **Define Agent Roles:** Assign roles and responsibilities to each agent based on their strengths. 
  * **Routing Layer:** Design a system for efficiently routing messages and tasks to the appropriate agents.
  * **Context Management:** Implement mechanisms for sharing and summarizing context between agents, especially in long conversations. 
* **Merging Models:** Explore techniques like **[MergeKit](https://github.com/arcee-ai/mergekit)** to combine layers from multiple models, potentially creating more capable composite models.

- **Cost and Latency Optimization:**  Consider using smaller, specialized models for specific tasks to reduce cost and latency.



### Comparison with Gorilla Project

- **Gorilla:**
  - Focuses on single-turn, forced function calling with Python signature generation.
  - Supports various function calling scenarios (single, parallel, nested).
  - Primarily designed for functions with simple parameters.
- **FireFunction:** 
  - Addresses real-world use cases involving complex system prompts and mixed conversations with function calling.
  - Handles functions with more complex parameters and instructions.
- **Benchmarks:** Gorilla leaderboard lacks tasks for complex system prompts and mixed conversation scenarios.



### Smallest Model for Local Smart Home Assistant

- **Challenges:**  Running a model locally with hundreds of functions on a resource-constrained device.
- **Potential Solutions:**
  - **Pre-populate KV Cache:** Pre-load function definitions into the model's KV cache to reduce inference time.
  - **Function Retrieval with RAG:** Use retrieval augmented generation (RAG) to dynamically select relevant functions based on user input, reducing the number of functions in the prompt.
  - **Smaller Models:** Explore smaller models like Phi or Qwen2 (2 billion parameters) that can potentially run without a GPU.



### Function Calling with GraphQL

- **GraphQL as Structured Data:** GraphQL can be treated as a structured data format similar to function call schemas.
- **Leveraging Function Calling Models:**  Explore using existing function calling models to generate or complete GraphQL queries by defining GraphQL operations as functions.
- **[Grammar Mode](https://docs.fireworks.ai/structured-responses/structured-output-grammar-based):**  Leverage the grammar enforcement capabilities of function calling models to ensure syntactically correct GraphQL queries.



### Handling API Changes

* **Canonical Data Format:** Store data in a format that can be easily translated to different API syntaxes.
* **Client-Side Translation:**  Implement a wrapper around the API to handle syntax conversions, allowing the model to remain agnostic to specific API changes.
* **Prompt-Based Function Definitions:**  Consider defining functions within the prompt itself.  This approach allows for easier updates when APIs change, eliminating the need for retraining.



### Synthetic Data Generation Best Practices

- **High-Quality Prompt and Seed Data:** Start with well-crafted prompts and a small, high-quality seed dataset.
- **Good Generation Model:** Utilize a capable language model for generation, balancing the legal constraints of using closed-source models with the effort required for filtering outputs from open-source models.
- **Data Variety over Quantity:** Prioritize diverse use cases and scenarios over a large number of examples for a single case.
- **Few-Shot Examples in Prompts:** Include examples of desired outputs in the prompts to guide the generation process.
- **Temperature Variation:** Experiment with different temperature settings to encourage creativity and diversity in generated samples.
- **Post-Filtering:** Implement filtering mechanisms to remove low-quality or incorrect samples.
- **DPO Alignment (Optional):** Use DPO to refine the model's behavior, especially for complex system prompts, by providing examples of both desired and undesired outputs.



### Importance of Data vs. Hyperparameters vs. Base Model

- **Data Quality:** As models become more intelligent and training data becomes smaller, the quality of the data becomes increasingly crucial.
- **Hyperparameter Sensitivity:** Smaller datasets often lead to increased sensitivity to hyperparameters, requiring careful tuning.
- **Base Model:** The choice of base model significantly impacts performance, especially for specialized tasks like Python code generation.
