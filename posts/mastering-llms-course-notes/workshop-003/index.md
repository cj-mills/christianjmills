---
title: "Workshop 3: Instrumenting & Evaluating LLMs"
date: 2024-6-20
image: /images/empty.gif
hide: false
search_exclude: false
categories: [notes, llms]
description: "Workshop #3 focuses on the crucial role of evaluation in fine-tuning and improving LLMs. It covers three main types of evaluations: unit tests, LLM as a judge, and human evaluation."

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





* [Introduction](#introduction)
* [Types of Evaluations](#types-of-evaluations)
* [Looking At Your Data](#looking-at-your-data)
* [Harrison Chase: Langsmith for Logging & Tests](#harrison-chase-langsmith-for-logging-tests)
* [Bryan Bischof: Spellgrounds for Prodigious Prestidigitation](#bryan-bischof-spellgrounds-for-prodigious-prestidigitation)
* [Eugene Yan: Evaluating LLM-Generated Summaries with Out-of-Domain Fine-tuning](#eugene-yan-evaluating-llm-generated-summaries-with-out-of-domain-fine-tuning)
* [Shreya Shankar: Scaling Up Vibe Checks for LLMs](#shreya-shankar-scaling-up-vibe-checks-for-llms)
* [Q&A Session](#qa-session)



## Introduction

* **Importance of Evaluation:** Evaluation is crucial for iteratively improving LLMs, whether through prompt engineering, fine-tuning, or other methods.
* **Data Flywheel:** A fast iteration cycle requires rapid feedback from evaluations, enabling you to experiment and improve your AI quickly.
  * **Blog Post:** [Your AI Product Needs Evals](https://hamel.dev/blog/posts/evals/)

* **Applied AI:** Evaluation and data analysis are key components of applied AI, allowing you to measure progress and make informed decisions.



## Types of Evaluations

* **Unit Tests:** 
  * Code-based tests that validate specific expectations about LLM responses. 
  * Typically fast to run and can catch basic errors.

* **LLM as a Judge:** 
  * Using another LLM to evaluate the quality of the primary LLM's response. 
  * Can be efficient but requires careful alignment with human judgment.

* **Human Evaluation:** 
  * Direct human assessment of LLM output. 
  * Considered the gold standard, but can be expensive and time-consuming.


### Example: Editing Out Stereotypes In Academic Writing

* **Goal:** Automate the process of identifying and removing subconscious biases and stereotypes from text.
  * **Original Text:** "Norway's mining economy flourished during the period due to Norwegian's natural hardiness."
  * **Desired Edit:** Remove the stereotype of "Norwegian's natural hardiness."
* **Approach:** Leverages the experience of a team that manually reviews and edits manuscripts for biases, highlighting the importance of considering existing workflows when designing evaluations.

### Unit Tests

* **Purpose:** First line of defense against basic errors in LLM output.

* **Identifying Failure Modes:** Even seemingly complex LLM tasks often have predictable failure modes that can be tested with code.

* **Abstraction and Reusability:** Unit tests should be abstracted and reusable, both during development and in production.

* **Logging and Tracking:** Log unit test results to a database or other system for tracking progress and identifying trends.

* ::: {.callout-note title="Example:" collapse="true"}

  ```python
  from transformers import pipeline, Pipeline
  import pytest
  
  # Unit Tests
  @pytest.fixture(scope="module")
  def llm_pipeline():
      return pipeline("text-generation", model="meta-llama/Llama-2-7b-chat-hf", device=0)
  
  def verify_answer_contains(p: Pipeline, query: str, expected: str):
      result = p(query, do_sample=False, truncation=True, return_full_text=False)[0]["generated_text"]
      assert expected in result, f"The result does not contain '{expected}'"
  
  def test_google_ceo(llm_pipeline):
      verify_answer_contains(llm_pipeline, "Who is the CEO of Google?", "Sundar Pichai")
  
  def test_2_plus_3(llm_pipeline):
      verify_answer_contains(llm_pipeline, "What is 2+3?", "5")
  ```

  :::

#### Generate Data For Each Scenario

* **Feature and Scenario Breakdown:** Break down the LLM application into features and scenarios to systematically generate test data.
* **Example:** A real estate CRM application with features like finding listings, each with scenarios like finding one listing, multiple listings, or no listings.

#### Use LLMs to synthetically generate inputs to the system

* **Synthetic Data Generation:** Use LLMs to generate synthetic test data for various scenarios, especially when real user data is limited.

* **Example:** Generating synthetic real estate agent instructions for testing a CMA (comparative market analysis) feature.

  * ::: {.callout-note title="Example:" collapse="true"}

    ~~~text
    Write an instruction that a real estate agent can give to his assistant to create CMA's for him. The results should be a string containing the instruction like so:
    
    ```json
    [
      "Create a CMA for 2430 Victory Park"
    ]
    ```
    
    If you need a listing you can use any of the following:
    
    <SELECT address FROM listings_filters;> (From minimal database)
    ~~~

    :::


#### Log Results to Database and Visualize

* Use existing tools to systematically track unit test results to monitor progress and identify areas for improvement.

  * ::: {.callout-note title="Example:" collapse="true"}

    ```text
    ----------
    Website - one-listing-found Results
    Success: 0 Fail 26 Total 26 Average Duration: 7
    Technical Tokens Leaked: 26
    ----------
    Website - multiple-listings-found Results
    Success: 0 Fail 22 Total 22 Average Duration: 4
    Unknown: 1
    Exposed UUIDs: 17
    Failed to format JSON Output: 4
    ----------
    Website - no-listing-found Results
    Success: 25 Fail 1 Total 26 Average Duration: 3
    Unknown: 1
    ----------
    ```

    :::

* Use bar charts or other visualizations to track error rates across different scenarios and iterations.

#### Unit Test Considerations

* **Strict vs. Leaderboard Approach:**
  - **Strict:** All unit tests must pass; otherwise, the pipeline is halted.
  - **Leaderboard:** Track the number of passing tests over iterations to measure progress.
* **Use Case Specificity:**
  - **Public-facing products:** Prioritize tests that prevent data leaks and ensure user privacy.
  - **Internal tools:** Focus on identifying major issues, as minor errors might be less critical.

### LLM as a Judge

* **Alignment with Human Standard:** LLM as a judge must be aligned with a trusted human standard to ensure reliable evaluation.
* **Iterative Alignment:** Continuously measure and improve the agreement between LLM as a judge and human evaluation.
* **Tips:** Use a powerful LLM, treat the judge as a mini-evaluation system, and periodically re-align with human judgment.

#### LLM-As-A-Judge Example: De-biasing Text Project

* **Challenge:** Lack of transitivity in LLM as a judge's evaluation, leading to unreliable results.
* **Solution:** Relying on human evaluation due to the limitations of LLM as a judge in this specific use case.

### Human Evaluation

* **Importance:** Human evaluation is always necessary, even when using other evaluation methods, to ensure alignment and prevent over-fitting.
* **Regular Data Analysis:** Continuously analyze human evaluations to identify patterns, biases, and areas for improvement.
* **Balancing Cost and Accuracy:** Determine the appropriate level of human evaluation based on project constraints and desired accuracy.

### What Worked

|                  | Writing Queries                              | Debiasing Text                              |
| ---------------- | -------------------------------------------- | ------------------------------------------- |
| Unit Tests       | Good                                         | Too Rigid                                   |
| LLM as a judge   | Pretty Good                                  | Not transitive                              |
| Human Evaluation | Some labor required, aided by LLM as a judge | Labor intensive, which was ok for this task |



### Evaluation Workflow

* **Iterative Feedback Loop:** Evaluation enables a fast feedback loop for prompt engineering, fine-tuning, and other improvements.
* **Hidden Complexity:** Building effective evaluation systems is not trivial and requires careful consideration of various factors.

### Human Eval Going Wrong in Alt Text Project

* **Challenge:** Human evaluators' standards can drift over time, leading to misleading results, even with a fixed rubric.
* **Example:** Alt text generation project where human evaluators' standards increased as they saw better models, making later models appear worse than they actually were.

#### A/B Testing

* **Solution:** A/B testing can control for changes in human judgment over time by randomly assigning evaluators to different models.
* **Limitations:** A/B testing requires sufficient data and human labelers, making it impractical for early-stage projects.



## Looking At Your Data

* **Crucial Importance:** Looking at your data is essential for understanding LLM behavior, identifying failure modes, and improving evaluation methods.
* **Common Pitfall:** Many practitioners do not look at their data enough, even when they think they do.

### What is a Trace?

* **Definition:** A trace is a sequence of events in an LLM pipeline, such as multi-turn conversations, RAG processes, or function calls.
* **Importance:** Traces are valuable for debugging, fine-tuning, and understanding LLM behavior.
* **Representation:** Traces are often represented as JSON-L files, but other formats are possible.

### Remove All Friction from Looking at Your Data

* **Ease of Access:** Make it easy to access, filter, and navigate your data to encourage regular inspection.
* **Custom Tools:** Consider building custom tools using Shiny, Gradio, Streamlit, or other frameworks to streamline data exploration.
  * **[langefree](https://github.com/parlance-labs/langfree):** Tools for extraction, transformation, and curation of `ChatOpenAI` runs from LangSmith. 

* **Key Considerations:** Ensure that your tools remove enough friction and provide the necessary information for effective analysis.

### Rendering & Logging Traces

* **Tools:** Use tools like LangSmith, IdenticLogFire, Braintrust, Weights&Biases Weave, OpenLLMetry, and Instruct to log and render traces.
  * **Commercial:**
    * [Langsmith](https://smith.langchain.com/)
    * [Pydantic LogFire](https://pydantic.dev/logfire)
    * [BrainTrust](https://www.braintrustdata.com/)
    * [W&B Weave](https://wandb.ai/site/weave)

  * **OSS:**
    * [Instruct](https://ukgovernmentbeis.github.io/inspect_ai/workflow.html)
    * [OpenLLMetry](https://github.com/traceloop/openllmetry)

* **LangSmith:** A platform for logging, testing, and visualizing LLM pipelines, with features like trace rendering, filtering, and feedback integration.

### It's Best to Use a Tool

* **Off-the-Shelf Solutions:** Leverage existing tools for logging traces and other evaluation tasks to focus on data analysis and model improvement.
* **Tool Exploration:** Explore the various tools available through workshops, office hours, and other resources to find the best fit for your needs.



## Harrison Chase: Langsmith for Logging & Tests

* [Langsmith Website](https://www.langchain.com/langsmith)
* [Langsmith Docs](https://docs.smith.langchain.com/)

### LangSmith Features

* **Data Visualization and Analysis:**  Log, visualize, and analyze interactions with your LLM applications, enabling deep dives into individual runs and identification of potential issues.
* **Dataset Management and Testing:** Create, manage, and test your LLM applications against diverse datasets, facilitating targeted improvements and robust evaluation.
* **Experiment Tracking and Comparison:** Track experiment results over time, compare different model versions, and gain insights into performance changes.
* **Leveraging LLM as a Judge:** Utilize LLMs for automated evaluation, streamline feedback loops, and align LLM judgments with human preferences.
* **Human-in-the-Loop Feedback:** Integrate human feedback seamlessly through annotation queues, enabling continuous improvement and refinement of your LLM applications.

### Observability: Looking at Your Data

* **Integration:** Langsmith integrates with Langchain via environment variables and offers various entry points for non-Langchain users, including decorators, direct span logging, and project-based organization.
* **Data Visualization:** Provides an interface to visualize logged data, including system messages, human and AI interactions, outputs, and relevant documents, all presented in an easily digestible format.
* **Transition to Playground:** Allows direct navigation from a specific trace to a playground environment, facilitating rapid iteration and prompt modification.

#### Filtering and Dissecting Data

* **Filtering Capabilities:** Offers robust filtering options based on errors, latency, status, tags (e.g., LLM provider), and user feedback, enabling focused analysis of specific data subsets.
* **Aggregate Statistics:** Provides aggregated statistics over time, allowing for the identification of trends and patterns in application performance.
* **A/B Testing and Metadata Grouping:** Enables A/B testing by grouping statistics based on metadata, such as LLM provider, to compare performance across different models or configurations.

### Datasets and Testing

* **Dataset Creation:** Supports manual example uploads, imports from existing traces (e.g., failed interactions), and the organization of data into distinct splits for targeted testing.
* **Split Testing:** Allows for the evaluation of LLM applications on specific data splits, enabling focused analysis and improvement of performance in identified problem areas.

#### Tracking Experiments Over Time

* **Experiment Tracking:** Automatically logs and displays experiment results, including metrics and performance over time, allowing for monitoring and identification of regressions.
* **Experiment Comparison:** Provides an interface to compare two or more experiments side-by-side, highlighting performance differences and facilitating detailed analysis of specific cases.

### LLM as a Judge

* **Automated Evaluation:** Supports the use of LLMs as judges for automated evaluation, streamlining the feedback process and reducing reliance on manual review.
* **Off-the-Shelf and Custom Evaluators:** Offers both pre-built evaluation prompts and the flexibility to define custom evaluation functions, catering to diverse use cases.
* **Aligning Human Preferences:** Facilitates the alignment of LLM judgments with human preferences through few-shot learning and an upcoming correction flow feature, enabling continuous improvement of evaluation accuracy.

### Human-in-the-Loop Feedback

* **Annotation Queues:** Provides annotation queues for efficient human feedback collection, allowing for the review, labeling, and categorization of data points to improve model performance.
* **Collaborative Features:** Includes features for adding notes, marking completion status, and collaborating on data annotation tasks, fostering teamwork and efficient feedback integration.



## Bryan Bischof: Spellgrounds for Prodigious Prestidigitation

* **Spellgrounds:** An internal library for developing and running evaluations, combining systematic and use-case-specific approaches.
* **Opinionated View on Evals:** Evals should help determine product readiness, ensure system reliability, and aid in debugging.
* [Google Slides](https://docs.google.com/presentation/d/1GC868XXjhxOpQEt1jUM79aW0RHjzxPp0XhpFHnYH760/edit#slide=id.p)
* [hex.tech](https://hex.tech/product/magic-ai/)

### Preamble

* **Three Purposes of Evals:**
  * Determine product readiness.
  * Ensure system reliability.
  * Aid in debugging.
* **Evals as Data Science:** LLM evaluations are not entirely new and should leverage existing data science principles and techniques.

### Miscats and Fizzled Spells: Things to avoid

#### Thinking LLM Evaluations Are Entirely New

* **Leverage Existing Expertise:** Data scientists have extensive experience in evaluating unpredictable outputs and mapping user problems to objective functions.
* **Examples:**
  * **Code generation:** Execution evaluation.
  * **Agents:** Planning as binary classification.
  * **Summarization:** Retrieval accuracy.

#### Failing to Include Use-Case Experts

* **Expert Input:** Users and domain experts provide valuable insights into what constitutes good output for specific use cases.
* **Example:** Collaborating with data scientists to define ideal chart outputs for an LLM-powered data visualization tool.

#### Waiting Too Long to Make Evaluations

* **Early Integration:** Evals should be part of the development cycle from the beginning, including RFC creation and design discussions.

#### Not Recognizing Product Metrics vs. Evaluation Metrics

* **Distinct but Related:**
  * Product metrics track overall system performance, while evaluation metrics assess specific LLM components or functionalities. 
  * Product metrics provide valuable insights for designing evals, but they are not sufficient for comprehensive evaluation.

* **Custom Environments:** Create custom datasets and environments that reflect real-world use cases, even when access to production data is limited.

#### Buying an Evaluation Framework Doesn't Make It Easy

* **Focus on Fundamentals:** The hard part of evals is understanding user stories and input diversity, not the framework itself.
* **Jupyter Notebooks:** Jupyter Notebooks are powerful tools for interactive data exploration and evaluation.
* **Invest Wisely:** Prioritize understanding user needs and building effective evals before investing in complex frameworks.

#### Reacing Too Early for LLM-Assisted Evaluation

* **LLM Judging as a Tool:** LLM judging can be valuable for scaling evaluations and identifying potential issues, but it is not a replacement for human judgment.
* **Systematic Approach:** 
  * Establish a solid foundation of traditional evaluations before incorporating LLM-assisted methods.
  * Use multiple judges and periodically check for alignment with human evaluation.


### Moderating Magic: How to build your eval system

#### Magic

* **Definition:** An AI copilot for data science that generates code, reacts to edits, and creates visualizations.
* [Product Page](https://hex.tech/product/magic-ai/)

#### RAG Evals

* **Treat RAG as Retrieval:** Evaluate RAG systems like traditional retrieval systems, focusing on hit rate and relevance.
* **Baselines and Calibration:** Establish clear baselines and avoid treating retrieval scores as absolute confidence estimates.

#### Planning Evals

* **State Machine as Classifier:** Evaluate agent planning as a binary classification task, checking the correctness of each step.
* **Prompt Quality:** Evaluate the quality of downstream prompts generated by the planning stage, as they can be suboptimal.

#### Agent-Specific Evals

* **Structured Output:** Encourage and evaluate the use of structured output from agents to facilitate integration and consistency.
* **API Interfaces:** Design tightly coupled API interfaces between agent components and evaluate their consistency.

#### Final Stage Evals

* **Topic:** Evaluating the final output or summary generated by an agent chain.
* **Recommendation:** Ensure the summary accurately reflects the agent's actions and avoid providing excessive context that can introduce noise.

#### Experiments are Repeated-Measure Designs

* **Treat Updates as Experiments:** Evaluate the impact of updates and bug fixes as experiments, measuring significance and comparing to historical data.
* **Production Event Reruns:** Rerun historical production events through updated models and use automated evals to assess improvements.

#### Production Endpoints Minimize Drift

* **Direct Connection:** Connect your evals framework directly to your production environment to minimize drift and ensure consistency.
* **Endpoint Exposure:** Expose each step of the production workflow as an endpoint to facilitate testing and debugging.

### Q&A

* Leverage Jupyter Notebooks for reproducible evaluation orchestration and detailed log analysis.
* Focus on evaluating the most critical and informative aspects of the LLM system, prioritizing evaluations that exhibit variability and potential for improvement.
* Use bootstrap sampling to efficiently assess performance and identify areas for improvement.
* Strive for an evaluation suite with a passing rate of 60-70% to ensure sufficient sensitivity to changes and improvements.



## Eugene Yan: Evaluating LLM-Generated Summaries with Out-of-Domain Fine-tuning

### Introduction

* **Problem:** Evaluating the factual accuracy of LLM-generated summaries and detecting hallucinations.
* **Solution:** Develop an evaluator model that predicts the probability of a summary being factually inconsistent with the source document.
* **Approach:** Frame the problem as a Natural Language Inference (NLI) task, treating "contradiction" as factual inconsistency.

* **GitHub Repository:** [eugeneyan/visualizing-finetunes](eugeneyan/visualizing-finetunes)
* **Blog Post:** [Out-of-Domain Finetuning to Bootstrap Hallucination Detection](https://eugeneyan.com/writing/finetuning/)

### Methodology

1. **Data Preparation:**

   * Exclude low-quality data (e.g., CNN Daily Mail from FIB).
   * Split data into train, validation, and test sets, ensuring no data leakage.
   * Balance classes within each set.

2. **Model Fine-tuning:**

   * Use a pre-trained NLI model (DistilBART fine-tuned on MNLI).
   * Fine-tune on FIB data alone and evaluate performance.
   * Fine-tune on USB data, then FIB data, and evaluate performance on both datasets.

3. **Evaluation Metrics:**

   * **Standard metrics:** 

     * **ROC AUC:** Area Under the Receiver Operating Characteristic Curve. 

       * Measures the ability of a binary classification model to distinguish between the positive and negative classes across all possible thresholds. 
       * Higher values indicate better performance, with 1 being perfect and 0.5 indicating no better performance than random guessing.

       **PR AUC:** Area Under the Precision-Recall Curve. 

       * Evaluates the trade-off between precision (the accuracy of positive predictions) and recall (the ability to find all positive instances) across different thresholds. 
       * Useful when dealing with imbalanced datasets. 
       * A higher PR AUC indicates better performance in identifying the positive class.

   * **Custom metrics:**

     * **Recall:** Measures the proportion of actual positive cases that are correctly identified by the model
     * **Precision:**  Measures the proportion of positive predictions that are actually correct.

   * **Visualizations:** Distribution overlap of predicted probabilities for consistent and inconsistent summaries.

### [Overview Notebook](https://github.com/eugeneyan/visualizing-finetunes/blob/main/0_overview.ipynb)

* **Evaluator Model:** A model trained to detect factual inconsistencies in summaries, framed as a natural language inference (NLI) task.
* **NLI for Factual Inconsistency:** Using the "contradiction" label in NLI to identify factual inconsistencies in summaries.
* **Objective:** Fine-tune an evaluator model to catch hallucinations and evaluate its performance through each epoch.
* **Data Blending:** Demonstrating how blending data from different benchmarks can improve the evaluator model's performance.

### [Prepare Data Notebook](https://github.com/eugeneyan/visualizing-finetunes/blob/main/1_prep_data.ipynb)

* **[Factual Inconsistency Benchmark (FIB)](https://huggingface.co/datasets/r-three/fib):** A dataset containing one-sentence summaries from news articles, with labels indicating factual consistency.
* **[Unified Summarization Benchmark (USB)](https://huggingface.co/datasets/kundank/usb):** A dataset containing summaries of Wikipedia articles, with labels indicating factual consistency.
* **Data Splitting and Balancing:** Splitting the data into train, validation, and test sets, ensuring no data leakage and balancing positive and negative examples.

### [Finetune FIB Notebook](https://github.com/eugeneyan/visualizing-finetunes/blob/main/2_ft_fib.ipynb)

* **Model:** Distilled BART, a pre-trained encoder-decoder model fine-tuned on MNLI.
* **Fine-Tuning:** Fine-tuning the model on the FIB dataset, tracking custom metrics like ROC AUC, recall, and precision.
* **Results:** Fine-tuning on FIB alone shows limited improvement in ROC AUC and recall, indicating the need for more data.

### [Finetune USB then FIB Notebook](https://github.com/eugeneyan/visualizing-finetunes/blob/main/3_ft_usb_then_fib.ipynb)

* **Data Blending:** Fine-tuning the model on the larger USB dataset first, followed by fine-tuning on the FIB dataset.
* **Results:** Fine-tuning on USB significantly improves performance on both USB and FIB, demonstrating the benefits of data blending.
* **Evaluator Model as a Tool:** The fine-tuned evaluator model can be used to evaluate generative models, acting as a fast and scalable hallucination detector.

### Advantages of the Evaluator Model

* **Fast and Scalable:** Evaluates summaries in milliseconds, making it suitable for real-time applications.
* **Controllable:** Allows setting thresholds to prioritize precision or recall based on specific needs.
* **Versatile:** Can be adapted to evaluate other aspects of summaries, such as relevance and information density.

### Evaluating Agents

* Break down complex tasks into smaller, evaluable steps.
* Use a combination of classification, extraction, and potentially reward model-based metrics.
* **Example:** Evaluating a meeting transcript summarization agent:
  * Step 1: Evaluate the extraction of decisions, actions, and owners (classification).
  * Step 2: Evaluate the factual consistency of extracted information against the transcript (classification using the hallucination detection model).
  * Step 3: Evaluate the quality of the final summary in terms of information density and writing style (potentially using a reward model).



## Shreya Shankar: Scaling Up Vibe Checks for LLMs

* **Focus:** Using LLMs to scale up human evaluation and create task-specific assertions or guardrails.
* **Evaluation Assistants:** Tools that aid humans in creating and refining evaluations for LLM pipelines.
* **Longer Talk:**  [Scaling Up “Vibe Checks” for LLMs - Shreya Shankar | Stanford MLSys #97](https://www.youtube.com/watch?v=eGVDKegRdgM) 

### LLM Pipelines

* **Zero-Shot Capabilities:** LLM pipelines can perform complex tasks without explicit training, using prompt templates and instructions.
  * **Examples:** 
    * **[julia/podcaster-tweet-thread](https://smith.langchain.com/hub/julia/podcaster-tweet-thread):** Take a podcast episode transcript and turn into a tweet thread.
    * **[homanp/github-code-reviews](https://smith.langchain.com/hub/homanp/github-code-reviews):** This prompt reviews pull request on GitHub.
    * **[matu/customer_satisfaction](https://smith.langchain.com/hub/matu/customer_satisfaction):** This prompt is being use to extract services and sentiments from a customer answer to a survey.
    * **[muhsinbashir/youtube-transcript-to-article:](muhsinbashir/youtube-transcript-to-article:)** Convert any Youtube Video Transcript into an Article.


### LLMs Make Unpredictable Mistakes

* **Instruction Following:** LLMs may not always follow instructions perfectly, leading to unexpected errors and inconsistencies.
* **Need for Guardrails:** Evaluation and assertions are crucial for detecting and correcting LLM errors, ensuring reliable output.

### Vibe Checks: Custom Evaluation for LLMs

* **Vibe Checks:** Task-specific constraints, guidelines, or assertions that define "good" output based on human judgment.
* **Challenges:**
  * **Subjectivity:**  Different users may have different expectations for the same task.
  * **Complexity:** Metrics like "tone" are difficult to quantify and evaluate.
  * **Scalability:** Manual vibe checks by humans are effective but don't scale well.
* **Spectrum of Vibe Checks:**
  * **Generic:** Common ML performance metrics provided by model developers.
  * **Architecture-Specific:** Metrics relevant to specific LLM architectures (e.g., faithfulness in RAG pipelines).
  * **Task-Specific:** Fine-grained constraints tailored to the exact requirements of a task.
* **Goal:** Develop scalable, codified vibe checks (validators, assertions, guardrails) that capture task-specific requirements.

### Evaluation Assistants: Using LLMs to Build Vibe Checks

* **Evaluation Assistants:** Tools that help humans define and implement task-specific evaluations and assertions.
* **Key Idea:** Leverage LLMs to scale, not replace, human judgment.
* **Workflow Components:**
  * **Auto-generate criteria and implementations:** Use LLMs to suggest potential evaluation criteria and ways to implement them.
  * **Mixed Initiative Interface:** Allow humans to interact with and refine LLM-generated criteria and provide feedback.

### Auto-Generated Assertions: Learning from Prompt History

* **Challenge:** Identifying relevant assertion criteria and ensuring coverage of potential failures.
* **[SPADE System](https://arxiv.org/abs/2401.03038):** A two-step workflow for generating assertions.
  1. **Generate candidate assertions:** Use LLMs to propose potential assertions.
  2. **Filter based on human preferences:** Allow humans to select and refine the most relevant assertions.
* **Insight:** Prompt version history reveals information about developer priorities and common LLM errors.
  * **Example:** Repeated edits to instructions related to sensitive information indicate a need for a corresponding assertion.
* **Categorizing Prompt Deltas:** Analyzing how humans modify prompts helps identify common categories of edits, which can inform assertion generation.
* **From Taxonomy to Assertions:** LLMs can use the categorized prompt deltas and the current prompt to suggest relevant assertion criteria.
* **Lessons from Deployment:**
  * Inclusion and exclusion assertions are most common.
  * LLM-generated assertions may be redundant, incorrect, or require further refinement.



### EvalGen: A Mixed Initiative Interface for Evaluation

* **Paper:** [Who Validates the Validators? Aligning LLM-Assisted Evaluation of LLM Outputs with Human Preferences](https://arxiv.org/abs/2404.12272)
* **Motivation:** Streamline the process of creating and refining assertions, making it more efficient and user-friendly.
* **Key Features:**
  * **Minimize wait time:** Provide rapid feedback and iteration cycles.
  * **Human-in-the-loop:** Allow users to edit, refine, and grade LLM outputs and criteria.
  * **Interactive grading:** Enable users to provide thumbs-up/thumbs-down feedback on LLM outputs.
  * **Report card:** Summarize evaluation results and highlight areas for improvement.
* **Qualitative Study Findings:**
  * **Starting point:** EvalGen provides a useful starting point for assertion development, even if initial suggestions require refinement.
  * **Iterative process:** Evaluation is an iterative process that benefits from ongoing human feedback.
  * **Criteria drift:** User definitions of "good" and "bad" output evolve over time and with exposure to more examples.
  * **Code-based vs. LLM-based evals:** 
    * Users have different expectations and use cases for these two types of evaluations.
    * Preferred for fuzzy criteria, dirty data, and situations where humans struggle to articulate clear rules.
* **EvalGen v2:** Incorporates lessons learned from the study, including:
  * Dynamic criteria list for easier iteration.
  * Natural language feedback for refining criteria.
  * Support for per-criteria feedback.

### Overall Takeaways

* **Mistakes are inevitable:** LLMs will make mistakes, especially at scale.
* **LLMs can assist in evaluation:** By leveraging prompt history and human feedback, LLMs can help create effective evaluation metrics.
* **Evaluation is iterative:** Continuous monitoring, feedback, and refinement are crucial for maintaining LLM accuracy and alignment with user expectations.
* **Evaluation assistants are valuable:** Tools like EvalGen can significantly streamline the process of developing and refining LLM evaluations.



## Q&A Session

### Using Prompt History for Generating Assertions

* **Benefit:** Focusing LLM's attention when generating evaluation criteria. Instead of designing a unit test for every sentence in a long prompt, providing prompt history helps focus on key criteria.
* **Focus on Iteration:** Start with 2-3 criteria, refine them, and then add more, rather than starting with an overwhelming number.
* **Challenges in Writing Assertions:**  The difficulty lies in aligning assertions with what constitutes "good" or "bad" output. This definition evolves over time and requires analyzing model output and user feedback. 
* **Value of Evaluation Assistants:**  Assist in drawing conclusions from data and defining "good" output, aiding in the continuous improvement process.

### Generalizability of Assertion Criteria

* **Generalization Across Models:** Prompt edits and the way people interact with LLMs are similar across different models, regardless of the specific model used (Mistral, LLAMA2, ChatGPT, Claude).

### Unit Tests for Specific LLM Tasks

* **Applicability of Unit Tests:**  While straightforward for tasks with clear data structures (e.g., query validity), unit tests are less effective for general-purpose language models or tasks like text rewriting or summarization.

### Temperature Parameter in Open-Source LLMs

* **Open-Source LLMs and Temperature:** Similar to OpenAI's models, open-source LLMs have a temperature parameter. Setting it to zero ensures deterministic output, which is often desirable in production settings.

### Importance of Evaluation Methods

* **Iterative Approach to Evaluation:**  Start building the product without extensive upfront evaluation. Implement evaluations as you learn more about the task, identify edge cases, and seek improvements. 
* **Don't Let Evals Hinder Progress:** Avoid evaluation paralysis. Focus on creating a minimal product and then iteratively refine it based on evaluations and user feedback.

### Fine-tuning LLM as a Judge

* **Using Off-The-Shelf Models:**  It is generally recommended to use publicly available, off-the-shelf LLMs as judges instead of fine-tuning separate judge models.
* **Complexity and Alignment:**  Fine-tuning judge models can lead to complexity and make it challenging to align them with human judgment.

### Starting the Data Flywheel

* **Start with a Prompt:** Begin with a simple prompt and an off-the-shelf LLM to build a basic product and gather user data.
* **Leverage Synthetic Data:** Utilize the LLM's capabilities to generate synthetic data, enabling faster iteration and unblocking progress.

### "Do Sample" Parameter in Production

* **Deterministic vs. Varied Output:**  Setting `do_sample` to `false` (or using zero temperature) ensures deterministic, consistent output, often preferred for production systems requiring predictable behavior. 
* **Use Case Dependency:** For creative applications like character AI, where variety is desired, `do_sample` can be set to `true` or a non-zero temperature can be used.

### Preparing Data for A/B Testing with LLMs

* **Human Evaluation vs. LLM as Judge:** While LLMs can potentially be used to choose between options, human evaluation is often more reliable.
* **Context-Specific Data Preparation:**  Data preparation depends heavily on the specific task and why an LLM is used for A/B testing.

### Evaluating Retriever Performance in RAG

* **Key Metrics:**
  * **Recall@10:** Measures how many relevant documents are retrieved within the top 10 results.
  * **Ranking (NDCG):**  Evaluates if the most relevant documents are ranked higher.
  * **Ability to Return Zero Results:** Important for identifying queries with no relevant information in the index, preventing the LLM from generating incorrect responses based on irrelevant data.
* **Importance of Handling Irrelevant Data:** Ensuring the retriever can effectively identify and handle queries with no relevant information is crucial for avoiding inaccurate or nonsensical responses from the LLM.

### Filtering Documents for Factuality and Bias in RAG

* **Challenges:** Identifying factually incorrect or biased content within the document corpus is a complex challenge.
* **Content Moderation and Exclusion:**  Employ content moderation techniques to identify and exclude toxic, biased, or offensive content from the retrieval index.
* **Open Problem:** Detecting subtle misinformation or bias remains an open research problem.

### Running Unit Tests during CI/CD

* **Local vs. CI/CD Execution:** Running tests locally provides faster feedback during development, while integrating with CI/CD ensures consistent testing and prevents accidental deployments without proper testing.
* **Use Case Dependency:** The choice depends on the purpose of the tests (quality assurance vs. safety checks) and the sensitivity of the application.

### Checking for Contamination of Base Models

* **Contextual Reasoning:**  Analyze the likelihood of overlap between the evaluation data and the base model's training data based on the nature and recency of the data.
* **Performance Monitoring:** Be wary of unexpectedly high performance, which could indicate data leakage.
* **No Foolproof Solution:**  Data contamination is a difficult problem with no universal solution. Careful consideration and context-specific analysis are essential. 























