---
title: "Workshop 1: When and Why to Fine-Tune an LLM"
date: 2024-5-31
image: /images/empty.gif
hide: false
search_exclude: false
categories: [notes, llms]
description: "Workshop #1 provides a practical overview of fine-tuning large language models, focusing on when it is and is not beneficial, emphasizing a workflow of simplification, prototyping, and iterative improvement using  evaluations."

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





* [Key Takeaways](#key-takeaways)
* [Course Overview](#course-overview)
* [When to Fine-Tune](#when-to-fine-tune)
* [Understanding Fine-Tuning](#understanding-fine-tuning)
* [Case Study: Logistics Company Regression Problem](#case-study-logistics-company-regression-problem)
* [Case Study: Honeycomb Natural Language Query Assistant](#case-study-honeycomb-natural-language-query-assistant)
* [Q&A Session #1](#qa-session-1)
* [Chatbots](#chatbots)
* [Preference Optimization](#preference-optimization)
* [Evaluating Use Cases for Fine-Tuning](#evaluating-use-cases-for-fine-tuning)
* [Q&A Session #2](#qa-session-2)



## Key Takeaways

* **Start simple:** Focus on prompt engineering and using pre-trained models like those from OpenAI before jumping into the complexity of fine-tuning.
* **Fine-tune strategically:** Consider fine-tuning when you need bespoke behavior, have unique data, or require data privacy.
* **Templating is crucial:** Pay close attention to consistency in templating between training and inference to avoid unexpected model behavior.
* **Evaluate rigorously:** Use domain-specific evaluations and metrics to measure model performance and guide fine-tuning decisions.
* **Preference optimization shows promise:** Techniques like Direct Preference Optimization (DPO) can train models to outperform even human experts by learning from comparative feedback.



## Course Overview 

- **Focus:** Actionable insights and practical guidance from real-world experience in deploying LLMs for various business needs.
- **Philosophy:** 
  - Prioritize practical value over project ideas that only sound cool.
  - Start with simple, straightforward solutions and progressively refine them.
  - Ship prototypes quickly for rapid iteration and feedback.

- **Workflow:** 
  - Start with prompt engineering before considering fine-tuning.
    - Prompt engineering provides much faster iteration and experimentation.
    - The results from prompt engineering will help inform whether fine-tuning is necessary.
  - Iterate quickly with simple prototypes.
    - Build and show people concrete things, so they can provide feedback.
    - Simple prototypes almost always work well enough to start making progress.
  - Incorporate evaluations (Evals) to measure and improve model performance.
    - Blog Post: [Your AI Product Needs Evals](https://hamel.dev/blog/posts/evals/)



## When to Fine-Tune 

- **Don't fine-tune for generic behavior:** 
  - Use existing powerful models like OpenAI's GPT or Anthropic's models via API for tasks where they excel.
  - Increasingly larger context windows allows us to fit more examples to fit into a prompt.
  - You should have some minimal evaluation system that you hit a wall on with prompting alone, before considering fine-tuning.

- **Do fine-tune for bespoke behavior:** 
  - When you need specific outputs or behavior not achievable through prompt engineering alone.
  - When you have a narrow, well-defined problem domain and sufficient data for training.
    - Fine-tuning requires examples of desired inputs and outputs for supervised learning.
  - When data privacy and model ownership are critical.
  - When you need improved quality and lower latency compared to large pre-trained models.
  - Requires proper operational setup and significant value use cases.
- **Iteration Speed & Complexity:** Fine-tuning involves slower iteration cycles and operational complexities compared to using pre-trained models.



## Understanding Fine-Tuning

* **Pre-training:** Training LLMs on massive text datasets to learn language fundamentals and next-token prediction.
* **Building on Pre-trained Models:** 
  * Fine-tuning adapts pre-trained models with vast general language knowledge to excel in specific domains.
  * Fine-tuning harnesses the next-token prediction mechanism used in pre-training to generate desired outputs.
* **Importance of Input-Output Examples**
  - Fine-tuning requires clear examples of desired inputs and outputs.
  - Documentation alone isn't sufficient; practical examples are necessary.
  - Mixed quality of training data (e.g., varied quality of human-written summaries) can lead to mediocre model performance.
* **Templating for Inference Control:**  
  * Guides the model to produce specific outputs by short-circuiting pre-trained behavior.
  * Inputs and outputs are placed within a consistent template to guide the model during inference.
  * Crucial for aligning training and inference.
  * Defines the structure of input and output text to guide the model.  
  * Inconsistencies in templating are a major source of errors.
    * Templates must be identical between training and inference.
    * There are many kinds of templates and it is easy to misinterpret them. 
    * Many tools try to abstract away and automate building templates and something often goes wrong.
    * Blog Post: [Tokenization Gotchas](https://hamel.dev/notes/llm/finetuning/05_tokenizer_gotchas.html)



## Case Study: Logistics Company Regression Problem

### Overview

* **Task:** Logistics company (e.g., UPS, DHL, USPS) needed to predict item value based on an 80-character description.
* **Takeaways:** Highlights the importance of understanding and preparing the training data, the limitations of fine-tuning for specific regression tasks, and the practical issues encountered with this approach.

### Traditional NLP and ML Approaches

- **Classical Techniques:** Initial consideration to use traditional NLP and ML methods.
- **Bag of Words Representation:** Highlighted issue where models fail to recognize unseen words or infrequent words due to limited data.

### Fine-Tuning Large Language Models

- **Initial Approach:** Attempted to use a large language model (LLM) with and without fine-tuning for regression.
- **Outcome:** The model learned patterns in the data that were not ideal for the task.

### Key Observations from Fine-Tuning

- **Round Numbers:** The model tended to predict round numbers frequently because past entries often used round numbers.
- **Mismatch in Values:** Conventional ML models can predict approximate values (e.g., $97 vs. $100), which is often more useful than exact but less frequent round number predictions by the LLM.
- **Training Data Limitations:** Training data often contained inaccuracies, such as undervalued entries to avoid insurance costs.

### Data Representation and Preprocessing

- **Description Complexity:** Corporate descriptions were often abbreviated or used acronyms, making them hard to interpret both for humans and models.
- **Pre-trained Model Limitations:** Pre-trained models struggled with unknown abbreviations or context-specific terms not encountered during pre-training.

### Conclusion

- **Unsuccessful Case Study:** The fine-tuning approach was largely unsuccessful due to predictable data issues.

### Insights and Recommendations

- **Data Quality:** Emphasized the importance of high-quality, representative training data for desired future behavior.
- **Raw Data Examination:** Stressed the need to carefully inspect raw data, a common yet frequently overlooked step in data science.
- **Practicality of ML Solutions:** For this case, traditional ML and NLP techniques did not provide satisfactory results, leading to the retention of the manual workflow.



## Case Study: Honeycomb Natural Language Query Assistant

### Overview

* **Task:** Building a system for Honeycomb, an observability platform that logs telemetry data about software applications, that translates natural language queries into the platform's domain-specific query language.

* **Takeaways:** Highlights the importance of fine-tuning in addressing domain-specific challenges, improving model performance, and meeting business requirements such as data privacy and operational efficiency.

### Honeycomb Platform Overview

- Honeycomb is an observability platform.
- Logs telemetry data like page load times, database response times, and application bottlenecks.
- Users query this data using a domain-specific query language.

### Initial Solution: Natural Language Query Assistant

- **Problem:** Users must learn a specific query language to use Honeycomb effectively.
- **Solution:** Create a natural language query assistant that translates user queries into Honeycomb's query language using large language models (LLMs).
- **Initial Approach:**
  - User provides a query and schema (list of column names from the user's data).
  - Prompt assembled with user input and schema sent to GPT-3/GPT-3.5.
  - Generated a Honeycomb query based on the prompt.

### Prompt Structure

1. **System Message:**
   - "Honeycomb AI suggests queries based on user input."
2. **Columns Section:**
   - Schema from the user's data inserted here.
3. **Query Spec:**
   - Simplified programming manual for Honeycomb's query language.
   - Contains operations and comments on their usage.
4. **Tips Section:**
   - Guidelines to handle different failure modes and edge cases.
   - Example: Handling time ranges correctly.
5. **Few-Shot Examples:**
   - Examples of natural language queries and corresponding Honeycomb query outputs.

### Challenges with Initial Solution

- **Expressing Query Language Nuances:**
  - Hard to capture all idioms and best practices of the query language.
  - GPT-3.5 lacks extensive exposure to Honeycomb's specific query language.
- **Tips Section Complexity:**
  - Tips devolved into numerous if-then statements.
  - Difficult for the language model to follow multiple conditionals.
- **Few-Shot Examples Limitations:**
  - Hard to cover all edge cases.
  - Dynamic few-shot examples could help but were not implemented.

### Business Challenges

- **Data Privacy:**
  - Need permission to send customer data to OpenAI.
  - Preference to keep data within a trusted boundary.
- **Quality vs. Latency Tradeoff:**
  - GPT-4 offered higher quality but was too slow and expensive.
  - Goal: Train a smaller, faster model with comparable quality.
- **Narrow Domain Problem:**
  - Honeycomb queries are a focused, narrow domain ideal for fine-tuning.
- **Impracticality of Extensive Prompt Engineering:**
  - Hard to manually encode all nuances of the query language.
  - Fine-tuning with many examples is more practical.

### Fine-Tuning Solution

- **Advantages:**
  - Faster, more compliant with data privacy needs.
  - Higher quality responses compared to GPT-3.5.
- **Implementation:**
  - Fine-tuned a model using synthetic data provided by Honeycomb.
  - The process and challenges encountered during fine-tuning will be simulated in the course.

### Recommendations

1. **Implement Fine-Tuning:**
   - Use synthetic data to replicate and improve the model.
   - Focus on capturing edge cases and nuances in the training data.
2. **Optimize for Performance:**
   - Balance model size and latency to ensure quick responses without sacrificing quality.
3. **Ensure Data Privacy:**
   - Keep data within a trusted boundary to comply with customer privacy requirements.
4. **Regularly Update Few-Shot Examples:**
   - Dynamically generate examples to cover new edge cases and improve model accuracy.
5. **Monitor and Iterate:**
   - Continuously monitor model performance and iteratively improve based on user feedback and new data.



## Q&A Session #1

This Q&A session covers various aspects of fine-tuning machine learning models, particularly focusing on fine-tuning versus retrieval-augmented generation (RAG), function calling, and synthetic data generation. It also touches upon the use of base models versus instruction-tuned models and the appropriate amount of data for fine-tuning. 

### Fine-Tuning vs. RAG

- **Definitions**:
  - **Fine-Tuning**: Adjusting a pre-trained model with additional data to improve performance in specific tasks.
  - **RAG (Retrieval-Augmented Generation)**: Combines information retrieval with generation to produce responses based on external documents.
- **Key Point**: Fine-tuning and RAG are not mutually exclusive; they can complement each other.
- **Process**: Validate the need for fine-tuning by ensuring good prompts and effective RAG.

### Fine-Tuning for Function Calls

- **Capability**: Models can be fine-tuned to improve at making function calls.
- **Examples**: Open models like LLaMA 3 and LLaMA 2 have been fine-tuned for function calling.
- **Challenges**: Identify and use good training data with successful function call examples while filtering out failures.

### Data Requirements for Fine-Tuning

- **Amount of Data**: Success with as few as 100 samples, though this varies by problem scope.
- **Broad Scope Problems**: Require more data to cover the problem space adequately.
- **Narrow Scope Problems**: Can often be fine-tuned with relatively little data.

### Synthetic Data Generation

- **Importance**: Helps overcome data scarcity in specific domains.
- **Methods**: Use powerful models to generate synthetic data, perturb existing data, and create test cases.
- **Practical Example**: Honeycomb example shows generating synthetic data to test and train models.

### Base Models vs. Instruction-Tuned Models

- **Base Models**: Not fine-tuned for specific instructions, allowing more control over fine-tuning processes.
- **Instruction-Tuned Models**: Pre-fine-tuned to respond to instructions, useful in broader chat-based applications.
- **Preference**: Often uses base models to avoid template conflicts and ensure specific fine-tuning needs.

### Model Size for Fine-Tuning

- **Preferred Size**: Starts with smaller models (e.g., 7 billion parameters) and scales up based on complexity and performance needs.
- **Trade-Offs**: Larger models require more resources and justification due to higher costs and hosting difficulties.

### Multimodal Fine-Tuning

- **Example Project**: Fine-tuning models to write alt text for images to assist visually impaired users.
- **Tools**: The LLaVA model is recommended for fine-tuning multimodal tasks.

### Recommendations

1. **Validate the Need for Fine-Tuning**: Before starting, ensure you have good prompts and effective RAG if applicable.
2. **Choose the Right Data**: Use high-quality, successful examples for fine-tuning and filter out poor results.
3. **Start Small**: Begin with smaller models and incrementally increase size based on performance needs.
4. **Leverage Synthetic Data**: Generate and use synthetic data to supplement training data, especially in data-scarce domains.
5. **Understand Model Types**: Choose between base models and instruction-tuned models based on the specific use case and desired control over fine-tuning.
6. **Explore Multimodal Capabilities**: Consider multimodal fine-tuning for tasks that require handling both text and images, utilizing models like LLaVA.



## Chatbots

### Overview

* **Topic:** Delves into the common pitfalls and considerations when working with LLM-powered chatbots.
* **Takaways:** Highlights why general-purpose chatbots are often a bad idea, with unrealistic expectations and overly broad scope leading to poor user experiences and significant challenges in development.

### Importance of Saying No to General-Purpose Chatbots

- **Prevalence of Chatbot Requests**: When working with LLMs, most clients will request a chatbot.
- **Need for Caution**: It's often necessary to push back on these requests due to potential complications.

### Case Study: Rechat Real Estate CRM Tool

- **Initial Concept**: A CRM tool for real estate that integrated multiple functionalities (appointments, listings, social media marketing).
- **Initial Implementation**: Started with a broad chat interface labeled "Ask Lucy anything."
  - **Problems with Broad Scope**: 
    - Unmanageable surface area.
    - User expectations mismatched with capabilities.
    - Difficult to make progress on scoped tasks.

### Lessons from Rechat Case Study

- **Scoped Interfaces**: Guide users towards specific tasks.
- **Fine-Tuning Challenges**: Difficult to fine-tune against a large and varied set of functions.

### Managing User Expectations

- **High User Expectations**: Users often assume chatbots can handle any request, leading to disappointment.
- **Setting Realistic Boundaries**: Important to guide users on what the chatbot can realistically do.

### Real-World Example: DPD Chatbot Incident

- **Background**: A chatbot released for a package delivery company, DPD, faced issues on launch.
- **Incident**: The chatbot swore in response to a user's prompt, leading to negative publicity.
  - **Media Coverage**: The incident was widely reported, causing significant concern within the company.
- **Lesson Learned**: 
  - **Expectations vs. Reality**: Even harmless errors can become major issues if they attract public attention.
  - **Guardrails**: Conventional software has clear input validation; free-form text input in chatbots is harder to manage.

### Guardrails and Prompt Injections

- **Challenges with Guardrails**: Tools to check for prompt injections are imperfect.
- **Importance of Reviewing Prompts**: Critical to understand and review the prompts used by guardrails to ensure safety.

### Recommendations

1. **Scoped Interfaces Over General Chatbots**: Focus on integrating chatbot functionalities into specific parts of the application rather than creating a general-purpose chatbot.
2. **User Expectation Management**: Clearly communicate what the chatbot can and cannot do to manage user expectations effectively.
3. **Modular Functionality**: Break down the chatbot’s functionalities into specific modules that can be fine-tuned individually.
4. **Review Guardrails**: Regularly review and understand the prompts and guardrails to ensure they are functioning correctly.
5. **Careful Rollout**: Test chatbots extensively before public release to avoid unexpected behaviors that could lead to negative publicity.



## Preference Optimization

Discusses the effectiveness of Direct Preference Optimization (DPO) in fine-tuning LLMs to produce superior outputs. By leveraging human preferences in comparing two responses to the same prompt, DPO can significantly improve the quality of model outputs.

### Preference Optimization Algorithms

- **Challenge:** Human-generated data is often imperfect, and training models solely on this data can lead to suboptimal results.
- **Human Preference Evaluation**: Humans excel at choosing between two options based on preference.
- **Preference Optimization Algorithms**: These techniques leverage human preferences to fine-tune models.

### Direct Preference Optimization (DPO)

- **Definition**: DPO involves using human preference data to guide model fine-tuning.
- Comparison to Supervised Fine-Tuning:
  - **Supervised Fine-Tuning**: Model learns to imitate responses based on a prompt-response pair.
  - **DPO**: Model learns from human preference data by comparing two responses to the same prompt and determining which is better.

### Process of Direct Preference Optimization

- Data Collection:
  - **Prompt**: Initial input or question.
  - **Responses**: Two different responses to the prompt.
  - **Human Evaluation**: Determining which response is better.
- **Model Update**: Model adjusts weights to favor better responses, potentially exceeding the quality of the best human-generated responses.

### Case Study: Customer Service Email Project

- Project Overview:
  - **Data**: 200 customer service emails.
  - **Responses**: Two responses per email from different agents.
  - **Manager Evaluation**: Manager chose the preferred response from each pair.
- **Model Used**: Fine-tuned on Zephyr (base model).

### Performance Comparison

- Methods Compared:
  1. **GPT-4 Response Generation**: Direct use of GPT-4 for generating responses.
  2. **Supervised Fine-Tuning**: Model fine-tuned on pairs of input-output data.
  3. **Human Agents**: Responses generated by human customer service agents.
  4. **DPO Model**: Model fine-tuned using direct preference optimization.
- Results:
  - **GPT-4**: Produced the lowest quality responses.
  - **Supervised Fine-Tuning**: Better than GPT-4 but worse than human agents.
  - **Human Agents**: Better than the supervised fine-tuned model.
  - **DPO Model**: Outperformed human agents, producing responses preferred 2 to 1 over human responses in blind comparisons.

### Advantages of Direct Preference Optimization

- **Superhuman Performance**: DPO models can generate responses superior to those of human experts.
- **Flexibility with Data Quality**: Effective even with imperfect or messy data.

### Recommendations

1. **Adopt DPO for Fine-Tuning**: Implement DPO in model fine-tuning processes to achieve superior performance.
2. **Leverage Human Preferences**: Collect and utilize human preference data to guide model improvements.
3. **Evaluate Model Performance**: Regularly compare DPO model outputs with human-generated outputs to ensure quality.
4. **Explore Variations of DPO**: Investigate slight tweaks and alternative algorithms related to DPO to further enhance model performance.



## Evaluating Use Cases for Fine-Tuning

This discussion focuses on evaluating different use cases for fine-tuning large language models (LLMs). The primary aim is to determine when fine-tuning is beneficial for the target use case compared to using a general model like ChatGPT.

### 1. Customer Service Automation for a Fast Food Chain

- **Use Case**: Automating responses to most customer service emails, with unusual requests routed to a human.
- Evaluation:
  - **Fit for Fine-Tuning**: Strong fit.
  - **Reasoning**: The company likely has a substantial dataset from past customer interactions. Fine-tuning can capture the specific nuances of the company’s customer service style and common issues.
  - **Example**: Handling specific inquiries about menu items, store locations, or promotions that are frequently encountered.

### 2. Classification of Research Articles for a Medical Publisher

- **Use Case**: Classifying new research articles into a complex ontology, facilitating trend analysis for various organizations.
- Evaluation:
  - **Fit for Fine-Tuning**: Excellent fit.
  - **Reasoning**: The ontology is complex with many subtle distinctions that are hard to convey in a prompt. The publisher likely has extensive historical data for training.
  - **Example**: Classifying articles into one of 10,000 categories, focusing on the most common 500 categories initially for efficiency.
  - **Implementation Detail**: Used a JSON array output for multi-class classification.

### 3. Short Fiction Generation for a Startup

- **Use Case**: Creating the world's best short fiction writer.
- Evaluation:
  - **Fit for Fine-Tuning**: Potentially good fit.
  - **Reasoning**:
    - General models like ChatGPT can write good short stories. 
    - Fine-tuning can help the model learn specific preferences in storytelling that go beyond what a general LLM can offer. The startup can gather user preferences on generated stories to continually improve the model.
  - **Example**: Generating two different story versions on a given topic and having users rate them to inform future fine-tuning.
  - **Considerations**: The feedback loop involving user ratings can help refine and optimize the storytelling quality.

### 4. Automated News Summarization for Employees

- **Use Case**: Providing employees with summaries of new articles on specific topics daily.
- Evaluation:
  - **Fit for Fine-Tuning**: Potentially unnecessary.
  - **Reasoning**: General LLMs like ChatGPT can already provide high-quality summaries. The benefit of fine-tuning depends on the availability of unique internal data to improve the summarization process.
  - **Example**: Summarizing a wide range of news articles without a significant internal dataset may not justify the effort of fine-tuning.
  - **Alternative**: Using preference-based optimization (DPO) to gather feedback on summary quality and improve the model if news summarization is a critical business function.

### Important Considerations

- **Data Availability**: Fine-tuning is more effective when there is a large, high-quality dataset available from past interactions or classifications.
- **Complexity and Specificity**: Use cases with complex, nuanced requirements are better candidates for fine-tuning compared to general tasks.
- **Resource Commitment**: The decision to fine-tune should consider the resources required for collecting and annotating additional data, as well as the importance of the task within the organization.

### Recommendations

1. **Assess Data Quality and Quantity**: Ensure sufficient and relevant data is available for fine-tuning.
2. **Evaluate Task Complexity**: Use fine-tuning for tasks that require specific knowledge or subtle distinctions that a general model might not capture.
3. **Consider Cost-Benefit**: Weigh the benefits of improved performance against the costs of data collection and model training.
4. **Iterate and Improve**: Continuously gather feedback to refine and improve the fine-tuned model, especially for user-preference-driven tasks.



## Q&A Session #2

This Q&A session addressed various questions related to model quantization, handling hallucinations in language models, and the importance of data annotation. 

### Quantization

- **Definition**: Quantization is a technique used to reduce the precision of models.
- **Performance Impact**: Over-quantization can lead to performance degradation.
- **Testing**: It is crucial to test the quantized models to ensure performance is not adversely affected.

### Hallucination in Language Models

- **Issue**: When classifying academic or scientific articles, ensuring that the language model (LM) only outputs valid classes is critical.
- **Solution**: Providing enough examples with specific sets of classes to train the model effectively.
- **Metrics**: Continuous monitoring and treating misclassifications as part of the expected process.

### Fine-Tuning Large Language Models

- **Use Case Evaluation**: The skill of evaluating use cases for fine-tuning is essential for data scientists.
- **Example**: Fine-tuning can outperform even human experts in specific, well-defined tasks, such as customer service for companies like McDonald's.

### Optimizing Prompts

- **Efficiency**: Static elements in prompts that don’t change should be removed in favor of more dynamic elements.
- **Few-Shot Examples**: These should be minimized or eliminated with extensive fine-tuning.
- **Prompt Engineering**: A critical technique in making language models more efficient and effective.

### Data Annotation and Evaluation

- **Human in the Loop**: Essential for evaluating LLMs and curating data for training and fine-tuning.
- **Tool Building**: Custom tools are often more effective than generic ones for specific domains.









{{< include /_about-author-cta.qmd >}}
