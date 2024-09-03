---
title: "Conference Talk 18: Fine-Tuning OpenAI Models - Best Practices"
date: 2024-8-29
image: /images/empty.gif
hide: false
search_exclude: false
categories: [notes, llms]
description: "In this talk, **Steven Heidel** from OpenAI's fine-tuning team covers best practices, use cases, and recent updates for fine-tuning OpenAI models."

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







### What is fine-tuning? When to use it?

- **What is fine-tuning?**
  - Training a model to follow a set of input-output examples. 
  - Teaching the model to output specific responses for given inputs.
- **When to use fine-tuning?**
  - **Following a specific format or tone:**  e.g., writing in a consistent style.
  - **Processing input in a particular way:**  e.g., extracting specific information from text.
  - **Following complex instructions:** When the base model struggles with specific instructions.
  - **Improving latency (speed) and reducing token usage (cost):**  Compared to multi-shot prompting, fine-tuning can be faster and cheaper. 
- **When NOT to use fine-tuning?**
  - **Teaching the model new knowledge:** Fine-tuning is not effective for adding new information to the model's knowledge base. Use RAG (Retrieval Augmented Generation) or custom models for this purpose.
  - **Performing multiple, unrelated tasks:** A single fine-tuned model is best suited for one task. Create separate models for different tasks or use prompt engineering. 
  - **Including up-to-date information:** Fine-tuning is not suitable for incorporating real-time or constantly changing data.



### Custom Models and Real-World Fine-tuning Examples

- **Question:** What are "custom models"?
  - **Answer:**
    - **Custom models** involve a deeper level of collaboration with OpenAI.
    - OpenAI works with select partners to train and refine models using large datasets over several months.
    - **Example:** OpenAI collaborated with Harvey, a legal tech company, to create a model trained on case law. This model excels in legal tasks and significantly reduces hallucinations compared to the base GPT model.
      - **Blog Post:** [Harvey](https://openai.com/index/harvey/)
- **Question:** Are there detailed examples of fine-tuning for real-world use cases, including training data and prompts?
  - **Answer:**
    - OpenAI's cookbook ([cookbook.openai.com](https://cookbook.openai.com/)) offers a Q&A model example.
      - The model is trained to respond with "I don't know" when it lacks relevant knowledge.
      - This example demonstrates how fine-tuning can improve a model's ability to recognize its limitations.
      - **GitHub Repository:** [examples/fine-tuned_qa](examples/fine-tuned_qa)
      - **Notebook 1:** [Fine-Tuned Q&A - collect data](https://cookbook.openai.com/examples/fine-tuned_qa/olympics-1-collect-data)
      - **Notebook 2:** [Fine-Tuned Q&A - create Q&A](https://cookbook.openai.com/examples/fine-tuned_qa/olympics-2-create-qa)
      - **Notebook 3:** [Fine-Tuned Q&A - train](https://cookbook.openai.com/examples/fine-tuned_qa/olympics-3-train-qa)
    - The "[Optimizing LLMs for Accuracy](https://platform.openai.com/docs/guides/optimizing-llm-accuracy)" guide on the OpenAI developer site provides a helpful framework for deciding when to use fine-tuning, RAG, or both. 
      - The guide includes a chart illustrating different approaches based on the need for adding context versus optimizing responses.



### A Combined Approach

- **Question:** What does it mean to combine fine-tuning and RAG, as shown in the upper-right quadrant of the "Optimizing LLMs" chart?
  - **Answer:**
    - The chart depicts a general progression, but specific applications may not require all steps.
    - Combining fine-tuning and RAG involves:
      - Using RAG to introduce new knowledge or context.
      - Using fine-tuning to modify the model's response style or instruction-following capabilities.



### Importance of Evals (Evaluation Metrics)

- **Key Takeaway:**  Define your own set of **evals** (evaluation metrics) tailored to your application's specific needs.
- **Why not use standard benchmarks?**
  - Standard benchmarks might not accurately reflect your application's requirements.
- **Best Practice:** Create evals based on real prompts and desired responses from your application.



### A Cautionary Tale: When Fine-tuning Goes Wrong

- **Example:** A company tried to create a Slack bot using fine-tuning on their entire Slack corpus.
  - **Goal:** The bot was intended to answer onboarding questions from new employees.
  - **Problem:** The model learned the format of Slack responses but not the underlying information.
    - It often responded with deferrals like "I'll do that tomorrow" or "Okay."
  - **Reason for Failure:** 
    - The model focused on replicating common responses instead of understanding the content.
    - This approach violated the principle of not using fine-tuning to add new knowledge.
  - **Better Approach:** Using RAG on the Slack data to provide the model with relevant information.



### Preparing Data for Fine-tuning

- **Data Format:**
  - Similar to the chat completions API, using system, user, and assistant messages.
  - Include the desired assistant response for each user message.
  - **Documentation:** [Training format for chat models](https://platform.openai.com/docs/api-reference/fine-tuning/chat-input)
  - **Documentation:** [Training format for completion models](https://platform.openai.com/docs/api-reference/fine-tuning/completions-input)
- **Recommendations:**
  - **Number of Examples:** 50-100 examples per task.
  - **Focus on a Single Task:** Use a single fine-tuned model for one specific task.
  - **Consistency in Prompt Structure:** Maintain consistency between the prompt structure used in fine-tuning and in actual application calls.



### Handling Multi-Turn Conversations and Internal Thoughts

- **Question:** How to handle multi-turn conversations with RAG, function calling, and internal thoughts (generated by tools like Langchain) during fine-tuning data preparation? Should all internal thoughts be included, even if they are not always present in production?
  - **Answer:**
    - Experiment with the **weight parameter** to control which parts of the conversation the model should focus on learning.
    - The weight parameter allows you to assign different levels of importance to different messages in the training data.
    - **General Advice:**
      - Avoid excessive complexity in fine-tuning. If multi-turn conversations prove too complex, consider simplifying or using alternative approaches.
- **Follow-up Question:** Is a weight parameter of 0 equivalent to setting the label of those tokens to -100 (effectively ignoring them)?
  - **Answer:**
    - A weight of 0 means the model won't learn how to generate those specific messages (e.g., user and system messages).
    - However, it will still use them as context when learning how to generate other messages.



### Evolution of the Weight Parameter and OpenAI's Development Process

- **Observation:** The introduction of the weight parameter was a significant development. However, its rollout wasn't widely publicized.
- **Question:** How does OpenAI decide on new features and gather feedback? Could this process be more transparent?
  - **Answer:**
    - OpenAI gathers feedback from various sources:
      - OpenAI Developer Forum
      - Direct customer interactions
      - Account managers
      - Events, conferences, and talks
    - The weight parameter was a direct result of this feedback process. 
    - OpenAI acknowledges the importance of a more prominent announcement for such features.
    - **Future Direction:** OpenAI plans to continue adding models, methods, and customization options to its fine-tuning platform.







### OpenAI's Fine-tuning Best Practices and Hyperparameters

#### Curate examples carefully

* Datasets can be difficult to build, start small and invest intentionally. Optimize for fewer high-quality training examples.
  - Consider "prompt baking", or using a basic prompt to generate your initial examples
  - If your conversations are multi-turn, ensure your examples are representative
  - Collect examples to target issues detected in evaluation
  - Consider the balance & diversity of data
  - Make sure your examples contain all the information needed in the response

#### Iterate on hyperparameters

* Start with the defaults and adjust based on performance.
  - If the model does not appear to converge, increase the learning rate multiplier
  - If the model does not follow the training data as much as expected, increase the number of epochs
  - If the model becomes less diverse than expected, decrease the number of epochs by 1-2

#### Establish a baseline

* Often users start with a zero-shot or few-shot prompt to build a baseline evaluation before graduating to fine-tuning.

#### Optimize for latency and token efficiency

* When using GPT-4, once you have a baseline evaluation and training examples, consider fine-tuning 3.5 to get similar performance for less cost and latency.
  - Experiment with reducing or removing system instructions with subsequent fine-tuned model versions.

#### Automate your feedback pipeline

* Introduce automated evaluations to highlight potential problem cases to clean up and use as training data.
  - Consider the G-Eval approach of using GPT-4 to perform automated testing using a scorecard.

#### Hyperparameters

- **Epochs:** The number of times the training data is iterated over.
  - Impacts training the most.
  - Higher epochs risk overfitting, lower epochs might lead to underfitting.
  - OpenAI automatically chooses a default based on dataset size.
- **Batch Size and Learning Rate Multiplier:** Have less impact but can be adjusted for optimization.
  - Larger batch sizes tend to work better for larger datasets.
  - Experiment with learning rates between `0.02-0.2`.
  - Larger learning rates often perform better with larger batch sizes 
- **Recommendation:** Start with defaults and adjust based on eval results.



### Success Stories and Performance Improvements

- **Case Study 1:** Fine-tuning a larger model (GPT-4) for an Icelandic government project resulted in significant improvements in grammar correction compared to the base model and other approaches.
- **Case Study 2:** Fine-tuning GPT-3.5 outperformed GPT-4 with few-shot prompting in a specific task, highlighting potential cost and latency benefits.



### Function Calling and Fine-tuning Considerations

- **Use Case:** Fine-tuning can improve the model's ability to select and call the correct functions.
- **Caution:** When working with complex or lengthy function definitions, it might be necessary to include them in the prompt even after fine-tuning.
  - Unlike scenarios where fine-tuning can replace lengthy prompts, function calling might require those definitions to be present during runtime. 



### Reducing Hallucinations

- **Example:** The Q&A model in OpenAI's cookbook demonstrates how to reduce hallucinations.
  - **Approach:** The model is trained on Olympic-related questions. If the question falls outside this domain, it's designed to respond with "I don't know." 
  - **Result:** Fine-tuning effectively minimizes false positives (incorrect answers) in this scenario.



### Adoption Rate of Fine-tuning and Open Source Alternatives

- **Observation:** Less than 1% of OpenAI API users utilize fine-tuning.
  - **Reason:**
    - Fine-tuning is a more advanced technique used for optimization or when other methods are insufficient.
    - Many users find success with base models and simpler approaches.
- **Question:** What are the advantages of fine-tuning OpenAI models compared to open-source models?
  - **Answer:**
    - **Advantages of OpenAI models:**
      - Access to state-of-the-art models like GPT-4 (for select partners).
      - Advanced features like tool calling, function calling, and the assistance API.
      - Simplified deployment and infrastructure management.
    - **Consider Open Source When:**
      - OpenAI's offerings don't meet specific needs or violate terms of service.
- **Additional Insights:**
  - OpenAI's models, even GPT-3.5, often outperform open-source alternatives, especially for complex tasks and at scale.
  - OpenAI's platform offers greater ease of use, especially with features like tool calling and handling API rate limits.



### Moving from Fine-tuning to Newer Models

- **Observation:** The release of more powerful base models sometimes leads users to abandon fine-tuning in favor of the improved base models. 
- **Factors Influencing the Decision:**
  - **Performance Difference:** If the new base model offers substantial improvement, switching might be preferable.
  - **Cost and Latency:** If the performance difference is minimal, sticking with a fine-tuned 3.5 model might be more cost-effective and faster. 



### Sustainability and Future of Fine-tuning

- **Concern:** Given Google's history of discontinuing products, is there a risk of OpenAI doing the same with fine-tuning?
- **Reassurance:**
  - OpenAI is committed to supporting applications that are successful with fine-tuning.
  - The company recognizes the investment users make in data preparation and training, and aims to avoid disruption.
  - **Key Takeaway:** OpenAI understands the higher switching cost associated with fine-tuning and aims to provide continued support.



### Data Ownership and Licensing for Fine-tuning

- **Question:** What happens to the data used for fine-tuning in terms of IP and licensing?
- **Answer:**
  - **Data Privacy:** OpenAI never uses customer data for training its foundation models. This is explicitly stated in their privacy policy and terms of service.
  - **Data Control:** Users have complete control over their data's lifecycle. They can delete it after fine-tuning or retain it for future models. 



### Language Model Agents and Function Calling Internals

- **Question:** Are there any impressive examples of OpenAI being used to create successful language model agents?
- **Answer:**
  - Steven defers to another team specializing in agents and tool calling.
  - **Personal Interest:** Steven finds the GitHub AI workspaces and their agent-like capabilities for coding tasks promising.
- **Question:** Is there tension between the abstraction provided by higher-level services (like the assistance API) and fine-tuning, especially regarding data access and transparency?
  - **Answer:**
    - Fine-tuned models can be used within the assistance API.
    - The assistance API primarily aids in context management and tool access, not necessarily replacing fine-tuning.
- **Follow-up Question:**  If the assistance API modifies or truncates context, wouldn't that impact fine-tuning?
  - **Answer:**
    - Steven acknowledges the potential issue but lacks specific details about the assistance API's internal context handling.
- **Question:** Regarding function calling, is it true that JSON schema definitions are translated into a Hyperscript-like format before processing? Can users leverage this for simpler function definitions?
  - **Answer:**
    - Steven confirms that all inputs, including JSON schemas, are converted into OpenAI's internal token format.
    - He can't share specifics about the internal representation or translation process.
    - **General Direction:** OpenAI aims to make the default function calling experience (using JSON schemas) the most effective, without requiring users to rely on workarounds or internal knowledge. 
