---
title: "Conference Talk 7: Best Practices For Fine Tuning Mistral"
date: 2024-7-18
image: /images/empty.gif
hide: false
search_exclude: false
categories: [notes, llms]
description: "In this talk, **Sophia Yang** from Mistal AI covers best practices for fine-tuning Mistral language models. It covers Mistral's capabilities, the benefits of fine-tuning over prompting, and provides practical demos using the Mistral Fine-tuning API and open-source codebase."

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





* [Mistral Overview](#mistral-overview)
* [Customization](#customization)
* [Demos](#demos)





## Mistral Overview

* **[Mistral AI](https://mistral.ai/):** Paris-based team (50+ people) specializing in large language models (LLMs).
* **Model Timeline:**
  * **Sept 2023:** Mistral 7b released
    * **Blog Post:** [Mistral 7B](https://mistral.ai/news/announcing-mistral-7b/)
  * **Dec 2023:** Mistral 8x7b, Mistral Medium (commercial), API platform launched
    * **Blog Post:** [Mixtral of experts](https://mistral.ai/news/mixtral-of-experts/)
  * **Feb 2024:** Mistral Small and Mistral Large (flagship) released
    * **Blog Post:** [Au Large](https://mistral.ai/news/mistral-large/)
  * **Feb 2024:** Le Chat - free conversational AI interface launched
    * **Blog Post:** [Le Chat](https://mistral.ai/news/le-chat-mistral/)
  * **Apr 2024:** Open-source 8x22b model released
    * **Blog Post:** [Cheaper, Better, Faster, Stronger](https://mistral.ai/news/mixtral-8x22b/)
  * **May 2024:** Codestral - specialized model for code generation (80+ languages)
    * **Blog Post:** [Codestral: Hello, World!](https://mistral.ai/news/codestral/)
    * **LangChain Tutorial:**  [Self-correcting code assistants with Codestral](https://www.youtube.com/watch?v=zXFxmI9f06M) 
* **Model Offerings:**
  * **Open-source (Apache 2 License):** Mistral 7b, 8x7b, 8x22b
    * **Homepage:** [Open source models](https://mistral.ai/technology/#models)
    * **Docs:** [Open-weight models](https://docs.mistral.ai/getting-started/open_weight_models/)
  * **Enterprise-Grade:** Mistral Small, Mistral Large (supports fine-tuning)
  * **Specialized:** Codestral for coding, Embedding model
* **Fine-Tuning:**
  * **Blog Post:** [My Tailor is Mistral](https://mistral.ai/news/customization/)
  * **GitHub Repository:** [mistral-finetune](https://github.com/mistralai/mistral-finetune)




## Customization

* **Blog Post:** [My Tailor is Mistral](https://mistral.ai/news/customization/)
* **GitHub Repository:** [mistral-finetune](https://github.com/mistralai/mistral-finetune)
* **Documentation:** [Model customization](https://docs.mistral.ai/getting-started/customization/)
* **Developer Examples:** [Model customization](https://docs.mistral.ai/getting-started/stories/)

### Benefits of Prompting

* **Documentation:** [Prompting Capabilities](https://docs.mistral.ai/guides/prompting_capabilities/)
* **Out-of-the-box functionality:** No data or training required.
* **Easy updates:** Adaptable to new workflows and prototyping. 

### Benefits of Fine-Tuning

* **Guide:** [Fine-tuning](https://docs.mistral.ai/guides/finetuning/)
* **Performance:** Often outperforms prompting and even larger models.
* **Efficiency:** Faster and cheaper than using large prompts.
* **Task Alignment:** Tailored to specific tasks and behaviors.
* **Knowledge Integration:**  Ability to teach new facts and information.





## Demos

* **GitHub Repository:** [Mistral Cookbook](https://github.com/mistralai/cookbook)
* **GitHub Repository:** [Mistral Inference](https://github.com/mistralai/mistral-inference)



### Mistral API

* **GitHub Repository:** [mistralai/client-python](https://github.com/mistralai/client-python)

* ```sh
  pip install -U mistral-api>=0.4.2
  ```



::: {.callout-note title="Model Name Structure"}

* Fine-tuned model names have a specific structure:

  * ```text
    ft:open-mistral-7b:b6e34a5e:20240531:a29e61db
    ```

* `ft`: Indicates a fine-tuned model.
* `open-mistral-7b`: Specifies the base model used.
* `b6e34a5e:20240531:a29e61db`: Represents the specific fine-tuned version.

:::



###  Abstract Generator Demo

* Generates abstracts based on inputted research paper titles. 

* It was trained on title-abstract pairs from [arxiv.org](https://arxiv.org/).

* Highlights how effective fine-tuning can be for specific tasks.

* ::: {.callout-note title="Example"}

  ```python
  client.chat(
    model='ft:mistral-small-latest:8e2706f0:20240604:d861257a',
    messages=[
      {"role": "user", "content": "Fine-tuning is all you need"}
    ],
    temperature=0.2,
  ).choices[0].message.content
  
  ```

  ```text
  'Fine-tuning is a common practice in deep learning, where a pre-trained model is fine-tuned on a downstream task.\nIn this paper, we show that fine-tuning is all you need for a wide range of tasks.\nWe propose a simple yet effective fine-tuning recipe that works well for both vision and language tasks.\nOur recipe includes a learning rate warmup stage, a cosine learning rate decay schedule, and a stochastic weight averaging (SWA) stage.\nWe show that our recipe can significantly improve the performance of fine-tuning on a wide range of tasks, including image classification, object detection, instance segmentation, semantic segmentation, and language understanding.\nWe also show that our recipe can improve the performance of fine-tuning on small datasets, where the performance of fine-tuning is usually worse than training from scratch.\nOur recipe is simple and easy to implement, and we hope it will be useful for the deep learning community.'
  ```

  :::



### Medical Chatbot Demo

* Trained on the HuggingFace dataset for AI medical chatbots. 
  * **HuggingFace Dataset:** [ruslanmv/ai-medical-chatbot](https://huggingface.co/datasets/ruslanmv/ai-medical-chatbot)

* ::: {.callout-note title="Example:"}

  ```python
  client.chat(
    model='ft:open-mistral-7b:b6e34a5e:20240531:a29e61db',
    messages=[
      {"role": "user", "content": "Hello doctor, My reverse elbow armpits have developed a darker (my skin color is fair) pigmentation. This pigmentation has also affected the whole of my ..."}
    ],
    temperature=0.2,
  ).choices[0].message.content
  ```

  ```text
  'Hi, It seems that you might be having some fungal infection. Apply clotrimazole cream locally. Take tablet fluconazole 150 mg once a week for three weeks. Keep local part clean and dry. Avoid oily and spicy food. Ok and take care.'
  ```

  :::



### News Article Stylist (Economist Style Guide) Demo

* Showcases how to generate training data using a larger model (e.g., Mistral Large) when you don't have an existing dataset. 

* **Process:**

  1. **Define Prompt:** "You are a news article stylist following the Economist style guide."

  1. **Generate Data:** Use Mistral Large to rewrite news articles in the Economist style, providing guidelines and examples.

  1. **Fine-tune:** Train a smaller model (e.g., Mistral 7B) on the generated data.


* ::: {.callout-note title="Example"}

  

  ```python
  news = "Incoming Florida Senate President Bill Galvano named the Naples Republican the Senate's majority leader for the upcoming legislative session. Kathleen Passidomo was unimpressed ..."
  response = client.chat(
    model='ft:mistral-small-latest:b6e34a5e:20240604:ee1ab18b',
    messages=[
      {"role": "user", "content": news}
    ],
    temperature=0.2,
  )
  print(response.choices[0].message.content)
  
  ```

  ```text
  Kathleen Passidomo, a Naples Republican, has been named the Senate's majority leader for the upcoming legislative session by incoming Florida Senate President Bill Galvano. Passidomo was impressed with the appointment because of her good rapport with both Senate Republicans and Democrats and her ability to build consensus. Passidomo agrees that her people skills will be key in shaping the Senate agenda for the coming session, which begins on March 5. Committee meetings will start on December 11. Although the legislative priorities will be the same as last year, including water quality and mental health. In a message to other senators, Galvano praised Passidomo's ability to not only extract favorable consideration from other senators but to maintain the respect of the members. Committee chairmen recognize leadership's ability to guide the Senate to a top spot now sets Passidomo up for even greater things in the future. Assuming she keeps her seat in the 2022 election and Republicans maintain control of the Senate in ...
  ```

  

  ```python
  prompt = "Please carefully review the provided news article and identify all stylistic issues that detract from its clarity, precision, and sophistication. To ensure the accuracy of your analysis, quote the specific text and explain the reason for your suggested improvement."
  response = client.chat(
    model='ft:mistral-small-latest:b6e34a5e:20240604:ee1ab18b',
    messages=[
      {"role": "user", "content": prompt+news}
    ],
    temperature=0.2,
  )
  print(response.choices[0].message.content)
  ```

  ```text
  1. The article starts with a repetitive sentence, "Incoming Florida Senate President Bill Galvano named the Naples Republican the Senate's majority leader for the upcoming legislative session." This can be simplified to avoid repetition.
  2. The use of informal language, such as "you can put my office in the men's room for all I care," detracts from the article's professional tone.
  3. The article uses indirect speech in some instances, such as "What he was saying is that he was naming her senate majority leader," which can be replaced with direct speech for clarity and conciseness.
  4. The sentence "Galvano says he picked Passidomo for the post because she has a good rapport with both Senate Republicans and Democrats and is a consensus builder" can be rephrased to avoid redundancy.
  5. The use of parentheses in the sentence "Most of it (the job) is member relations, helping members get their bills heard" can be avoided by rephrasing the sentence.
  6. The sentence "The Senate majority leader only gets one vote, just like other senators, so what's the upside for constituents?" can be rephrased to avoid the use of a question and make it more assertive.
  7. The sentence "That table includes the Senate president, president pro tempore, in this case Sen. David Simmons, Passidomo as majority leader and perhaps one or two others" can be rephrased for clarity.
  8. The sentence "Traditionally, legislators in leadership positions don't file as many bills as rank and file members" can be rephrased to avoid the use of "don't" and make it more assertive.
  9. The sentence "Some recent predecessors in the post haven't filed any" can be rephrased to avoid the use of "haven't" and make it more assertive.
  10. The sentence "In a message to other senators, Galvano praised Passidomo's efforts on the latter" can be rephrased to avoid the use of "the latter" and make it more precise.
  11. The sentence "The role of the Senate majority leader isn't to extract favorable consideration from other senators" can be rephrased to avoid the use of "isn't" and make it more assertive.
  12. The sentence "Being named to a top spot now sets Passidomo up for even greater things in the future" can be rephrased to avoid the use of "sets up" and make it more assertive.
  13. The sentence "Assuming she keeps her seat in the 2022 election and Republicans maintain control of the Senate in 2022 and beyond, she could be Senate president herself in the session that follows" can be rephrased for clarity and conciseness.
  14. The sentence "Passidomo said she was taken by surprise by her sudden promotion" can be rephrased to avoid the use of "was taken by surprise" and make it more assertive.
  15. The sentence "Third floor, to be exact" can be avoided as it does not add any significant information to the article.
  ```

  :::



### Mistral Fine-tune API Walkthrough

* **Documentation:** [https://docs.mistral.ai/guides/finetuning/](https://docs.mistral.ai/guides/finetuning/)
* **Jupyter Notebook:** [mistral/fine_tune/mistral_finetune_api.ipynb](https://github.com/mistralai/cookbook/blob/main/mistral/fine_tune/mistral_finetune_api.ipynb)

1. **Data Preparation:**
   * **Format:** Data should be in JSON format.
     * **Size Limits:**
       * Training data: Each file <= 512 MB (multiple files allowed).
       * Evaluation data: Total size <= 1 MB.
   * **Reformatting:** Use provided scripts to adapt data from sources like HuggingFace.
     * **GitHub:** [mistral-finetune/utils/reformat_data.py](https://github.com/mistralai/mistral-finetune/blob/main/utils/reformat_data.py)
   * **Validation:** The `mistral-finetune` repository includes a data validation script.
     * **GitHub:** [mistral-finetune/utils/validate_data.py](https://github.com/mistralai/mistral-finetune/blob/main/utils/validate_data.py)
2. **Uploading Data:**
   * **Documentation:** [Upload dataset](https://docs.mistral.ai/guides/finetuning/#upload-dataset)
   * Use the `files.create` function, specifying file name and purpose ("fine-tune").
3. **Creating a Fine-tuning Job:**
   * Provide file IDs for training and evaluation data.
   * Choose the base model (Mistral 7B or Mistral Small).
   * Set hyperparameters (e.g., learning rate, number of steps).
4. **Monitoring Progress:**
   * Retrieve job status and metrics using the job ID.
5. **Using the Fine-tuned Model:**
   * Access the fine-tuned model using the provided model name (retrieved from the completed job).
6. **Weight & Biases Integration (Optional)**:
   * Configure API key for tracking metrics and visualizations.



### Getting Started Fine-Tuning Mistral 7B (Local)

* **Jupyter Notebook:** [tutorials/mistral_finetune_7b.ipynb](https://github.com/mistralai/mistral-finetune/blob/main/tutorials/mistral_finetune_7b.ipynb)

* Covers fine-tuning Mistral 7B

**Steps:**

1. **Clone Repository:** `git clone https://github.com/mistralai/mistral-finetune.git`
2. **Install Dependencies:** Follow [instructions](https://github.com/mistralai/mistral-finetune?tab=readme-ov-file#mistral-finetune) in the repository.
3. **Download Model:** Download the desired Mistral model (e.g., 7Bv3).
4. **Prepare Data:** Similar to the API walkthrough.
5. **Configure Training:**
   * Use a configuration file (`.yaml`) to specify data paths, model parameters, and hyperparameters.
   * Adjust sequence length based on available GPU memory.
6. **Start Training:** Execute the training script.
7. **Inference:**
   * Utilize the `mistral-inference` package.
   
     * **GitHub Repository:** [mistral-inference](https://github.com/mistralai/mistral-inference)
     * ```sh
       pip install mistral-inference
       ```
   * Load the tokenizer, base model, and fine-tuned LoRA weights.
   * Generate text. 













{{< include /_about-author-cta.qmd >}}
