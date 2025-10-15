---
title: "Notes on ICML 2024 Tutorial: Physics of Language Models"
date: 2025-02-17
image: /images/empty.gif
hide: false
search_exclude: false
categories: [notes, llms]
description: "My notes from **Zeyuan Allen-Zhu's** ICML presentation, outlining a physics of language models framework using synthetic data, controlled experiments, and probing to reveal how LLMs learn, store, and manipulate knowledge, perform multi-step reasoning, and depend on training conditions for robust, generalizable capabilities."

twitter-card:
  creator: "@cdotjdotmills"
  site: "@cdotjdotmills"
  image: /images/default-preview-image-black.png
open-graph:
  image: /images/default-preview-image-black.png
---





* [Introduction](#introduction)
* [Part 3: Knowledge](#part-3-knowledge)
* [Part 2: Reasoning](#part-2-reasoning)
* [Part 1: Language Structures](#part-1-language-structures)



::: {.callout-tip title="Resource Links"}

* **YouTube Recording:** [ICML 2024 Tutorial: Physics of Language Models](https://www.youtube.com/watch?v=yBL7J0kgldU) 
* **Project Page:** [Physics of Language Models](https://physics.allen-zhu.com/home)
* **Speaker:** [Zeyuan Allen-Zhu](http://zeyuan.allen-zhu.com/)

:::



## Introduction

### Spectrum of "Theory" in Language Models

- The term "theory" in the context of language models encompasses a broad spectrum, ranging from rigorous mathematical proofs to empirical observations (ethology).
- **Mathematical Theory:** Involves proving theorems about learnability, often with idealistic assumptions and limited applicability to real-world, deep networks. Progress is slow.
- **Ethology (Animal Behavior Science):** Involves experimenting with large language models (LLMs) like GPT-4 through APIs, leading to discoveries like "chain of thought". Progress is very rapid.
- **Pros and Cons:**
  - **Mathematical Theory:**
    - **Pros:** Rigorous theorems.
    - **Cons:** Idealistic assumptions, shallow networks, slow progress, limited practical relevance.
  - **Ethology:**
    - **Pros:** Accessible to everyone, potential for significant discoveries (e.g., chain of thought).
    - **Cons:** Concerns about scientific rigor (data contamination, lack of control, model specificity, limited internal understanding).
- **Historical Context:** The slow, patient progress of scientific discovery in the past (e.g., Newton's laws building upon Kepler's laws, which in turn built upon Tycho Brahe's observations) contrasts sharply with the rapid pace of current AI development.
- **Analogy:** The analogy of Newton's laws to mathematical theory and Tycho's observations to ethology is not entirely accurate. There's a gap between simply observing LLM behavior and developing a true "physics" of language models.

### Concerns with Purely Ethological Approaches

1.  **Data Concerns:** Studying models trained on internet data may lack scientific rigor due to biases, bugs (e.g., parity check failures in GPT-4), and the need for controlled studies.
2.  **Model Specificity:** Observations might be specific to a particular model version (e.g., a bug in a specific GPT-4 release) and not generalizable.
3.  **Data Contamination:** Benchmarks like [GSM8K](https://huggingface.co/datasets/openai/gsm8k) can be compromised by unintentional data leakage (e.g., translating problems into other languages and posting them online).
4.  **Lack of Internal Understanding:** Observing external behavior reveals little about the internal workings and failure modes of LLMs. Geocentrism analogy: Observing the sun and moon's movement doesn't reveal the true heliocentric model.

### The Physics of Language Models: A Proposed Approach

- **Decomposition:** Break down "intelligence" into building blocks (language structures, knowledge, reasoning) and study them individually.
- **Synthetic Data:** Use controlled, idealized synthetic datasets to manipulate variables (difficulty, type, amount, format) and understand their impact.
- **Repeatability:** Focus on smaller models (e.g., 100 million parameters) to enable repeated, controlled experiments, which are infeasible with multi-billion parameter models. Universal laws can still be derived.
- **Probing:** Investigate the inner workings of language models to understand how they function.

### Presentation Structure

- The presentation covers three main parts, presented in reverse order:
  1.  **Language Structures:** How LLMs learn language structures, focusing on context-free grammars (CFGs) (joint work with Professor Yuanzhi Li).
  2.  **Reasoning:** How LLMs perform reasoning, specifically at the level of grade-school math (joint work with Tian Ye, Zicheng Xu, and Yuanzhi Li ).
  3.  **Knowledge:** How LLMs acquire and manipulate knowledge (joint work with Professor Yuanzhi Li).



## Part 3: Knowledge

::: {.callout-tip title="Resource Links"}

* **YouTube Recording:** [Physics of Language Models: Part 3.1 + 3.2, Knowledge Storage, Extraction and Manipulation](https://www.youtube.com/watch?v=YSHzKmEianc) 
* **Paper:** [Physics of Language Models: Part 3.1, Knowledge Storage and Extraction](https://arxiv.org/abs/2309.14316)
* **Paper:** [Physics of Language Models: Part 3.2, Knowledge Manipulation](https://arxiv.org/abs/2309.14402)
* **Paper:** [Physics of Language Models: Part 3.3, Knowledge Capacity Scaling Laws](https://arxiv.org/abs/2404.05405)

:::

### 3.1 Knowledge Extraction

#### Introduction

-   Problem: LLMs often fail simple knowledge manipulation tasks (e.g., parity checks on birth years, comparing celebrity birth dates).
-   Prerequisite to Studying Manipulation: Before assessing manipulation, it's crucial to determine if the model can even *extract* the relevant knowledge from its pre-training data. Can the model retrieve a celebrity's birth year?
-   **Controlled Experiments:** It is essential to conduct controlled experiments to determine the model's ability to:
    -   A. Extract knowledge.
    -   B. Avoid data contamination (e.g., the question being revealed through publication).
    -   C. Manipulate that knowledge.
    -   D. Understand the concepts needed for manipulation (e.g., even/odd).

#### Synthetic Biography Dataset

- **Data Generation:** Create synthetic biography data for *N* individuals, using sentence templates or LLMs.

  - **Example Biography:**

    > Anya Briar Forger was born on **October 2, 1996**. She spent her early years in **Princeton, NJ**. 
    > She received mentorship and guidance from faculty members at **MIT**. She completed her education with a focus on **Communications**. 
    > She had a professional role at **Meta Platforms**. She was employed in **Menlo Park, CA**.

- **Attributes:** Each person has six attributes: 

  1. birth date
  2. birth city
  3. university
  4. major
  5. employer
  6. work city

- **Question-Answer (QA) Data:** Generate six QA pairs per person, one for each attribute. This acts as instruction fine-tuning data.

  - **Example QA:**

    > **What is the birth date of Anya Briar Forger?** 
    > *Answer: October 2, 1996.*
    >
    > **Which university did Anya Briar Forger study?** 
    > *Answer: MIT.*
    >
    > **Which company did Anya Briar Forger work for?** 
    > *Answer: Meta Platforms.*
    >
    > **What is the birth city of Anya Briar Forger?** 
    > *Answer: Princeton, NJ.*
    >
    > **What major did Anya Briar Forger study?** 
    > *Answer: Communications.*
    >
    > **Where did Anya Briar Forger work?** 
    > *Answer: Menlo Park, CA.*


#### Experiment Setup

-   **Training/Test Split:** Reveal only half of the QA data during training.
-   **Out-of-Distribution Evaluation:** Evaluate the model on the remaining half of the individuals.
-   **Knowledge Extraction:** If the model performs well on the test set, it demonstrates *knowledge extraction* – generalizing the ability to answer questions to new individuals based on their biographies. Performance on the training set only demonstrates memorization.

#### Result: Mixed Training

-   **Mixed Training:** If biography data and QA data are mixed during pre-training, the model achieves high accuracy (`86.6%`) on out-of-distribution knowledge extraction.
-   **Practical Scenario (Not Mixed):** In practice, pre-training (e.g., on Wikipedia) and instruction fine-tuning are separate. This leads to very poor knowledge extraction.
-   **Universality:** This failure is independent of model size, architecture (GPT, GPT-2, LLaMA), data size, and training parameters. Over `500` experiments consistently showed near-`0%` accuracy.

#### Result: Knowledge Augmentation

-   **Catch:** The initial experiments used only *one* biography per person.
-   **Knowledge Augmentation:** Generate multiple biography entries per person, using different writing styles, permutations, or translations.
-   **Impact:** With knowledge augmentation (e.g., five biographies per person), accuracy dramatically increases (`96%)`.
-   **Conclusion:** Unless mixed training is used, knowledge augmentation is *absolutely necessary* for knowledge extraction.

#### Probing: Where and How is Knowledge Stored?

-   **Probing Technique:** Feed a pre-trained model (e.g., GPT-2) with a biography entry and examine the hidden states of the last layer.
-   **Focus:** Probe for specific knowledge (e.g., employer name) at different token positions.
-   **Observation (No Augmentation):** 
    -   Without knowledge augmentation, previous token positions (before the employer name) show near-zero probing accuracy. 
    -   The model learns the "wrong logic", storing information jointly with preceding values, rather than associating it directly with the key (person's name). 
        -   **Example:** The model may store that *someone* born on October 2nd, 1996, in Princeton, who studied communications at MIT works for Meta.

-   **Mathematical Form (No Augmentation):** `[value 5]` (employer) is stored in a tuple defined by the key *and* all preceding values.
-   **Observation (With Augmentation):** With knowledge augmentation, the model stores knowledge differently. The hidden state *immediately after the person's name* already encodes the employer name.
-   **Mathematical Form (With Augmentation):** `[value 5]` is directly stored with the key (person's name).
-   **Conclusion:** Knowledge augmentation changes how knowledge is stored, which in turn affects its extractability via instruction fine-tuning.

#### Result: Celebrity Helps Minorities

-   **Controlled Experiment:** Consider a dataset with celebrities (multiple biographies per person) and minorities (one biography per person).
-   **Training:** Pre-train on both groups, but fine-tune only on the celebrities' QA data.
-   **Observation:** Knowledge extraction accuracy for the *minorities* is high, even though they had no knowledge augmentation and weren't part of the fine-tuning data.
-   **Explanation (Probing):** The inclusion of celebrity data teaches the model to store knowledge in the correct format, benefiting even the minorities.
-   **"Donald Trump Effect":** The existence of multiple Donald Trump biographies improves LLMs' ability to extract knowledge about minorities.
-   **Conclusion:** Augmenting only *part* of the data (e.g., celebrities) can lead to knowledge extraction for *all* individuals.

#### Summary of 3.1

-   **Distinction:** There's a crucial difference between knowledge *storage* and knowledge *extraction*. Memorization doesn't guarantee extractability.
-   **Extractability Requirements:**
    -   Mixed training.
    -   Knowledge augmentation.
-   **Bidirectional Models (BERT, DeBERTa):** Fail at knowledge extraction even with mixed training and augmentation. ([paper](https://arxiv.org/abs/2309.14316))

### 3.2 Knowledge Manipulation

#### Introduction

-   **Assumption:** Assume knowledge is fully extractable (based on the findings of 3.1).
-   **Focus:** Study LLMs' ability to *manipulate* knowledge.
-   **Simplest Task:** Knowledge *classification* (e.g., classifying months into even/odd categories).

#### Knowledge Classification Experiment

- **Setup:** Pre-train on biographies, fine-tune to extract birth dates.

- **Classification Task:** Classify the 12 months into two categories (even/odd).

-   **With and Without Chain of Thought (COT):**
    - **Without COT:** Direct answer (yes/no).
    
      > Was Anya Briar Forger born in an even month? 
      > Answer (without CoT): **Yes**
    
    - **With COT:** Explicitly state the birth month, *then* answer yes/no.
    
      > Was Anya Briar Forger born in an even month? 
      > Answer (with CoT): **October**; so it is **Yes**
    
- **Fine-tuning:** Fine-tune sufficiently to achieve perfect accuracy on the training set.

-   **Out-of-Distribution Evaluation:** Evaluate on the remaining half of the individuals.

#### Result: COT is Crucial for Knowledge Manipulation

-   **Observation (Without COT):** Out-of-distribution accuracy is extremely low (near random guessing).
-   **Observation (With COT in Training):** Including COT in training *does not* improve accuracy during evaluation *without* COT.
-   **Conclusion:** Knowledge manipulation (even the simplest form) requires COT *both* during training and inference. The model must explicitly state the knowledge before manipulating it.
-   **Contrast with Reasoning:** This is different from reasoning tasks (e.g., adding small numbers), where LLMs can skip steps.
-   This is a statement only discoverable via controlled experiments.

#### Result: Knowledge Inverse Search is Impossible

- **Inverse Search Task:** Fine-tune the model to answer questions like "Who was born on [date] in [city] and works for [employer]?"

  > **Question**: **Who** was born on October 2, 1996, in Princeton, NJ, studied Communications at MIT, and worked for Meta Platforms at Menlo Park, CA?
  >
  > **Answer**: **Anya Briar Forger**

- **Out-of-Distribution Evaluation:** Evaluate on the remaining half of the individuals.

- **Observation:** 

  - Zero accuracy, regardless of model size, data size, training method (mixed training, fine-tuning, knowledge augmentation), or fine-tuning parameters. 
  - Hundreds of pre-training regimes were tested.

- **Exception:** Inverse search is only possible if knowledge is *already reversed* in the pre-training data (e.g., person's name at the end of the biography).

- **Paper on Knowledge Reversal:** 

  - A separate paper with Meta colleagues explores how to practically reverse knowledge. 
    - Reversal must happen in the pre-training phase. 
      - Changing to a bi-directional model (like BERT) does not solve this.

    - **Paper:** [Reverse Training to Nurse the Reversal Curse](https://arxiv.org/abs/2403.13799)

- **Conclusion:** Knowledge inverse search is generally impossible without pre-training data modification.

#### Connections to Practice

-   **Parity Checks and Ranking:** GPT-4 and LLaMA also fail at parity checks and ranking tasks (comparing celebrity birth dates) without COT. With COT, accuracy improves significantly.
-   **Chinese Idiom Task:** GPT-4 fails at filling in missing characters in Chinese idioms (a form of inverse search), demonstrating the practical limitations.
-   **Turing Test:** These failures can distinguish current AI models from humans, who can perform these tasks mentally without explicit statements.

#### Result (Skipped): Knowledge Partial Search

- Language models might be able to fully extract knowledge (e.g., birthday) but not the individual component words (like the birth *year*).
- Related to "multi-token prediction" work from Meta colleagues: Predicting multiple future tokens can change knowledge storage and improve capabilities.
  - **Paper:** [Better & Faster Large Language Models via Multi-token Prediction](https://arxiv.org/abs/2404.19737)


#### Summary of 3.2

- The model must state knowledge explicitly before manipulating it.
- Knowledge inverse search is impossible unless the knowledge is reversed in the pre-trained data.
- A concurrent work refers to this as the "reversal curse": If a model learns "A is B", it doesn't learn "B is A."
  - **Paper:** [Reverse Training to Nurse the Reversal Curse](https://arxiv.org/abs/2403.13799)


### 3.3 Scaling Laws for Knowledge Capacity

#### Introduction

-   **Goal:** Determine the relationship between model size and knowledge storage capacity.
-   **"Bit" Definition:** Information-theoretic bits in the dataset.

#### Measuring Information Bits in Synthetic Data

- **Random Generation:**

  > If birthdates are uniformly drawn from $( 12$ (months) $\times 28$ (days) $\times 200 (years) )$ possibilities, this is $\log_2(12 \times 28 \times 200) = 60.21$ bits.
  >
  > If cities are uniformly drawn from $300$ US cities, this is $\log_2(300) = 8.23$ bits.

- **General Formula:** A formula can be created to calculate the information content of any synthetic knowledge dataset, regardless of writing style variations.

  > **bioD**: a synthetic data with hyperparameters:
  >
  > - \( N \) — distinct names from \( N_0 \) possible names
  > - \( K \) — number of knowledge attributes
  > - \( T \) — vocabulary size
  > - \( C, L \) — values in \( C \) chunks, each of length \( L \)
  > - \( D \) — value has diversity \( D \)
  >
  > $\log_2 \binom{N_0}{N} + NKC \log_2 D + K \log_2 \binom{T^L}{D}$

#### Scaling Law Experiment

-   **Pre-training:** Pre-train a language model on synthetically generated knowledge data.
-   **Knowledge Measurement:** Calculate the amount of knowledge learned by the model (accounting for partial correctness).
-   **Major Discovery:** LLMs consistently achieve *two bits per parameter* in knowledge storage, if sufficiently trained.

#### Universality of the Two Bits Per Parameter Scaling Law

-   **Model Size, Depth, Width:** Holds for a wide range of model sizes, depths, and widths (as long as the transformer has at least two layers).
-   **Data Types:** Regardless of the specific parameters of the synthetic knowledge data.
-   **Rewriting:** Independent of how the data is rewritten.
-   **Training Parameters:** Holds for a wide range of training parameters.

#### Conjecture: 7 Billion Parameters for Human Knowledge

-   Based on an estimate of the information content of English Wikipedia and textbooks, a 7-billion parameter model should be sufficient to store all such knowledge.

#### Sufficient Training: 1000 Exposures

-   **Definition:** Each piece of knowledge needs to be exposed approximately 1000 times during pre-training to reach the two bits per parameter capacity.
-   **Exposure:** Doesn't mean 1000 training passes; it means the same knowledge, possibly in different writing styles, is seen 1000 times.
-   **Controlled Experiment:** If each piece of knowledge is exposed the same number of times (e.g., 1000), the two bits per parameter scaling law holds.
-   Fixing data size, increasing model size doesn't increase knowledge learned beyond the data's inherent information content. 
    -   Before that point, the model's knowledge capacity closely follows two bits per parameter.


#### Insufficient Training: Rare Knowledge

-   **100 Exposures:** If knowledge is exposed only 100 times (rare knowledge), the capacity decreases to approximately one bit per parameter.
-   **Architecture Differences:** With rare knowledge, differences between model architectures emerge.
    -   GPT-2: Performs better.
    -   LLaMA, Mistral: Perform worse (by a factor of 1.3).
-   **MLP Layers:** Reducing the size of GPT-2's MLP layers doesn't significantly affect capacity, but *removing* them does.
-   **Disclaimers:** This comparison is *only* for knowledge capacity and *only* for rare knowledge.

#### Gated MLP is the Culprit

-   **Controlled Experiment:** By systematically comparing GPT-2 (rotary version) and LLaMA (which have several architectural differences), it's found that the [*gated MLP*](https://arxiv.org/abs/2105.08050) in LLaMA is responsible for the reduced knowledge capacity.
-   **Fix:** Replacing LLaMA's gated MLP with a standard MLP restores the one bit per parameter capacity (a `30%` improvement).

#### Result: Mixed Quality Data

-   **Controlled Experiment:** Compare training on:
    -   Scenario 1: Only "good" data (rich in knowledge, 100 exposures per piece).
    -   Scenario 2: "Good" data (100 exposures) *and* "bad" data (junk data).
-   **Observation:** A *20-fold* difference in the amount of "good" knowledge stored. The mere presence of junk data significantly harms the LLM's ability to learn from the good data.
-   Increasing training time on the "good" data in Scenario 2, does *not* fully compensate for the harm caused by the junk data.

#### Solution: Domain Tokens

-   **Technique:** Prepend each piece of pre-training data with a domain token (e.g., the domain name or URL).
-   **Impact:** Significantly mitigates the negative impact of junk data.
    -   20x worse becomes 10x worse.
    -   3x worse becomes fully restored.
-   **Mechanism:** LLMs automatically learn to prioritize high-quality domains without explicit instruction.

#### Summary of 3.3

-   **Sufficient Training:** Two bits per parameter capacity, regardless of architecture.
-   **Insufficient Training (Rare Knowledge):** Architecture matters; GPT-2's standard MLP outperforms LLaMA's gated MLP.
-   **Quantization and MOEs (Skipped):** Int8 quantization maintains the two bits per parameter capacity (4:1 compression ratio).
-   **Mixed Quality Data:** Domain tokens are crucial for mitigating the negative impact of junk data.

### Reflection on the "Physics of Language Models" Approach (End of Part 3)

-   **Knowledge Focus:** Part 3 focused *solely* on knowledge, using synthetic data.
-   **Small Models:** 
    -   Most results are replicable with 100-million parameter models, enabling extensive controlled experiments (data variations, training process tweaks, architecture modifications). 
    -   An H100 can pretrain in a day. 
    -   Eight V100s can pretrain in a day. 
    -   Even scaling down the synthetic data by 5x maintains the validity of the results.

-   **Probing:** All statements are supported by probing, revealing the internal workings of the models.



## Part 2: Reasoning

::: {.callout-tip title="Resource Links"}

* **YouTube Recording:** [Physics of Language Models: Part 2.1, Grade-School Math and the Hidden Reasoning Process](https://www.youtube.com/watch?v=bpp6Dz8N2zY)
* **YouTube Recording:** [Physics of Language Models: Part 2.2, How to Learn From Mistakes on Grade-School Math Problems](https://www.youtube.com/watch?v=yBgxxvQ76_E&list=PLIZhMKKbVX6JmdngPRKvAS4u4L97odbGp&index=5) 
* **Paper:** [Physics of Language Models: Part 2.1, Grade-School Math and the Hidden Reasoning Process](https://arxiv.org/abs/2407.20311)
* **Paper:** [Physics of Language Models: Part 2.2, How to Learn From Mistakes on Grade-School Math Problems](https://arxiv.org/abs/2408.16293)
* **GitHub Repository:** [iGSM](https://github.com/facebookresearch/iGSM)

:::

### 2.1 Hidden Reasoning Process

#### Introduction

-   **Focus:** Reasoning at the level of grade-school math.
-   **Goal:** Understand the hidden reasoning process of LLMs, their mental processes, and why they make mistakes.
-   **Synthetic Math Dataset:** Create a synthetic math dataset that simulates GSM8K.
-   **Probing:** All statements are supported by probing.

#### Limitations of Existing Approaches

-   **GSM8K:** Too small for thorough analysis.
-   **GPT-4 Augmentation:** Using GPT-4 to generate similar problems from GSM8K leads to biased and limited data, lacking hard problems.

#### Assumptions for the Synthetic Math Dataset

-   **GitHub Repository:** [iGSM](https://github.com/facebookresearch/iGSM)
-   **Direct Pre-training:** Design the dataset for direct pre-training, removing common-sense elements from GSM8K (e.g., that a burning candle shrinks).
-   **Key Elements to Keep:**
    -   **Direct Dependency:** Relationships between parameters (e.g., one parameter is the sum of two others).
    -   **Instant Dependency:** Relationships like "X classrooms with Y bags each have X\*Y bags".
    -   **Implicit Dependency:** Relationships like "Bob has three times more fruits than Alice, then X are not fruits".

#### Data

-   **Structure Graph:** Defines the possible parameters (e.g., number of film studios in a school).
-   **Dependency Description:** Each sentence in the problem description specifies a dependency between parameters, represented as a directed acyclic graph (DAG).
-   **Solution:** A chain-of-thought solution, step-by-step computation from the leaves of the DAG to the final question.
-   **Modular Arithmetic (Mod 23):** All calculations are performed modulo 23 to focus on reasoning, not arithmetic skills.
-   **Shuffled Sentences:** Problem description sentences are randomly shuffled.
-   **Number of Operations (OP):** A key parameter representing the difficulty of the reasoning problem (number of steps in the solution).

#### Pre-training and Testing

-   **Data Families:**
    -   **Medium:** Problems with OP ≤ 15.
    -   **Hard:** Problems with OP ≤ 21.
-   **Solution Templates:**
    -   Medium: At least 7 billion solution templates.
    -   Hard: At least 90 trillion solution templates.
-   **Pre-training:** Train a language model (e.g., GPT-2) on this data.
-   **Testing:**
    -   **In-Distribution:** Test on problems of the same difficulty.
    -   **Out-of-Distribution:** Test on *harder* problems.
-   **Observation:** LLMs can generalize out-of-distribution.

#### Claim: LLMs Learn Reasoning Skills

-   LLMs are capable of learning reasoning skills, not just memorizing solution templates. This is demonstrated by out-of-distribution generalization.

#### What Skills Did They Learn?

-   **Level 0 Reasoning:** Brute-force computation of all possible parameters.
-   **Level 1 Reasoning:** Topological sort, ignoring unnecessary parameters.
-   **Discovery:** LLMs learn *level 1 reasoning*, producing the shortest solutions almost always.
-   It is a difficult task for the model to understand which parameters are neccessary *before* it even generates the first token of the solution. Chain of thought is not simply breaking down problems; it requires mental processing *before* the first step.

#### Probing: Mental Pre-computation

-   **Probing Tasks:**
    -   Before solution generation: Does the model know if a parameter is necessary?
    -   Between solution sentences: Does the model know which parameters can be computed next?
    -   Before the question is asked: Does the model know the dependencies between parameters?
-   **Observation:** The model has mentally computed all of these with `>99%` accuracy.
-   **Level 1 Reasoning Mechanism:** The model knows necessary parameters and computable parameters; the logical AND of these determines the next step, leading to the shortest solution.

#### Level 2 Reasoning: Beyond Humans

-   **Surprising Finding:** The GPT-2 model also pre-computes dependencies for *unnecessary* parameters. It learns the all-pairs dependency graph even before the question is asked.
-   **AGI Signal:** This is a preliminary signal of generalization (the "G" in AGI) – learning skills not explicitly taught in the training set. This ability is crucial for future fine-tuning on other tasks.

#### How LLMs Make Mistakes

-   **Two Types of Mistakes:**
    1.  Computing unnecessary parameters (rare, but occurs with extremely hard problems).
    2.  Getting stuck because a defined parameter isn't ready for computation.
-   **Correlation with Probing:**
    -   Mistake Type 1: High correlation with the model *wrongly* believing a parameter is necessary *before* generation. 
        -   Some mistakes are *systematic*, not due to generation randomness.
    -   Mistake Type 2: Correlation with the model believing a parameter is ready to compute when it's not.
-   **Improving Reasoning:** Improving the model's mental computation of the "`can_next`" quantity (parameters ready for computation) is crucial.

#### Scaling Laws: Depth Matters for Reasoning

-   **Contrast with Previous Findings:** Unlike knowledge capacity (where only size matters), *depth* matters significantly for reasoning.
-   **Experiment:** Compare a tall, skinny model (smaller) with a shallow, wide model (larger). The tall model performs much better on reasoning tasks.
-   **Explanation (Probing):** 
    -   The accuracy of probing the "necessary" parameter decreases with the distance of the parameter from the question. 
    -   Deeper networks are needed for longer reasoning chains. 
        -   This cannot be compensated by the use of chain of thought.

-   Even before using chain of thought, mental thinking is neccessary to decide what to compute *first*. This requires depth.

#### Summary of 2.1

-   Synthetic math dataset to simulate GSM8K.
-   LLMs exhibit level 2 reasoning (beyond human capabilities).
-   Probing reveals how models reason and make mistakes.
-   Model depth is crucial for reasoning due to mental computation.
-   GPT-4o, even today, likely cannot perform > 10 step reasoning, which means that synthetic math data may be neccessary to improve reasoning.

### 2.2 Learning from Mistakes

#### Introduction

-   **Discovery:** LLMs often *know* they have made mistakes.

#### Regretful Behavior

-   **Mistake Type:** The model starts to compute a parameter but then realizes it's not ready.
-   **Probing:** Probing at the point of the mistake reveals the model's internal state shows "regret" – it wants to go back.

#### Experiment: Allowing the Model to Go Back

-   **Error Detector:** A model pre-trained on correct data can act as an error detector (through probing or fine-tuning).
-   **Assisted Generation:** Use the error detector to trigger backtracking during generation.
-   **Result:** Only a small improvement (`2%`).
-   **Drawbacks:**
    -   Requires two models (generator and detector).
    -   Limited improvement because it relies on *randomness* for correction (regeneration), similar to beam search (which gives zero improvement).

#### Pre-training with Mistakes and Corrections

- **Data Modification:** Introduce mistakes (with probability *p*) and corrections ("`[BACK]`" token) into the synthetic math dataset.

- **Autoregressive Training:** The model still uses autoregressive language modeling; it sees its previous mistakes.

- **Result:** Significant accuracy gain.

  > 78% :arrow_right: **95%** (med, op=23)
  >
  > 84% :arrow_right: **96%** (hard, op=32)

#### Properties of Training with Mistakes

- **Higher $p$ is Better:** More mistakes during training lead to better performance.

  | $p$    | 0.05 | 0.1  | 0.2  | 0.5  |
  | ------ | ---- | ---- | ---- | ---- |
  | Medium | 78%  | 84%  | 91%  | 92%  |
  | Hard   | 84%  | 89%  | 88%  | 93%  |

- **No Inference-Time Mistakes:** Even with high *p*, the model doesn't make more mistakes during inference (due to temperature 0 or beam search).

- **No Label Masking Needed:** Label masking (preventing the model from learning from mistakes) is unnecessary.

-   **Shortest Solutions:** The model still generates the shortest solutions (level 1 and 2 reasoning).

#### Pre-training is Crucial

-   **Fine-tuning Fails:** Fine-tuning a model (pre-trained on correct data) with mistake/correction data does *not* improve performance. Error correction is a much harder skill than error detection and must be learned during pre-training.

#### Generating Fake Mistakes in Practice

- **Dumber Idea (Works):** Create fake mistakes by inserting a *future* sentence from the solution into an earlier position.

  > 78% :arrow_right: **91%** (med, op=23) 
  > 84% :arrow_right: **92%** (hard, op=32)

- **Smarter Idea (Doesn't Work):** Create fake mistakes by inserting a random unused problem parameters.

  > 78% :arrow_right: **87%** (med, op=23) 
  > 84% :arrow_right: **87%** (hard, op=32)

- **Conclusion:** The dumber, cheaper method is more effective.

-   **Slogan:** "Pre-train with fake mistakes and no more regret."

#### Summary of 2.2

-   LLMs exhibit regret when making mistakes.
-   Pre-training with mistakes and corrections is crucial for learning error correction.
-   Fine-tuning and beam search are insufficient.
-   Fake mistakes can be easily generated and are effective.

### Reflection on the "Physics of Language Models" Approach (End of Part 2)

-   **Reasoning Focus:** Part 2 focused solely on reasoning, using synthetic data.
-   **Small Models:** 100-million parameter models were sufficient.
-   **Controlled Experiments:** Manipulated data difficulty, mistake types, and training processes.
-   **Probing:** Used probing to understand reasoning, mistakes, and the relationship between model depth and reasoning length.



## Part 1: Language Structures

::: {.callout-tip title="Resource Links"}

* **YouTube Recording:** [Physics of Language Models: Part 1, Context-Free Grammar](https://www.youtube.com/watch?v=kf_eGgVtOcs) 
* **Paper:** [Physics of Language Models: Part 1, Learning Hierarchical Language Structures](https://arxiv.org/abs/2309.14316)

:::

### Introduction

-   **Two Goals:**
    1.  **Interpretation Beyond Tokens:** Provide precise interpretations of how LLMs learn non-trivial, hierarchical algorithms, going beyond simple token-level interpretations (like induction heads).
    2.  **Learning Language Structures:** Understand how LLMs learn complex language structures, addressing the question of "format learning" (hallucination).
        * hallucination (learn "format" faster than "task")

### Context-Free Grammars (CFGs)

-   **Approach:** Study how LLMs learn CFGs, using synthetic CFGs that are intentionally difficult.
-   **CFG Generation:** Generate sentences from a CFG tree by recursively applying rules, starting from the root.
-   **Synthetic CFG Design:**
    -   Small vocabulary size (e.g., 1, 2, 3) to make local parsing difficult.
    -   Large number of possible sentences (e.g., 10^80) to prevent memorization.
-   **CFGs vs. English Grammar:** Synthetic CFGs are much harder than English grammar, requiring dynamic programming for parsing (not just greedy approaches).

### Experiment: Pre-training on CFG Data

-   **Models:**
    -   GPT (vanilla, absolute positional embedding).
    -   GPT ([rotary embedding](https://arxiv.org/abs/2104.09864)).
    -   GPT (relative attention).
    -   "GPT Stupid" (uniform attention with exponentially increasing spans).
-   **Metrics:**
    1. Accuracy (generating valid sentences from a valid prefix)
    2. Diversity
    3. Distribution difference (KL divergence)
-   **Observation:**
    -   Relative attention and rotary embedding GPTs perform well.
    -   Vanilla GPT performs poorly.
    -   "GPT Stupid" performs surprisingly well.

### Conclusion: Importance of Relative Attention

-   Strong connection between rotary embedding/relative attention and the ability to learn language structures.
-   Rotary embedding is preferred in practice (LLaMA, Mistral) for efficiency, but relative attention is slightly better.
-   "GPT Stupid" demonstrates that even uniform attention with varying spans is beneficial, suggesting that future attention-free models should incorporate this concept.

### How LLMs Learn CFGs: Probing

-   **Hidden CFG Trees:** The model doesn't see the underlying CFG tree, only the generated sentences.
-   **Probing:** Does the model secretly learn to parse the CFG trees? Are the parsing trees encoded in the hidden embeddings?
-   **Answer:** Yes, the model learns the CFG trees, and the information is stored *locally* in the hidden states. The information about each subtree is linearly encoded around its ending position.
-   **BERT Doesn't:** BERT (encoder-based models) *do not* learn the CFGs in this way. Masked language modeling is an easier task than language modeling, not requiring full parsing.

### Dynamic Programming (DP)

-   **Human Parsing:** Humans use dynamic programming to parse CFGs.
-   **DP States:** `DP(i,j,a)` represents whether symbol `a` can generate the subsequence from `i` to `j`.
-   **DP Transition Functions:** Connect DP states to determine larger subtrees.
-   **Observation (Probing):**
    -   DP states are locally stored in the hidden states.
    -   Attention patterns in the transformer precisely serve as DP transition functions.

### Two Levels of Dynamic Programming

-   **Parsing DP:** Determining if a symbol can generate a subsequence.
-   **Generation DP:** Determining the next token and its probability given a prefix. This requires another, less-known level of dynamic programming.
-   **Observation (Probing):** Both levels of DP (states and transition functions) are present in the trained transformer.

### Summary of Part 1

-   GPTs can learn long, synthetic CFGs, requiring non-trivial planning and dynamic programming (harder than topological sort).
-   Probing reveals DP states in hidden states and DP transition functions in attention mechanisms.
-   BERT doesn't learn CFGs in the same way; language modeling is a harder task.
-   GPTs can learn implicit/corrupted CFGs (details in the [paper](https://arxiv.org/abs/2309.14316)).
-   The dynamic programming used is non-trivial, unknown to many software engineers and interview candidates. This surpasses the speaker's abilities at age 17. GPT-4 is likely to perform well, but it has seen dynamic programming in training materials. However, GPT-2 has learned this *without* seeing any definitions of dynamic programming.

### Final Thoughts: Future Science

-   **Synthetic Data:** Synthetic data is becoming increasingly important as real-world data becomes exhausted.
-   **GPT-5/GPT-6:** To surpass current limitations (e.g., GPT-4's reasoning limit), synthetic data will be necessary.
-   **Research Questions:** What are the optimal formats for synthetic data to maximize knowledge acquisition and reasoning abilities?
-   **AGI:** This research is crucial for building language models that approach AGI.







{{< include /_about-author-cta.qmd >}}
