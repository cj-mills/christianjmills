---
title: "Conference Talk 20: Back to Basics for RAG"
date: 2024-8-30
image: /images/empty.gif
hide: false
search_exclude: false
categories: [notes, llms]
description: "In this talk, **Jo Kristian Bergum** from **Vespa.ai** explores practical strategies for building better Retrieval Augmented Generation (RAG) applications, emphasizing the importance of robust evaluation methods and understanding the nuances of information retrieval beyond simple vector embeddings."

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





::: {.callout-tip title="Presentation Resources:"}

* **Slides:** [Back to basics?](https://docs.google.com/presentation/d/19L2j2-fBC_iPGswwER3Cfsy7N3HSqUrVfxh2xvcBU2Y/edit#slide=id.p)

:::



### About Jo Kristian Bergum

* **Distinguished Engineer** at Vespa.ai
* 18 years at Vespa.ai, 20 years in search and recommendation.
* **[Vespa.ai](https://vespa.ai/):**
  * Serving platform spun out of Yahoo.
  * Open source since 2017.
  * **Blog:** [https://blog.vespa.ai/](https://blog.vespa.ai/)
* Active on Twitter ([@jobergum](https://x.com/jobergum)), enjoys posting memes.



### Talk Overview

* **Stuffing Text into Language Model Prompts:** Using RAG beyond question answering, e.g., for classification by retrieving relevant training examples. 
* **Information Retrieval (The R in RAG):** Exploring the core concepts of retrieval and its importance in RAG pipelines.
* **Evaluation of IR Systems:**
  * Building your own evaluation systems to measure and improve search performance.
  * Demonstrating the impact of changes to your CTO.
* **Representational Approaches for IR:**
  * Discussing **sparse** and **dense** representations (BM25, vectors, embeddings).
  * Examining baselines for comparison.



### Demystifying RAG

* **RAG (Retrieval Augmented Generation):** A technique for enhancing language model outputs by retrieving relevant context from external knowledge sources.
* **Common Use Cases:** Question answering, chatbots, generating grounded responses.
  * ::: {.callout-note title="Example:"}

    **Current date is {date}, Don’t be rude. I’ll tip $5. Think step-by-step.**

    I want you to classify the text input as positive, negative or neutral. Examples:

    **Input:** I’m very happy today  
    **Output:** positive

    **Input:** I’m sad today  
    **Output:** negative

    **Input:** I don’t know what to feel today  
    **Output:** neutral

    ✨ **{many_retrieved_context_sensitive_examples}** ✨

    **Input:** {input}  
    **Output:** 

    :::
  * ::: {.callout-note title="Example:"}

    **Current date is {date}, Don’t be rude. I’ll tip $5. Think step-by-step.**

    I want you to summarize and answer the question using context retrieved by a search engine.  
    **Context:** [1] BERT: Pre-training of deep bidirectional transformers for language understanding...  
    **Question:** What is a bidirectional transformer model?  
    **Helpful answer:** bidirectional means that tokens attend to all other tokens in the input sequence [1].

    ✨ **{retrieved_context_sensitive_examples}** ✨

    **Context:** {retrieved_context_question}  
    **Question:** {question}  
    **Helpful answer:** 

    :::
* **Basic Architecture:**
  * **Orchestration Component:** Manages the flow of data and interactions between components.
  * **Input:** User queries or prompts.
  * **Output:** Generated responses.
  * **Evaluation:** Measures the quality of the output.
  * **Prompting:** Techniques for interacting with language models.
  * **Language Models:** The core component for generating text.
  * **State:** Data storage, including files, search engines, vector databases, and databases.



### Cutting Through the Hype

* **Challenges in the RAG Landscape:**
  * Constant stream of new models, components, and tricks.
  * Oversimplification of RAG as just vector embeddings and language models.
  * Lack of focus on evaluating performance on specific data.
* **Importance of Information Retrieval:**
  * Retrieval is a deep and well-studied field, crucial for many applications.
  * It's more complex than simply encoding text into a single vector representation.
  * Building effective RAG solutions requires understanding and leveraging existing retrieval techniques.



### Evaluating Information Retrieval Systems

* **Information Retrieval System as a Black Box:**
  * Input: Query
  * Output: Ranked list of documents
* **Evaluation Based on Relevance:**
  * Human annotators judge the relevance of retrieved documents to the query.
  * **Binary Judgment:** Relevant or not relevant.
  * **Graded Judgment:** Levels of relevance (e.g., 0 - irrelevant, 1 - slightly relevant, 2 - highly relevant).
* **Established IR Research and Benchmarks:**
  * **[TREC (Text Retrieval Conference)](https://trec.nist.gov/):**  Evaluates various retrieval tasks, including news retrieval.
  * **[MS MARCO](https://microsoft.github.io/msmarco/):**  Large-scale dataset from Bing with real-world annotated data used for training embedding models.
    * **HuggingFace Hub:** [microsoft/ms_marco](https://huggingface.co/datasets/microsoft/ms_marco)
  * **[BEIR](https://github.com/beir-cellar/beir):** Evaluates models in a zero-shot setting without training data. 
* **Common IR Metrics:**
  * **Recall@K:** Measures the proportion of relevant documents retrieved within the top K positions.
  * **Precision@K:** Measures the proportion of relevant documents among the top K retrieved documents.
  * **nDCG (Normalized Discounted Cumulative Gain):**  Rank-aware metric that considers graded relevance judgments.
  * **Reciprocal Rank:** Measures the position of the first relevant hit.
  * **LGTM (Looks Good To Me):** Informal but common metric in industry.
* **Production System Metrics:** Engagement (clicks, dwell time), add-to-cart rate, revenue.
* **Benchmark Limitations:** 
  * Often compare flat lists, not personalized results.
  * Don't always transfer well to specific domains or use cases. 



### Building Your Own Relevancy Dataset

* **The Importance of Measuring:**  To improve RAG performance, measure relevance on your specific data.

* **Creating a Relevancy Dataset:**
  * **Leverage Existing Traffic:** Log user searches and judge the relevance of results.
  * **Bootstrap with Language Models:**  Use LLMs to generate questions based on your content and judge the relevance of retrieved passages.
  
* **Dataset Format:** 
  
  * Simple TSV file is sufficient.
  * **Query ID, Document ID, Relevance Label**
  
    * | qid           | docid | relevance label | comment |
      | ------------- | ----- | --------------- | ------- |
      | 3 (how to ..) | 4     | 2               |         |
      | 3 (where ..)  | 2     | 0               |         |
  
* **Static vs. Dynamic Collections:** 
  * Static collections are preferred for consistent evaluation.
  * Dynamic collections can introduce noise when evaluating metrics.
  
* **Using Language Models for Relevance Judgments:**
  
  * **Paper:** [Large language models can accurately predict searcher preferences](https://arxiv.org/abs/2309.10621)
  
  * LLMs can be prompted to assess the relevance of queries and passages.
  
  * Find a prompt that correlates well with your golden dataset.
  
  * Enables cheaper and larger-scale evaluation.
  
  * **Example:** Microsoft research demonstrating LLM effectiveness in judging relevance.
  
    * **Paper:** [UMBRELA: UMbrela is the (Open-Source Reproduction of the) Bing RELevance Assessor](https://arxiv.org/abs/2406.06519)
  
    * ::: {.callout-note title="Figure 1: Prompt used for relevance assessment."}
  
      ```text
      Given a query and a passage, you must provide a score on an
      integer scale of 0 to 3 with the following meanings:
      0 = represent that the passage has nothing to do with the query,
      1 = represents that the passage seems related to the query but
      does not answer it,
      2 = represents that the passage has some answer for the query,
      but the answer may be a bit unclear, or hidden amongst extraneous
      information and
      3 = represents that the passage is dedicated to the query and
      contains the exact answer.
      
      Important Instruction: Assign category 1 if the passage is
      somewhat related to the topic but not completely, category 2 if
      passage presents something very important related to the entire
      topic but also has some extra information and category 3 if the
      passage only and entirely refers to the topic. If none of the
      above satisfies give it category 0.
      
      Query: {query}
      Passage: {passage}
      
      Split this problem into steps:
      Consider the underlying intent of the search.
      Measure how well the content matches a likely intent of the query
      (M).
      Measure how trustworthy the passage is (T).
      Consider the aspects above and the relative importance of each,
      and decide on a final score (O). Final score must be an integer
      value only.
      Do not provide any code in result. Provide each score in the
      format of: ##final score: score without providing any reasoning.
      ```
  
      :::
  
* **Benefits of Custom Relevancy Datasets:**
  
  * Iterate and measure the impact of changes to your retrieval system.
  * Track improvements in metrics like nDCG.
  * **Example:** Vespa documentation search showing improvement in nDCG with hybrid retrieval methods.
  
    



### Representational Approaches and Scoring Functions

* **Motivation for Efficient Retrieval:** Avoid scoring all documents in the collection for each query.
* **Sparse Representations:**
  * **Term-based:** Documents and queries are represented by the presence and weight of terms.
  * **Efficient Retrieval:** Inverted indexes, algorithms like WAND and MaxScore.
  * **Example:** Keyword search technologies like Elasticsearch and Vespa.
* **Dense Representations:**
  * **Embedding-based:** Documents and queries are represented by vectors in a latent space.
  * **Neural/Sparse Embedding Models:** Learn term weights using transformer models.
  * **Efficient Retrieval:** Approximate Nearest Neighbor search, vector databases.
  * **Example:** Text embedding models, semantic search.
* **Advantages of Dense Representations:** 
  * Capture semantic relationships between words and concepts.
  * Enable search based on meaning rather than exact keyword matches.
* **Challenges of Dense Representations:** 
  * **Transfer Learning:** Off-the-shelf models may not perform well on specific data.
  * **Diluted Representations:** Averaging token embeddings into a single vector can lose information.
  * **Fixed Vocabulary:** Out-of-vocabulary words can be mapped to incorrect concepts.
  * **Chunking:** Long documents need to be chunked to maintain precision.
* **Baselines for Comparison:** 
  * **BM25:**  
    * **Term-based scoring function.**
    * **Unsupervised, based on corpus statistics.**
    * **Cheap, small index footprint.**
    * **Strong baseline for many tasks.**
    * **Limitations:** Requires language-specific tokenization, struggles with long context.
  * **Example:** BM25 outperforming embedding models on long context documents in ColBERT evaluation.
    * **Blog Post:** [Announcing Vespa Long-Context ColBERT](https://blog.vespa.ai/announcing-long-context-ColBERT-in-vespa/)



### Hybrid Approaches

* **Combining Sparse and Dense Representations:**
  * Can overcome limitations of individual approaches.
  * Example: Combining keyword search with embedding retrieval.
* **Challenges of Hybrid Approaches:**
  * Calibration of different scoring functions.
  * Determining when to ignore embedding results.
  * Requires careful tuning and evaluation.



### Long Context and Chunking

* **Desire for Long Context Models:**  Eliminate the need for chunking.
* **Reality of Chunking:** 
  * Necessary for meaningful representations in high-precision search.
    * Dense representation beyond 256 tokens are bad for high-precision search.
  * **Pooling operations dilute representations in long contexts.**
  * **Limited training data for long context models.**
* **Chunking Strategies:** 
  * Split long documents into smaller segments.
  * Index multiple vectors per row in a database.



### Real-World RAG Considerations

* **Google Search Signals:** 
  * **Text Similarity:** BM25, vector cosine similarity.
  * **Freshness:** Recency of content.
  * **Authority:** Trustworthiness of the source.
  * **Quality:** Overall content quality.
  * **PageRank:** Link analysis algorithm.
  * **Revenue:** In advertising-based search.
* **GBDT (Gradient Boosted Decision Trees):**
  * Effective for combining tabular features.
  * Still relevant for real-world search.



### Summary

* **Information retrieval is more than just vector representations.**
* **Build your own evaluations to improve retrieval.**
* **Don't ignore the BM25 baseline.**
* **Choose technologies with hybrid capabilities.**
* **Real-world search involves more than just text similarity.**





### Q&A Session

#### Q1: Metadata for Vector DB in RAG

* **Question:** What kind of metadata is most valuable to put into a vector DB for doing RAG?
* **Answer:**
  * **Context-Dependent:** The most valuable metadata depends on the specific use case and domain.
  * **Text-Only Use Cases:**
    * **Authority/Source Filtering:**  Important in domains like healthcare where trustworthiness of sources is crucial (e.g., filter out Reddit posts in favor of medical journals).
    * **Title and other basic metadata:** Can provide additional context for retrieval.
  * **Real-World Use Cases:** 
    * Consider factors beyond text, such as freshness, authority, quality, and even revenue.

#### Q2: Calibration of Different Indices

* **Question:** Do you have any thoughts on calibration of different indices? How can we obtain confidence scores for recommendations?
* **Answer:**
  * **Challenge:** Different scoring functions (e.g., BM25, cosine similarity) have different distributions and ranges, making calibration difficult. Scores are not probabilities.
  * **Learning Task:** Combining scores effectively is a learning problem. 
  * **GBDT's Strength:**  Gradient Boosted Decision Trees (GBDT) can learn non-linear combinations of features, including different scoring functions.
  * **Need for Training Data:** Calibration and learning require training data.
  * **Options for Training Data:**
    * **Evaluation Data:**  The evaluation datasets described earlier can be used to generate training data.
    * **Real User Interactions:** Gather data from user searches and clicks (like Google).
    * **Synthetic Data:**  Use large language models to generate synthetic training data.
  * **No Easy Tricks:** There's no universal solution for calibration without training data and evaluation.

#### Q3: Efficacy of Re-rankers

* **Question:** What are your observations on the efficacy of re-rankers? Do you recommend using them?
* **Answer:**
  * **Phased Retrieval and Ranking:** Re-rankers are valuable in multi-stage pipelines. They allow you to invest more compute into fewer, more promising hits retrieved in earlier stages.
  * **Benefits of Re-rankers:**
    * **Token-Level Interaction:** Re-rankers like Cohere or cross-encoders enable deeper interaction between the query and document at the token level, improving accuracy.
    * **Latency Management:**  By focusing on a smaller set of candidates, re-rankers can help meet latency requirements.
  * **Trade-offs:** Re-rankers add computational cost and latency.
  * **Recommendation:**  If accuracy is a priority and the cost is acceptable, re-rankers are recommended.

#### Q4: Combining Usage Data and Semantic Similarity

* **Question:**  Do you have advice on combining usage data (e.g., number of views) with semantic similarity?
* **Answer:**
  * **Learning to Rank Problem:**  Integrating usage data turns it into a learning to rank problem.
  * **Label Generation:** Convert interaction data into labeled training data. Different interactions (e.g., views, clicks, add-to-cart) may have different weights in the label generation process.
  * **Model Training:**  Train a ranking model (e.g., GBDT) using the labeled data, including semantic similarity scores and usage data as features.

#### Q5: Jason Liu's Post on Structured Summaries 

* **Question:** What are your thoughts on Jason Liu's post about the value of generating structured summaries and reports for decision makers instead of doing RAG as commonly done today?
* **Answer:**  Jo was not familiar with the specific post.

#### Q6: Recent Advancements in Text Embedding Models

* **Question:** What are some of your favorite advancements recently in text embedding models or other search technologies?
* **Answer:**
  * **Larger Vocabularies:** Jo hopes for embedding models with larger vocabularies to better handle out-of-vocabulary words, especially in specialized domains. BERT's vocabulary is outdated.
  * **Improved Pre-trained Models:** Better pre-training techniques and data can lead to more robust and generalizable embedding models.
  * **Caution on Long Context:** Jo is not overly enthusiastic about increasing context length for embedding models. Research suggests diminishing returns for high-precision search with very long contexts.

#### Q7: Query Expansion with BM25

* **Question:** Does query expansion of out-of-vocabulary words with BM25 work better at search? Are people utilizing classical search techniques like query expansion enough?
* **Answer:**
  * **BM25 and Re-ranking:** Combining BM25 with a re-ranker can yield excellent results. While BM25 may struggle with single-word queries that are out-of-vocabulary, it avoids the severe failure modes of relying solely on embedding-based search.
  * **Query Expansion's Potential:** Query expansion and understanding are powerful techniques. Language models can be used for query expansion, and tools for prompting LLMs for this purpose are improving.
  * **Importance of Evaluation:** Building your own evaluation setup allows you to systematically test different techniques like query expansion on your specific data and determine their effectiveness.

#### Q8: Handling Jargon and Tokenization Issues

* **Question:** How do you overcome limitations in fixed vocabulary and poor tokenization in domains with a lot of jargon, when using an out-of-the-box model?
* **Answer:**
  * **Hybrid Approach:** Combine keyword search with embedding retrieval to mitigate vocabulary limitations.
  * **Challenge of Ignoring Embedding Results:** Embedding retrieval always returns results, even if they are not semantically relevant. It's crucial to identify and filter out these irrelevant results.
  * **Fine-tuning:** Fine-tuning your own embedding model on domain-specific data can help, but vocabulary limitations may persist.
  * **Pre-training from Scratch:**  Training a BERT-like model from scratch with a custom vocabulary tailored to the domain is becoming more feasible. 
    * This is a common practice in e-commerce.

#### Q9: ColBERT and Tokenizer Problems

* **Question:** Would ColBERT-based methods improve retrieval when we are concerned with tokenizer problems?
* **Answer:**
  * **ColBERT's Approach:** ColBERT learns token-level vector representations instead of a single vector for the whole passage or query. It offers high accuracy while being computationally less expensive than cross-encoders.
  * **Vocabulary Limitations:** ColBERT still relies on the same vocabulary as other BERT-based models, so it's not a complete solution to tokenizer problems.
  * **Future Direction:** Better pre-trained models with larger vocabularies would benefit ColBERT and other embedding models.

