---
title: "Conference Talk 10: Systematically Improving RAG Applications"
date: 2024-7-20
image: /images/empty.gif
hide: false
search_exclude: false
categories: [notes, llms]
description: "In this talk, **Jason Liu** covers a a systematic approach to improving Retrieval Augmented Generation (RAG) applications."

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





* [The RAG Playbook](#the-rag-playbook)
* [Identifying and Addressing Issues](#identifying-and-addressing-issues)
* [Real-Time Monitoring and Classifiers](#real-time-monitoring-and-classifiers)
* [Low-Hanging Fruit and Synthetic Data Generation](#low-hanging-fruit-and-synthetic-data-generation)
* [Importance of Full Text Search and Metadata](#importance-of-full-text-search-and-metadata)
* [Conclusion](#conclusion)
* [Q&A Session](#qa-session)







## The RAG Playbook

* **Blog Posts:** [Jason Liu RAG](https://jxnl.co/writing/category/rag/)

### Importance of Feedback Mechanisms

*  Define clear objectives for your RAG application. What user behavior are you trying to drive?
*  Don't try to improve everything at once. Focus on specific areas identified through data analysis.

### Capturing Feedback: User Satisfaction

*  Implement simple feedback mechanisms like thumbs up/thumbs down buttons.
*  Carefully choose the copy for feedback prompts to ensure you're measuring the intended metric (e.g., answer correctness vs. overall experience).
*  Example: Changing feedback prompt from "How did we do?" to "Did we answer your question?" led to a 5x increase in feedback volume and improved data quality.

### Measuring Relevancy: Cosine and Re-ranker Scores

*  Track objective relevancy metrics alongside user feedback.
*  Use cosine similarity of embedding scores and re-ranker scores as cost-effective measures of relevancy.
*  **reranker:** A type of model that, given a query and document pair, will output a similarity score.
*  **Tool:**
   *  **[rerankers](https://github.com/AnswerDotAI/rerankers):** A lightweight unified API for various reranking models.


### Unsupervised Learning for Topic Clustering

*  Employ unsupervised learning techniques like LDA or BERTopic to group similar queries into topics.
   *  **LDA:**
      *  **Paper:** [Latent Dirichlet Allocation](https://ai.stanford.edu/~ang/papers/jair03-lda.pdf)

   *  **BERTopic:**
      *  **Paper:** [BERTopic: Neural topic modeling with a class-based TF-IDF procedure](https://arxiv.org/abs/2203.05794)
      *  **Documentation:** [BERTopic](https://maartengr.github.io/BERTopic/index.html)

*  Analyze topic clusters based on:
   *  Number of questions within the topic
   *  Mean cosine similarity score
   *  Mean user satisfaction score

### Prioritizing Improvement Based on Topic Analysis

*  Prioritize topics with:
   *  High volume (many questions)
   *  Low relevancy scores
   *  Low user satisfaction scores
*  Consider deprioritizing or explicitly excluding topics with low volume, low relevancy, and low satisfaction.
*  Different combinations of volume, relevancy, and satisfaction provide insights into different types of issues requiring specific solutions.



## Identifying and Addressing Issues

### Content vs. Capability Topics

*  **Content topics:** Issues stem from insufficient or inadequate content in the knowledge base.
   *  Example: Users asking about pricing, but not enough pricing documents available.
*  **Capability topics:** Issues relate to limitations in the system's functionality.
   *  Example: Users asking for the last modified date of a document, but this information is not available to the language model.

### Examples of Capability Topics and Solutions

*  **Modified dates:** Include last modified date metadata in text chunks.
*  **Comparing and contrasting:** Implement parallel search functionality.
*  **Recency and latency:** Add date range filtering capabilities.
*  **Financial year variations:** Account for industry-specific fiscal year definitions.

### Addressing Inventory Issues and Building Rules

*  **Address content topics by:**
   *  Identifying and filling inventory gaps.
*  **Build rules to:**
   *  Alert content management teams when questions frequently lack relevant documents.
   *  Inform users when insufficient data is available to answer their query.



## Real-Time Monitoring and Classifiers

### Building Classifiers for Real-Time Question Categorization

*  Develop classifiers to categorize new questions into previously identified topics and capability clusters in real-time.

### Monitoring Question Distribution and Prioritization Over Time

*  Use tools like Amplitude or Datadog to monitor the distribution of question types over time.
   *  **Amplitude:** [https://amplitude.com/](https://amplitude.com/)
   *  **Datadog:** [https://www.datadoghq.com/](https://www.datadoghq.com/)

*  Identify shifts in user needs and adjust prioritization accordingly.
*  **Example:** Onboarding a new client significantly increases the volume of "comparing and contrasting" questions, highlighting the need to prioritize that capability.



## Low-Hanging Fruit and Synthetic Data Generation

### Focusing on Specific Improvements

*  Translate ambiguous goals like "improving RAG" into concrete, measurable objectives (e.g., "improving datetime filtering").
*  Break down large improvements into smaller, manageable experiments.

### Synthetic Data Generation for Baseline Evaluation

*  Generate synthetic data for specific topics to:
   *  Establish baselines for evaluating the impact of system changes.
   *  Test the effectiveness of new datasets or models.



## Importance of Full Text Search and Metadata

### Benefits of Full Text Search and Re-Rankers

*  Incorporate full text search capabilities for robust retrieval.
*  Utilize re-ranker models to improve the ranking of retrieved results.

### Importance of Metadata Filtering

*  Include relevant metadata (e.g., author, date, document type) to enable granular filtering.
*  Example:  "Show me the latest pricing document for Fortune 500 companies" requires filtering by date, document type, and client category.

### Measuring Impact and Driving Business Outcomes

*  Track relevant metrics (e.g., character length, latency) and correlate them with business outcomes (e.g., user conversion, engagement).



## Conclusion

* The key to improving RAG applications is to adopt a systematic, data-driven approach.
* By analyzing user feedback, clustering queries, and monitoring question distribution, developers can identify and prioritize areas for improvement.
* Focusing on specific, measurable goals and utilizing synthetic data generation enables efficient testing and iteration.







## Q&A Session 



### How much data is needed for these techniques to be worthwhile?

* Even with small datasets (e.g., 100 queries), clustering techniques can reveal patterns and differences in query types (e.g., time-sensitive queries).
* Clustering enables focusing on specific query groups and tailoring optimization efforts.
* Early clustering allows for proactive identification of potential issues and areas for improvement.

### How to build a data ingestion pipeline for complex data?

* **Understand user questions:** Determine how users interact with different data types (e.g., extracting specific information from images or answering questions about charts).
* **Generate text summaries:** Create text representations of images, charts, and tables to enable text-based search and retrieval.
* **Use specialized libraries:** Leverage libraries for extracting structured data like tables from PDFs, addressing potential challenges like nested headers.

### How to handle tables in RAG applications?

* **Understand query intent:** Determine whether users seek aggregate statistics or specific rows within tables.
* **Choose appropriate processing:** Utilize Text-to-SQL engines for aggregate queries or chunk tables strategically for row-based searches.
* **Consider table size and complexity:** Large tables may necessitate alternative approaches like using SQLite or optimized search techniques.

### How to capture feedback when RAG is hidden from the user?

* **Analyze user interactions:** Track edits to report fields to identify areas requiring frequent corrections.
* **Monitor internal RAG metrics:** Analyze cosine distances and re-ranker scores across different fields to pinpoint areas for improvement.
* **Design for feedback:** Ensure reports allow for user feedback and facilitate mapping feedback to specific report elements.

### How to capture feedback and iterate quickly?

* **Write assertions for extractable facts:** Create tests to verify the accuracy of extracted information against ground truth, enabling rapid iteration without deploying the system.
* **Use automated testing:** Automate the evaluation process to speed up development and identify regressions.

### Why use cosine distance as a metric when it's relative?

* **Provides a quantifiable measure:** Cosine distance offers a numerical representation of relevance, even if relative, allowing for comparisons and tracking progress.
* **Available for every request:** Unlike user feedback, which can be sparse, cosine distance can be calculated for every query, providing more data points for analysis.
* **Useful for relative comparisons:** Focus on the relative differences in cosine distances between queries and text chunks to identify areas for improvement.

### How to incorporate metadata filtering in RAG?

* **Use language models for structured outputs:** Utilize models like Instructor to generate structured representations of queries, including metadata filters like date ranges and allowed domains.
* **Customize metadata extraction:** Define custom metadata fields based on the specific data and user needs.
* **Leverage metadata in search:** Integrate metadata filters into the search process to narrow down relevant documents and improve accuracy.

### How to use language models for metadata creation?

* **Extract specific information:** Use language models to extract relevant metadata from documents, such as converting financial reporting periods based on industry-specific rules.
* **Generate synthetic queries from complex content:** Extract complex diagrams and tables, then use language models to generate related questions, embedding those questions as metadata for improved retrieval.

### Favorite platforms for building RAG systems?

* **LanceDB:** Favored for its ability to combine full-text search, SQL, and vector search in a single database, simplifying the development process.
  * **[LanceDB](https://lancedb.com/):** The Database for Multimodal AI

* **Custom solutions:** Building custom query engines and processors provides greater control and flexibility, especially for handling specific data structures and complex search requirements.

### How to develop intuition for user questions at the start of a project?

* **Generate synthetic queries:** Utilize language models to generate synthetic queries from the data, revealing potential search challenges and guiding UI design.
* **Start with a hypothesis:** Base initial development on assumptions about user questions, then refine based on real-world data and feedback.
* **Iterate based on feedback:** Continuously analyze user queries, identify gaps in the system's understanding, and adapt the system accordingly.

### When does BM25 outperform semantic search?

* **Wikipedia:** [Okapi BM25](https://en.wikipedia.org/wiki/Okapi_BM25)
* **Stanford IR Book:** [Okapi BM25: a non-binary model](https://nlp.stanford.edu/IR-book/html/htmledition/okapi-bm25-a-non-binary-model-1.html) 
* **Document search:** When users are searching for specific documents they authored or are familiar with, BM25 excels due to its keyword-matching approach.
* **Exact word matches:** BM25 performs well when users use the same terminology as the document authors, common when searching personal notes or transcripts.

### How to design a good UX for report editing in RAG applications?

* **Utilize structured outputs:** Structure report data using keys and values to enable granular editing and facilitate attributing edits to specific report elements.
* **Avoid monolithic markdown outputs:** Markdown, while seemingly structured, poses challenges in tracking edits and associating them with underlying data.

### How to implement citations in RAG systems?

* **Cite entire text chunks:** Include text chunk IDs as citations within the generated response.
* **Leverage markdown formatting:** Format citations as markdown URLs for improved readability and potential integration with UI elements for displaying cited text.

### High-dimension embedding models vs. cross-encoder models for re-ranking?

* **Use both:** Employ a multi-stage re-ranking approach using both vector databases and cross-encoders to balance speed and accuracy.
* **Prioritize speed with vector databases:** Utilize vector databases for fast initial retrieval, narrowing down the candidate pool for the more computationally expensive cross-encoder.
* **Leverage cross-encoders for nuanced relevance:** Employ cross-encoders to capture subtle semantic similarities and differences that vector databases might miss.

### Thoughts on hierarchical retrievers and fine-tuning embeddings?

* **Hierarchical retrievers:** Potentially less crucial with increasing context lengths in language models, but still valuable for ensuring retrieval of related information across chunks.
* **Fine-tuning embeddings:** Crucial for improving relevance in domain-specific applications, outperforming generic embedding models, especially with sufficient training data.

### Tips for improving data extraction from tables and PDFs?

* **Utilize specialized tools:** Leverage tools like LlamaParse, which combine language models and traditional text extraction techniques for improved accuracy.
  * **LlamaParse:** [Documentation](https://docs.llamaindex.ai/en/stable/llama_cloud/llama_parse/)

* **Experiment with output formats:** Consider requesting markdown table outputs from language models instead of CSVs, as markdown often handles complex table structures better.
* **Leverage advanced language models:** Utilize models like GPT-4.0 or Opus to process images of tables and generate structured outputs like markdown tables.

### Guidelines for metrics and telemetry in RAG systems?

* **Treat RAG like recommendation systems:** Instrument RAG systems similarly to recommendation systems, tracking user interactions, retrieval metrics, and feedback.
* **Log relevant data:** Capture data on user queries, retrieved text chunks, citations, and user feedback for analysis and model improvement.
* **Utilize micro-interactions for feedback:** Treat interactions like citations as implicit feedback, leveraging it to fine-tune embedding models and improve retrieval relevance.

### Recommendations for picking embedding models?

* **Evaluate with synthetic data:** Create synthetic datasets with question-answer pairs and evaluate different embedding models to determine which performs best for the specific task.
* **Fine-tune with real data:** Once sufficient real-world data is available, fine-tune embedding models to further enhance relevance and outperform generic models.

### Balancing upfront data inspection with adaptation over time?

* **Iterative approach:** Combine upfront data analysis with ongoing monitoring and adaptation.
* **Clustering and labeling:** Cluster initial datasets to understand query patterns and create labels for different query types.
* **Monitor for drift:** Regularly monitor the percentage of uncategorized queries ("other") to identify shifts in user behavior and adapt the system accordingly.

### How to teach clients to analyze data for RAG applications?

* **Focus on the scientific method:** Guide clients through a process of hypothesis generation, data collection, experimentation, and iteration.
* **Provide clear visualizations and reports:** Present data analysis results in an easily understandable format to facilitate decision-making and identify areas for improvement.

### Choosing between vector databases and cross-encoders?

* **Consider both evaluation results and latency constraints:** Balance the trade-off between accuracy (often favoring cross-encoders) and speed (favoring vector databases).
* **Prioritize business outcomes:** Ultimately, the choice should align with achieving desired business outcomes, considering factors like user experience and cost.

### How to handle frequently occurring statements in RAG?

* **Create dedicated text chunks:** If a specific statement appears frequently and holds significance for users, create dedicated text chunks containing variations of that statement to ensure retrieval.
* **Augment the dataset:** Use language models to extract relevant clauses and statements from documents, creating augmented text chunks to improve retrieval of specific information.

### Recommendations for vector stores and metadata?

* **Think beyond vector stores:** View the problem as building a search engine, considering various components like BM25, SQL, and vector search, not just vector stores in isolation.
* **Choose tools based on specific needs:** Select technologies and approaches that best suit the project's requirements, data characteristics, and available resources.

### Finding early customers for RAG consulting?

* **Leverage experience from recommendation systems:** Position RAG expertise by drawing parallels to recommendation systems, highlighting transferable skills in data analysis, model development, and system optimization.
* **Target businesses with similar workflows:** Focus on businesses where users interact with information similarly to recommendation systems, such as those requiring information retrieval and synthesis based on user requests.
