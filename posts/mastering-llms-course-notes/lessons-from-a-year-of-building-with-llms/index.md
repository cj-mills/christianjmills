---
title: "Livestream: Lessons from a Year of Building with LLMs"
date: 2024-8-30
image: /images/empty.gif
hide: false
search_exclude: false
categories: [notes, llms]
description: "This live discussion between six AI experts and practitioners centers on the practical lessons learned from a year of building real-world applications with LLMs, emphasizing the critical importance of data literacy, rigorous evaluation, and iterative development processes."

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





### Introduction and Background

* **[Hugo Bowne-Anderson](https://hugobowne.github.io/)**: Host of the *Vanishing Gradients* podcast and live stream. Has a background in data science, machine learning, and education.
* **Live Stream Focus**: Discussion of a report co-authored by six AI professionals (Eugene Yan, Shreya Shankar, Hamel Husain, Brian Bischof, Charles Frye, and Jason Liu), focusing on practical lessons learned from building real-world applications with Large Language Models (LLMs) over the past year.
* **Report Link**: [https://applied-llms.org/](https://applied-llms.org/)
* **Recording Link**: [https://www.youtube.com/live/c0gcsprsFig](https://www.youtube.com/live/c0gcsprsFig)
* **Key Themes**:  The report covers tactical, operational, and strategic aspects of building LLM systems, including: 
  - **Tactical**: Prompt engineering, data management, evaluation strategies.
  - **Operational**: Building development pipelines, integrating LLMs into existing workflows.
  - **Strategic**: Understanding business use cases, defining success metrics, and building trust with stakeholders.

### Panelist Introductions and Motivations

* **Eugene Yan**:
  - Works at Amazon Books, focusing on recommendation systems and search.
  - Interested in using LLMs to enhance recommendations and search by improving customer understanding.
  - Co-authored several key resources on prompt engineering and LLM evaluation.
  - Blog: [https://eugeneyan.com](https://eugeneyan.com)
* **Shreya Shankar**:
  - Researcher and ML engineer pursuing a PhD focused on data management, UX, and HCI for machine learning.
  - Interested in building intelligent software with LLMs, particularly for small teams and early-stage products.
  - Emphasis on developing evaluation methods that are simple and accessible.
  - Conducts research on evaluating LLM output quality and validating evaluation methods.
* **Hamel Husain**:
  - 25 years of experience in machine learning, including work on developer tools and ML infrastructure at GitHub.
  - Led research at GitHub that contributed to Copilot.
  - Passionate about using LLMs to accelerate software development and launch applications faster.
  - Strong advocate for prioritizing evaluation (**evals**) as a core part of the AI development process.
  - Blog post: [https://hamel.dev/blog/posts/evals/](https://hamel.dev/blog/posts/evals/)
* **Brian Bischof**:
  - Head of AI at Hex, leads the team developing Magic (an AI-powered data science tool).
  - Extensive experience building data teams at Blue Bottle Coffee, Stitch Fix, and Weights & Biases.
  - Passionate about applying LLMs to answer questions with data, particularly in the context of data science workflows.
  - Strong advocate for using notebooks for exploring and understanding data.
* **Charles Frye**:
  - Works at Modal, focused on teaching people to build AI applications.
  - Background in psychopharmacology, neurobiology, and neural networks.
  - Interested in developing intelligent software and using LLMs to democratize cognition.
  - Key contributor to Modal's infrastructure and developer tools, including the TensorRT-LLM library.

### The Genesis of the Report

* **Origin**: The report originated from discussions within a group chat among the co-authors.
* **Initial Motivation**:
  - **Brian Bischof**:  Was considering writing about a year of LLMs.
  - **Eugene Yan**: Had already started drafting similar content.
  - **Charles Frye**: Suggested a collaborative effort.
  - **Hamel Husain and Jason Liu**: Enthusiastically joined the project.
  - **Shreya Shankar**: Was invited for her expertise in evaluation and editing.
* **Organization**:
  - **Brian Bischof**: Proposed structuring the report into tactical, operational, and strategic levels.
* **Collaborative Workflow**:
  - Authors contributed their ideas and experiences.
  - Content was synthesized and organized collaboratively.
  - Charles Frye played a key role in editing and consolidating the material.

### The Importance of Evaluation 

* **Traditional Evals vs. LLM Evals**:
  - **Shreya Shankar**:
    - Traditional evaluation methods often fail to uncover issues arising from real-world data idiosyncrasies (e.g., typos, inconsistent casing).
    - Canonical benchmarks and datasets often rely on clean, pre-processed data, which does not reflect real-world data challenges.
    - **Example**: Mistral struggles to retrieve names from documents if the name is in lowercase. 
    - Traditional evals often lack methods for validating the evaluation methods themselves. 
    - LLM-based evals can make evaluation more accessible to smaller teams who lack resources for traditional methods. 
    - **[EvalGen Paper](https://arxiv.org/abs/2404.12272)**: Discusses a flow-based approach to validating evaluation methods and constructing appropriate evals for deployment.
* **Overcoming Barriers to Evaluation**:
  - Developers often struggle with the concept of evals and how to get started.
  - Shreya Shankar's research tools provide a visual, intuitive approach to building and understanding evals, similar to Scratch for programming.
  - This "Scratch for evals" approach makes evals more accessible and understandable.
* **Evaluation as an Integral Part of AI Development**:
  - Evaluation is not optional, it's an essential part of the AI development process.
  - Evals are not separate from building AI, they are how you build AI.
  - Measuring progress and having a systematic approach to improvement are critical.
  - Focus on making evals frictionless and integrated into the workflow to enable continuous improvement.
* **Shifting the Focus from Tools to Process**:
  - Developers often focus on tools (e.g., vector databases, embeddings) rather than understanding the underlying process of building and improving AI applications.
  - It's essential to shift the focus from tools to the process of evaluation and data understanding.
* **Evaluation as a Proxy for Loss Functions**:
  - Traditional machine learning relies on explicit loss functions for optimization.
  - In generative AI, the loss functions are less defined, and evals act as a proxy for measuring progress and alignment with desired outcomes.

###  Underappreciated Aspects of LLM Development

* **Equipping Engineers with LLM Skills**:
  - Organizations need to focus on training existing software engineers to understand and effectively use LLMs.
  - Key areas for training:
    - Basic evaluation methods (e.g., using synthetic data, Kaggle datasets).
    - Understanding the autoregressive nature of LLM generation and its impact on latency.
    - Understanding context as conditioning.
* **Moving from Prototype to Production**:
  - The industry has focused heavily on demos and prototypes, and the bar for "production" has been lowered with generative AI.
  - **Prototype++**:  Products are being launched without rigorous evaluation or clear methods for quantifying improvements.
  - The definition of "production" should include systematic improvement processes and clear evidence of progress for stakeholders.
  - **Challenge**: Moving from prototype++ to true production-ready LLM applications with robust evals and improvement processes.
* **The Importance of Data Literacy**:
  - Data literacy is essential for working with LLMs, even if you are not training models from scratch.
  - Data literacy enables developers to:
    - **Analyze and debug systems**: By examining data, developers can identify issues and understand system behavior.
    - **Develop domain-specific evals**: Understanding the data is crucial for creating meaningful and effective evaluation methods.
  - **Challenge**:  Overcoming the misconception that AI automates everything and realizing the ongoing need to look at and understand data.
* **Specific Data Literacy Skills**:
  - **Assessing Output Quality**:
    - Develop methods for determining whether an output is good or bad.
    - Use a combination of binary indicators (e.g., conciseness, tone, presence of specific phrases) to simplify evaluation.
  - **Pairwise Comparisons**: Compare two models or pipelines directly instead of relying on individual ratings.
  - **Understanding Implicit Constraints**: Define criteria for "good" that align with user expectations.
  - **Simplifying Evaluation**: Break down complex evaluation tasks into smaller, more manageable binary assessments.
* **The Role of Human-in-the-Loop Evaluation**:
  - Importance of manual data labeling and review, especially in the early stages of development.
  - "Homework" approach at Hex:  Team members use interactive Hex applications to provide feedback on model outputs and assist with data labeling.
  - This process helps to bootstrap data sets and align LLM evaluations with human judgment.

### Misconceptions and Knowledge Gaps in Organizations

* **The "AI Engineer" Narrative and Its Limitations**:
  - **Popular Characterization**:
    - AI engineers are portrayed as primarily focused on tools, infrastructure, chains, and agents, with limited involvement in training, evals, inference, and data.
  - **Skills Gaps**:
    - Neglecting data literacy and evaluation skills creates significant challenges beyond the MVP stage.
    - AI engineers may find themselves unable to systematically improve their applications due to a lack of understanding of data and evaluation.
  - **Title Issues**:
    - The "AI engineer" title sets unrealistic expectations and places undue pressure on individuals when projects face challenges.
* **Addressing the Talent Gap**:
  - The most significant impact on AI product development is the talent and skills of the team.
  - Organizations need to hire for data literacy and evaluation expertise, not just tooling and infrastructure skills.
  - **Consulting Work**:  A significant portion of Hamel's consulting work arises from addressing this talent gap and helping organizations build effective AI teams.
* **Real-World Example of Successful "AI Engineer" Hiring**:
  - Successfully hired "AI engineers" by focusing on core data science skills.
  - Take-home exam focused on data cleaning, demonstrating the importance of data literacy in his definition of the role.
* **The Importance of Data Literacy in AI Engineering**:
  - Data literacy is essential for AI engineers to analyze outputs, understand failure modes, and develop effective evaluation strategies.
  - It's crucial for AI engineers to be able to examine data and draw conclusions about system performance without relying solely on AI tools.

### Promising Opportunities and Future Challenges

* **Focusing on Unsexy, Expensive, and Slow Tasks**:
  - LLMs offer the potential to automate tasks that are currently time-consuming and costly for humans.
  - **Examples**:
    - Classification.
    - Information Extraction.
    - Quiz Generation from Textbooks.
  - **Focus**:  Identify tasks that are currently unsexy but can be effectively delegated to LLMs, leading to cost savings and increased efficiency.
* **Developing More Thoughtful UX**:
  - LLMs should not be seen as one-shot wonders with perfect UX.
  - Graceful failure modes and user-friendly interfaces are essential.
  - **Co-pilot Mentality**:  Design systems that allow users to edit and interact with LLM outputs, turning failures into learning opportunities.
* **Empowering End Users as Programmers**:
  - AI will not be able to read minds, so empowering users to interact and refine LLM outputs is crucial.
  - **ChatGPT Example**:  Users act as programmers by refining their prompts and editing previous messages to achieve their desired outcomes.
  - **Notebook Interfaces**:  Offer a flexible workspace for both technical and non-technical users to interact with and programmatically control LLMs.

### Key Insights from Collaboration

* **Value of Community and Alignment**:
  - The most valuable aspect of the collaboration was the opportunity to connect with other experts, discuss ideas, and learn from each other's experiences.
  - Having a network of trusted peers to consult with and debate challenging questions is essential.
* **Surprising Alignment Despite Diverse Perspectives**:
  - The authors, despite working in different parts of the LLM stack (research, infrastructure, product development, etc.), found a remarkable degree of alignment in their key insights.
  - They had "grabbed different parts of the elephant" but had all come to similar conclusions about the core principles of building with LLMs.
* **The Collaborative Process as a "Raid Boss"**:
  - The collaborative effort was akin to battling a raid boss in an MMORPG, requiring a skilled and coordinated team effort.
  - Charles Frye played a critical role in editing and consolidating the vast amount of material.
  - Hamel Husain demonstrated high agency by quickly setting up a website for the report.
  - The collaborative process was enjoyable, inspiring, and ultimately led to a highly impactful resource for the community.
* **Impact Beyond Industry**:
  - The report's impact has extended beyond industry, reaching academics and researchers who traditionally do not engage with industry blogs.
  - The report's timing was critical, capturing the widespread interest in LLMs and their potential to transform computing.

### The Importance of Data-Centric AI Development

* **Data Literacy as a Core Skill**:
  - The ability to understand and work with data is essential for success with LLMs.
  - Traditional software engineering often focuses on the syntactic correctness of code, while data science emphasizes the semantic meaning of data.
  - **Challenge**: Bridging this gap in perspective and helping software engineers develop data literacy skills.
* **Developing Intuition for Data Distributions**:
  - Experience with data analysis and visualization leads to an intuitive understanding of data distributions.
  - This intuition allows for quickly identifying issues in data or model outputs, even without specific theoretical knowledge.

### Building Trust with Stakeholders and Users

* **Collaborative Design**:
  - Involve designers, UX professionals, and domain experts early in the process to understand user needs and build trust.
  - Co-designing with stakeholders helps identify potential pitfalls and ensure the application aligns with user expectations.
* **Slow Rollout and Iterative Feedback**:
  - Slowly roll out LLM applications to small groups of users to gather feedback and identify issues before wider deployment.
  - Repeated interactions with small groups provide more valuable insights than initial interactions with larger groups.
* **User Feedback as the Antidote to Demoitis**:
  - Gathering user feedback through beta programs and interactions at meetups is crucial for moving beyond the hype of demos.
  - Directly observing user reactions provides valuable insights into usability and helps identify areas for improvement.

### Systems Thinking vs. Model Focus

* **The Importance of Systems-Level Design**:
  - Building successful LLM applications requires a systems-level perspective, not just a focus on the model itself.
  - **Example**:  At Hex, the team started by designing the overall system architecture, including prompt templating, evaluation frameworks, and context construction (RAG).
  - These architectural decisions have proven durable and have guided the project's development.
* **Key System Design Considerations**:
  - **Evals**:  Prioritize building robust evaluation frameworks from the outset.
  - **Composability**: Design systems with composability in mind, especially for context construction and prompt generation.
  - **Meta-Programming**: Think of prompt construction as a form of meta-programming, allowing for flexible and adaptable system behavior.
* **Leveraging Prior Experience from ML**:
  - Much of the knowledge and best practices from traditional machine learning apply to building LLM systems.
  - **Example**:  The principles outlined in the book *Machine Learning Design Patterns* are directly relevant to LLM application development.

### Final Advice and Future Directions

* **Prioritize Evaluation**:
  - Start by creating sample inputs and ideal outputs, then build evaluation methods to measure progress and identify issues.
  - Focus on creating a gold standard dataset, even if it's small, to guide evaluation and refinement.
* **Read, Build, and Share**:
  - **Read**:  Engage with high-quality resources and articles that distill key concepts and best practices.
  - **Build**: Get hands-on experience by building demos and experimenting with LLMs to understand their capabilities and limitations.
  - **Share**: Contribute to the community by sharing your experiences, insights, and code to accelerate learning and adoption.
* **Focus on Iterative Improvement**:
  - Embrace a process of validated iterative improvement, similar to gradient descent in optimization.
  - LLMs introduce complexity and uncertainty, but the fundamental principles of building complex systems remain the same:  make small, validated steps toward improvement.
  - **Key Elements**:  Data, experimentation, operationalization, and rapid deployment to production.
  - Focus on iterative experimentation and building evaluation frameworks to guide development toward user-centric solutions.
    - **Zero-to-One Mentality**:  Break down the problem into smaller, manageable chunks and iteratively improve each component.
* **The Future of Knowledge Access**:
  - LLMs have the potential to make knowledge more accessible, personalized, and useful.
  - **Example**:  Memory extenders and personal Memex systems.
  - LLMs will transform how we interact with information, socialize, and carry out our daily lives.

### Conclusion

* The report and this discussion highlight the essential lessons learned from a year of building with LLMs. 
* **Key Takeaways**:
  - **Data Literacy**:  Prioritize understanding and working with data as a foundational skill.
  - **Evaluation**:  Make evaluation a core part of the development process, focusing on human-in-the-loop methods and iterative refinement.
  - **Systems Thinking**:  Build end-to-end systems that go beyond model focus, considering aspects like prompt design, context construction, and evaluation frameworks.
  - **Process over Tools**:  Focus on developing robust processes and methodologies rather than relying solely on tools. 
  - **User-Centricity**: Design applications with user needs in mind, prioritizing trust, feedback, and iterative improvement.
* The future of LLMs is bright, with the potential to transform how we interact with information and build intelligent software.




{{< include /_about-author-cta.qmd >}}
