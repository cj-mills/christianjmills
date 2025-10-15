---
title: "Conference Talk 17: Language Models on the Command-Line"
date: 2024-8-29
image: /images/empty.gif
hide: false
search_exclude: false
categories: [notes, llms]
description: "In this talk, **Simon Willison** showcases **LLM**, a command-line tool for interacting with large language models, including how to leverage its plugin system, local model support, embedding capabilities, and integration with other Unix tools for tasks like retrieval augmented generation."

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



::: {.callout-tip title="Presentation Resources"}

* **Talk Recording:**  [Language models on the command-line w/ Simon Willison](https://www.youtube.com/watch?v=QUXQNi6jQ30) 
* **Handout:** [Language models on the command-line](https://github.com/simonw/language-models-on-the-command-line/blob/main/README.md)

:::





### Introduction

- **[Simon Willison](https://simonwillison.net/)**, creator of [Datasette](https://datasette.io/), Django co-creator, and PSF board member, presents a case for using Unix command line with LLMs.

### Unix Command Line: A Perfect LLM Playground

- **Unix Philosophy**: Tools output information that gets piped into other tools as input.
- LLMs function similarly: Prompt input generates responses that can be further processed.
- **LLM Tool**: A Python command line tool for interacting with LLMs.
  - **GitHub Repository:** [https://github.com/simonw/llm](https://github.com/simonw/llm)
  - **Documentation:** [https://llm.datasette.io/en/stable/](https://llm.datasette.io/en/stable/)

### Installing LLM

- **Python Users**: `pip install llm` (recommended: `pipx install llm`).
- **Homebrew Users**: `brew install llm`.

### Using LLM with OpenAI

- **Setting API Key**: 
  - `llm keys set openAI <your_api_key>`
- **Running Prompts**:
  - `llm "five great names for a pet pelican"`
- **Saving Output**:
  - `llm "five great names for a pet pelican" > pelicans.txt`
- **Continuing Conversation**:
  - `llm -c "now do walruses"`
- **Accessing Logs**:
  - `llm logs -c <conversation_id>` (e.g., to retrieve past responses)
- **Viewing Logs in [Datasette](https://datasette.io/)**: 
  - ```bash
    pipx install datasette
    ```
  - `llm logs path` (provides path to SQLite database)
  - `datasette "<path_to_database>"` (opens a web interface for browsing conversations).
- **Changing Default Model**:
  - `llm models default` (shows the current default model)
  - `llm models default -m <model_name>` (sets a new default model, e.g., `chatGPT`).

### LLM Plugins

- **Plugin Directory**: [https://llm.datasette.io/en/stable/plugins/directory.html](https://llm.datasette.io/en/stable/plugins/directory.html)
- **Plugin Types**:
  - **Remote API Plugins**: Connect to various LLM providers like Claude, Rekha, Perplexity, AnyScale.
  - **Local Model Plugins**: Enable running models locally on your computer.
- **Installing Plugins**:
  - `llm install <plugin_name>` (e.g., `llm install llm-claude-3`)
- **Viewing Installed Plugins**:
  - `llm plugins`
- **Using Plugin Aliases**:
  -  `llm -m <plugin_alias>` (e.g., `llm -m haiku` to use `llm-claude-3-haiku`).

###  LLM and Local Models

- **Local Models**: Increasingly effective and accessible through LLM plugins.

- **LLM-GPT4all Plugin**: Wrapper around nomic's [gpt4all library](https://github.com/nomic-ai/gpt4all).
  - Install: `llm install llm-gpt4all`.
  - List Models: `llm models` (includes installed local models).
  
- **Example**: Running a local Mistral model:
  - ```bash
    llm chat -m mistral-7b-instruct-v0 "five great names for a pet seagull, explanations"
    ```
  
- **LLM Chat**: A command for persistent chat sessions with local models, avoids repeated loading.
  - `llm chat -m <model_name>` (e.g., `llm chat -m "mistral-7b"`)
  
- **Ollama Integration**:
  - **[Ollama](https://ollama.com/)**: A desktop app for managing and running local models.
  - **LLM-Olama Plugin**:  Allows using Ollama models within LLM.
    - Install: `llm install llm-olama`.
    - Access Ollama models: `llm models`.
  - Example: Using Mixtral through Ollama: 
    - `llm chat -m "mixtral-latest" "Hola en español"`
  
- **[LlamaFile](https://github.com/Mozilla-Ocho/llamafile)**:  
  - **Functionality**: A single binary containing both the LLM and the software to run it, compatible across multiple operating systems.
  - **Advantages**: Downloadable, self-contained, acts as an offline backup.
  - **Example**: Running Llama-370b:
    - Download the LlamaFile binary.
    - Make it executable: `chmod +x <llamafile_binary>`.
    - Run: `./<llamafile_binary>` (starts a web server for interaction).
  - **LLM-LamaFile Plugin**:  Allows using LlamaFile models within LLM.
    - Install: `llm install llm-llamafile`.
    - Access LlamaFile models: `llm models`.
  - **Lava Model**: A notable LlamaFile model, recommended for its multi-modal capabilities.

###  Command Line Scripts with LLM

- **[HN-Summary Script](https://til.simonwillison.net/llms/claude-hacker-news-themes)**: Summarizes Hacker News posts and conversations using a combination of:
  - `curl` to fetch data from the Hacker News API.
  - `jq` to process the JSON output.
  - `llm` with a system prompt to summarize the extracted text.
  
- **[Files-to-Prompt Command](https://simonwillison.net/2024/Apr/8/files-to-prompt/)**: Converts multiple files into a single prompt, including filenames and content.
  - Example: Using LLM to suggest tests for a project: 
    - ```bash
      files-to-prompt <project_directory> | llm -s "suggest tests to add to this project"
      ```

###  LLM and ShotScraper for RAG

- **[ShotScraper Tool](https://github.com/simonw/shot-scraper)**: A browser automation tool for taking screenshots and executing JavaScript from the command line.
- **Scraping Google Search Results**:
  - Example: Using ShotScraper and LLM to answer a question using Google search results:
    - ```bash
      shot-scraper javascript 'https://www.google.com/search?q=nytimes+slop' '
      Array.from(
        document.querySelectorAll("h3"),
        el => ({href: el.parentNode.href, title: el.innerText})
      )'
      ```
    - ```bash
      shot-scraper javascript 'https://www.google.com/search?q=nytimes+slop' '
      () => {
          function findParentWithHveid(element) {
              while (element && !element.hasAttribute("data-hveid")) {
                  element = element.parentElement;
              }
              return element;
          }
          return Array.from(
              document.querySelectorAll("h3"),
              el => findParentWithHveid(el).innerText
          );
      }' | llm -s 'describe slop'
      ```
    - This script scrapes Google search results for "NY Times slop", extracts relevant information using JavaScript, and pipes the results to LLM to answer the question "describe slop".

###  LLM and Embeddings

- **Embeddings**: Supported through plugins, both API-based and local.

- **Viewing Available Embedding Models**:
  - `llm embed models`.
  
- **Creating Embeddings**:
  - `llm embed -m <model_name> "<text_to_embed>"`.
  
- **Storing Embeddings in SQLite**:
  - **`llm embed multi` Command**: Creates a collection of embeddings and stores them in a SQLite database.
  - Example: Creating embeddings for bookmarks in a blog database:
    - ```bash
      curl -O https://datasette.simonwillison.net/simonwillisonblog.db
      llm embed-multi links \
        -d simonwillisonblog.db \
        --sql 'select id, link_url, link_title, commentary from blog_blogmark' \
        -m 3-small --store
      ```
    - This command creates a collection called "links", embeds bookmarks from the specified database, uses the "text-embedding-ada-002" model, and stores both text and embeddings in the database.
  
- **Searching with Embeddings**:
  - **`llm similar` Command**: Finds similar items in a collection based on embedding similarity.
  - Example: Searching for bookmarks related to "things that make me angry":
    - ```bash
      llm similar links \
        -d simonwillisonblog.db \
        -c 'things that make me angry'
      ```
  
- **Combining Embeddings and RAG**:
  - Example: Searching for data set plugins and summarizing the most interesting ones:
    - ```bash
      llm similar -c links -d <database_path> "datasette plugins" | llm -s "most interesting plugins"
      ```

###  Building a RAG System with LLM

- **`blog-answer` Script**: A bash script demonstrating a full RAG Q&A workflow using:
  - Embedding search against paragraphs in a blog.
  - JQ for data processing.
  - A local LlamaFile model (or other models) for question answering.
- **Limitations of SQLite for Large Datasets**:
  - Brute force search becomes inefficient for very large datasets.
  - Consider specialized vector indexes (SQLite-based or external like Pinecone) for better performance.
- **Future Goals**:
  - Integrating support for external vector indexes within LLM.
- **Python API**:
  - `pip install llm` provides a Python API for accessing LLM functionalities.







## Q&A Highlights

###  Hugging Face Hub Models

- **No plugins currently exist for Hugging Face Hub models** in LLM due to Simon's lack of an NVIDIA GPU (required for most Hugging Face models).
- **Opportunity for contribution**: Anyone with an NVIDIA GPU is encouraged to write an LLM plugin for Hugging Face models.
- Other serving technologies for Hugging Face models (e.g., vLLM) are also worth exploring.

###  Serverless Inference

- LLM supports various **API-based models** through plugins (e.g., AnyScale, Fireworks, Open Router).
- **OpenAI compatible models** can be configured directly within LLM without writing plugins.
- **Hugging Face Inference API**: Potentially exciting opportunity for a new LLM plugin.

###  Agentic Workflows

- **Not yet supported** in LLM due to the lack of function calling functionality.
- **Future plans**: Function calling support and potential LLM agents plugin.
- **Python API**: Offers a way to access LLM functionalities from Python code, but the interface is still under development.

###  Productivity Tips from Simon Willison

- **Blog Post:** [Coping strategies for the serial project hoarder](https://simonwillison.net/2022/Nov/26/productivity/)
- **Importance of Unit Tests and Documentation**: Enables revisiting and continuing projects easily.
- **Focus on Quick Projects**: Leverage existing expertise to build cool things rapidly.
- **GitHub Issues for Comprehensive Note-Taking**: Use GitHub issues to document every step of a project, facilitating TIL writing and future reference.
  - **Example:** [Figure out how to serve an AWS Lambda function with a Function URL from a custom subdomain](https://github.com/simonw/public-notes/issues/1)

###  Hardware and Local Model Performance

- **Simon's Hardware**: M2 Max with 64 GB RAM.
- **Local Model Performance**: Runs Mistral and other small models flawlessly, Llama-370b requires 40 GB RAM.
- **Apple Silicon's future**: Potentially a powerful platform for running local models with the development of Apple's MLX library.

###  Running LLM on iPhones

- **[MLC Chat App](https://apps.apple.com/us/app/mlc-chat/id6448482937)**: Allows running Mistral directly on iPhones, enabling offline LLM access.

###  Apple's Approach to LLMs

- **Focused on specific features**: Emphasizes using LLMs for functionalities like summarization and copy editing, avoiding the complexities of general chatbots.

###  Value of Running Local LLMs

- **Exploration and Learning**: Provides hands-on experience with different models, including less performant ones, which helps understand their strengths and weaknesses.
- **Privacy**: Keeps sensitive data on your local machine.
- **Offline Access**: Essential for situations without internet connectivity (e.g., flights, post-apocalyptic scenarios).

###  LLM Evaluation Tool (In Development)

- **Goals**:
  - Record evaluation results in SQLite.
  - Build an interface for comparing model responses and collecting human feedback (similar to [LM Arena](https://lmarena.ai/)).
- **Advantages of SQLite**:
  - Fast, free, universally available.
  - Works as a portable file format for sharing evaluation data.

###  Benchmarking and Logging

- **Planned Feature**: Log latency, token count, and duration of LLM operations in the SQLite database.
- **Challenges**: Determining token counts for models that don't provide this information.

###  Running LLM on Multiple Machines

- **Solution**: Leverage existing tools like [Ansible](https://www.ansible.com/) to run LLM commands in parallel on different machines within a LAN.
- **Unix Flexibility**: Highlights the ability to combine LLM with other Unix tools for customized workflows and automation.





{{< include /_about-author-cta.qmd >}}
