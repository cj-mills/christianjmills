---
title: Notes on Transformers Book Ch. 10
date: 2022-4-25
image: /images/empty.gif
title-block-categories: true
layout: post
toc: false
hide: false
search_exclude: false
comments:
  utterances:
    repo: cj-mills/christianjmills
description: Chapter 10 covers how to train a GPT-like model to generate Python source
  code from scratch.
categories: [ai, huggingface, nlp, notes]

aliases:
- /Notes-on-Transformers-Book-10/
---

* [Training Transformers from Scratch](#training-transformers-from-scratch)
* [Project: Python Source Code Generator](#project-python-source-code-generator)
* [Large Datasets and Where to Find Them](#large-datasets-and-where-to-find-them)
* [Building a Tokenizer](#building-a-tokenizer)
* [Training a Model from Scratch](#training-a-model-from-scratch)
* [Results and Analysis](#results-and-analysis)
* [References](#references)




------


```python
import transformers
import datasets
import accelerate

# Only print error messages
transformers.logging.set_verbosity_error()
datasets.logging.set_verbosity_error()

transformers.__version__, datasets.__version__, accelerate.__version__
```
```text
    ('4.18.0', '2.1.0', '0.5.1')
```

------

```python
import ast
# https://astor.readthedocs.io/en/latest/
import astor
import inspect
import textwrap
def print_source(obj, exclude_doc=True):
    
    # Get source code
    source = inspect.getsource(obj)
    # Remove any common leading whitespace from every line
    cleaned_source = textwrap.dedent(source)
    # Parse the source into an AST node.
    parsed = ast.parse(cleaned_source)

    for node in ast.walk(parsed):
        # Skip any nodes that are not class or function definitions
        if not isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
            continue
        
        if exclude_doc and len(node.body) > 1: node.body = node.body[1:]
        
    print(astor.to_source(parsed))
```

------



## Training Transformers from Scratch

* Efficiently training large models from scratch requires special tools for distributed training.



## Project: Python Source Code Generator

* The goal is to train a GPT-like model to generate Python source code.

### Existing AI Code Completion Products
* [GitHub Copilot](https://copilot.github.com/)
* [TabNine](https://www.tabnine.com/)
* [Kite](https://www.kite.com/)

### CodeParrot
* [GitHub Repository](https://github.com/huggingface/transformers/tree/main/examples/research_projects/codeparrot)
* CodeParrot is a GPT-2 model trained from scratch on Python code.



## Large Datasets and Where to Find Them

* Many domains often have large amounts of data available such as legal documents, biomedical databases, and programming codebases.
* Large datasets can usually only be labeled using heuristics or accompanying metadata.
* We can still use large unlabeled datasets to fine-tune language models for domain adaptation.
* Using a pretrained model forces you to use the model's corresponding tokenizer.
* Using a tokenizer trained on a corpus from a different domain is typically suboptimal.

### Challenges of Building a Large-Scale Corpus
* The model will inherit any defects in the pretraining corpus.
* It becomes more difficult to control or fully understand the contents of a dataset the larger it gets.
* Most exceedingly large datasets are not handcrafted.
* Creating large-scale datasets typically requires using data generated as a side effect of other activities.
* The high degree of automation used to create large-scale datasets means there is limited control over the content and the method to create them.
* There is an increased risk of training a model on lower-quality and biased data.
* A significant portion of the C4 corpus used to train T5 is machine-translated rather than human-translated.
* The stopword filtering in C4 disproportionately removed African-American English from the corpus.
* It is challenging to find a middle ground between including too much explicit content and erasing all mention of sexuality or gender.
* Common words like "sex" are absent from C4.
* There are many copyright violations in the Bookcorpus dataset used to train BERT.
* Bookcorpus also contains genre-skew toward "romance" novels.
* [Aligning Books and Movies: Towards Story-like Visual Explanations by Watching Movies and Reading Books](https://arxiv.org/abs/1506.06724)
* [Addressing "Documentation Debt" in Machine Learning Research: A Retrospective Datasheet for BookCorpus](https://arxiv.org/abs/2105.05241)



### Compare text generations from GPT and GPT-2
* The original GPT model trained predominately on BookCorpus.
* GPT-2 trained on web pages, blogs, and news articles linked from Reddit.

------


```python
from transformers import pipeline, set_seed
```

------

**Initialze text generation pipelines with the original GPT and GPT-2**


```python
generation_gpt = pipeline("text-generation", model="openai-gpt")
generation_gpt2 = pipeline("text-generation", model="gpt2")
```

**Note:** The main difference between the two models is the pretraining dataset.

------

**Compare the model sizes**


```python
def model_size(model):
    return sum(t.numel() for t in model.parameters())

print(f"GPT  size: {model_size(generation_gpt.model)/1000**2:.1f}M parameters")
print(f"GPT2 size: {model_size(generation_gpt2.model)/1000**2:.1f}M parameters")
```
```text
    GPT  size: 116.5M parameters
    GPT2 size: 124.4M parameters
```

**Note:** The original GPT model is approximately the same size as the smallest GPT-2 variant.

------

**Reset random seed**


```python
set_seed(1)
```

------

**Define a function to generate text using a prompt**


```python
def enum_pipeline_ouputs(pipe, prompt, num_return_sequences):
    out = pipe(prompt, num_return_sequences=num_return_sequences,
               clean_up_tokenization_spaces=True)
    return "\n".join(f"{i+1}." + s["generated_text"] for i, s in enumerate(out))
```

------

**Compare the output of the two models**


```python
prompt = "\nWhen they came back"
print("GPT completions:\n" + enum_pipeline_ouputs(generation_gpt, prompt, 3))
print("")
print("GPT-2 completions:\n" + enum_pipeline_ouputs(generation_gpt2, prompt, 3))
```
```text
    GPT completions:
    1.
    When they came back. 
     " we need all we can get, " jason said once they had settled into the back of the truck without anyone stopping them. " after getting out here, it 'll be up to us what to find. for now
    2.
    When they came back. 
     his gaze swept over her body. he 'd dressed her, too, in the borrowed clothes that she 'd worn for the journey. 
     " i thought it would be easier to just leave you there. " a woman like
    3.
    When they came back to the house and she was sitting there with the little boy. 
     " don't be afraid, " he told her. she nodded slowly, her eyes wide. she was so lost in whatever she discovered that tom knew her mistake
    
    GPT-2 completions:
    1.
    When they came back we had a big dinner and the other guys went to see what their opinion was on her. I did an hour and they were happy with it.
    2.
    When they came back to this island there had been another massacre, but he could not help but feel pity for the helpless victim who had been left to die, and that they had failed that day. And so was very, very grateful indeed.
    3.
    When they came back to our house after the morning, I asked if she was sure. She said, "Nope." The two kids were gone that morning. I thought they were back to being a good friend.
    
    When Dost
```

**Note:**
* The text generated with the original GPT model has a distinctive romance skew.
* GPT-2 generates more neutral text containing blog-like or adventure-related elements.
* A model reflects the language bias and over or underrepresentation of populations of the dataset used to train it.
* We need to consider the model's biases concerning the target audience.
* [Towards Accountability for Machine Learning Datasets: Practices from Software Engineering and Infrastructure](https://arxiv.org/abs/2010.13561)

------

### Building a Custom Code Dataset

* We can obtain a pretraining corpus of Python code from GitHub repositories.
* We can access GitHub repositories via the [GitHub REST API](https://docs.github.com/en/rest/guides/getting-started-with-the-rest-api) or public dataset inventories like [Google BigQuery](https://console.cloud.google.com/marketplace/product/github/github-repos?pli=1&project=majestic-vault-303101).
* The GitHub REST API is rate limited but provides access to additional attributes like star and downstream usage information.
* The [Libraries.io](https://libraries.io/) service monitors open source packages.

#### `bigquery-public-data.github_repos.contents` table
* The [`bigquery-public-data.github_repos.contents` table](https://console.cloud.google.com/bigquery?project=bigquery-public-data&page=table&t=contents&d=github_repos&p=bigquery-public-data&redirect_from_classic=true&ws=!1m5!1m4!4m3!1sbigquery-public-data!2sgithub_repos!3scontents) contains copies of all ASCII files less than 10MB in size.


#### CodeSearchNet corpus
* The CodeSearchNet corpus contains 2 million comment-code pairs from open-source libraries hosted on GitHub.
* It contains code and documentation for several programming languages.
* [Hugging Face Dataset Card](https://huggingface.co/datasets/code_search_net)

#### Creating a dataset with Google BigQuery
* [Unsupervised Translation of Programming Languages](https://arxiv.org/abs/2006.03511)

**Steps to export Python files**
1. Create a Google Cloud account.
2. Create a Google BigQuery project under your account.
3. Create a dataset inside the project.
4. Create a table in the dataset to store the results of the SQL request.
5. Prepare the following SQL query and specify a destination table
```sql
SELECT 
    f.repo_name, f.path, c.copies, c.size, c.content, l.license
FROM
    `bigquery-public-data.github_repos.files` AS f
JOIN
    `bigquery-public-data.github_repos.contents` AS c
ON
    f.id = c.id
JOIN
    `bigquery-public-data.github_repos.licenses` as l
ON
    f.repo_name = l.repo_name
WHERE
    NOT c.binary
    AND ((F.path LIKE '%.py')
        AND (c.size BETWEEN 1024 and 1048575))
```
6. Run the query

**Note:** Encoutered the following error when attempting to run the query
```text
Quota exceeded: Your project exceeded quota for free query bytes scanned. For more information, see https://cloud.google.com/bigquery/docs/troubleshoot-quotas 
```

* The above command processes about 2.6TB of data to extract 26.8 million files.
* The resulting dataset contains about 50 GB of compressed JSON files.
* The dataset is about 200GB when uncompressed.
* Each JSON file contains source code from Python files.
* The query filters empty files like `__init__.py` files and files larger than 1MB.
* The query includes the licenses for the files so we can filter the training data later on.

**Steps to download results from Google Cloud**
1. Export results to Google Cloud
    a. Create a bucket and a folder in Google Cloud Storage (GCS).
    b. Export your table to this bucket by selecting Export > Export to GCS, with a JSON export format and gzip compression.
2. Download the bucket to your local machine using [gsutil](https://cloud.google.com/storage/docs/gsutil)
 a. Install gsutil with pip install gsutil.
    b. Configure gsutil with your Google account: gsutil config.
    c. Copy your bucket on your machine: 
     ```bash
     gsutil -m -o "GSUtil:parallel_process_count=1" cp -r gs://<name_of_bucket>
     ```

 **Alternative: Download the dataset from Hugging Face Hub**

```bash
git clone https://huggingface.co/datasets/transformersbook/codeparrot
```

### To Filter the Noise or Not?
* Data preparation is crucial, and we should clean the dataset as much as possible. 
* The quality of code in GitHub repositories varies greatly.
* Having some noise in the training dataset makes our code generation system robust to noisy inputs at inference time but also makes predictions more random.
* The intended use case and whole-system integration determine whether you want more or less noisy data and add pre and post-filtering operations.

#### Potential steps to clean dataset
* Filter code based on stars or usage information.
* Code with more stars or higher usage is more likely to be higher quality.
* Remove duplicated code samples.
* Consider copyright information.
* Investigate the language used in the documentation, comments, or docstrings.
* Remove personal identifying information such as passwords or keys.

### Working with Large Datasets
* Working with large datasets requires additional considerations regarding disk space and RAM usage.
* It is common for datasets to be larger than the available RAM.
* The Hugging Face Datasets library provides memory mapping and streaming functionality to address RAM and disk space limitations.

#### Memory mapping
* Hugging Face Datasets uses a mechanism for zero-copy and zero-overhead memory mapping.
* The mechanism caches each dataset in a file that directly reflects the content in RAM.
* Hugging Face Datasets opens a read-only pointer to this file and uses it as a substitute for RAM.

------


```python
from datasets import load_dataset, DownloadConfig
```

------

**Decompress and load the downloaded dataset from the local folder**

> **Note:** The following code block assumes that you have downloaded the BigQuery dataset to a folder called `codeparrot`. We suggest skipping this step since it will unpack the compressed files and require ~180GB of disk space. This code is just for demonstration purposes and you can just continue below with the streamed dataset which will not consume that much disk space.

------


```python
download_config = DownloadConfig(delete_extracted=True, cache_dir="/mnt/980SSD/Datasets/codeparrot-cache")
dataset = load_dataset("/mnt/980SSD/Datasets/codeparrot", cache_dir="/mnt/980SSD/Datasets/codeparrot-cache", split="train",
                       download_config=download_config)
```
```text
    Dataset json downloaded and prepared to /mnt/980SSD/Datasets/codeparrot-cache/json/codeparrot-43fc192cc9f62326/0.0.0/ac0ca5f5289a6cf108e706efcf040422dbbfa8e658dee6a819f20d76bb84d26b. Subsequent calls will reuse this data.
```

**Note:** 
* The `delete_extracted=True` argument deletes the extracted files to free up disk space.
* The Hugging Face Datasets library extracted and read the compressed JSON files by loading them in a single optimized cache file.

------


```python
import psutil, os
```

------

**Check the size of the cached dataset**


```python
print(f"Number of python files code in dataset : {len(dataset)}")
ds_size = sum(os.stat(f["filename"]).st_size for f in dataset.cache_files)
# os.stat.st_size is expressed in bytes, so we convert to GB
print(f"Dataset size (cache file) : {ds_size / 2**30:.2f} GB")
# Process.memory_info is expressed in bytes, so we convert to MB
print(f"RAM used: {psutil.Process(os.getpid()).memory_info().rss >> 20} MB")
```
```text
    Number of python files code in dataset : 18695559
    Dataset size (cache file) : 183.68 GB
    RAM used: 4359 MB
```

**Note:**
* The dataset is much larger than the available RAM, but we can still load and access it.
* NLP data is typically lightweight to load compared to the model processing computations.
* The zero-copy/zero-overhead format uses Apache Arrow under the hood for efficiency.

------

#### Streaming

* Some datasets are too large to fit in most hard drives.
* The Hugging Face Datasets library supports streaming many compressed and uncompressed file formats that we can read line-by-line.
* Hugging Face Datasets opens and reads compressed JSON files on the fly in streaming mode.
* Streamed datasets are of the type [`IterableDataset`](https://huggingface.co/docs/datasets/v2.1.0/en/package_reference/main_classes#datasets.IterableDataset).
* We cannot access random elements and need to read them in order.
* Methods like `shuffle()` operate by fetching a buffer of examples and shuffling within this buffer.
* The samples of a streamed dataset are identical to those of a nonstreamed dataset.
* Streamed datasets do not generate a cache file on the drive or require significant RAM.
* Individual batches load into memory as requested, reducing the memory footprint.
* We can also stream remote datasets from the Hugging Face Hub, allowing us to use arbitrarily large datasets on small servers.

------


```python
streamed_dataset = load_dataset("/mnt/980SSD/Datasets/codeparrot", split="train", streaming=True)
```
```text
    AttributeError: '_io.BufferedReader' object has no attribute 'loc'
```

------

**Iterate through the streamed dataset**


```python
iterator = iter(streamed_dataset)

print(dataset[0] == next(iterator))
print(dataset[1] == next(iterator))
```
------

**Stream a remote dataset**


```python
remote_dataset = load_dataset('transformersbook/codeparrot', split="train",
                              streaming=True)
```

------

### Adding Datasets to the Hugging Face Hub

* Pushing our dataset to the Hugging Face Hub allows us to access it from a training server and share it with the community.

#### Command Line Steps
1. Log into Hugging Face account
```bash
huggingface-cli login
```
2. Create a new dataset repository on the Hub for the training split
```bash
huggingface-cli repo create --type dataset codeparrot-train
```
3. Create a new dataset repository on the Hub for the validation split
```bash
huggingface-cli repo create --type dataset codeparrot-valid
```
4. Clone the training repository
```bash
huggingface-cli repo create --type dataset codeparrot-train
```
5. Clone the validation repository
```bash
huggingface-cli repo create --type dataset codeparrot-valid
```
6. Copy all but the last GitHub file to the as the training set
```bash
cd codeparrot-train
cp ../codeparrot/*.json.gz .
rm ./file-000000000183.json.gz
```
7. Commit the files and push them to the Hub
```bash
git add .
git commit -m "Adding dataset files"
git push
```
8. Repeat the process for the validation set
```bash
cd ../codeparrot-valid
cp ../codeparrot/file-000000000183.json.gz
mv ./file-000000000183.json.gz ./file-000000000183_validation.json.gz
git add .
git commit -m "Adding dataset files"
git push
```
* It is good practice to add README cards that explain how the datasets were created and provide as much helpful information as possible.
* A well-documented dataset is more likely to be valuable to other people, including the future you.
* [Hugging Face Dataset Card Creation Guide](https://github.com/huggingface/datasets/blob/master/templates/README_guide.md)



## Building a Tokenizer

* It is crucial to stick with the same preprocessing design choices used during the pretraining process when using a pretrained model.
* Using a tokenizer prepared for another dataset when training a new model can be suboptimal.
    * The T5 tokenizer uses extensive stopword filtering and is unaware of some common English words like "sex." 
    * The CamemBERT tokenizer is only trained on French text and is unaware of common English words such as "being."

------


```python
from transformers import AutoTokenizer
```

------


```python
def tok_list(tokenizer, string):
    input_ids = tokenizer(string, add_special_tokens=False)["input_ids"]
    return [tokenizer.decode(tok) for tok in input_ids]
```

**Initialize tokenizers using the pretrained T5 and CamemBERT model vocabularies**


```python
tokenizer_T5 = AutoTokenizer.from_pretrained("t5-base")
tokenizer_camembert = AutoTokenizer.from_pretrained("camembert-base")
```

------

**Test the limitations of the T5 and CamemBERT tokenizers**


```python
print(f'T5 tokens for "sex": {tok_list(tokenizer_T5,"sex")}')
print(f'CamemBERT tokens for "being": {tok_list(tokenizer_camembert,"being")}')
```
```text
    T5 tokens for "sex": ['', 's', 'ex']
    CamemBERT tokens for "being": ['be', 'ing']
```

**Note:**

* Splitting such short and common words into subparts is often inefficient as it increases the sequence length of the model.
* It is essential to consider the domain and the preprocessing of the dataset used to train a tokenizer.
* The tokenizer and model can encode bias from the dataset that impacts the downstream behavior of the model.

------

### The Tokenizer Model

* Training a tokenizer is a way to create an optimal mapping from a string of text to a list of integers that the model can ingest.
* The optimal string-to-integer conversion involves a vocabulary consisting of a list of atomic strings and an associated method to convert, normalize, cut, or map a text string into a list of indices with this vocabulary.
* The list of indices is the input for the neural network.
* The tokenizer processing pipeline involves normalization, pre-tokenization, the tokenizer model, and postprocessing.
* The tokenizer model trains on a corpus.
* Several subword tokenization algorithms are available, such as BPE, WordPiece, and Unigram.
* BPE starts from a list of single characters and creates a vocabulary by progressively creating new tokens formed by merging the most frequently co-occurring basic units and adding them to the list.
* This process continues until we reach the predefined vocabulary size.
* Unigram initializes its base vocabulary with all the words in the corpus and potential subwords and progressively removes or splits the less helpful tokens until it reaches the target vocab size.
* The impact of the chosen tokenization algorithm on downstream performance varies based on the task.
* It is difficult to identify if one algorithm is better than the others.
* Both BPE and Unigram perform reasonably well in most cases.

### Measuring Tokenizer Performance
* It is challenging to measure a tokenizer's optimality and performance in practice.
* Subword fertility calculates the average number of subwords produced per tokenized word.
* The proportion of continued words refers to the amount of tokenized words in a corpus split into at least two subtokens.
* Coverage metrics track information like the proportion of unknown words or rarely used tokens in a tokenized corpus.
* We often estimate the robustness to misspelling or noise and model performance on such out-of-domain examples.
* These measures provide different views on tokenizer performance.
* However, they tend to ignore the interaction of the tokenizer with the model.
* The best way to evaluate tokenizers is using the downstream performance of the model.

### A Tokenizer for Python 
* Using a natural language pre-tokenizer for Python code might be suboptimal.
* Indentation has semantic meaning in Python code.
* Splitting on all whitespaces and removing them would remove valuable indentation information.
* Line breaks are not meaningful in Python code, and we can remove them without issue.
* Underscores can be part of single variable names and would not to use for splitting text.
* Byte-level tokenizers preserve spaces and might be a good candidate for tokenizing code.
* Python has a built-in tokenize module that splits Python code strings into meaningful units.
    * This approach is slow since it is Python-based and limited by the Python global interpreter lock (GIL).
* Most tokenizers provided by the Hugging Face Tokenizers library are in Rust and many orders of magnitude faster to train and use.

------


```python
from transformers import AutoTokenizer
```

------

**Test the byte-level GPT-2 tokenizer on Python code**


```python
python_code = r"""def say_hello():
    print("Hello, World!")
# Print it
say_hello()
"""
python_code
```
```text
    'def say_hello():\n    print("Hello, World!")\n# Print it\nsay_hello()\n'
```

------

```python
tokenizer = AutoTokenizer.from_pretrained("gpt2")
pd.DataFrame(tokenizer(python_code).tokens()).T
```
<div style="overflow-x:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>11</th>
      <th>12</th>
      <th>13</th>
      <th>14</th>
      <th>15</th>
      <th>16</th>
      <th>17</th>
      <th>18</th>
      <th>19</th>
      <th>20</th>
      <th>21</th>
      <th>22</th>
      <th>23</th>
      <th>24</th>
      <th>25</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>def</td>
      <td>Ġsay</td>
      <td>_</td>
      <td>hello</td>
      <td>():</td>
      <td>Ċ</td>
      <td>Ġ</td>
      <td>Ġ</td>
      <td>Ġ</td>
      <td>Ġprint</td>
      <td>("</td>
      <td>Hello</td>
      <td>,</td>
      <td>ĠWorld</td>
      <td>!"</td>
      <td>)</td>
      <td>Ċ</td>
      <td>#</td>
      <td>ĠPrint</td>
      <td>Ġit</td>
      <td>Ċ</td>
      <td>say</td>
      <td>_</td>
      <td>hello</td>
      <td>()</td>
      <td>Ċ</td>
    </tr>
  </tbody>
</table>
</div>
------


**Inspect the normalization step**


```python
print(tokenizer.backend_tokenizer.normalizer)
```
```text
    None
```

**Note:** The GPT-2 tokenizer does not use normalization and works directly on raw Unicode inputs.

------


```python
import pandas as pd
pd.set_option('max_colwidth', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
```

------

**Inspect the pre-tokenization step**


```python
pd.DataFrame(tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(python_code))
```
<div style="overflow-x:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>def</td>
      <td>(0, 3)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Ġsay</td>
      <td>(3, 7)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>_</td>
      <td>(7, 8)</td>
    </tr>
    <tr>
      <th>3</th>
      <td>hello</td>
      <td>(8, 13)</td>
    </tr>
    <tr>
      <th>4</th>
      <td>():</td>
      <td>(13, 16)</td>
    </tr>
    <tr>
      <th>5</th>
      <td>ĊĠĠĠ</td>
      <td>(16, 20)</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Ġprint</td>
      <td>(20, 26)</td>
    </tr>
    <tr>
      <th>7</th>
      <td>("</td>
      <td>(26, 28)</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Hello</td>
      <td>(28, 33)</td>
    </tr>
    <tr>
      <th>9</th>
      <td>,</td>
      <td>(33, 34)</td>
    </tr>
    <tr>
      <th>10</th>
      <td>ĠWorld</td>
      <td>(34, 40)</td>
    </tr>
    <tr>
      <th>11</th>
      <td>!")</td>
      <td>(40, 43)</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Ċ</td>
      <td>(43, 44)</td>
    </tr>
    <tr>
      <th>13</th>
      <td>#</td>
      <td>(44, 45)</td>
    </tr>
    <tr>
      <th>14</th>
      <td>ĠPrint</td>
      <td>(45, 51)</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Ġit</td>
      <td>(51, 54)</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Ċ</td>
      <td>(54, 55)</td>
    </tr>
    <tr>
      <th>17</th>
      <td>say</td>
      <td>(55, 58)</td>
    </tr>
    <tr>
      <th>18</th>
      <td>_</td>
      <td>(58, 59)</td>
    </tr>
    <tr>
      <th>19</th>
      <td>hello</td>
      <td>(59, 64)</td>
    </tr>
    <tr>
      <th>20</th>
      <td>()</td>
      <td>(64, 66)</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Ċ</td>
      <td>(66, 67)</td>
    </tr>
  </tbody>
</table>
</div>
**Note:**

* Hugging Face Tokenizers provides an offset tracking feature for switching between strings and tokens.
* Hugging Face Tokenizers tracks all operations on the input string so that it is possible to know what part of the input string corresponds to a token after tokenization.
* The numbers in the above output indicate where each token originates in the original string.
* The word "hello" corresponds to the characters 8 to 13 in the original string.
* Each Unicode character is composed of between 1 and 4 bytes.
* There are 143,859 Unicode characters and 256 elements in the byte alphabet.
* We can express each Unicode character as a sequence of bytes.
* We can have a model using an alphabet of only 256 words and process any Unicode string.

------

**Check the representations of some Unicode characters**


```python
a, e = u"a", u"€"
byte = ord(a.encode("utf-8"))
print(f'`{a}` is encoded as `{a.encode("utf-8")}` with a single byte: {byte}')
byte = [ord(chr(i)) for i in e.encode("utf-8")]
print(f'`{e}` is encoded as `{e.encode("utf-8")}` with three bytes: {byte}')
```
```text
    `a` is encoded as `b'a'` with a single byte: 97
    `€` is encoded as `b'\xe2\x82\xac'` with three bytes: [226, 130, 172]
```

**Note:**

* Building our vocabulary from the 143,859 Unicode characters would make the model's embedding layer extremely large.
* Using only the 256 byte-values as the vocabulary would result in longer input sequences.
    * [ByT5: Towards a token-free future with pre-trained byte-to-byte models](https://arxiv.org/abs/2105.13626)
        * The ByT5 paper provides a details study of the overhead from using byte values for our vocabulary.
* The BPE algorithm constructs a medium-sized vocabulary by extending the 256 byte-values with the most common combinations of bytes.
* The name, Byte-Pair Encoding, comes from a data compression technique proposed by Philip Gage in 1994, which operated on bytes.
    * [A New Algorithm for Data Compression Optimization](https://thesai.org/Publications/ViewPaper?Volume=3&Issue=8&Code=IJACSA&SerialNo=3)
* Standard BPE algorithms in NLP typically operate on Unicode strings rather than bytes.
    * A recent type of BPE that works specifically on bytes is called byte-level BPE.
* The BPE algorithms are designed to work with clean Unicode strings as inputs, not bytes, and expect regular ASCII characters in the inputs without spaces or control characters.
* Many Unicode control characters correspond to the 256 first bytes.
* The GPT-2 tokenizer maps all the 256 input bytes to printable Unicode characters, which the BPE algorithms can digest. 

------


```python
from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode
```

------

**Inspect the GPT-2 mapping of bytes to Unicode characters**


```python
byte_to_unicode_map = bytes_to_unicode()
unicode_to_byte_map = dict((v, k) for k, v in byte_to_unicode_map.items())
base_vocab = list(unicode_to_byte_map.keys())

print(f'Size of our base vocabulary: {len(base_vocab)}')
print(f'First element: `{base_vocab[0]}`, last element: `{base_vocab[-1]}`')
```

```text
Size of our base vocabulary: 256
First element: `!`, last element: `Ń`
```



------

**Examples of character mappings in BPE**


```python
byte_to_unicode_map = bytes_to_unicode()
unicode_to_byte_map = dict((v, k) for k, v in byte_to_unicode_map.items())
base_vocab = list(unicode_to_byte_map.keys())

examples = [
    ['Regular characters', '`a` and `?`', f'{ord("a")} and {ord("?")}' , f'`{byte_to_unicode_map[ord("a")]}` and `{byte_to_unicode_map[ord("?")]}`'],
    ['Nonprintable control character (carriage return)', '`U+000D`', f'13', f'`{byte_to_unicode_map[13]}`'],
    ['A space', '` `', f'{ord(" ")}', f'`{byte_to_unicode_map[ord(" ")]}`'],
    ['A nonbreakable space', '`\\xa0`', '160', f'`{byte_to_unicode_map[ord(chr(160))]}`'],
    ['A newline character', '`\\n`', '10', f'`{byte_to_unicode_map[ord(chr(10))]}`'],
]

pd.DataFrame(examples, columns = ['Description', 'Character', 'Bytes', 'Mapped bytes'])
```
<div style="overflow-x:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Description</th>
      <th>Character</th>
      <th>Bytes</th>
      <th>Mapped bytes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Regular characters</td>
      <td>`a` and `?`</td>
      <td>97 and 63</td>
      <td>`a` and `?`</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Nonprintable control character (carriage return)</td>
      <td>`U+000D`</td>
      <td>13</td>
      <td>`č`</td>
    </tr>
    <tr>
      <th>2</th>
      <td>A space</td>
      <td>` `</td>
      <td>32</td>
      <td>`Ġ`</td>
    </tr>
    <tr>
      <th>3</th>
      <td>A nonbreakable space</td>
      <td>`\xa0`</td>
      <td>160</td>
      <td>`ł`</td>
    </tr>
    <tr>
      <th>4</th>
      <td>A newline character</td>
      <td>`\n`</td>
      <td>10</td>
      <td>`Ċ`</td>
    </tr>
  </tbody>
</table>
</div>
------


**Inspect the pre-tokenization step again**


```python
pd.DataFrame(tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(python_code))
```
<div style="overflow-x:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>def</td>
      <td>(0, 3)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Ġsay</td>
      <td>(3, 7)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>_</td>
      <td>(7, 8)</td>
    </tr>
    <tr>
      <th>3</th>
      <td>hello</td>
      <td>(8, 13)</td>
    </tr>
    <tr>
      <th>4</th>
      <td>():</td>
      <td>(13, 16)</td>
    </tr>
    <tr>
      <th>5</th>
      <td>ĊĠĠĠ</td>
      <td>(16, 20)</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Ġprint</td>
      <td>(20, 26)</td>
    </tr>
    <tr>
      <th>7</th>
      <td>("</td>
      <td>(26, 28)</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Hello</td>
      <td>(28, 33)</td>
    </tr>
    <tr>
      <th>9</th>
      <td>,</td>
      <td>(33, 34)</td>
    </tr>
    <tr>
      <th>10</th>
      <td>ĠWorld</td>
      <td>(34, 40)</td>
    </tr>
    <tr>
      <th>11</th>
      <td>!")</td>
      <td>(40, 43)</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Ċ</td>
      <td>(43, 44)</td>
    </tr>
    <tr>
      <th>13</th>
      <td>#</td>
      <td>(44, 45)</td>
    </tr>
    <tr>
      <th>14</th>
      <td>ĠPrint</td>
      <td>(45, 51)</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Ġit</td>
      <td>(51, 54)</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Ċ</td>
      <td>(54, 55)</td>
    </tr>
    <tr>
      <th>17</th>
      <td>say</td>
      <td>(55, 58)</td>
    </tr>
    <tr>
      <th>18</th>
      <td>_</td>
      <td>(58, 59)</td>
    </tr>
    <tr>
      <th>19</th>
      <td>hello</td>
      <td>(59, 64)</td>
    </tr>
    <tr>
      <th>20</th>
      <td>()</td>
      <td>(64, 66)</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Ċ</td>
      <td>(66, 67)</td>
    </tr>
  </tbody>
</table>
</div>
**Note:**

* Consecutive spaces count as a single word.
* Each space preceding a word is attached to and considered part of the following word.

------

**Check the size of the GPT-2 vocabulary**


```python
print(f"Size of the vocabulary: {len(tokenizer)}")
```
```text
    Size of the vocabulary: 50257
```

**Note:** The GPT-2 vocabulary consists of the base vocabulary with the 256 values of the bytes, 50,000 additional tokens created by repeatedly merging the most commonly occurring tokens, and a special character to represent document boundaries.

------

**Run the GPT-2 tokenizer pipeline again** 


```python
pd.DataFrame(tokenizer(python_code).tokens()).T
```
<div style="overflow-x:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>11</th>
      <th>12</th>
      <th>13</th>
      <th>14</th>
      <th>15</th>
      <th>16</th>
      <th>17</th>
      <th>18</th>
      <th>19</th>
      <th>20</th>
      <th>21</th>
      <th>22</th>
      <th>23</th>
      <th>24</th>
      <th>25</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>def</td>
      <td>Ġsay</td>
      <td>_</td>
      <td>hello</td>
      <td>():</td>
      <td>Ċ</td>
      <td>Ġ</td>
      <td>Ġ</td>
      <td>Ġ</td>
      <td>Ġprint</td>
      <td>("</td>
      <td>Hello</td>
      <td>,</td>
      <td>ĠWorld</td>
      <td>!"</td>
      <td>)</td>
      <td>Ċ</td>
      <td>#</td>
      <td>ĠPrint</td>
      <td>Ġit</td>
      <td>Ċ</td>
      <td>say</td>
      <td>_</td>
      <td>hello</td>
      <td>()</td>
      <td>Ċ</td>
    </tr>
  </tbody>
</table>
</div>
**Note:**

* The tokenizer keeps most of the words but splits indentations into several consecutive spaces.
* The training corpus for the tokenizer mostly contained text where consecutive spaces are rare.
* The BPE model does not include a specific token for indentation, meaning it is not well suited for Python code.

------

### Training a Tokenizer

* A tokenizer learns which letter combinations are the most frequent in a target corpus.
* The corpus does not need to be very large, just representative of the target domain.
* We can train a tokenizer on a target corpus using the [`tokenizer.train_new_from_iterator()`](https://huggingface.co/docs/transformers/main/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.train_new_from_iterator) method.
* We need to specify a target vocab size and prepare an iterator to supply lists of input strings.
* The tokenizer might store unusual character sequences depending on the vocab size and the exact texts in the corpus.

**Check the longest words in the GPT-2 tokenizer vocabulary**


```python
tokens = sorted(tokenizer.vocab.items(), key=lambda x: len(x[0]), reverse=True)
pd.DataFrame([f'{tokenizer.convert_tokens_to_string(t)}' for t, _ in tokens[:8]]).style.hide(axis='columns')
```
<div style="overflow-x:auto;">
<table id="T_72966">
  <thead>
  </thead>
  <tbody>
    <tr>
      <th id="T_72966_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_72966_row0_col0" class="data row0 col0" >ÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂ</td>
    </tr>
    <tr>
      <th id="T_72966_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_72966_row1_col0" class="data row1 col0" > =================================================================</td>
    </tr>
    <tr>
      <th id="T_72966_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_72966_row2_col0" class="data row2 col0" > ----------------------------------------------------------------</td>
    </tr>
    <tr>
      <th id="T_72966_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_72966_row3_col0" class="data row3 col0" >================================================================</td>
    </tr>
    <tr>
      <th id="T_72966_level0_row4" class="row_heading level0 row4" >4</th>
      <td id="T_72966_row4_col0" class="data row4 col0" >________________________________________________________________</td>
    </tr>
    <tr>
      <th id="T_72966_level0_row5" class="row_heading level0 row5" >5</th>
      <td id="T_72966_row5_col0" class="data row5 col0" >----------------------------------------------------------------</td>
    </tr>
    <tr>
      <th id="T_72966_level0_row6" class="row_heading level0 row6" >6</th>
      <td id="T_72966_row6_col0" class="data row6 col0" >ÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂ</td>
    </tr>
    <tr>
      <th id="T_72966_level0_row7" class="row_heading level0 row7" >7</th>
      <td id="T_72966_row7_col0" class="data row7 col0" >................................................................</td>
    </tr>
  </tbody>
</table>
</div>
**Note:** These tokens look like separator lines used on forums.

------

**Check the least frequent words**


```python
tokens = sorted(tokenizer.vocab.items(), key=lambda x: x[1], reverse=True)
pd.DataFrame([f'{tokenizer.convert_tokens_to_string(t)}' for t, _ in tokens[:12]])
```
<div style="overflow-x:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>&lt;|endoftext|&gt;</td>
    </tr>
    <tr>
      <th>1</th>
      <td>gazed</td>
    </tr>
    <tr>
      <th>2</th>
      <td>informants</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Collider</td>
    </tr>
    <tr>
      <th>4</th>
      <td>regress</td>
    </tr>
    <tr>
      <th>5</th>
      <td>ominated</td>
    </tr>
    <tr>
      <th>6</th>
      <td>amplification</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Compar</td>
    </tr>
    <tr>
      <th>8</th>
      <td>…."</td>
    </tr>
    <tr>
      <th>9</th>
      <td>(/</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Commission</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Hitman</td>
    </tr>
  </tbody>
</table>
</div>
**Note:**

* The `<|endoftext|>` token specifies the end of a text sequence and is not from the training corpus.
* The model has to learn an associated word embedding for each token.
* This tokenizer embeds some highly time and space-specific knowledge of the world by granting these words separate tokens.
* Overly specific tokens can indicate the target vocab size is too large or that the corpus contains peculiar tokens.
* We don't want the embedding matrix to contain too many noisy words.

------


```python
from tqdm.auto import tqdm
```

------

**Train a fresh tokenizer on 100,000 documents**


```python
length = 100000
dataset_name = 'transformersbook/codeparrot-train'
dataset = load_dataset(dataset_name, split="train", streaming=True)
iter_dataset = iter(dataset)

def batch_iterator(batch_size=10):
    for _ in tqdm(range(0, length, batch_size)):
        yield [next(iter_dataset)['content'] for _ in range(batch_size)]

new_tokenizer = tokenizer.train_new_from_iterator(batch_iterator(), 
                                                  vocab_size=12500,
                                                  initial_alphabet=base_vocab)
```
------

**Examine the first tokens added by the BPE algorithm**


```python
tokens = sorted(new_tokenizer.vocab.items(), key=lambda x: x[1], reverse=False)
pd.DataFrame([f'{tokenizer.convert_tokens_to_string(t)}' for t, _ in tokens[257:280]]).T
```
<div style="overflow-x:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>11</th>
      <th>12</th>
      <th>13</th>
      <th>14</th>
      <th>15</th>
      <th>16</th>
      <th>17</th>
      <th>18</th>
      <th>19</th>
      <th>20</th>
      <th>21</th>
      <th>22</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td>se</td>
      <td>in</td>
      <td></td>
      <td>re</td>
      <td>on</td>
      <td>te</td>
      <td>\n</td>
      <td>\n</td>
      <td>or</td>
      <td>st</td>
      <td>de</td>
      <td>\n</td>
      <td>th</td>
      <td>le</td>
      <td>=</td>
      <td>lf</td>
      <td>self</td>
      <td>me</td>
      <td>al</td>
    </tr>
  </tbody>
</table>
</div>
**Note:**

* There are various standard levels of indentation and whitespace tokens and short common Python keywords.
* The BPE algorithm is working as intended.

------

**Examine the last tokens added by the BPE algorithm**


```python
pd.DataFrame([f'{new_tokenizer.convert_tokens_to_string(t)}' for t,_ in tokens[-12:]]).T
```
<div style="overflow-x:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>11</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>capt</td>
      <td>embedded</td>
      <td>regarding</td>
      <td>Bundle</td>
      <td>355</td>
      <td>recv</td>
      <td>dmp</td>
      <td>vault</td>
      <td>Mongo</td>
      <td>possibly</td>
      <td>implementation</td>
      <td>Matches</td>
    </tr>
  </tbody>
</table>
</div>
**Note:**

* There are still some relatively common words like the  [`recv`](https://docs.python.org/3/library/socket.html#socket.socket.recv) method.
* There are also some more noisy words potentially from comments.

------

**Test the custom tokenizer on the sample code**


```python
pd.DataFrame(new_tokenizer(python_code).tokens()).T
```
<div style="overflow-x:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>11</th>
      <th>12</th>
      <th>13</th>
      <th>14</th>
      <th>15</th>
      <th>16</th>
      <th>17</th>
      <th>18</th>
      <th>19</th>
      <th>20</th>
      <th>21</th>
      <th>22</th>
      <th>23</th>
      <th>24</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>def</td>
      <td>Ġs</td>
      <td>ay</td>
      <td>_</td>
      <td>hello</td>
      <td>():</td>
      <td>ĊĠĠĠ</td>
      <td>Ġprint</td>
      <td>("</td>
      <td>Hello</td>
      <td>,</td>
      <td>ĠWor</td>
      <td>ld</td>
      <td>!")</td>
      <td>Ċ</td>
      <td>#</td>
      <td>ĠPrint</td>
      <td>Ġit</td>
      <td>Ċ</td>
      <td>s</td>
      <td>ay</td>
      <td>_</td>
      <td>hello</td>
      <td>()</td>
      <td>Ċ</td>
    </tr>
  </tbody>
</table>
</div>
**Note:** The tokenize splits common English words like "World" and "say."

------


```python
import keyword
```

------

#### `keyword`

* [Documentation](https://docs.python.org/3/library/keyword.html)
* Determine if a string is a [keyword](https://docs.python.org/3/reference/lexical_analysis.html#keywords) or [soft keyword](https://docs.python.org/3/reference/lexical_analysis.html#soft-keywords).

**Check if all the Python reserved words are in the vocabulary**


```python
print(f'There are in total {len(keyword.kwlist)} Python keywords.')
for keyw in keyword.kwlist:
    if keyw not in new_tokenizer.vocab:
        print(f'No, keyword `{keyw}` is not in the vocabulary')
```
```text
    There are in total 36 Python keywords.
    No, keyword `__peg_parser__` is not in the vocabulary
    No, keyword `await` is not in the vocabulary
    No, keyword `finally` is not in the vocabulary
    No, keyword `nonlocal` is not in the vocabulary
```

**Note:** Several frequent keywords like "finally" are not in the vocabulary.

------

**Reset random seed**


```python
set_seed(1)
```

------

**Train a tokenizer using a larger target vocab size and dataset sample**


```python
length = 200000
new_tokenizer_larger = tokenizer.train_new_from_iterator(batch_iterator(),
    vocab_size=32768, initial_alphabet=base_vocab)
```
------

**Check the last tokens added**


```python
tokens = sorted(new_tokenizer_larger.vocab.items(), key=lambda x: x[1],
                reverse=False)
pd.DataFrame([f'{tokenizer.convert_tokens_to_string(t)}' for t, _ in tokens[-12:]]).T
```
<div style="overflow-x:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>11</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>组</td>
      <td>typically</td>
      <td>ARGIN</td>
      <td>Termination</td>
      <td>StaticText</td>
      <td>interesting</td>
      <td>Circular</td>
      <td>combinatorics</td>
      <td>)([</td>
      <td>969</td>
      <td>EAR</td>
      <td>Gap</td>
    </tr>
  </tbody>
</table>
</div>
**Note:** The group of least-frequent tokens does not contain any  Python keywords.

------

**Test the new tokenizer on the sample code**


```python
pd.DataFrame(new_tokenizer_larger(python_code).tokens()).T
```
<div style="overflow-x:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>11</th>
      <th>12</th>
      <th>13</th>
      <th>14</th>
      <th>15</th>
      <th>16</th>
      <th>17</th>
      <th>18</th>
      <th>19</th>
      <th>20</th>
      <th>21</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>def</td>
      <td>Ġsay</td>
      <td>_</td>
      <td>hello</td>
      <td>():</td>
      <td>ĊĠĠĠ</td>
      <td>Ġprint</td>
      <td>("</td>
      <td>Hello</td>
      <td>,</td>
      <td>ĠWorld</td>
      <td>!")</td>
      <td>Ċ</td>
      <td>#</td>
      <td>ĠPrint</td>
      <td>Ġit</td>
      <td>Ċ</td>
      <td>say</td>
      <td>_</td>
      <td>hello</td>
      <td>()</td>
      <td>Ċ</td>
    </tr>
  </tbody>
</table>
</div>
**Note:** The new tokenizer keeps the indents in the vocabulary and does not split common English words.

------


```python
for keyw in keyword.kwlist:
    if keyw not in new_tokenizer_larger.vocab:
        print(f'No, keyword `{keyw}` is not in the vocabulary')
```
```text
    No, keyword `__peg_parser__` is not in the vocabulary
    No, keyword `nonlocal` is not in the vocabulary
```

**Note:**
* The new tokenizer vocabulary is still missing a couple of rare Python keywords, neither of which are relevant for most Python code.
* The `__peg_parser__` keyword is an easter egg for the new [PEG parser](https://peps.python.org/pep-0617/) and [will not be in Python 3.10](https://bugs.python.org/issue40939).
* The `nonlocal` keyword causes listed identifiers to refer to previously bound variables in the nearest enclosing scope, excluding globals.
* The new tokenizer is more efficient than the standard GPT-2 tokenizer as it uses fewer tokens to encode a given code sample.

------

**Disable Tokenizers Parallelism**


```python
%env TOKENIZERS_PARALLELISM=false
```
```text
    env: TOKENIZERS_PARALLELISM=false
```

------

### Saving a Custom Tokenizer on the Hub

**Log into Hugging Face account**


```python
from huggingface_hub import notebook_login
```


```python
notebook_login()
```
```text
    Login successful
    Your token has been saved to /home/innom-dt/.huggingface/token
```

------

**Push custom tokenizer to Hugging Face Hub**


```python
model_ckpt = "codeparrot"
```


```python
# org = "transformersbook"
new_tokenizer_larger.push_to_hub(model_ckpt)
```
```text
    'https://huggingface.co/cj-mills/codeparrot/commit/97c7905ef55cb4139e88f9b9d17225c372fc8f55'
```

------

**Load the custom tokenizer from the Hub repository**


```python
# reloaded_tokenizer = AutoTokenizer.from_pretrained(org + "/" + model_ckpt)
reloaded_tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
pd.DataFrame(reloaded_tokenizer(python_code).tokens()).T
```
<div style="overflow-x:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>11</th>
      <th>12</th>
      <th>13</th>
      <th>14</th>
      <th>15</th>
      <th>16</th>
      <th>17</th>
      <th>18</th>
      <th>19</th>
      <th>20</th>
      <th>21</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>def</td>
      <td>Ġsay</td>
      <td>_</td>
      <td>hello</td>
      <td>():</td>
      <td>ĊĠĠĠ</td>
      <td>Ġprint</td>
      <td>("</td>
      <td>Hello</td>
      <td>,</td>
      <td>ĠWorld</td>
      <td>!")</td>
      <td>Ċ</td>
      <td>#</td>
      <td>ĠPrint</td>
      <td>Ġit</td>
      <td>Ċ</td>
      <td>say</td>
      <td>_</td>
      <td>hello</td>
      <td>()</td>
      <td>Ċ</td>
    </tr>
  </tbody>
</table>
</div>



**Push the smaller tokenizer to Hugging Face Hub**


```python
new_tokenizer.push_to_hub(model_ckpt+ "-small-vocabulary")
```
```text
    'https://huggingface.co/cj-mills/codeparrot-small-vocabulary/commit/b4efe8c9692ce772175b97b01cffc9f1924ae706'
```

------



## Training a Model from Scratch

* [CodeParrot Trainng Script and Instructions](https://github.com/huggingface/transformers/tree/main/examples/research_projects/codeparrot)

### A Tale of Pretraining Objectives
* The large-scale pretraining corpus allows us to tackle several downstream tasks.
* The selected task will influence which pretraining objective we choose.

#### Causal language modeling
* Causal language modeling is a self-supervised approach that does not require annotations.
* Code autocompletion is a directly related downstream task.
* We can provide a model with the beginning of a code sample and have it generate possible completions.
* A decoder-only architecture like the GPT family is usually best suited for this task.

#### Masked language modeling
* Masked language modeling (also called denoising) is a self-supervised training objective.
* We can provide a model with a noisy code sample (e.g., by replacing a code instruction with a random or masked word) and have it reconstruct the original clean sequence.
* Masked language modeling is not directly related to a downstream task like autocompletion, but it is a practical pretraining objective for learning general representations.
* We can combine masked language modeling with fine-tuning the model on a downstream task.
* Encoder architectures are best suited to this pretraining objective.

#### Sequence-to-sequence training
* Sequence-to-sequence training is a supervised learning objective where one category serves as input while another serves as labels.
* We can use a heuristic like regular expressions to separate comments or docstrings from code and build a large-scale annotated dataset of code-comment pairs.
* We can then use this dataset to train a model to transcript comments in code or vice versa.
* Document generation from code and code generation from comments are directly-related downstream tasks.
* Encoder decoder architectures are best suited to sequence-to-sequence objectives.

### Initializing the Model

> **NOTE**: In the following code block, a large GPT-2 checkpoint is loaded into memory. On platforms like Colab and Kaggle, this can cause the instance to crash due to insufficient RAM or GPU memory. You can still run the example if you use the small checkpoint by replacing the configuration with `config = AutoConfig.from_pretrained("gpt2", vocab_size=len(tokenizer))`.

------


```python
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
```

------

**Instantiate a tokenizer using the custom checkpoint**


```python
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
```

------

**Start with the hyperparameters for training the 1.5 billion-parameter GPT-2 variant**


```python
config = AutoConfig.from_pretrained("gpt2-xl", vocab_size=len(tokenizer))
config
```
```text
    GPT2Config {
      "_name_or_path": "gpt2-xl",
      "activation_function": "gelu_new",
      "architectures": [
        "GPT2LMHeadModel"
      ],
      "attn_pdrop": 0.1,
      "bos_token_id": 50256,
      "embd_pdrop": 0.1,
      "eos_token_id": 50256,
      "initializer_range": 0.02,
      "layer_norm_epsilon": 1e-05,
      "model_type": "gpt2",
      "n_ctx": 1024,
      "n_embd": 1600,
      "n_head": 25,
      "n_inner": null,
      "n_layer": 48,
      "n_positions": 1024,
      "output_past": true,
      "reorder_and_upcast_attn": false,
      "resid_pdrop": 0.1,
      "scale_attn_by_inverse_layer_idx": false,
      "scale_attn_weights": true,
      "summary_activation": null,
      "summary_first_dropout": 0.1,
      "summary_proj_to_labels": true,
      "summary_type": "cls_index",
      "summary_use_proj": true,
      "task_specific_params": {
        "text-generation": {
          "do_sample": true,
          "max_length": 50
        }
      },
      "transformers_version": "4.18.0",
      "use_cache": true,
      "vocab_size": 32768
    }
```

------

**Free unoccupied cached memory**


```python
import torch
torch.cuda.empty_cache()
```

------

**Initialize a GPT-2 XL model using the custom tokenizer**


```python
model = AutoModelForCausalLM.from_config(config)
```

------

**Check the model size**


```python
print(f'GPT-2 (xl) size: {model_size(model)/1000**2:.1f}M parameters')
```
```text
    GPT-2 (xl) size: 1529.6M parameters
```

**Note:** Large models are generally more efficient to train as long as the dataset is reasonably large.

------


```python
!git lfs install
```
```text
    Updated Git hooks.
    Git LFS initialized.
```

------

**Save the newly initialized model to the Hub**


```python
model.save_pretrained("models/" + model_ckpt+"-large", push_to_hub=True)
```
```text
    OSError: EOF
    error: failed to push some refs to 'https://user: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@huggingface.co/cj-mills/codeparrot-large'
```

------

**Initialize a smaller GPT-2 variant using the custom tokenizer**


```python
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
config_small = AutoConfig.from_pretrained("gpt2", vocab_size=len(tokenizer))
model_small = AutoModelForCausalLM.from_config(config_small)
```

------

**Check smaller model size**


```python
print(f'GPT-2 size: {model_size(model_small)/1000**2:.1f}M parameters')
```

```text
GPT-2 size: 111.0M parameters
```



------

**Push the smaller model to the Hub**


```python
model_small.save_pretrained("models/" + model_ckpt + "-small", push_to_hub=True)
```
------

### Implementing the Dataloader

* We want to supply our model with sequences that fill its context length for maximal efficiency.
* Some code examples might be shorter or longer than the 1,024 token context length.
* We can concatenate several examples to create a long sequence using the EOS token as a separator.
* We then split this sequence into equally sized chunks that fill the context length.

```python
input_characters = number_of_sequences * sequence_length * characters_per_token
```
* `input_characters`: the number of characters in the string input to the tokenizer
* `number_of_seqeunces`: the number of (truncated) sequences returned by the tokenizer
* `sequence_length`: the number of tokens per sequence returned by the tokenizer
* `characters_per_token`: the average number of characters per output token that we first need to estimate

**Estimate the average character length per token**


```python
examples, total_characters, total_tokens = 500, 0, 0
dataset = load_dataset('transformersbook/codeparrot-train', split='train',
                       streaming=True)

for _, example in tqdm(zip(range(examples), iter(dataset)), total=examples):
    total_characters += len(example['content'])
    total_tokens += len(tokenizer(example['content']).tokens())

characters_per_token = total_characters / total_tokens
```
------


```python
print(characters_per_token)
```

```text
3.621530410894045
```

**Note:** We'll round this to $3.6$.

------


```python
import torch
from torch.utils.data import IterableDataset
```

------

**Define an IterableDataset class for preparing constant-length inputs**


```python
class ConstantLengthDataset(IterableDataset):
    
    def __init__(self, tokenizer, dataset, seq_length=1024,
                 num_of_sequences=1024, chars_per_token=3.6):
        self.tokenizer = tokenizer
        self.concat_token_id = tokenizer.eos_token_id
        self.dataset = dataset
        self.seq_length = seq_length
        self.input_characters = num_of_sequences * seq_length * chars_per_token
    
    def __iter__(self):
        iterator = iter(self.dataset)
        more_examples = True
        while more_examples:
            buffer, buffer_len = [], 0
            while True:
                # Check if the buffer is full
                if buffer_len >= self.input_characters:
                    m=f"Buffer full: {buffer_len}>={self.input_characters:.0f}"
                    print(m)
                    break
                # Try to add the next code sample to the buffer
                try:
                    m=f"Fill buffer: {buffer_len}<{self.input_characters:.0f}"
                    print(m)
                    buffer.append(next(iterator)["content"])
                    buffer_len += len(buffer[-1])
                # Reset iterator
                except StopIteration:
                    iterator = iter(self.dataset)
            
            all_token_ids = []
            # Tokenize the code samples in the buffer
            tokenized_inputs = self.tokenizer(buffer, truncation=False)
            # Concatenate the tokenized code samples
            for tokenized_input in tokenized_inputs['input_ids']:
                all_token_ids.extend(tokenized_input + [self.concat_token_id])
            # Split the sequence into equally sized chunks
            for i in range(0, len(all_token_ids), self.seq_length):
                input_ids = all_token_ids[i : i + self.seq_length]
                if len(input_ids) == self.seq_length:
                    yield torch.tensor(input_ids)
```

**Note:** We don't need attention masks here since all sequences precisely fill the context length of 1024 tokens.

------

**Prepare the constant-length dataset**


```python
shuffled_dataset = dataset.shuffle(buffer_size=100)
constant_length_dataset = ConstantLengthDataset(tokenizer, shuffled_dataset,
                                                num_of_sequences=10)
```

**Note:** We can't shuffle iterable datasets as a whole, so we need to use a buffer instead.

------

**Verify the dataset yields equal length chunks**


```python
dataset_iterator = iter(constant_length_dataset)

lengths = [len(b) for _, b in zip(range(5), dataset_iterator)]
print(f"Lengths of the sequences: {lengths}")
```
```text
    Fill buffer: 0<36864
    Fill buffer: 4344<36864
    Fill buffer: 5460<36864
    Fill buffer: 7467<36864
    Fill buffer: 13812<36864
    Fill buffer: 16142<36864
    Fill buffer: 17571<36864
    Fill buffer: 25693<36864
    Fill buffer: 27359<36864
    Fill buffer: 28903<36864
    Fill buffer: 32076<36864
    Buffer full: 49996>=36864
    Lengths of the sequences: [1024, 1024, 1024, 1024, 1024]
```

------

### Defining the Training Loop

* Even modern GPUs can't train a model at GPT-2 scale in a reasonable time.
* We need to use data parallelism to utilize several GPUs for training.
* The Hugging Face Accelerate library makes distributed training and changing the underlying hardware for training easier.
* Hugging Face Accelerate provides an API to make training scripts run with mixed precision and in any distributed setting.
* The same code can run seamlessly on your local machine for debugging and a beefy training cluster for a final training run.

------


```python
from argparse import Namespace
```

------

**Define the hyperparameters**


```python
# Commented parameters correspond to the small model
config = {"train_batch_size": 2, # 12
          "valid_batch_size": 2, # 12
          "weight_decay": 0.1,
          "shuffle_buffer": 1000,
          "learning_rate": 2e-4, # 5e-4
          "lr_scheduler_type": "cosine",
          "num_warmup_steps": 750, # 2000
          "gradient_accumulation_steps": 16, # 1
          "max_train_steps": 50000, # 150000
          "max_eval_steps": -1,
          "seq_length": 1024,
          "seed": 1,
          "save_checkpoint_steps": 50000} # 15000

args = Namespace(**config)
```

------


```python
from torch.utils.tensorboard import SummaryWriter
import logging
import wandb
```

------

#### `logging.getLogger()`

* [Documentation](https://docs.python.org/3/library/logging.html#logging.getLogger)
* Create a [Logger object](https://docs.python.org/3/library/logging.html#logger-objects).

#### `torch.utils.tensorboard.writer.SummaryWriter`
* [Documentation](https://pytorch.org/docs/stable/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter)
* Write entries directly to event files for [TensorBoard](https://github.com/tensorflow/tensorboard)

#### `wandb`
* [GitHub Repository](https://github.com/wandb/client)
* [Documentation](https://docs.wandb.ai/)
* A tool for visualizing and tracking machine learning experiements.

------

**Define a method to initialize the loggers for the training process**


```python
def setup_logging(project_name):
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO, handlers=[
        logging.FileHandler(f"log/debug_{accelerator.process_index}.log"),
        logging.StreamHandler()])
    if accelerator.is_main_process: # We only want to set up logging once
        wandb.init(project=project_name, config=args)
        run_name = wandb.run.name
        tb_writer = SummaryWriter()
        tb_writer.add_hparams(vars(args), {'0': 0})
        logger.setLevel(logging.INFO)
        datasets.utils.logging.set_verbosity_debug()
        transformers.utils.logging.set_verbosity_info()
    else:
        tb_writer = None
        run_name = ''
        logger.setLevel(logging.ERROR)
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
    return logger, tb_writer, run_name
```

**Note:**
* Each worker gets a unique `accelerator.process_index`, which we use with the FileHandler to write the logs of each worker to an individual file.
* We'll use the unique `run_name` to name our experiment branch on the Hub.

------

**Define function to log metrics with TensorBoard and Weights and Biases**


```python
def log_metrics(step, metrics):
    logger.info(f"Step {step}: {metrics}")
    if accelerator.is_main_process:
        wandb.log(metrics)
        [tb_writer.add_scalar(k, v, step) for k, v in metrics.items()]
```

------


```python
from torch.utils.data.dataloader import DataLoader
```

------

**Define a function to create dataloaders for the training and validation sets**


```python
def create_dataloaders(dataset_name):
    train_data = load_dataset(dataset_name+'-train', split="train",
                              streaming=True)
    train_data = train_data.shuffle(buffer_size=args.shuffle_buffer,
                                    seed=args.seed)
    valid_data = load_dataset(dataset_name+'-valid', split="validation",
                              streaming=True)
    
    train_dataset = ConstantLengthDataset(tokenizer, train_data,
                                          seq_length=args.seq_length)
    valid_dataset = ConstantLengthDataset(tokenizer, valid_data,
                                          seq_length=args.seq_length)
    
    train_dataloader=DataLoader(train_dataset, batch_size=args.train_batch_size)
    eval_dataloader=DataLoader(valid_dataset, batch_size=args.valid_batch_size)
    return train_dataloader, eval_dataloader
```

**Note:** Hugging Face Accelerate takes care of distributing batches to each worker.

------

**Define a helper function to differentiate the parameters that should receive weight decay**

* Biases and LayerNorm weights are generally not subject to weight decay.


```python
def get_grouped_params(model, no_decay=["bias", "LayerNorm.weight"]):
    params_with_wd, params_without_wd = [], []
    for n, p in model.named_parameters():
        if any(nd in n for nd in no_decay):
            params_without_wd.append(p)
        else:
            params_with_wd.append(p)
    return [{'params': params_with_wd, 'weight_decay': args.weight_decay},
            {'params': params_without_wd, 'weight_decay': 0.0}]
```

------

**Define a function to evaluate the model on the validation set**


```python
def evaluate():
    model.eval()
    losses = []
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            outputs = model(batch, labels=batch)
        loss = outputs.loss.repeat(args.valid_batch_size)
        losses.append(accelerator.gather(loss))
        if args.max_eval_steps > 0 and step >= args.max_eval_steps: break
    loss = torch.mean(torch.cat(losses))
    try:
        perplexity = torch.exp(loss)
    except OverflowError:
        perplexity = torch.tensor(float("inf"))
    return loss.item(), perplexity.item()
```

**Note:**
* The perplexity measures how well the model's output probability distributions predict the targeted tokens.
* A lower perplexity corresponds to better performance.
* We compute the perplexity by exponentiating the cross-entropy loss from the model's output.

------

**Training session**


```python
# Reset random seed
set_seed(args.seed)

# Accelerator
accelerator = Accelerator()
samples_per_step = accelerator.state.num_processes * args.train_batch_size

# Logging
logger, tb_writer, run_name = setup_logging(project_name.split("/")[1])
logger.info(accelerator.state)

# Load model and tokenizer
if accelerator.is_main_process:
    # Check out a new branch for the current run
    hf_repo = Repository("./", clone_from=project_name, revision=run_name)
model = AutoModelForCausalLM.from_pretrained("./", gradient_checkpointing=True)
tokenizer = AutoTokenizer.from_pretrained("./")

# Load dataset and dataloader
train_dataloader, eval_dataloader = create_dataloaders(dataset_name)

# Prepare the optimizer and learning rate scheduler
optimizer = AdamW(get_grouped_params(model), lr=args.learning_rate)
lr_scheduler = get_scheduler(name=args.lr_scheduler_type, optimizer=optimizer,
                             num_warmup_steps=args.num_warmup_steps,
                             num_training_steps=args.max_train_steps,)
def get_lr():
    return optimizer.param_groups[0]['lr']

# Prepare everything with our `accelerator` (order of args is not important)
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader)

# Train model
model.train()
completed_steps = 0
for step, batch in enumerate(train_dataloader, start=1):
    loss = model(batch, labels=batch).loss
    log_metrics(step, {'lr': get_lr(), 'samples': step*samples_per_step,
                       'steps': completed_steps, 'loss/train': loss.item()})
    loss = loss / args.gradient_accumulation_steps
    accelerator.backward(loss)
    # Use gradient accumulation to imitate larger batch sizes
    if step % args.gradient_accumulation_steps == 0:
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        completed_steps += 1
    if step % args.save_checkpoint_steps == 0:
        logger.info('Evaluating and saving model checkpoint')
        # Evaluate the model every time we save a new checkpoint
        eval_loss, perplexity = evaluate()
        log_metrics(step, {'loss/eval': eval_loss, 'perplexity': perplexity})
        # Synchronize the model before storing the latest checkpoint
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        if accelerator.is_main_process:
            # Save the latest checkpoint to disk
            unwrapped_model.save_pretrained("./")
            # Push the latest checkpoint to the Hub
            hf_repo.push_to_hub(commit_message=f'step {step}')
        model.train()
    if completed_steps >= args.max_train_steps:
        break

# Evaluate and save the last checkpoint
logger.info('Evaluating and saving model after training')
eval_loss, perplexity = evaluate()
log_metrics(step, {'loss/eval': eval_loss, 'perplexity': perplexity})
accelerator.wait_for_everyone()
unwrapped_model = accelerator.unwrap_model(model)
if accelerator.is_main_process:
    unwrapped_model.save_pretrained("./")
    hf_repo.push_to_hub(commit_message=f'final model')
```

**Note:**
* here are several approaches to distributed training depending on the model size and volume of data.
* Hugging Face Accelerate uses [DataDistributedParalellism (DDP)](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html).
* DDP allows you to train models faster with larger batch sizes that wouldn't fit into any single GPU.
* Hugging Face Accelerate prepares batches of data and sends them to the workers.
* Each worker consists of a GPU and calculates the loss and their respective accumulated gradients from forward and backward passes with a local copy of the model. 
* We average the gradients from each node with a `reduce` pattern and send the average back to each worker.
* We apply the gradients using the optimizer on each node to avoid transferring copies of the large models between nodes.
* We repeat the process after updating the models for each worker.
* DDP requires that the model fits on a single GPU.
* [Fitting larger networks into memory.](https://medium.com/tensorflow/fitting-larger-networks-into-memory-583e3c758ff9)
* [Model Paralellism](https://huggingface.co/docs/transformers/main/en/parallelism)

------

### The Training Run

* We can save the training steps to a script and push them to a repository on the Hub.
* We can then execute the training script on a training server using the `accelerate launch` command.

```bash
git clone https://huggingface.co/transformerbook/codeparrot
cd codeparrot
pip install -r requirements.txt
wandb login
accelerate config
accelerate launch codparrot_training.py
```

* The `accelerate config` command guides you through setting up the infrastructure.
* Hugging Face uses [`a2-megagpu-16g`](https://cloud.google.com/blog/products/compute/announcing-google-cloud-a2-vm-family-based-on-nvidia-a100-gpu) instances on Google Cloud for experiments ([pricing](https://www.economize.cloud/gcp/pricing/a2/a2-megagpu-16g)).
* [Reducing 90% in costs with Spot VMs for Machine Learning on Google Kubernetes Engine in GCP](https://spltech.co.uk/reducing-90-in-costs-with-spot-vms-for-machine-learning-on-google-kubernetes-engine-in-gcp/)

**Configuration used to train CodeParrot models**

| Setting              | Value     |
| -------------------- | --------- |
| Compute Environment? | multi-gpu |
| How many machines?   | 1         |
| DeepSpeed?           | No        |
| How many processes?  | 16        |
| Use FP16?            | Yes       |

* Running the training script with the above settings takes about 24 hours for the small model and seven days for the large model.
* Test the code on smaller infrastructure before using expensive cloud instances.
* We can merge the experiment branch back into the main one after training completes.
```bash
git checkout main
git merge <RUN_NAME>
git push
```



## Results and Analysis

* The training loss and validation perplexity should go down continuously during training.
* The large model converges with fewer processed tokens, but training takes longer overall.
* Qualitative analysis involves looking at concrete examples and trying to better understand in which cases the model succeeds and fails.
* Quantitative analysis involves evaluating model performance statistically on a large set of test cases. 

------


```python
from transformers import pipeline, set_seed
```

------

**Wrap the small model in a text generation pipeline**


```python
model_ckpt = 'transformersbook/codeparrot-small'
generation = pipeline('text-generation', model=model_ckpt, device=0)
```
------


```python
import re
from transformers import set_seed 
```

------

**Define a function to extract the first code block from the model output**


```python
def first_block(string):
    return re.split('\nclass|\ndef|\n#|\n@|\nprint|\nif', string)[0].rstrip()
```

------

**Define a function to print out generated code completions**


```python
def complete_code(pipe, prompt, max_length=64, num_completions=4, seed=1):
    set_seed(seed)
    gen_kwargs = {"temperature":0.4, "top_p":0.95, "top_k":0, "num_beams":1,
                  "do_sample":True,}
    code_gens = generation(prompt, num_return_sequences=num_completions, 
                            max_length=max_length, **gen_kwargs)
    code_strings = []
    for code_gen in code_gens:
        generated_code = first_block(code_gen['generated_text'][len(prompt):])
        code_strings.append(generated_code)
    print(('\n'+'='*80 + '\n').join(code_strings))
```

------

**Test the model on a simple task**


```python
prompt = '''def area_of_rectangle(a: float, b: float):
    """Return the area of the rectangle."""'''
complete_code(generation, prompt)
```
```text

        return math.sqrt(a * b)
    ================================================================================
    
        return a * b / 2.0
    ================================================================================
    
        return a * b
    ================================================================================
    
        return a * b / 2.0
```

**Note:** The generated outputs look convincing, but not all of them are correct.

------

**Test the model on a more complex task**


```python
prompt = '''def get_urls_from_html(html):
    """Get all embedded URLs in a HTML string."""'''
complete_code(generation, prompt)
```
```text

        if not html:
            return []
        return [url for url in re.findall(r'<a href="(/[^/]+/[^"]+?)">', html)]
    ================================================================================
    
        return [url for url in re.findall(r'<a href="(.*?)"', html)
                if url]
    ================================================================================
    
        return [url for url in re.findall(r'<a href="(.*?)"', html)]
    ================================================================================
    
        return re.findall(r'<a href="([^"]+)">', html)
```

**Note:** The second attempt is not quite right, but the other three generations are correct.

------

**Test the generated code**


```python
import requests

def get_urls_from_html(html):
    return [url for url in re.findall(r'<a href="(.*?)"', html) if url]

pd.DataFrame(get_urls_from_html(requests.get('https://hf.co/').text))
```
<div style="overflow-x:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>https://huggingface.co/bigscience/tr11-176B-ml-logs</td>
    </tr>
    <tr>
      <th>1</th>
      <td>https://github.com/huggingface/transformers</td>
    </tr>
    <tr>
      <th>2</th>
      <td>/join</td>
    </tr>
    <tr>
      <th>3</th>
      <td>/tasks</td>
    </tr>
    <tr>
      <th>4</th>
      <td>https://huggingface.co/transformers</td>
    </tr>
    <tr>
      <th>5</th>
      <td>/inference-api</td>
    </tr>
    <tr>
      <th>6</th>
      <td>/distilbert-base-uncased</td>
    </tr>
    <tr>
      <th>7</th>
      <td>/dbmdz/bert-large-cased-finetuned-conll03-english</td>
    </tr>
    <tr>
      <th>8</th>
      <td>https://bigscience.huggingface.co/</td>
    </tr>
    <tr>
      <th>9</th>
      <td>https://bigscience.huggingface.co/blog/t0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>https://medium.com/huggingface/distilbert-8cf3380435b5</td>
    </tr>
    <tr>
      <th>11</th>
      <td>https://arxiv.org/abs/1811.06031</td>
    </tr>
    <tr>
      <th>12</th>
      <td>https://arxiv.org/abs/1803.10631</td>
    </tr>
    <tr>
      <th>13</th>
      <td>/coref</td>
    </tr>
    <tr>
      <th>14</th>
      <td>https://transformer.huggingface.co/</td>
    </tr>
  </tbody>
</table>
</div>
**Note:** The URLs starting with `https` are external pages, while the others are subpages of the main website.

------

**Wrap the large model in a text generation pipeline**


```python
model_ckpt = 'transformersbook/codeparrot'
generation = pipeline('text-generation', model=model_ckpt, device=0)
```

------

**Try to translate a function from pure Python to NumPy using the large model**


```python
prompt = '''# a function in native python:
def mean(a):
    return sum(a)/len(a)

# the same function using numpy:
import numpy as np
def mean(a):'''
complete_code(generation, prompt, max_length=64)
```
```text

        return np.mean(a)
    ================================================================================
    
        return sum(a)/len(a)
    ================================================================================
    
        return np.mean(a)
    ================================================================================
    
        return sum(a)/len(a)
```

**Note:** It worked.

------

**Try building a Scilit-learn model**


```python
prompt = '''X = np.random.randn(100, 100)
y = np.random.randint(0, 1, 100)

# fit random forest classifier with 20 estimators'''
complete_code(generation, prompt, max_length=96)
```
```text

    reg = DummyRegressor()
    
    forest = RandomForestClassifier(n_estimators=20)
    
    forest.fit(X, y)
    ================================================================================
    
    clf = ExtraTreesClassifier(n_estimators=100, max_features='sqrt')
    clf.fit(X, y)
    ================================================================================
    
    clf = RandomForestClassifier(n_estimators=20, n_jobs=n_jobs, random_state=1)
    clf.fit(X, y)
    ================================================================================
    
    clf = RandomForestClassifier(n_estimators=20, n_jobs=n_jobs, random_state=1)
    clf.fit(X, y)
```

**Note:**
* The second attempt used an [extra-trees classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html), but the other three generated what we asked.
* The BLEU score is not well suited for measuring the quality of generated code as it would punish a generation that deviates from the reference naming.
* The success of a program does not depend on the naming scheme as long as it is consistent.
* We can use traditional software development methods like unit tests to measure the quality of generated code.
* OpenAI evaluated Codex models by running several code generations for coding tasks through some unit tests and calculating the fraction that passes the tests.
* [Evaluating Large Language Models Trained on Code](https://arxiv.org/abs/2107.03374)


------



## References

* [Natural Language Processing with Transformers Book](https://transformersbook.com/)
* [The Transformers book GitHub Repository](https://github.com/nlp-with-transformers/notebooks)





<!-- Cloudflare Web Analytics --><script defer src='https://static.cloudflareinsights.com/beacon.min.js' data-cf-beacon='{"token": "56b8d2f624604c4891327b3c0d9f6703"}'></script><!-- End Cloudflare Web Analytics -->





















































































































