---
title: Notes on Transformers Book Ch. 1
date: 2022-3-30
image: /images/empty.gif
title-block-categories: true
hide: false
search_exclude: false
comments:
  utterances:
    repo: cj-mills/christianjmills
description: Chapter 1 covers essential advancements for transformers, recurrent architectures, the encoder-decoder framework, attention mechanisms, transfer learning in NLP, and the HuggingFace ecosystem.
categories: [ai, huggingface, nlp, notes]

aliases:
- /Notes-on-Transformers-Book-01/


twitter-card:
  creator: "@cdotjdotmills"
  site: "@cdotjdotmills"
  image: /images/default-preview-image-black.png
open-graph:
  image: /images/default-preview-image-black.png

---



::: {.callout-tip}
## This post is part of the following series:
* [**Natural Language Processing with Transformers**](/series/notes/transformers-book-notes.html)
:::



* [Key Advancements](#key-advancements)
* [Recurrent Architectures](#recurrent-architectures)
* [The Encoder-Decoder Framework](#the-encoder-decoder-framework)
* [Attention Mechanisms](#attention-mechanisms)
* [Transfer Learning in NLP](#transfer-learning-in-nlp)
* [Bridging the Gap With Hugging Face Transformers](#bridging-the-gap-with-hugging-face-transformers)
* [A Tour of Transformer Applications](#a-tour-of-transformer-applications)
* [The Hugging Face Ecosystem](#the-hugging-face-ecosystem)
* [Main Challenges with Transformers](#main-challenges-with-transformers)
* [References](#references)



## Key Advancements

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
    - published in June 2017 by researchers at Google
    - introduced the Transformer architecture for sequence modeling
    - outperformed Recurrent Neural Networks (RNNs) on machine translation tasks, both in terms of translation quality and training cost
- [Universal Language Model Fine-tuning for Text Classification](https://arxiv.org/abs/1801.06146)
    - published in January 2018 by Jeremy Howard and Sebastian Ruder
    - introduced an effective training method called ULMFiT
    - showed that training Long Short-Term Memory Networks (LSTMs) on a very large and diverse corpus could produce state-of-the-art text classifiers with little labeled data
    - inspired other research groups to combine transformers with unsupervised learning
- [Improving Language Understanding with Unsupervised Learning](https://openai.com/blog/language-unsupervised/)
    - published by OpenAI in June 2018
    - introduced Generative Pretrained Transformer (GPT)
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
    - published by researchers at Google in October 2018
- Combining the Transformer architecture with unsupervised learning removed the need to train task-specific architectures from scratch.
- Pretrained Transformers broke almost every benchmark in NLP by a significant margin.


## Recurrent Architectures

- Recurrent architectures such as LSTMs were state of the art in Natural Language Processing (NLP) before Transformers.
- Recurrent architectures contain a feedback loop that allows information to propagate from one step to another.
    - ideal for sequential data like text.
- A Recurrent Neural Network receives an input token and feeds it through the network.
- The network outputs a vector called a hidden state, and it passes some information back to itself through a feedback loop.
- The information passed through the feedback loop allows an RNN to keep track of details from previous steps and use it to make predictions.
- Many still use recurrent architectures for NLP, speech processing, and time-series tasks.
- [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
    - provides an overview of RNNs and demonstrates how to train a language model on several datasets
- RNNs were critical in developing systems to translate a sequence of words from one language to another.
    - known as machine translation
- The computations for recurrent models are inherently sequential and cannot parallelize across the input.
    - The inability to parallelize computations is a fundamental shortcoming of recurrent models.


## The Encoder-Decoder Framework

- [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215)
    - published in 2014 by researchers at Google
- An encoder-decoder is also known as a sequence-to-sequence architecture.
- This type of architecture is well-suited for situations where the input and output are both sequences of arbitrary length.
- An encoder encodes information from an input sequence into a numerical representation.
    - This numerical representation is often called the last hidden state.
- The decoder uses the numerical representation to generate the output sequence.
- The encoder and decoder can use any neural network architecture that can model sequences.
- The final hidden state of the encoder is the only information the decoder has access to when generating the output.
    - It has to represent the meaning of the whole input sequence.
    - This requirement creates an information bottleneck that can limit performance for longer sequences.



## Attention Mechanisms

- Attention mechanisms allow the decoder to access all of the encoder's hidden states, not just the last one.
- The decoder assigns a different amount of weight, called attention, to each state at every decoding time-step.
- The attention values allow the decoder to prioritize which encoder state to use.
- Attention-based models focus on which input tokens are most relevant at each time step.
    - They can learn nontrivial alignments between the words in a generated translation and those in a source sentence.
        - Example: An attention-based decoder can align the words "zone" and "Area" even when ordered differently in the source sentence and the translation.
- Transformers use a special kind of attention called self-attention and do not use any form of recurrence.
    - Self-attention operates on all states in the same layer of a neural network.
    - The outputs of the self-attention mechanisms serve as input to feed-forward networks.
    - This architecture trains much faster than recurrent models.


## Transfer Learning in NLP

- Transfer learning involves using the knowledge a model learned from a previous task on a new one.
    - Computer vision models first train on large-scale datasets such as [ImageNet](https://image-net.org/) to learn the basic features of images before being fine-tuned on a downstream task.
    - It was not initially clear how to perform transfer learning for NLP tasks.
- Fine-tuned models are typically more accurate than supervised models trained from scratch on the same amount of labeled data.
- We adapt a pretrained model to a new task by splitting the model into a body and a head.
- The head is the task-specific portion of the network.
- The body contains broad features from the source domain learned during training.
- We can use the body weights to initialize a new model head for a new task.
- Transfer learning typically produces high-quality models that we can efficiently train on many downstream tasks.
- The ULMFit paper provided a general framework to perform transfer learning with NLP models.
    1. A model first trains to predict the next word based on those preceding it in a large-scale generic corpus to learn the basic features of the language.
        - This task is called language modeling.
    2. The pretrained model then trains on the same task using an in-domain corpus.
    3. Lastly, we fine-tune the model with a classification layer for the target task.
- The ULMFit transfer learning framework provided the missing piece for transformers to take off.
- Both GPT and BERT combine self-attention with transfer learning and set a new state of the art across many NLP benchmarks.
- GPT only uses the decoder part of the Transformer architecture and the language modeling approach as ULMFiT.
- BERT uses the encoder part of the Transformer architecture and a form of language modeling called masked language modeling.
    - Masked language modeling requires the model to fill in randomly missing words in a text.
- GPT trained on the BookCorpus dataset while BERT trained on the BookCorpus dataset and English Wikipedia.
    - The [BookCorpus dataset](https://arxiv.org/abs/1506.06724) consists of thousands of unpublished books across many genres.



## Bridging the Gap With Hugging Face Transformers

- Applying a novel machine learning architecture to a new application can be complicated and requires custom logic for each model and task.
    1. Implement the model architecture in code. 
        - PyTorch and TensorFlow are the most common frameworks for this.
    2. Load pretrained weights if available.
    3. Preprocess the inputs, pass them through the model, and apply task-specific postprocessing.
    4. Implement data loaders and define loss functions and optimizers to train the model.
- Code released by research groups is rarely standardized and often requires days of engineering to adapt to new use cases.
    - Different research labs often release their models in incompatible frameworks, making it difficult for practitioners to port these models to their applications.
- Hugging Face Transformers provides a standardized interface to a wide range of transformer models, including code and tools to adapt these models to new applications.
    - The availability of a standardized interface catalyzed the explosion of research into transformers and made it easy for NLP practitioners to integrate these models into real-life applications.
- The library supports the PyTorch, TensorFlow, and JAX deep learning frameworks and provides task-specific model heads to fine-tune transformers on downstream tasks.



## A Tour of Transformer Applications

- Hugging Face Transformers has a layered API that allows users to interact with the library at various levels of abstraction.

### [Pipelines](https://huggingface.co/docs/transformers/main_classes/pipelines)

- Pipelines abstract away all the steps needed to convert raw text into a set of predictions from a fine-tuned model.
- Hugging Face provides pipelines for several tasks.
- Instantiate a pipeline by calling the `pipeline()` function and providing the name of the desired task.
    - `'audio-classification'`
    - `'automatic-speech-recognition'`
    - `'feature-extraction'`
    - `'text-classification'`
    - `'token-classification'`
    - `'question-answering'`
    - `'table-question-answering'`
    - `'fill-mask'`
    - `'summarization'`
    - `'translation'`
    - `'text2text-generation'`
    - `'text-generation'`
    - `'zero-shot-classification'`
    - `'conversational'`
    - `'image-classification'`
    - `'object-detection'`
- The names for the supported tasks are available in the `transformers.pipelines.SUPPORTED_TASKS` dictionary.
- The pipeline automatically downloads the model weights for the selected task and caches them for future use.
- Each pipeline takes a string of text or a list of strings as input and returns a list of predictions.
    - Each prediction is in a Python dictionary along with the corresponding confidence score.



```python
import transformers
import datasets
import pandas as pd
from transformers import pipeline

transformers.logging.set_verbosity_error()
datasets.logging.set_verbosity_error()
```

```python
pipeline
```


```text
<function transformers.pipelines.pipeline(task: str, model: Optional = None, config: Union[str, transformers.configuration_utils.PretrainedConfig, NoneType] = None, tokenizer: Union[str, transformers.tokenization_utils.PreTrainedTokenizer, NoneType] = None, feature_extractor: Union[str, ForwardRef('SequenceFeatureExtractor'), NoneType] = None, framework: Optional[str] = None, revision: Optional[str] = None, use_fast: bool = True, use_auth_token: Union[bool, str, NoneType] = None, model_kwargs: Dict[str, Any] = {}, **kwargs) -> transformers.pipelines.base.Pipeline>
```




```python
list(transformers.pipelines.SUPPORTED_TASKS.keys())
```


```text
['audio-classification',
 'automatic-speech-recognition',
 'feature-extraction',
 'text-classification',
 'token-classification',
 'question-answering',
 'table-question-answering',
 'fill-mask',
 'summarization',
 'translation',
 'text2text-generation',
 'text-generation',
 'zero-shot-classification',
 'conversational',
 'image-classification',
 'object-detection']
```




```python
transformers.pipelines.TASK_ALIASES
```


```text
{'sentiment-analysis': 'text-classification', 'ner': 'token-classification'}
```




##### Sample Text: Customer Review

```python
text = """Dear Amazon, last week I ordered an Optimus Prime action figure \
from your online store in Germany. Unfortunately, when I opened the package, \
I discovered to my horror that I had been sent an action figure of Megatron \
instead! As a lifelong enemy of the Decepticons, I hope you can understand my \
dilemma. To resolve the issue, I demand an exchange of Megatron for the \
Optimus Prime figure I ordered. Enclosed are copies of my records concerning \
this purchase. I expect to hear from you soon. Sincerely, Bumblebee."""
```



#### Text Classification Pipeline

- [Documentation](https://huggingface.co/docs/transformers/v4.17.0/en/main_classes/pipelines#transformers.TextClassificationPipeline)
- The text-classification pipeline supports sentiment analysis, multiclass, and multilabel classification and performs sentiment analysis by default.


```python
classifier = pipeline("text-classification")
```


```python
# Classify the customer review as positive or negative
outputs = classifier(text)
pd.DataFrame(outputs)    
```
<div style="overflow-x:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>label</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>NEGATIVE</td>
      <td>0.901546</td>
    </tr>
  </tbody>
</table>
</div>




#### Named Entity Recognition Pipeline

- [Documentation](https://huggingface.co/docs/transformers/v4.17.0/en/main_classes/pipelines#transformers.TokenClassificationPipeline)
- Named entity recognition (NER) involves extracting real-world objects like products, places, and people from a piece of text.
- Default Entity Labels
    - `MISC`: Miscellaneous
    - `PER`: Person
    - `ORG`: Organization
    - `LOC`: Location


```python
# Create a named entity recognizer that groups words according to the model's predictions
ner_tagger = pipeline("ner", aggregation_strategy="simple")
```

**Note:** The `simple` aggregation strategy might end up splitting words undesirably.


```python
ner_tagger.model.config.id2label
```
```text
{0: 'O',
 1: 'B-MISC',
 2: 'I-MISC',
 3: 'B-PER',
 4: 'I-PER',
 5: 'B-ORG',
 6: 'I-ORG',
 7: 'B-LOC',
 8: 'I-LOC'}
```




```python
outputs = ner_tagger(text)
pd.DataFrame(outputs)    
```
<div style="overflow-x:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>entity_group</th>
      <th>score</th>
      <th>word</th>
      <th>start</th>
      <th>end</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ORG</td>
      <td>0.879010</td>
      <td>Amazon</td>
      <td>5</td>
      <td>11</td>
    </tr>
    <tr>
      <th>1</th>
      <td>MISC</td>
      <td>0.990859</td>
      <td>Optimus Prime</td>
      <td>36</td>
      <td>49</td>
    </tr>
    <tr>
      <th>2</th>
      <td>LOC</td>
      <td>0.999755</td>
      <td>Germany</td>
      <td>90</td>
      <td>97</td>
    </tr>
    <tr>
      <th>3</th>
      <td>MISC</td>
      <td>0.556570</td>
      <td>Mega</td>
      <td>208</td>
      <td>212</td>
    </tr>
    <tr>
      <th>4</th>
      <td>PER</td>
      <td>0.590256</td>
      <td>##tron</td>
      <td>212</td>
      <td>216</td>
    </tr>
    <tr>
      <th>5</th>
      <td>ORG</td>
      <td>0.669693</td>
      <td>Decept</td>
      <td>253</td>
      <td>259</td>
    </tr>
    <tr>
      <th>6</th>
      <td>MISC</td>
      <td>0.498349</td>
      <td>##icons</td>
      <td>259</td>
      <td>264</td>
    </tr>
    <tr>
      <th>7</th>
      <td>MISC</td>
      <td>0.775362</td>
      <td>Megatron</td>
      <td>350</td>
      <td>358</td>
    </tr>
    <tr>
      <th>8</th>
      <td>MISC</td>
      <td>0.987854</td>
      <td>Optimus Prime</td>
      <td>367</td>
      <td>380</td>
    </tr>
    <tr>
      <th>9</th>
      <td>PER</td>
      <td>0.812096</td>
      <td>Bumblebee</td>
      <td>502</td>
      <td>511</td>
    </tr>
  </tbody>
</table>
</div>

**Note:** The words `Megatron`, and `Decepticons` were split into separate words.

**Note:** The `##` symbols are produced by the model's tokenizer.


```python
ner_tagger.tokenizer.vocab_size
```
```text
28996
```




```python
pd.DataFrame(ner_tagger.tokenizer.vocab, index=[0]).T.head(10)
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
      <th>Rees</th>
      <td>24646</td>
    </tr>
    <tr>
      <th>seeded</th>
      <td>14937</td>
    </tr>
    <tr>
      <th>Ruby</th>
      <td>11374</td>
    </tr>
    <tr>
      <th>Libraries</th>
      <td>27927</td>
    </tr>
    <tr>
      <th>foil</th>
      <td>20235</td>
    </tr>
    <tr>
      <th>collapsed</th>
      <td>7322</td>
    </tr>
    <tr>
      <th>membership</th>
      <td>5467</td>
    </tr>
    <tr>
      <th>Birth</th>
      <td>20729</td>
    </tr>
    <tr>
      <th>Texans</th>
      <td>25904</td>
    </tr>
    <tr>
      <th>Saul</th>
      <td>18600</td>
    </tr>
  </tbody>
</table>
</div>




#### Question Answering Pipeline

- [Documentation](https://huggingface.co/docs/transformers/v4.17.0/en/main_classes/pipelines#transformers.QuestionAnsweringPipeline)
- Question answering involves having a model find the answer to a specified question using a given passage of text.

```python
reader = pipeline("question-answering")
```

```python
question = "What does the customer want?"
outputs = reader(question=question, context=text)
pd.DataFrame([outputs])    
```
<div style="overflow-x:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>score</th>
      <th>start</th>
      <th>end</th>
      <th>answer</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.631291</td>
      <td>335</td>
      <td>358</td>
      <td>an exchange of Megatron</td>
    </tr>
  </tbody>
</table>
</div>


**Note:** This particular kind of question answering is called extractive question answering. The answer is extracted directly from the text.



#### Summarization Pipeline

- [Documentation](https://huggingface.co/docs/transformers/v4.17.0/en/main_classes/pipelines#transformers.SummarizationPipeline)
- Text summarization involves generating a short version of a long passage of text while retaining all the relevant facts.
- Tasks requiring a model to generate new text are more challenging than extractive ones.

```python
summarizer = pipeline("summarization")
```

```python
# Limit the generated summary to 45 words
outputs = summarizer(text, max_length=45, clean_up_tokenization_spaces=True)
print(outputs[0]['summary_text'])
```
```text
 Bumblebee ordered an Optimus Prime action figure from your online store in Germany. Unfortunately, when I opened the package, I discovered to my horror that I had been sent an action figure of Megatron instead.
```


**Note:** The model captured the essence of the customer message but directly copied some of the original text.



#### Translation Pipeline

- [Documentation](https://huggingface.co/docs/transformers/v4.17.0/en/main_classes/pipelines#transformers.TranslationPipeline)
- The model generates a translation of a piece of text in the target language.

```python
# Create a translator that translates English to German
# Override the default model selection
translator = pipeline("translation_en_to_de", model="Helsinki-NLP/opus-mt-en-de")
```

```python
# Require the model to generate a translation at least 100 words long
outputs = translator(text, clean_up_tokenization_spaces=True, min_length=100)
print(outputs[0]['translation_text'])
```
```text
Sehr geehrter Amazon, letzte Woche habe ich eine Optimus Prime Action Figur aus Ihrem Online-Shop in Deutschland bestellt. Leider, als ich das Paket öffnete, entdeckte ich zu meinem Entsetzen, dass ich stattdessen eine Action Figur von Megatron geschickt worden war! Als lebenslanger Feind der Decepticons, Ich hoffe, Sie können mein Dilemma verstehen. Um das Problem zu lösen, Ich fordere einen Austausch von Megatron für die Optimus Prime Figur habe ich bestellt. Anbei sind Kopien meiner Aufzeichnungen über diesen Kauf. Ich erwarte, bald von Ihnen zu hören. Aufrichtig, Bumblebee.
```

**Note:** The model supposedly did a good job translating the text. (I don't speak German.)



#### Text Generation Pipeline

- [Documentation](https://huggingface.co/docs/transformers/v4.17.0/en/main_classes/pipelines#transformers.TextGenerationPipeline)
- The model generates new text to complete a provided text prompt.

```python
from transformers import set_seed
# Set the random seed to get reproducible results
set_seed(42)
```


```python
generator = pipeline("text-generation")
```


```python
response = "Dear Bumblebee, I am sorry to hear that your order was mixed up."
prompt = text + "\n\nCustomer service response:\n" + response
outputs = generator(prompt, max_length=200)
print(outputs[0]['generated_text'])
```
```text
Dear Amazon, last week I ordered an Optimus Prime action figure from your online store in Germany. Unfortunately, when I opened the package, I discovered to my horror that I had been sent an action figure of Megatron instead! As a lifelong enemy of the Decepticons, I hope you can understand my dilemma. To resolve the issue, I demand an exchange of Megatron for the Optimus Prime figure I ordered. Enclosed are copies of my records concerning this purchase. I expect to hear from you soon. Sincerely, Bumblebee.

Customer service response:
Dear Bumblebee, I am sorry to hear that your order was mixed up. The order was completely mislabeled, which is very common in our online store, but I can appreciate it because it was my understanding from this site and our customer service of the previous day that your order was not made correct in our mind and that we are in a process of resolving this matter. We can assure you that your order
```



## The Hugging Face Ecosystem

- Hugging Face Transformers is surrounded by an ecosystem of helpful tools that support the modern machine learning workflow.
- This ecosystem consists of a family of code libraries and a hub of pretrained model weights, datasets, scripts for evaluation, other resources.



### The Hugging Face Hub

- The Hub hosts over 20,000 freely available models plus datasets and scripts for computing metrics.
- Model and dataset cards document the contents of the models and datasets.
- Filters are available for tasks, frameworks, datasets, and more designed to help quickly navigate the Hub.
- Users can directly try out any model through task-specific widgets.



### Hugging Face Tokenizers

- [Documentation](https://huggingface.co/docs/transformers/v4.17.0/en/main_classes/tokenizer)
- Tokenizers split the raw input text into smaller pieces called tokens.
- Tokens can be words, parts of words, or single characters.
- Hugging Face Tokenizers takes care of all the preprocessing and postprocessing steps, such as normalizing the inputs and transforming the model outputs to the required format.
- The [Tokenizers library](https://github.com/huggingface/tokenizers) uses a [Rust](https://www.rust-lang.org/) backend for fast tokenization.



### Hugging Face Datasets

- [Documentation](https://huggingface.co/docs/datasets/index)
- The Datasets library provides a standard interface for [thousands](https://huggingface.co/datasets) of datasets to simplify loading, processing, and storing datasets.
- Smart caching removes the need to perform preprocessing steps each time your run your code.
- Memory mapping helps avoid RAM limitations by storing the contents of a file in virtual memory and enables multiple processes to modify the file more efficiently.
- The library is interoperable with frameworks like [Pandas](https://pandas.pydata.org/) and [NumPy](https://numpy.org/).
- Scripts are available for many metrics to help make experiments more reproducible and trustworthy.



### Hugging Face Accelerate

- [Documentation](https://huggingface.co/docs/accelerate/index)
- The Accelerate library adds a layer of abstraction to training loops, which takes care of all the custom logic necessary for the training infrastructure.



## Main Challenges with Transformers

#### Language

- It is hard to find pretrained models for languages other than English.

#### Data Availability

- Even with transfer learning, transformers still need a lot of data compared to humans to perform a task.

#### Working With Long Documents

- Self-attention becomes computationally expensive when working on full-length documents.

#### Opacity

- It is hard or impossible to determine precisely why a model made a given prediction.

#### Bias

- Biases present in the training data imprint into the model.




## References

* [Natural Language Processing with Transformers Book](https://transformersbook.com/)
* [The Transformers book GitHub Repository](https://github.com/nlp-with-transformers/notebooks)



**Next:** [Notes on Transformers Book Ch. 2](../chapter-2/)





{{< include /_about-author-cta.qmd >}}
