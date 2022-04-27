---
title: Notes on Transformers Book Ch. 11
layout: post
toc: false
comments: true
description: My full notes for chapter 11 of Natural Language Processing with Transformers
categories: [ai, huggingface, nlp, notes]
hide: false
permalink: /:title/
search_exclude: false
---




* [Scaling Transformers](#scaling-transformers)
* [Going Beyond Text](#going-beyond-text)
* [Multimodal Transformers](#multimodal-transformers)
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



## Scaling Transformers

* **[The Bitter Lesson](http://incompleteideas.net/IncIdeas/BitterLesson.html)**
    * [Richard Sutton](http://www.incompleteideas.net/) argued that general methods that leverage computation are far more effective in AI than methods that leverage domain knowledge.
    * The human knowledge approach tends to complicate things, making them less suited to taking advantage of general methods leveraging computation.
    * Search methods and learning methods seem to scale arbitrarily with computation power.
* Large language models perform better on downstream tasks.
* Interesting capabilities like zero-shot and few-shot learning emerge in the 10 to 100-billion parameter range.
* Computing power and training data must also scale with parameter count.
* Large language models like GPT-3 are estimated to cost [$4.6 million](https://lambdalabs.com/blog/demystifying-gpt-3/) to train.
* The high cost of training large models means we need a way to estimate the model's performance in advance.
* [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)
    * The performance of language models appears to obey a power-law relationship with model size and other factors.

------


```python
import pandas as pd
import matplotlib.pyplot as plt
```

------

**Plot parameter counts over time for prominent Transformer architectures**


```python
model_data = [
    {'date': '12-06-2017', 'name': 'Transformer', 'size': 213*1e6},
    {'date': '11-06-2018', 'name': 'GPT', 'size': 110*1e6},
    {'date': '11-10-2018', 'name': 'BERT', 'size': 340*1e6},
    {'date': '14-02-2019', 'name': 'GPT-2', 'size': 1.5*1e9},
    {'date': '23-10-2019', 'name': 'T5', 'size': 11*1e9},
    {'date': '17-09-2019', 'name': 'Megatron', 'size': 8.3*1e9},
    {'date': '13-02-2020', 'name': 'Turing-NLG', 'size': 17*1e9},
    {'date': '30-06-2020', 'name': 'GShard', 'size': 600*1e9},
    {'date': '28-05-2020', 'name': 'GPT-3', 'size': 175*1e9},
    {'date': '11-01-2021', 'name': 'Switch-C', 'size': 1.571*10e12},
]

def label_point(x, y, val, ax):
    a = pd.concat({"x": x, "y": y, "val": val}, axis=1)
    for i, point in a.iterrows():
        ax.text(
            point["x"],
            point["y"],
            str(point["val"]),
            horizontalalignment="center",
            verticalalignment="bottom",
        )

df_lm = pd.DataFrame.from_records(model_data)
df_lm["date"] = pd.to_datetime(df_lm["date"], dayfirst=True)

fig, ax = plt.subplots(1, 1, figsize=(12, 4))
df_lm.plot(x="date", y="size", kind="scatter", s=15, ax=ax)
ax.set_yscale("log")
label_point(df_lm["date"], df_lm["size"], df_lm["name"], ax)
ax.set_xlabel("Release date")
ax.set_ylabel("Number of parameters")
ax.grid(True)
plt.subplots_adjust(top=1.2)
plt.show()
```
![png](../images/notes-transformers-book/chapter-11/output_5_0.png)

------

### Scaling Laws

* Scaling laws allow us to empirically quantify the "bigger is better" paradigm for language models by studying their behavior with varying compute budgets $C$, dataset sizes $D$, and model sizes $N$.
* We measure dataset size in the number of tokens.
* The model size excludes parameters from the embedding layers.
* We chart the dependence of the cross-entropy loss on these three factors to determine if a relationship emerges.
* Scaling laws imply that increasing compute budget, dataset size, and model size in tandem is more productive than architectural tweaks or hyperparameter optimization to improve performance.
* The test loss has a power-law relationship with computation budget, dataset size, and model size across several orders of magnitude.
* We can express $L\left( X \right) \sim 1/X^{\alpha}$ for $X = N, C, D$ where $\alpha$ is a scaling exponent determined by a fit to the loss curve.
    * Typical values for $\alpha$ lie in the range $\left[0.05,0.095 \right]$.
    * [Scaling Laws for Autoregressive Generative Modeling](https://arxiv.org/abs/2010.14701)
* These power laws mean we can extrapolate the early part of a loss curve to predict the approximate loss from training longer.
* Larger models can achieve the same performance as smaller models with fewer training steps.
* Scaling laws are also present for other modalities like images, videos, and mathematical problem-solving.

### Challenges with Scaling
* Provisioning and managing hundreds or thousands of GPU nodes typically requires specialized engineers familiar with running large-scale, distributed experiments.
* Most companies cannot afford the teams and resources to train models at the largest scales.  
* A recently proposed distributed deep learning framework enables smaller groups to pool their computational resources and pre-train models.
    * [Distributed Deep Learning in Open Collaborations](https://arxiv.org/abs/2106.10207)
* Large models require large, high-quality datasets.
    * It is hard to curate only high-quality training examples when the dataset contains terabytes of text.
    * We need a way to control common biases in the dataset to prevent the model from inheriting them.
    * There are potential licensing issues when using large-scale web-text corpora.
    * Large-scale text datasets might contain personal information.
* Evaluating trained models on downstream tasks requires additional time and resources.
    * We need to probe the model for biased and toxic output, even when using a cleaned dataset.
* Optimization approaches like distillation, pruning, and quantization might not be enough when starting with a model that is hundreds of gigabytes in size.
* [OpenAI API](https://openai.com/api/)
* [Hugging Face Accelerated Inference API](https://huggingface.co/docs/api-inference/index)
* [BigScience](https://bigscience.huggingface.co/) is a one-year-long research workshop meant to foster discussions and reflections on the research questions surrounding large language models, the challenges of creating and sharing them, and datasets used for research.
    * The collaborative tasks involve creating, sharing, and evaluating a massive multilingual dataset and language model.
* [EleutherAI](https://www.eleuther.ai/) is a decentralized collective of volunteers focused on AI alignment, scaling, and open-source AI research.
    * EleutherAI wants to train and open-source a GPT-3-sized model.
    * [GPT-Neo 2.7B](https://huggingface.co/EleutherAI/gpt-neo-2.7B)
    * [GPT-J 6B](https://huggingface.co/EleutherAI/gpt-j-6B)

### Attention Please!
* Self-attention involves performing pairwise comparisons of all the tokens in a sequence, which becomes a computational bottleneck.
* The self-attention layer of the Transformer architecture naively scales like $O(n^{2})$, where n is the length of the sequence.
* A recent paper from Google shows we can reduce the memory complexity to $O \left( \log{n} \right)$ via a simple reordering of the operations.
    * [Self-attention Does Not Need $O(n^{2})$ Memory](https://arxiv.org/abs/2112.05682)
* Much of the recent research on transformers focuses on making self-attention more efficient.
    * [Efficient Transformers: A Survey](https://arxiv.org/abs/2009.06732)
* Common approaches to making attention more efficient involve introducing sparsity into the attention mechanism or applying kernels to the attention matrix.

### Sparse Attention
* We can reduce the number of computations performed in the self-attention layer by limiting the number of query-key pairs it generates according to a predefined pattern.
* There are a handful of popular "atomic" sparsity patterns.
    * [A Survey of Transformers](https://arxiv.org/abs/2106.04554)
* **Global attention** defines a few tokens in the sequence that are allowed to attend to all others.
* **Band attention** computes attention over a diagonal band.
* **Dilated attention** skips some query-key pairs using a dilated window with gaps.
* **Random attention** samples a few keys for each query to compute attention scores.
* **Block local attention** divides the sequence into blocks and restricts attention to within these blocks.
* Most transformer models with sparse attention use a mix of atomic sparsity patterns to generate the final attention matrix.
* Models like [Longformer](https://huggingface.co/allenai/longformer-base-4096) use a mix of global and band attention, while [Bigbird](https://huggingface.co/google/bigbird-roberta-base) adds random attention.
* Introducing sparsity into the attention matrix enables models to process much longer sequences.
* It is also possible to learn the sparsity pattern by clustering the tokens into chunks.
    * [Reformer](https://huggingface.co/google/reformer-crime-and-punishment) uses a hash function to cluster similar tokens.

### Linearized Attention
* Linearized attention involves changing the order of operations for computing attention scores.
* We compute the self-attention score of the queries and keys using a similarity function like the dot product.
* For a general similarity function $sim \left( q_{i},k_{j} \right)$, we can express the attention outputs as the following equation:
### $$y_{i} = \sum_{j}{\frac{sim \left( Q_{i}, K_{j} \right)}{\sum_{k}{sim\left( Q_{i}, K_{k} \right)}}V_{j}}$$
* The trick behind linearized attention mechanisms is to express the similarity function as a kernel function that decomposes the operation into two pieces:
### $$sim \left( Q_{j}, K_{j} \right) = \phi \left(Q_{i} \right)^{T} \phi \left( K_{j} \right)$$
* where $\phi$ is typically a high-dimensional feature map.
* $\phi \left( Q_{i} \right)$ is independent of $j$ and $k$, so we can pull it under the sums to write the attention output as follows:
### $$y_{i} = \frac{\phi \left(Q_{i} \right)^{T} \sum_{j}{\phi \left( K_{j} \right)} V_{j}^{T}}{\phi \left(Q_{i} \right)^{T} \sum_{k}{\phi \left( K_{k} \right)}}$$
* By first computing $\sum_{j}{\phi \left( K_{j} \right)} V_{j}^{T}$ and $\sum_{k}{\phi \left( K_{k} \right)}$, we can effectively linearize the space and time complexity of self-attention.
* Popular methods that implement linearized self-attention include Linear Transformer and Performer.
    * [Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention](https://arxiv.org/abs/2006.16236)
    * [Rethinking Attention with Performers](https://arxiv.org/abs/2009.14794)

## Going Beyond Text
* Developing effective strategies for common textual tasks like classification and question answering allows us to address many types of real-world problems.


### Limitations to using text

#### Human reporting bias
* The frequencies of events in the training text my not represent their actual frequencies.
    * [Reporting Bias and Knowledge Acquisition](https://openreview.net/pdf?id=AzxEzvpdE3Wcy)
* A model trained exclusively on text from the internet might have a distorted image of the world.

#### Common Sense
* Most do not document their reasoning based on common sense.
* Language models trained on text might know many facts about the world but lack basic common-sense reasoning.

#### Facts
* A probabilistic language model cannot reliably store facts and can produce factually incorrect text.
* Such models can detect named entities but have no direct way to access information about them.

#### Modality
* Language models can't connect to other modalities, such as audio, visual signals, or tabular data, that might address some of these limitations.

### Vision
* Transformers are now achieving efficiency similar to or better than Convolutional Neural Networks (CNNs).

#### iGPT
* iGPT (short for image GPT) uses the GPT architecture and autoregressive pretraining objective to predict future pixel values by viewing images as sequences of pixels.
* [Generative Pretraining From Pixels](https://proceedings.mlr.press/v119/chen20s.html)
* Pretraining on large image datasets enables iGPT to "autocomplete" partial images.
* iGPT achieves performant results on classification tasks when using a classification head.

#### ViT
* Vision Transformer (Vit) is a BERT-style take on transformers for vision.
* We split the image into smaller patches and then embed each of these patches with a linear projection.
* We combine the patch embeddings with position embeddings and feed them through an ordinary transformer encoder.
* We mask or distort some of the patches during training, and the objective is to predict the average color of the masked patch.
* This approach did not produce better results when pretrained on the standard ImageNet dataset, but it scaled significantly better than Convolutional Neural Networks on larger datasets.
* [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
* The Hugging Face Transformers library includes Vision Transformer.

------


```python
from PIL import Image
import matplotlib.pyplot as plt
```

------

**Load an image of a dog**


```python
image = Image.open("dog.jpg")
plt.imshow(image)
plt.axis("off")
plt.show()
```
![png](../images/notes-transformers-book/chapter-11/output_17_0.png)

------

```python
import pandas as pd
pd.set_option('max_colwidth', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
from transformers import pipeline
```

------

#### `ImageClassificationPipeline`

* [Documentation](https://huggingface.co/docs/transformers/main/en/main_classes/pipelines#transformers.ImageClassificationPipeline)
* Create an image classification pipeline

**Create an image classification pipeline**


```python
image_classifier = pipeline("image-classification")
```

------

**Get the model architecture**


```python
image_classifier.model.config.architectures
```
```text
    ['ViTForImageClassification']
```

------

#### `ViTForImageClassification`

* [Documentation](https://huggingface.co/docs/transformers/main/en/model_doc/vit#transformers.ViTForImageClassification)
* Create a ViT Model transformer with an image classification head for ImageNet.

**Get the link to the Hugging Face model card**


```python
print(f"https://huggingface.co/{image_classifier.model.config._name_or_path}")
```

    https://huggingface.co/google/vit-base-patch16-224

------

**View potential Image classes**


```python
pd.DataFrame(list(image_classifier.model.config.id2label.values())).T
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
      <th>26</th>
      <th>27</th>
      <th>28</th>
      <th>29</th>
      <th>30</th>
      <th>31</th>
      <th>32</th>
      <th>33</th>
      <th>34</th>
      <th>35</th>
      <th>36</th>
      <th>37</th>
      <th>38</th>
      <th>39</th>
      <th>40</th>
      <th>41</th>
      <th>42</th>
      <th>43</th>
      <th>44</th>
      <th>45</th>
      <th>46</th>
      <th>47</th>
      <th>48</th>
      <th>49</th>
      <th>50</th>
      <th>51</th>
      <th>52</th>
      <th>53</th>
      <th>54</th>
      <th>55</th>
      <th>56</th>
      <th>57</th>
      <th>58</th>
      <th>59</th>
      <th>60</th>
      <th>61</th>
      <th>62</th>
      <th>63</th>
      <th>64</th>
      <th>65</th>
      <th>66</th>
      <th>67</th>
      <th>68</th>
      <th>69</th>
      <th>70</th>
      <th>71</th>
      <th>72</th>
      <th>73</th>
      <th>74</th>
      <th>75</th>
      <th>76</th>
      <th>77</th>
      <th>78</th>
      <th>79</th>
      <th>80</th>
      <th>81</th>
      <th>82</th>
      <th>83</th>
      <th>84</th>
      <th>85</th>
      <th>86</th>
      <th>87</th>
      <th>88</th>
      <th>89</th>
      <th>90</th>
      <th>91</th>
      <th>92</th>
      <th>93</th>
      <th>94</th>
      <th>95</th>
      <th>96</th>
      <th>97</th>
      <th>98</th>
      <th>99</th>
      <th>100</th>
      <th>101</th>
      <th>102</th>
      <th>103</th>
      <th>104</th>
      <th>105</th>
      <th>106</th>
      <th>107</th>
      <th>108</th>
      <th>109</th>
      <th>110</th>
      <th>111</th>
      <th>112</th>
      <th>113</th>
      <th>114</th>
      <th>115</th>
      <th>116</th>
      <th>117</th>
      <th>118</th>
      <th>119</th>
      <th>120</th>
      <th>121</th>
      <th>122</th>
      <th>123</th>
      <th>124</th>
      <th>125</th>
      <th>126</th>
      <th>127</th>
      <th>128</th>
      <th>129</th>
      <th>130</th>
      <th>131</th>
      <th>132</th>
      <th>133</th>
      <th>134</th>
      <th>135</th>
      <th>136</th>
      <th>137</th>
      <th>138</th>
      <th>139</th>
      <th>140</th>
      <th>141</th>
      <th>142</th>
      <th>143</th>
      <th>144</th>
      <th>145</th>
      <th>146</th>
      <th>147</th>
      <th>148</th>
      <th>149</th>
      <th>150</th>
      <th>151</th>
      <th>152</th>
      <th>153</th>
      <th>154</th>
      <th>155</th>
      <th>156</th>
      <th>157</th>
      <th>158</th>
      <th>159</th>
      <th>160</th>
      <th>161</th>
      <th>162</th>
      <th>163</th>
      <th>164</th>
      <th>165</th>
      <th>166</th>
      <th>167</th>
      <th>168</th>
      <th>169</th>
      <th>170</th>
      <th>171</th>
      <th>172</th>
      <th>173</th>
      <th>174</th>
      <th>175</th>
      <th>176</th>
      <th>177</th>
      <th>178</th>
      <th>179</th>
      <th>180</th>
      <th>181</th>
      <th>182</th>
      <th>183</th>
      <th>184</th>
      <th>185</th>
      <th>186</th>
      <th>187</th>
      <th>188</th>
      <th>189</th>
      <th>190</th>
      <th>191</th>
      <th>192</th>
      <th>193</th>
      <th>194</th>
      <th>195</th>
      <th>196</th>
      <th>197</th>
      <th>198</th>
      <th>199</th>
      <th>200</th>
      <th>201</th>
      <th>202</th>
      <th>203</th>
      <th>204</th>
      <th>205</th>
      <th>206</th>
      <th>207</th>
      <th>208</th>
      <th>209</th>
      <th>210</th>
      <th>211</th>
      <th>212</th>
      <th>213</th>
      <th>214</th>
      <th>215</th>
      <th>216</th>
      <th>217</th>
      <th>218</th>
      <th>219</th>
      <th>220</th>
      <th>221</th>
      <th>222</th>
      <th>223</th>
      <th>224</th>
      <th>225</th>
      <th>226</th>
      <th>227</th>
      <th>228</th>
      <th>229</th>
      <th>230</th>
      <th>231</th>
      <th>232</th>
      <th>233</th>
      <th>234</th>
      <th>235</th>
      <th>236</th>
      <th>237</th>
      <th>238</th>
      <th>239</th>
      <th>240</th>
      <th>241</th>
      <th>242</th>
      <th>243</th>
      <th>244</th>
      <th>245</th>
      <th>246</th>
      <th>247</th>
      <th>248</th>
      <th>249</th>
      <th>250</th>
      <th>251</th>
      <th>252</th>
      <th>253</th>
      <th>254</th>
      <th>255</th>
      <th>256</th>
      <th>257</th>
      <th>258</th>
      <th>259</th>
      <th>260</th>
      <th>261</th>
      <th>262</th>
      <th>263</th>
      <th>264</th>
      <th>265</th>
      <th>266</th>
      <th>267</th>
      <th>268</th>
      <th>269</th>
      <th>270</th>
      <th>271</th>
      <th>272</th>
      <th>273</th>
      <th>274</th>
      <th>275</th>
      <th>276</th>
      <th>277</th>
      <th>278</th>
      <th>279</th>
      <th>280</th>
      <th>281</th>
      <th>282</th>
      <th>283</th>
      <th>284</th>
      <th>285</th>
      <th>286</th>
      <th>287</th>
      <th>288</th>
      <th>289</th>
      <th>290</th>
      <th>291</th>
      <th>292</th>
      <th>293</th>
      <th>294</th>
      <th>295</th>
      <th>296</th>
      <th>297</th>
      <th>298</th>
      <th>299</th>
      <th>300</th>
      <th>301</th>
      <th>302</th>
      <th>303</th>
      <th>304</th>
      <th>305</th>
      <th>306</th>
      <th>307</th>
      <th>308</th>
      <th>309</th>
      <th>310</th>
      <th>311</th>
      <th>312</th>
      <th>313</th>
      <th>314</th>
      <th>315</th>
      <th>316</th>
      <th>317</th>
      <th>318</th>
      <th>319</th>
      <th>320</th>
      <th>321</th>
      <th>322</th>
      <th>323</th>
      <th>324</th>
      <th>325</th>
      <th>326</th>
      <th>327</th>
      <th>328</th>
      <th>329</th>
      <th>330</th>
      <th>331</th>
      <th>332</th>
      <th>333</th>
      <th>334</th>
      <th>335</th>
      <th>336</th>
      <th>337</th>
      <th>338</th>
      <th>339</th>
      <th>340</th>
      <th>341</th>
      <th>342</th>
      <th>343</th>
      <th>344</th>
      <th>345</th>
      <th>346</th>
      <th>347</th>
      <th>348</th>
      <th>349</th>
      <th>350</th>
      <th>351</th>
      <th>352</th>
      <th>353</th>
      <th>354</th>
      <th>355</th>
      <th>356</th>
      <th>357</th>
      <th>358</th>
      <th>359</th>
      <th>360</th>
      <th>361</th>
      <th>362</th>
      <th>363</th>
      <th>364</th>
      <th>365</th>
      <th>366</th>
      <th>367</th>
      <th>368</th>
      <th>369</th>
      <th>370</th>
      <th>371</th>
      <th>372</th>
      <th>373</th>
      <th>374</th>
      <th>375</th>
      <th>376</th>
      <th>377</th>
      <th>378</th>
      <th>379</th>
      <th>380</th>
      <th>381</th>
      <th>382</th>
      <th>383</th>
      <th>384</th>
      <th>385</th>
      <th>386</th>
      <th>387</th>
      <th>388</th>
      <th>389</th>
      <th>390</th>
      <th>391</th>
      <th>392</th>
      <th>393</th>
      <th>394</th>
      <th>395</th>
      <th>396</th>
      <th>397</th>
      <th>398</th>
      <th>399</th>
      <th>400</th>
      <th>401</th>
      <th>402</th>
      <th>403</th>
      <th>404</th>
      <th>405</th>
      <th>406</th>
      <th>407</th>
      <th>408</th>
      <th>409</th>
      <th>410</th>
      <th>411</th>
      <th>412</th>
      <th>413</th>
      <th>414</th>
      <th>415</th>
      <th>416</th>
      <th>417</th>
      <th>418</th>
      <th>419</th>
      <th>420</th>
      <th>421</th>
      <th>422</th>
      <th>423</th>
      <th>424</th>
      <th>425</th>
      <th>426</th>
      <th>427</th>
      <th>428</th>
      <th>429</th>
      <th>430</th>
      <th>431</th>
      <th>432</th>
      <th>433</th>
      <th>434</th>
      <th>435</th>
      <th>436</th>
      <th>437</th>
      <th>438</th>
      <th>439</th>
      <th>440</th>
      <th>441</th>
      <th>442</th>
      <th>443</th>
      <th>444</th>
      <th>445</th>
      <th>446</th>
      <th>447</th>
      <th>448</th>
      <th>449</th>
      <th>450</th>
      <th>451</th>
      <th>452</th>
      <th>453</th>
      <th>454</th>
      <th>455</th>
      <th>456</th>
      <th>457</th>
      <th>458</th>
      <th>459</th>
      <th>460</th>
      <th>461</th>
      <th>462</th>
      <th>463</th>
      <th>464</th>
      <th>465</th>
      <th>466</th>
      <th>467</th>
      <th>468</th>
      <th>469</th>
      <th>470</th>
      <th>471</th>
      <th>472</th>
      <th>473</th>
      <th>474</th>
      <th>475</th>
      <th>476</th>
      <th>477</th>
      <th>478</th>
      <th>479</th>
      <th>480</th>
      <th>481</th>
      <th>482</th>
      <th>483</th>
      <th>484</th>
      <th>485</th>
      <th>486</th>
      <th>487</th>
      <th>488</th>
      <th>489</th>
      <th>490</th>
      <th>491</th>
      <th>492</th>
      <th>493</th>
      <th>494</th>
      <th>495</th>
      <th>496</th>
      <th>497</th>
      <th>498</th>
      <th>499</th>
      <th>500</th>
      <th>501</th>
      <th>502</th>
      <th>503</th>
      <th>504</th>
      <th>505</th>
      <th>506</th>
      <th>507</th>
      <th>508</th>
      <th>509</th>
      <th>510</th>
      <th>511</th>
      <th>512</th>
      <th>513</th>
      <th>514</th>
      <th>515</th>
      <th>516</th>
      <th>517</th>
      <th>518</th>
      <th>519</th>
      <th>520</th>
      <th>521</th>
      <th>522</th>
      <th>523</th>
      <th>524</th>
      <th>525</th>
      <th>526</th>
      <th>527</th>
      <th>528</th>
      <th>529</th>
      <th>530</th>
      <th>531</th>
      <th>532</th>
      <th>533</th>
      <th>534</th>
      <th>535</th>
      <th>536</th>
      <th>537</th>
      <th>538</th>
      <th>539</th>
      <th>540</th>
      <th>541</th>
      <th>542</th>
      <th>543</th>
      <th>544</th>
      <th>545</th>
      <th>546</th>
      <th>547</th>
      <th>548</th>
      <th>549</th>
      <th>550</th>
      <th>551</th>
      <th>552</th>
      <th>553</th>
      <th>554</th>
      <th>555</th>
      <th>556</th>
      <th>557</th>
      <th>558</th>
      <th>559</th>
      <th>560</th>
      <th>561</th>
      <th>562</th>
      <th>563</th>
      <th>564</th>
      <th>565</th>
      <th>566</th>
      <th>567</th>
      <th>568</th>
      <th>569</th>
      <th>570</th>
      <th>571</th>
      <th>572</th>
      <th>573</th>
      <th>574</th>
      <th>575</th>
      <th>576</th>
      <th>577</th>
      <th>578</th>
      <th>579</th>
      <th>580</th>
      <th>581</th>
      <th>582</th>
      <th>583</th>
      <th>584</th>
      <th>585</th>
      <th>586</th>
      <th>587</th>
      <th>588</th>
      <th>589</th>
      <th>590</th>
      <th>591</th>
      <th>592</th>
      <th>593</th>
      <th>594</th>
      <th>595</th>
      <th>596</th>
      <th>597</th>
      <th>598</th>
      <th>599</th>
      <th>600</th>
      <th>601</th>
      <th>602</th>
      <th>603</th>
      <th>604</th>
      <th>605</th>
      <th>606</th>
      <th>607</th>
      <th>608</th>
      <th>609</th>
      <th>610</th>
      <th>611</th>
      <th>612</th>
      <th>613</th>
      <th>614</th>
      <th>615</th>
      <th>616</th>
      <th>617</th>
      <th>618</th>
      <th>619</th>
      <th>620</th>
      <th>621</th>
      <th>622</th>
      <th>623</th>
      <th>624</th>
      <th>625</th>
      <th>626</th>
      <th>627</th>
      <th>628</th>
      <th>629</th>
      <th>630</th>
      <th>631</th>
      <th>632</th>
      <th>633</th>
      <th>634</th>
      <th>635</th>
      <th>636</th>
      <th>637</th>
      <th>638</th>
      <th>639</th>
      <th>640</th>
      <th>641</th>
      <th>642</th>
      <th>643</th>
      <th>644</th>
      <th>645</th>
      <th>646</th>
      <th>647</th>
      <th>648</th>
      <th>649</th>
      <th>650</th>
      <th>651</th>
      <th>652</th>
      <th>653</th>
      <th>654</th>
      <th>655</th>
      <th>656</th>
      <th>657</th>
      <th>658</th>
      <th>659</th>
      <th>660</th>
      <th>661</th>
      <th>662</th>
      <th>663</th>
      <th>664</th>
      <th>665</th>
      <th>666</th>
      <th>667</th>
      <th>668</th>
      <th>669</th>
      <th>670</th>
      <th>671</th>
      <th>672</th>
      <th>673</th>
      <th>674</th>
      <th>675</th>
      <th>676</th>
      <th>677</th>
      <th>678</th>
      <th>679</th>
      <th>680</th>
      <th>681</th>
      <th>682</th>
      <th>683</th>
      <th>684</th>
      <th>685</th>
      <th>686</th>
      <th>687</th>
      <th>688</th>
      <th>689</th>
      <th>690</th>
      <th>691</th>
      <th>692</th>
      <th>693</th>
      <th>694</th>
      <th>695</th>
      <th>696</th>
      <th>697</th>
      <th>698</th>
      <th>699</th>
      <th>700</th>
      <th>701</th>
      <th>702</th>
      <th>703</th>
      <th>704</th>
      <th>705</th>
      <th>706</th>
      <th>707</th>
      <th>708</th>
      <th>709</th>
      <th>710</th>
      <th>711</th>
      <th>712</th>
      <th>713</th>
      <th>714</th>
      <th>715</th>
      <th>716</th>
      <th>717</th>
      <th>718</th>
      <th>719</th>
      <th>720</th>
      <th>721</th>
      <th>722</th>
      <th>723</th>
      <th>724</th>
      <th>725</th>
      <th>726</th>
      <th>727</th>
      <th>728</th>
      <th>729</th>
      <th>730</th>
      <th>731</th>
      <th>732</th>
      <th>733</th>
      <th>734</th>
      <th>735</th>
      <th>736</th>
      <th>737</th>
      <th>738</th>
      <th>739</th>
      <th>740</th>
      <th>741</th>
      <th>742</th>
      <th>743</th>
      <th>744</th>
      <th>745</th>
      <th>746</th>
      <th>747</th>
      <th>748</th>
      <th>749</th>
      <th>750</th>
      <th>751</th>
      <th>752</th>
      <th>753</th>
      <th>754</th>
      <th>755</th>
      <th>756</th>
      <th>757</th>
      <th>758</th>
      <th>759</th>
      <th>760</th>
      <th>761</th>
      <th>762</th>
      <th>763</th>
      <th>764</th>
      <th>765</th>
      <th>766</th>
      <th>767</th>
      <th>768</th>
      <th>769</th>
      <th>770</th>
      <th>771</th>
      <th>772</th>
      <th>773</th>
      <th>774</th>
      <th>775</th>
      <th>776</th>
      <th>777</th>
      <th>778</th>
      <th>779</th>
      <th>780</th>
      <th>781</th>
      <th>782</th>
      <th>783</th>
      <th>784</th>
      <th>785</th>
      <th>786</th>
      <th>787</th>
      <th>788</th>
      <th>789</th>
      <th>790</th>
      <th>791</th>
      <th>792</th>
      <th>793</th>
      <th>794</th>
      <th>795</th>
      <th>796</th>
      <th>797</th>
      <th>798</th>
      <th>799</th>
      <th>800</th>
      <th>801</th>
      <th>802</th>
      <th>803</th>
      <th>804</th>
      <th>805</th>
      <th>806</th>
      <th>807</th>
      <th>808</th>
      <th>809</th>
      <th>810</th>
      <th>811</th>
      <th>812</th>
      <th>813</th>
      <th>814</th>
      <th>815</th>
      <th>816</th>
      <th>817</th>
      <th>818</th>
      <th>819</th>
      <th>820</th>
      <th>821</th>
      <th>822</th>
      <th>823</th>
      <th>824</th>
      <th>825</th>
      <th>826</th>
      <th>827</th>
      <th>828</th>
      <th>829</th>
      <th>830</th>
      <th>831</th>
      <th>832</th>
      <th>833</th>
      <th>834</th>
      <th>835</th>
      <th>836</th>
      <th>837</th>
      <th>838</th>
      <th>839</th>
      <th>840</th>
      <th>841</th>
      <th>842</th>
      <th>843</th>
      <th>844</th>
      <th>845</th>
      <th>846</th>
      <th>847</th>
      <th>848</th>
      <th>849</th>
      <th>850</th>
      <th>851</th>
      <th>852</th>
      <th>853</th>
      <th>854</th>
      <th>855</th>
      <th>856</th>
      <th>857</th>
      <th>858</th>
      <th>859</th>
      <th>860</th>
      <th>861</th>
      <th>862</th>
      <th>863</th>
      <th>864</th>
      <th>865</th>
      <th>866</th>
      <th>867</th>
      <th>868</th>
      <th>869</th>
      <th>870</th>
      <th>871</th>
      <th>872</th>
      <th>873</th>
      <th>874</th>
      <th>875</th>
      <th>876</th>
      <th>877</th>
      <th>878</th>
      <th>879</th>
      <th>880</th>
      <th>881</th>
      <th>882</th>
      <th>883</th>
      <th>884</th>
      <th>885</th>
      <th>886</th>
      <th>887</th>
      <th>888</th>
      <th>889</th>
      <th>890</th>
      <th>891</th>
      <th>892</th>
      <th>893</th>
      <th>894</th>
      <th>895</th>
      <th>896</th>
      <th>897</th>
      <th>898</th>
      <th>899</th>
      <th>900</th>
      <th>901</th>
      <th>902</th>
      <th>903</th>
      <th>904</th>
      <th>905</th>
      <th>906</th>
      <th>907</th>
      <th>908</th>
      <th>909</th>
      <th>910</th>
      <th>911</th>
      <th>912</th>
      <th>913</th>
      <th>914</th>
      <th>915</th>
      <th>916</th>
      <th>917</th>
      <th>918</th>
      <th>919</th>
      <th>920</th>
      <th>921</th>
      <th>922</th>
      <th>923</th>
      <th>924</th>
      <th>925</th>
      <th>926</th>
      <th>927</th>
      <th>928</th>
      <th>929</th>
      <th>930</th>
      <th>931</th>
      <th>932</th>
      <th>933</th>
      <th>934</th>
      <th>935</th>
      <th>936</th>
      <th>937</th>
      <th>938</th>
      <th>939</th>
      <th>940</th>
      <th>941</th>
      <th>942</th>
      <th>943</th>
      <th>944</th>
      <th>945</th>
      <th>946</th>
      <th>947</th>
      <th>948</th>
      <th>949</th>
      <th>950</th>
      <th>951</th>
      <th>952</th>
      <th>953</th>
      <th>954</th>
      <th>955</th>
      <th>956</th>
      <th>957</th>
      <th>958</th>
      <th>959</th>
      <th>960</th>
      <th>961</th>
      <th>962</th>
      <th>963</th>
      <th>964</th>
      <th>965</th>
      <th>966</th>
      <th>967</th>
      <th>968</th>
      <th>969</th>
      <th>970</th>
      <th>971</th>
      <th>972</th>
      <th>973</th>
      <th>974</th>
      <th>975</th>
      <th>976</th>
      <th>977</th>
      <th>978</th>
      <th>979</th>
      <th>980</th>
      <th>981</th>
      <th>982</th>
      <th>983</th>
      <th>984</th>
      <th>985</th>
      <th>986</th>
      <th>987</th>
      <th>988</th>
      <th>989</th>
      <th>990</th>
      <th>991</th>
      <th>992</th>
      <th>993</th>
      <th>994</th>
      <th>995</th>
      <th>996</th>
      <th>997</th>
      <th>998</th>
      <th>999</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>tench, Tinca tinca</td>
      <td>goldfish, Carassius auratus</td>
      <td>great white shark, white shark, man-eater, man-eating shark, Carcharodon carcharias</td>
      <td>tiger shark, Galeocerdo cuvieri</td>
      <td>hammerhead, hammerhead shark</td>
      <td>electric ray, crampfish, numbfish, torpedo</td>
      <td>stingray</td>
      <td>cock</td>
      <td>hen</td>
      <td>ostrich, Struthio camelus</td>
      <td>brambling, Fringilla montifringilla</td>
      <td>goldfinch, Carduelis carduelis</td>
      <td>house finch, linnet, Carpodacus mexicanus</td>
      <td>junco, snowbird</td>
      <td>indigo bunting, indigo finch, indigo bird, Passerina cyanea</td>
      <td>robin, American robin, Turdus migratorius</td>
      <td>bulbul</td>
      <td>jay</td>
      <td>magpie</td>
      <td>chickadee</td>
      <td>water ouzel, dipper</td>
      <td>kite</td>
      <td>bald eagle, American eagle, Haliaeetus leucocephalus</td>
      <td>vulture</td>
      <td>great grey owl, great gray owl, Strix nebulosa</td>
      <td>European fire salamander, Salamandra salamandra</td>
      <td>common newt, Triturus vulgaris</td>
      <td>eft</td>
      <td>spotted salamander, Ambystoma maculatum</td>
      <td>axolotl, mud puppy, Ambystoma mexicanum</td>
      <td>bullfrog, Rana catesbeiana</td>
      <td>tree frog, tree-frog</td>
      <td>tailed frog, bell toad, ribbed toad, tailed toad, Ascaphus trui</td>
      <td>loggerhead, loggerhead turtle, Caretta caretta</td>
      <td>leatherback turtle, leatherback, leathery turtle, Dermochelys coriacea</td>
      <td>mud turtle</td>
      <td>terrapin</td>
      <td>box turtle, box tortoise</td>
      <td>banded gecko</td>
      <td>common iguana, iguana, Iguana iguana</td>
      <td>American chameleon, anole, Anolis carolinensis</td>
      <td>whiptail, whiptail lizard</td>
      <td>agama</td>
      <td>frilled lizard, Chlamydosaurus kingi</td>
      <td>alligator lizard</td>
      <td>Gila monster, Heloderma suspectum</td>
      <td>green lizard, Lacerta viridis</td>
      <td>African chameleon, Chamaeleo chamaeleon</td>
      <td>Komodo dragon, Komodo lizard, dragon lizard, giant lizard, Varanus komodoensis</td>
      <td>African crocodile, Nile crocodile, Crocodylus niloticus</td>
      <td>American alligator, Alligator mississipiensis</td>
      <td>triceratops</td>
      <td>thunder snake, worm snake, Carphophis amoenus</td>
      <td>ringneck snake, ring-necked snake, ring snake</td>
      <td>hognose snake, puff adder, sand viper</td>
      <td>green snake, grass snake</td>
      <td>king snake, kingsnake</td>
      <td>garter snake, grass snake</td>
      <td>water snake</td>
      <td>vine snake</td>
      <td>night snake, Hypsiglena torquata</td>
      <td>boa constrictor, Constrictor constrictor</td>
      <td>rock python, rock snake, Python sebae</td>
      <td>Indian cobra, Naja naja</td>
      <td>green mamba</td>
      <td>sea snake</td>
      <td>horned viper, cerastes, sand viper, horned asp, Cerastes cornutus</td>
      <td>diamondback, diamondback rattlesnake, Crotalus adamanteus</td>
      <td>sidewinder, horned rattlesnake, Crotalus cerastes</td>
      <td>trilobite</td>
      <td>harvestman, daddy longlegs, Phalangium opilio</td>
      <td>scorpion</td>
      <td>black and gold garden spider, Argiope aurantia</td>
      <td>barn spider, Araneus cavaticus</td>
      <td>garden spider, Aranea diademata</td>
      <td>black widow, Latrodectus mactans</td>
      <td>tarantula</td>
      <td>wolf spider, hunting spider</td>
      <td>tick</td>
      <td>centipede</td>
      <td>black grouse</td>
      <td>ptarmigan</td>
      <td>ruffed grouse, partridge, Bonasa umbellus</td>
      <td>prairie chicken, prairie grouse, prairie fowl</td>
      <td>peacock</td>
      <td>quail</td>
      <td>partridge</td>
      <td>African grey, African gray, Psittacus erithacus</td>
      <td>macaw</td>
      <td>sulphur-crested cockatoo, Kakatoe galerita, Cacatua galerita</td>
      <td>lorikeet</td>
      <td>coucal</td>
      <td>bee eater</td>
      <td>hornbill</td>
      <td>hummingbird</td>
      <td>jacamar</td>
      <td>toucan</td>
      <td>drake</td>
      <td>red-breasted merganser, Mergus serrator</td>
      <td>goose</td>
      <td>black swan, Cygnus atratus</td>
      <td>tusker</td>
      <td>echidna, spiny anteater, anteater</td>
      <td>platypus, duckbill, duckbilled platypus, duck-billed platypus, Ornithorhynchus anatinus</td>
      <td>wallaby, brush kangaroo</td>
      <td>koala, koala bear, kangaroo bear, native bear, Phascolarctos cinereus</td>
      <td>wombat</td>
      <td>jellyfish</td>
      <td>sea anemone, anemone</td>
      <td>brain coral</td>
      <td>flatworm, platyhelminth</td>
      <td>nematode, nematode worm, roundworm</td>
      <td>conch</td>
      <td>snail</td>
      <td>slug</td>
      <td>sea slug, nudibranch</td>
      <td>chiton, coat-of-mail shell, sea cradle, polyplacophore</td>
      <td>chambered nautilus, pearly nautilus, nautilus</td>
      <td>Dungeness crab, Cancer magister</td>
      <td>rock crab, Cancer irroratus</td>
      <td>fiddler crab</td>
      <td>king crab, Alaska crab, Alaskan king crab, Alaska king crab, Paralithodes camtschatica</td>
      <td>American lobster, Northern lobster, Maine lobster, Homarus americanus</td>
      <td>spiny lobster, langouste, rock lobster, crawfish, crayfish, sea crawfish</td>
      <td>crayfish, crawfish, crawdad, crawdaddy</td>
      <td>hermit crab</td>
      <td>isopod</td>
      <td>white stork, Ciconia ciconia</td>
      <td>black stork, Ciconia nigra</td>
      <td>spoonbill</td>
      <td>flamingo</td>
      <td>little blue heron, Egretta caerulea</td>
      <td>American egret, great white heron, Egretta albus</td>
      <td>bittern</td>
      <td>crane</td>
      <td>limpkin, Aramus pictus</td>
      <td>European gallinule, Porphyrio porphyrio</td>
      <td>American coot, marsh hen, mud hen, water hen, Fulica americana</td>
      <td>bustard</td>
      <td>ruddy turnstone, Arenaria interpres</td>
      <td>red-backed sandpiper, dunlin, Erolia alpina</td>
      <td>redshank, Tringa totanus</td>
      <td>dowitcher</td>
      <td>oystercatcher, oyster catcher</td>
      <td>pelican</td>
      <td>king penguin, Aptenodytes patagonica</td>
      <td>albatross, mollymawk</td>
      <td>grey whale, gray whale, devilfish, Eschrichtius gibbosus, Eschrichtius robustus</td>
      <td>killer whale, killer, orca, grampus, sea wolf, Orcinus orca</td>
      <td>dugong, Dugong dugon</td>
      <td>sea lion</td>
      <td>Chihuahua</td>
      <td>Japanese spaniel</td>
      <td>Maltese dog, Maltese terrier, Maltese</td>
      <td>Pekinese, Pekingese, Peke</td>
      <td>Shih-Tzu</td>
      <td>Blenheim spaniel</td>
      <td>papillon</td>
      <td>toy terrier</td>
      <td>Rhodesian ridgeback</td>
      <td>Afghan hound, Afghan</td>
      <td>basset, basset hound</td>
      <td>beagle</td>
      <td>bloodhound, sleuthhound</td>
      <td>bluetick</td>
      <td>black-and-tan coonhound</td>
      <td>Walker hound, Walker foxhound</td>
      <td>English foxhound</td>
      <td>redbone</td>
      <td>borzoi, Russian wolfhound</td>
      <td>Irish wolfhound</td>
      <td>Italian greyhound</td>
      <td>whippet</td>
      <td>Ibizan hound, Ibizan Podenco</td>
      <td>Norwegian elkhound, elkhound</td>
      <td>otterhound, otter hound</td>
      <td>Saluki, gazelle hound</td>
      <td>Scottish deerhound, deerhound</td>
      <td>Weimaraner</td>
      <td>Staffordshire bullterrier, Staffordshire bull terrier</td>
      <td>American Staffordshire terrier, Staffordshire terrier, American pit bull terrier, pit bull terrier</td>
      <td>Bedlington terrier</td>
      <td>Border terrier</td>
      <td>Kerry blue terrier</td>
      <td>Irish terrier</td>
      <td>Norfolk terrier</td>
      <td>Norwich terrier</td>
      <td>Yorkshire terrier</td>
      <td>wire-haired fox terrier</td>
      <td>Lakeland terrier</td>
      <td>Sealyham terrier, Sealyham</td>
      <td>Airedale, Airedale terrier</td>
      <td>cairn, cairn terrier</td>
      <td>Australian terrier</td>
      <td>Dandie Dinmont, Dandie Dinmont terrier</td>
      <td>Boston bull, Boston terrier</td>
      <td>miniature schnauzer</td>
      <td>giant schnauzer</td>
      <td>standard schnauzer</td>
      <td>Scotch terrier, Scottish terrier, Scottie</td>
      <td>Tibetan terrier, chrysanthemum dog</td>
      <td>silky terrier, Sydney silky</td>
      <td>soft-coated wheaten terrier</td>
      <td>West Highland white terrier</td>
      <td>Lhasa, Lhasa apso</td>
      <td>flat-coated retriever</td>
      <td>curly-coated retriever</td>
      <td>golden retriever</td>
      <td>Labrador retriever</td>
      <td>Chesapeake Bay retriever</td>
      <td>German short-haired pointer</td>
      <td>vizsla, Hungarian pointer</td>
      <td>English setter</td>
      <td>Irish setter, red setter</td>
      <td>Gordon setter</td>
      <td>Brittany spaniel</td>
      <td>clumber, clumber spaniel</td>
      <td>English springer, English springer spaniel</td>
      <td>Welsh springer spaniel</td>
      <td>cocker spaniel, English cocker spaniel, cocker</td>
      <td>Sussex spaniel</td>
      <td>Irish water spaniel</td>
      <td>kuvasz</td>
      <td>schipperke</td>
      <td>groenendael</td>
      <td>malinois</td>
      <td>briard</td>
      <td>kelpie</td>
      <td>komondor</td>
      <td>Old English sheepdog, bobtail</td>
      <td>Shetland sheepdog, Shetland sheep dog, Shetland</td>
      <td>collie</td>
      <td>Border collie</td>
      <td>Bouvier des Flandres, Bouviers des Flandres</td>
      <td>Rottweiler</td>
      <td>German shepherd, German shepherd dog, German police dog, alsatian</td>
      <td>Doberman, Doberman pinscher</td>
      <td>miniature pinscher</td>
      <td>Greater Swiss Mountain dog</td>
      <td>Bernese mountain dog</td>
      <td>Appenzeller</td>
      <td>EntleBucher</td>
      <td>boxer</td>
      <td>bull mastiff</td>
      <td>Tibetan mastiff</td>
      <td>French bulldog</td>
      <td>Great Dane</td>
      <td>Saint Bernard, St Bernard</td>
      <td>Eskimo dog, husky</td>
      <td>malamute, malemute, Alaskan malamute</td>
      <td>Siberian husky</td>
      <td>dalmatian, coach dog, carriage dog</td>
      <td>affenpinscher, monkey pinscher, monkey dog</td>
      <td>basenji</td>
      <td>pug, pug-dog</td>
      <td>Leonberg</td>
      <td>Newfoundland, Newfoundland dog</td>
      <td>Great Pyrenees</td>
      <td>Samoyed, Samoyede</td>
      <td>Pomeranian</td>
      <td>chow, chow chow</td>
      <td>keeshond</td>
      <td>Brabancon griffon</td>
      <td>Pembroke, Pembroke Welsh corgi</td>
      <td>Cardigan, Cardigan Welsh corgi</td>
      <td>toy poodle</td>
      <td>miniature poodle</td>
      <td>standard poodle</td>
      <td>Mexican hairless</td>
      <td>timber wolf, grey wolf, gray wolf, Canis lupus</td>
      <td>white wolf, Arctic wolf, Canis lupus tundrarum</td>
      <td>red wolf, maned wolf, Canis rufus, Canis niger</td>
      <td>coyote, prairie wolf, brush wolf, Canis latrans</td>
      <td>dingo, warrigal, warragal, Canis dingo</td>
      <td>dhole, Cuon alpinus</td>
      <td>African hunting dog, hyena dog, Cape hunting dog, Lycaon pictus</td>
      <td>hyena, hyaena</td>
      <td>red fox, Vulpes vulpes</td>
      <td>kit fox, Vulpes macrotis</td>
      <td>Arctic fox, white fox, Alopex lagopus</td>
      <td>grey fox, gray fox, Urocyon cinereoargenteus</td>
      <td>tabby, tabby cat</td>
      <td>tiger cat</td>
      <td>Persian cat</td>
      <td>Siamese cat, Siamese</td>
      <td>Egyptian cat</td>
      <td>cougar, puma, catamount, mountain lion, painter, panther, Felis concolor</td>
      <td>lynx, catamount</td>
      <td>leopard, Panthera pardus</td>
      <td>snow leopard, ounce, Panthera uncia</td>
      <td>jaguar, panther, Panthera onca, Felis onca</td>
      <td>lion, king of beasts, Panthera leo</td>
      <td>tiger, Panthera tigris</td>
      <td>cheetah, chetah, Acinonyx jubatus</td>
      <td>brown bear, bruin, Ursus arctos</td>
      <td>American black bear, black bear, Ursus americanus, Euarctos americanus</td>
      <td>ice bear, polar bear, Ursus Maritimus, Thalarctos maritimus</td>
      <td>sloth bear, Melursus ursinus, Ursus ursinus</td>
      <td>mongoose</td>
      <td>meerkat, mierkat</td>
      <td>tiger beetle</td>
      <td>ladybug, ladybeetle, lady beetle, ladybird, ladybird beetle</td>
      <td>ground beetle, carabid beetle</td>
      <td>long-horned beetle, longicorn, longicorn beetle</td>
      <td>leaf beetle, chrysomelid</td>
      <td>dung beetle</td>
      <td>rhinoceros beetle</td>
      <td>weevil</td>
      <td>fly</td>
      <td>bee</td>
      <td>ant, emmet, pismire</td>
      <td>grasshopper, hopper</td>
      <td>cricket</td>
      <td>walking stick, walkingstick, stick insect</td>
      <td>cockroach, roach</td>
      <td>mantis, mantid</td>
      <td>cicada, cicala</td>
      <td>leafhopper</td>
      <td>lacewing, lacewing fly</td>
      <td>dragonfly, darning needle, devil's darning needle, sewing needle, snake feeder, snake doctor, mosquito hawk, skeeter hawk</td>
      <td>damselfly</td>
      <td>admiral</td>
      <td>ringlet, ringlet butterfly</td>
      <td>monarch, monarch butterfly, milkweed butterfly, Danaus plexippus</td>
      <td>cabbage butterfly</td>
      <td>sulphur butterfly, sulfur butterfly</td>
      <td>lycaenid, lycaenid butterfly</td>
      <td>starfish, sea star</td>
      <td>sea urchin</td>
      <td>sea cucumber, holothurian</td>
      <td>wood rabbit, cottontail, cottontail rabbit</td>
      <td>hare</td>
      <td>Angora, Angora rabbit</td>
      <td>hamster</td>
      <td>porcupine, hedgehog</td>
      <td>fox squirrel, eastern fox squirrel, Sciurus niger</td>
      <td>marmot</td>
      <td>beaver</td>
      <td>guinea pig, Cavia cobaya</td>
      <td>sorrel</td>
      <td>zebra</td>
      <td>hog, pig, grunter, squealer, Sus scrofa</td>
      <td>wild boar, boar, Sus scrofa</td>
      <td>warthog</td>
      <td>hippopotamus, hippo, river horse, Hippopotamus amphibius</td>
      <td>ox</td>
      <td>water buffalo, water ox, Asiatic buffalo, Bubalus bubalis</td>
      <td>bison</td>
      <td>ram, tup</td>
      <td>bighorn, bighorn sheep, cimarron, Rocky Mountain bighorn, Rocky Mountain sheep, Ovis canadensis</td>
      <td>ibex, Capra ibex</td>
      <td>hartebeest</td>
      <td>impala, Aepyceros melampus</td>
      <td>gazelle</td>
      <td>Arabian camel, dromedary, Camelus dromedarius</td>
      <td>llama</td>
      <td>weasel</td>
      <td>mink</td>
      <td>polecat, fitch, foulmart, foumart, Mustela putorius</td>
      <td>black-footed ferret, ferret, Mustela nigripes</td>
      <td>otter</td>
      <td>skunk, polecat, wood pussy</td>
      <td>badger</td>
      <td>armadillo</td>
      <td>three-toed sloth, ai, Bradypus tridactylus</td>
      <td>orangutan, orang, orangutang, Pongo pygmaeus</td>
      <td>gorilla, Gorilla gorilla</td>
      <td>chimpanzee, chimp, Pan troglodytes</td>
      <td>gibbon, Hylobates lar</td>
      <td>siamang, Hylobates syndactylus, Symphalangus syndactylus</td>
      <td>guenon, guenon monkey</td>
      <td>patas, hussar monkey, Erythrocebus patas</td>
      <td>baboon</td>
      <td>macaque</td>
      <td>langur</td>
      <td>colobus, colobus monkey</td>
      <td>proboscis monkey, Nasalis larvatus</td>
      <td>marmoset</td>
      <td>capuchin, ringtail, Cebus capucinus</td>
      <td>howler monkey, howler</td>
      <td>titi, titi monkey</td>
      <td>spider monkey, Ateles geoffroyi</td>
      <td>squirrel monkey, Saimiri sciureus</td>
      <td>Madagascar cat, ring-tailed lemur, Lemur catta</td>
      <td>indri, indris, Indri indri, Indri brevicaudatus</td>
      <td>Indian elephant, Elephas maximus</td>
      <td>African elephant, Loxodonta africana</td>
      <td>lesser panda, red panda, panda, bear cat, cat bear, Ailurus fulgens</td>
      <td>giant panda, panda, panda bear, coon bear, Ailuropoda melanoleuca</td>
      <td>barracouta, snoek</td>
      <td>eel</td>
      <td>coho, cohoe, coho salmon, blue jack, silver salmon, Oncorhynchus kisutch</td>
      <td>rock beauty, Holocanthus tricolor</td>
      <td>anemone fish</td>
      <td>sturgeon</td>
      <td>gar, garfish, garpike, billfish, Lepisosteus osseus</td>
      <td>lionfish</td>
      <td>puffer, pufferfish, blowfish, globefish</td>
      <td>abacus</td>
      <td>abaya</td>
      <td>academic gown, academic robe, judge's robe</td>
      <td>accordion, piano accordion, squeeze box</td>
      <td>acoustic guitar</td>
      <td>aircraft carrier, carrier, flattop, attack aircraft carrier</td>
      <td>airliner</td>
      <td>airship, dirigible</td>
      <td>altar</td>
      <td>ambulance</td>
      <td>amphibian, amphibious vehicle</td>
      <td>analog clock</td>
      <td>apiary, bee house</td>
      <td>apron</td>
      <td>ashcan, trash can, garbage can, wastebin, ash bin, ash-bin, ashbin, dustbin, trash barrel, trash bin</td>
      <td>assault rifle, assault gun</td>
      <td>backpack, back pack, knapsack, packsack, rucksack, haversack</td>
      <td>bakery, bakeshop, bakehouse</td>
      <td>balance beam, beam</td>
      <td>balloon</td>
      <td>ballpoint, ballpoint pen, ballpen, Biro</td>
      <td>Band Aid</td>
      <td>banjo</td>
      <td>bannister, banister, balustrade, balusters, handrail</td>
      <td>barbell</td>
      <td>barber chair</td>
      <td>barbershop</td>
      <td>barn</td>
      <td>barometer</td>
      <td>barrel, cask</td>
      <td>barrow, garden cart, lawn cart, wheelbarrow</td>
      <td>baseball</td>
      <td>basketball</td>
      <td>bassinet</td>
      <td>bassoon</td>
      <td>bathing cap, swimming cap</td>
      <td>bath towel</td>
      <td>bathtub, bathing tub, bath, tub</td>
      <td>beach wagon, station wagon, wagon, estate car, beach waggon, station waggon, waggon</td>
      <td>beacon, lighthouse, beacon light, pharos</td>
      <td>beaker</td>
      <td>bearskin, busby, shako</td>
      <td>beer bottle</td>
      <td>beer glass</td>
      <td>bell cote, bell cot</td>
      <td>bib</td>
      <td>bicycle-built-for-two, tandem bicycle, tandem</td>
      <td>bikini, two-piece</td>
      <td>binder, ring-binder</td>
      <td>binoculars, field glasses, opera glasses</td>
      <td>birdhouse</td>
      <td>boathouse</td>
      <td>bobsled, bobsleigh, bob</td>
      <td>bolo tie, bolo, bola tie, bola</td>
      <td>bonnet, poke bonnet</td>
      <td>bookcase</td>
      <td>bookshop, bookstore, bookstall</td>
      <td>bottlecap</td>
      <td>bow</td>
      <td>bow tie, bow-tie, bowtie</td>
      <td>brass, memorial tablet, plaque</td>
      <td>brassiere, bra, bandeau</td>
      <td>breakwater, groin, groyne, mole, bulwark, seawall, jetty</td>
      <td>breastplate, aegis, egis</td>
      <td>broom</td>
      <td>bucket, pail</td>
      <td>buckle</td>
      <td>bulletproof vest</td>
      <td>bullet train, bullet</td>
      <td>butcher shop, meat market</td>
      <td>cab, hack, taxi, taxicab</td>
      <td>caldron, cauldron</td>
      <td>candle, taper, wax light</td>
      <td>cannon</td>
      <td>canoe</td>
      <td>can opener, tin opener</td>
      <td>cardigan</td>
      <td>car mirror</td>
      <td>carousel, carrousel, merry-go-round, roundabout, whirligig</td>
      <td>carpenter's kit, tool kit</td>
      <td>carton</td>
      <td>car wheel</td>
      <td>cash machine, cash dispenser, automated teller machine, automatic teller machine, automated teller, automatic teller, ATM</td>
      <td>cassette</td>
      <td>cassette player</td>
      <td>castle</td>
      <td>catamaran</td>
      <td>CD player</td>
      <td>cello, violoncello</td>
      <td>cellular telephone, cellular phone, cellphone, cell, mobile phone</td>
      <td>chain</td>
      <td>chainlink fence</td>
      <td>chain mail, ring mail, mail, chain armor, chain armour, ring armor, ring armour</td>
      <td>chain saw, chainsaw</td>
      <td>chest</td>
      <td>chiffonier, commode</td>
      <td>chime, bell, gong</td>
      <td>china cabinet, china closet</td>
      <td>Christmas stocking</td>
      <td>church, church building</td>
      <td>cinema, movie theater, movie theatre, movie house, picture palace</td>
      <td>cleaver, meat cleaver, chopper</td>
      <td>cliff dwelling</td>
      <td>cloak</td>
      <td>clog, geta, patten, sabot</td>
      <td>cocktail shaker</td>
      <td>coffee mug</td>
      <td>coffeepot</td>
      <td>coil, spiral, volute, whorl, helix</td>
      <td>combination lock</td>
      <td>computer keyboard, keypad</td>
      <td>confectionery, confectionary, candy store</td>
      <td>container ship, containership, container vessel</td>
      <td>convertible</td>
      <td>corkscrew, bottle screw</td>
      <td>cornet, horn, trumpet, trump</td>
      <td>cowboy boot</td>
      <td>cowboy hat, ten-gallon hat</td>
      <td>cradle</td>
      <td>crane</td>
      <td>crash helmet</td>
      <td>crate</td>
      <td>crib, cot</td>
      <td>Crock Pot</td>
      <td>croquet ball</td>
      <td>crutch</td>
      <td>cuirass</td>
      <td>dam, dike, dyke</td>
      <td>desk</td>
      <td>desktop computer</td>
      <td>dial telephone, dial phone</td>
      <td>diaper, nappy, napkin</td>
      <td>digital clock</td>
      <td>digital watch</td>
      <td>dining table, board</td>
      <td>dishrag, dishcloth</td>
      <td>dishwasher, dish washer, dishwashing machine</td>
      <td>disk brake, disc brake</td>
      <td>dock, dockage, docking facility</td>
      <td>dogsled, dog sled, dog sleigh</td>
      <td>dome</td>
      <td>doormat, welcome mat</td>
      <td>drilling platform, offshore rig</td>
      <td>drum, membranophone, tympan</td>
      <td>drumstick</td>
      <td>dumbbell</td>
      <td>Dutch oven</td>
      <td>electric fan, blower</td>
      <td>electric guitar</td>
      <td>electric locomotive</td>
      <td>entertainment center</td>
      <td>envelope</td>
      <td>espresso maker</td>
      <td>face powder</td>
      <td>feather boa, boa</td>
      <td>file, file cabinet, filing cabinet</td>
      <td>fireboat</td>
      <td>fire engine, fire truck</td>
      <td>fire screen, fireguard</td>
      <td>flagpole, flagstaff</td>
      <td>flute, transverse flute</td>
      <td>folding chair</td>
      <td>football helmet</td>
      <td>forklift</td>
      <td>fountain</td>
      <td>fountain pen</td>
      <td>four-poster</td>
      <td>freight car</td>
      <td>French horn, horn</td>
      <td>frying pan, frypan, skillet</td>
      <td>fur coat</td>
      <td>garbage truck, dustcart</td>
      <td>gasmask, respirator, gas helmet</td>
      <td>gas pump, gasoline pump, petrol pump, island dispenser</td>
      <td>goblet</td>
      <td>go-kart</td>
      <td>golf ball</td>
      <td>golfcart, golf cart</td>
      <td>gondola</td>
      <td>gong, tam-tam</td>
      <td>gown</td>
      <td>grand piano, grand</td>
      <td>greenhouse, nursery, glasshouse</td>
      <td>grille, radiator grille</td>
      <td>grocery store, grocery, food market, market</td>
      <td>guillotine</td>
      <td>hair slide</td>
      <td>hair spray</td>
      <td>half track</td>
      <td>hammer</td>
      <td>hamper</td>
      <td>hand blower, blow dryer, blow drier, hair dryer, hair drier</td>
      <td>hand-held computer, hand-held microcomputer</td>
      <td>handkerchief, hankie, hanky, hankey</td>
      <td>hard disc, hard disk, fixed disk</td>
      <td>harmonica, mouth organ, harp, mouth harp</td>
      <td>harp</td>
      <td>harvester, reaper</td>
      <td>hatchet</td>
      <td>holster</td>
      <td>home theater, home theatre</td>
      <td>honeycomb</td>
      <td>hook, claw</td>
      <td>hoopskirt, crinoline</td>
      <td>horizontal bar, high bar</td>
      <td>horse cart, horse-cart</td>
      <td>hourglass</td>
      <td>iPod</td>
      <td>iron, smoothing iron</td>
      <td>jack-o'-lantern</td>
      <td>jean, blue jean, denim</td>
      <td>jeep, landrover</td>
      <td>jersey, T-shirt, tee shirt</td>
      <td>jigsaw puzzle</td>
      <td>jinrikisha, ricksha, rickshaw</td>
      <td>joystick</td>
      <td>kimono</td>
      <td>knee pad</td>
      <td>knot</td>
      <td>lab coat, laboratory coat</td>
      <td>ladle</td>
      <td>lampshade, lamp shade</td>
      <td>laptop, laptop computer</td>
      <td>lawn mower, mower</td>
      <td>lens cap, lens cover</td>
      <td>letter opener, paper knife, paperknife</td>
      <td>library</td>
      <td>lifeboat</td>
      <td>lighter, light, igniter, ignitor</td>
      <td>limousine, limo</td>
      <td>liner, ocean liner</td>
      <td>lipstick, lip rouge</td>
      <td>Loafer</td>
      <td>lotion</td>
      <td>loudspeaker, speaker, speaker unit, loudspeaker system, speaker system</td>
      <td>loupe, jeweler's loupe</td>
      <td>lumbermill, sawmill</td>
      <td>magnetic compass</td>
      <td>mailbag, postbag</td>
      <td>mailbox, letter box</td>
      <td>maillot</td>
      <td>maillot, tank suit</td>
      <td>manhole cover</td>
      <td>maraca</td>
      <td>marimba, xylophone</td>
      <td>mask</td>
      <td>matchstick</td>
      <td>maypole</td>
      <td>maze, labyrinth</td>
      <td>measuring cup</td>
      <td>medicine chest, medicine cabinet</td>
      <td>megalith, megalithic structure</td>
      <td>microphone, mike</td>
      <td>microwave, microwave oven</td>
      <td>military uniform</td>
      <td>milk can</td>
      <td>minibus</td>
      <td>miniskirt, mini</td>
      <td>minivan</td>
      <td>missile</td>
      <td>mitten</td>
      <td>mixing bowl</td>
      <td>mobile home, manufactured home</td>
      <td>Model T</td>
      <td>modem</td>
      <td>monastery</td>
      <td>monitor</td>
      <td>moped</td>
      <td>mortar</td>
      <td>mortarboard</td>
      <td>mosque</td>
      <td>mosquito net</td>
      <td>motor scooter, scooter</td>
      <td>mountain bike, all-terrain bike, off-roader</td>
      <td>mountain tent</td>
      <td>mouse, computer mouse</td>
      <td>mousetrap</td>
      <td>moving van</td>
      <td>muzzle</td>
      <td>nail</td>
      <td>neck brace</td>
      <td>necklace</td>
      <td>nipple</td>
      <td>notebook, notebook computer</td>
      <td>obelisk</td>
      <td>oboe, hautboy, hautbois</td>
      <td>ocarina, sweet potato</td>
      <td>odometer, hodometer, mileometer, milometer</td>
      <td>oil filter</td>
      <td>organ, pipe organ</td>
      <td>oscilloscope, scope, cathode-ray oscilloscope, CRO</td>
      <td>overskirt</td>
      <td>oxcart</td>
      <td>oxygen mask</td>
      <td>packet</td>
      <td>paddle, boat paddle</td>
      <td>paddlewheel, paddle wheel</td>
      <td>padlock</td>
      <td>paintbrush</td>
      <td>pajama, pyjama, pj's, jammies</td>
      <td>palace</td>
      <td>panpipe, pandean pipe, syrinx</td>
      <td>paper towel</td>
      <td>parachute, chute</td>
      <td>parallel bars, bars</td>
      <td>park bench</td>
      <td>parking meter</td>
      <td>passenger car, coach, carriage</td>
      <td>patio, terrace</td>
      <td>pay-phone, pay-station</td>
      <td>pedestal, plinth, footstall</td>
      <td>pencil box, pencil case</td>
      <td>pencil sharpener</td>
      <td>perfume, essence</td>
      <td>Petri dish</td>
      <td>photocopier</td>
      <td>pick, plectrum, plectron</td>
      <td>pickelhaube</td>
      <td>picket fence, paling</td>
      <td>pickup, pickup truck</td>
      <td>pier</td>
      <td>piggy bank, penny bank</td>
      <td>pill bottle</td>
      <td>pillow</td>
      <td>ping-pong ball</td>
      <td>pinwheel</td>
      <td>pirate, pirate ship</td>
      <td>pitcher, ewer</td>
      <td>plane, carpenter's plane, woodworking plane</td>
      <td>planetarium</td>
      <td>plastic bag</td>
      <td>plate rack</td>
      <td>plow, plough</td>
      <td>plunger, plumber's helper</td>
      <td>Polaroid camera, Polaroid Land camera</td>
      <td>pole</td>
      <td>police van, police wagon, paddy wagon, patrol wagon, wagon, black Maria</td>
      <td>poncho</td>
      <td>pool table, billiard table, snooker table</td>
      <td>pop bottle, soda bottle</td>
      <td>pot, flowerpot</td>
      <td>potter's wheel</td>
      <td>power drill</td>
      <td>prayer rug, prayer mat</td>
      <td>printer</td>
      <td>prison, prison house</td>
      <td>projectile, missile</td>
      <td>projector</td>
      <td>puck, hockey puck</td>
      <td>punching bag, punch bag, punching ball, punchball</td>
      <td>purse</td>
      <td>quill, quill pen</td>
      <td>quilt, comforter, comfort, puff</td>
      <td>racer, race car, racing car</td>
      <td>racket, racquet</td>
      <td>radiator</td>
      <td>radio, wireless</td>
      <td>radio telescope, radio reflector</td>
      <td>rain barrel</td>
      <td>recreational vehicle, RV, R.V.</td>
      <td>reel</td>
      <td>reflex camera</td>
      <td>refrigerator, icebox</td>
      <td>remote control, remote</td>
      <td>restaurant, eating house, eating place, eatery</td>
      <td>revolver, six-gun, six-shooter</td>
      <td>rifle</td>
      <td>rocking chair, rocker</td>
      <td>rotisserie</td>
      <td>rubber eraser, rubber, pencil eraser</td>
      <td>rugby ball</td>
      <td>rule, ruler</td>
      <td>running shoe</td>
      <td>safe</td>
      <td>safety pin</td>
      <td>saltshaker, salt shaker</td>
      <td>sandal</td>
      <td>sarong</td>
      <td>sax, saxophone</td>
      <td>scabbard</td>
      <td>scale, weighing machine</td>
      <td>school bus</td>
      <td>schooner</td>
      <td>scoreboard</td>
      <td>screen, CRT screen</td>
      <td>screw</td>
      <td>screwdriver</td>
      <td>seat belt, seatbelt</td>
      <td>sewing machine</td>
      <td>shield, buckler</td>
      <td>shoe shop, shoe-shop, shoe store</td>
      <td>shoji</td>
      <td>shopping basket</td>
      <td>shopping cart</td>
      <td>shovel</td>
      <td>shower cap</td>
      <td>shower curtain</td>
      <td>ski</td>
      <td>ski mask</td>
      <td>sleeping bag</td>
      <td>slide rule, slipstick</td>
      <td>sliding door</td>
      <td>slot, one-armed bandit</td>
      <td>snorkel</td>
      <td>snowmobile</td>
      <td>snowplow, snowplough</td>
      <td>soap dispenser</td>
      <td>soccer ball</td>
      <td>sock</td>
      <td>solar dish, solar collector, solar furnace</td>
      <td>sombrero</td>
      <td>soup bowl</td>
      <td>space bar</td>
      <td>space heater</td>
      <td>space shuttle</td>
      <td>spatula</td>
      <td>speedboat</td>
      <td>spider web, spider's web</td>
      <td>spindle</td>
      <td>sports car, sport car</td>
      <td>spotlight, spot</td>
      <td>stage</td>
      <td>steam locomotive</td>
      <td>steel arch bridge</td>
      <td>steel drum</td>
      <td>stethoscope</td>
      <td>stole</td>
      <td>stone wall</td>
      <td>stopwatch, stop watch</td>
      <td>stove</td>
      <td>strainer</td>
      <td>streetcar, tram, tramcar, trolley, trolley car</td>
      <td>stretcher</td>
      <td>studio couch, day bed</td>
      <td>stupa, tope</td>
      <td>submarine, pigboat, sub, U-boat</td>
      <td>suit, suit of clothes</td>
      <td>sundial</td>
      <td>sunglass</td>
      <td>sunglasses, dark glasses, shades</td>
      <td>sunscreen, sunblock, sun blocker</td>
      <td>suspension bridge</td>
      <td>swab, swob, mop</td>
      <td>sweatshirt</td>
      <td>swimming trunks, bathing trunks</td>
      <td>swing</td>
      <td>switch, electric switch, electrical switch</td>
      <td>syringe</td>
      <td>table lamp</td>
      <td>tank, army tank, armored combat vehicle, armoured combat vehicle</td>
      <td>tape player</td>
      <td>teapot</td>
      <td>teddy, teddy bear</td>
      <td>television, television system</td>
      <td>tennis ball</td>
      <td>thatch, thatched roof</td>
      <td>theater curtain, theatre curtain</td>
      <td>thimble</td>
      <td>thresher, thrasher, threshing machine</td>
      <td>throne</td>
      <td>tile roof</td>
      <td>toaster</td>
      <td>tobacco shop, tobacconist shop, tobacconist</td>
      <td>toilet seat</td>
      <td>torch</td>
      <td>totem pole</td>
      <td>tow truck, tow car, wrecker</td>
      <td>toyshop</td>
      <td>tractor</td>
      <td>trailer truck, tractor trailer, trucking rig, rig, articulated lorry, semi</td>
      <td>tray</td>
      <td>trench coat</td>
      <td>tricycle, trike, velocipede</td>
      <td>trimaran</td>
      <td>tripod</td>
      <td>triumphal arch</td>
      <td>trolleybus, trolley coach, trackless trolley</td>
      <td>trombone</td>
      <td>tub, vat</td>
      <td>turnstile</td>
      <td>typewriter keyboard</td>
      <td>umbrella</td>
      <td>unicycle, monocycle</td>
      <td>upright, upright piano</td>
      <td>vacuum, vacuum cleaner</td>
      <td>vase</td>
      <td>vault</td>
      <td>velvet</td>
      <td>vending machine</td>
      <td>vestment</td>
      <td>viaduct</td>
      <td>violin, fiddle</td>
      <td>volleyball</td>
      <td>waffle iron</td>
      <td>wall clock</td>
      <td>wallet, billfold, notecase, pocketbook</td>
      <td>wardrobe, closet, press</td>
      <td>warplane, military plane</td>
      <td>washbasin, handbasin, washbowl, lavabo, wash-hand basin</td>
      <td>washer, automatic washer, washing machine</td>
      <td>water bottle</td>
      <td>water jug</td>
      <td>water tower</td>
      <td>whiskey jug</td>
      <td>whistle</td>
      <td>wig</td>
      <td>window screen</td>
      <td>window shade</td>
      <td>Windsor tie</td>
      <td>wine bottle</td>
      <td>wing</td>
      <td>wok</td>
      <td>wooden spoon</td>
      <td>wool, woolen, woollen</td>
      <td>worm fence, snake fence, snake-rail fence, Virginia fence</td>
      <td>wreck</td>
      <td>yawl</td>
      <td>yurt</td>
      <td>web site, website, internet site, site</td>
      <td>comic book</td>
      <td>crossword puzzle, crossword</td>
      <td>street sign</td>
      <td>traffic light, traffic signal, stoplight</td>
      <td>book jacket, dust cover, dust jacket, dust wrapper</td>
      <td>menu</td>
      <td>plate</td>
      <td>guacamole</td>
      <td>consomme</td>
      <td>hot pot, hotpot</td>
      <td>trifle</td>
      <td>ice cream, icecream</td>
      <td>ice lolly, lolly, lollipop, popsicle</td>
      <td>French loaf</td>
      <td>bagel, beigel</td>
      <td>pretzel</td>
      <td>cheeseburger</td>
      <td>hotdog, hot dog, red hot</td>
      <td>mashed potato</td>
      <td>head cabbage</td>
      <td>broccoli</td>
      <td>cauliflower</td>
      <td>zucchini, courgette</td>
      <td>spaghetti squash</td>
      <td>acorn squash</td>
      <td>butternut squash</td>
      <td>cucumber, cuke</td>
      <td>artichoke, globe artichoke</td>
      <td>bell pepper</td>
      <td>cardoon</td>
      <td>mushroom</td>
      <td>Granny Smith</td>
      <td>strawberry</td>
      <td>orange</td>
      <td>lemon</td>
      <td>fig</td>
      <td>pineapple, ananas</td>
      <td>banana</td>
      <td>jackfruit, jak, jack</td>
      <td>custard apple</td>
      <td>pomegranate</td>
      <td>hay</td>
      <td>carbonara</td>
      <td>chocolate sauce, chocolate syrup</td>
      <td>dough</td>
      <td>meat loaf, meatloaf</td>
      <td>pizza, pizza pie</td>
      <td>potpie</td>
      <td>burrito</td>
      <td>red wine</td>
      <td>espresso</td>
      <td>cup</td>
      <td>eggnog</td>
      <td>alp</td>
      <td>bubble</td>
      <td>cliff, drop, drop-off</td>
      <td>coral reef</td>
      <td>geyser</td>
      <td>lakeside, lakeshore</td>
      <td>promontory, headland, head, foreland</td>
      <td>sandbar, sand bar</td>
      <td>seashore, coast, seacoast, sea-coast</td>
      <td>valley, vale</td>
      <td>volcano</td>
      <td>ballplayer, baseball player</td>
      <td>groom, bridegroom</td>
      <td>scuba diver</td>
      <td>rapeseed</td>
      <td>daisy</td>
      <td>yellow lady's slipper, yellow lady-slipper, Cypripedium calceolus, Cypripedium parviflorum</td>
      <td>corn</td>
      <td>acorn</td>
      <td>hip, rose hip, rosehip</td>
      <td>buckeye, horse chestnut, conker</td>
      <td>coral fungus</td>
      <td>agaric</td>
      <td>gyromitra</td>
      <td>stinkhorn, carrion fungus</td>
      <td>earthstar</td>
      <td>hen-of-the-woods, hen of the woods, Polyporus frondosus, Grifola frondosa</td>
      <td>bolete</td>
      <td>ear, spike, capitulum</td>
      <td>toilet tissue, toilet paper, bathroom tissue</td>
    </tr>
  </tbody>
</table>
</div>


------

**Perform image classification**


```python
preds = image_classifier(image)
preds_df = pd.DataFrame(preds)
preds_df
```
<div style="overflow-x:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>score</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.989680</td>
      <td>golden retriever</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.002968</td>
      <td>Labrador retriever</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.000502</td>
      <td>kuvasz</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.000402</td>
      <td>Irish setter, red setter</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.000345</td>
      <td>tennis ball</td>
    </tr>
  </tbody>
</table>
</div>
**Note:** 

* The model correctly classifies the dog as a Golden Retriever.
* Video models are a natural extension of image models and add a temporal dimension on top of the spatial dimension.
* Video tasks are more challenging as the volume of data gets much larger, and we need to deal with an extra dimension.
* Models such as TimeSformer introduce a spatial and temporal attention mechanism.
    * [Is Space-Time Attention All You Need for Video Understanding?](https://arxiv.org/abs/2102.05095)
    * Such models can help build tools for many tasks such as video classification or annotation.

------

### Tables

* Lots of data is in structured databases instead of raw text.
* Table Parser (TAPAS) applies the Transformer architecture to tables by combining the tabular information with the query.
* [TAPAS: Weakly Supervised Table Parsing via Pre-training](https://arxiv.org/abs/2004.02349)

<img alt="tapas-architecture" width="800" caption="Architecture of TAPAS (courtesy of Jonathan Herzig)" src="../images/notes-transformers-book/chapter-11/chapter11_tapas-architecture.png" id="tapas-architecture"/>

**Create some sample table data**


```python
book_data = [
    {"chapter": 0, "name": "Introduction", "start_page": 1, "end_page": 11},
    {"chapter": 1, "name": "Text classification", "start_page": 12, 
     "end_page": 48},
    {"chapter": 2, "name": "Named Entity Recognition", "start_page": 49,
     "end_page": 73},
    {"chapter": 3, "name": "Question Answering", "start_page": 74, 
     "end_page": 120},
    {"chapter": 4, "name": "Summarization", "start_page": 121, 
     "end_page": 140},
    {"chapter": 5, "name": "Conclusion", "start_page": 141, 
     "end_page": 144}
]

table = pd.DataFrame(book_data)
table['number_of_pages'] = table['end_page']-table['start_page']
```

**Note:** We need to make all columns of type `str` to play nicely with TAPAS.

------


```python
table = table.astype(str)
table
```
<div style="overflow-x:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>chapter</th>
      <th>name</th>
      <th>start_page</th>
      <th>end_page</th>
      <th>number_of_pages</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>Introduction</td>
      <td>1</td>
      <td>11</td>
      <td>10</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>Text classification</td>
      <td>12</td>
      <td>48</td>
      <td>36</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>Named Entity Recognition</td>
      <td>49</td>
      <td>73</td>
      <td>24</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>Question Answering</td>
      <td>74</td>
      <td>120</td>
      <td>46</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>Summarization</td>
      <td>121</td>
      <td>140</td>
      <td>19</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>Conclusion</td>
      <td>141</td>
      <td>144</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>
------


**Create a table question answering pipeline**


```python
table_qa = pipeline("table-question-answering")
table_qa.model.config
```
```text
    TapasConfig {
      "_name_or_path": "google/tapas-base-finetuned-wtq",
      "aggregation_labels": {
        "0": "NONE",
        "1": "SUM",
        "2": "AVERAGE",
        "3": "COUNT"
      },
      "aggregation_loss_weight": 1.0,
      "aggregation_temperature": 1.0,
      "allow_empty_column_selection": false,
      "answer_loss_cutoff": 0.664694,
      "answer_loss_importance": 1.0,
      "architectures": [
        "TapasForQuestionAnswering"
      ],
      "attention_probs_dropout_prob": 0.1,
      "average_approximation_function": "ratio",
      "average_logits_per_cell": false,
      "cell_selection_preference": 0.207951,
      "disable_per_token_loss": false,
      "gradient_checkpointing": false,
      "hidden_act": "gelu",
      "hidden_dropout_prob": 0.1,
      "hidden_size": 768,
      "huber_loss_delta": 0.121194,
      "init_cell_selection_weights_to_zero": true,
      "initializer_range": 0.02,
      "intermediate_size": 3072,
      "layer_norm_eps": 1e-12,
      "max_num_columns": 32,
      "max_num_rows": 64,
      "max_position_embeddings": 1024,
      "model_type": "tapas",
      "no_aggregation_label_index": 0,
      "num_aggregation_labels": 4,
      "num_attention_heads": 12,
      "num_hidden_layers": 12,
      "pad_token_id": 0,
      "positive_label_weight": 10.0,
      "reset_position_index_per_cell": true,
      "select_one_column": true,
      "softmax_temperature": 1.0,
      "temperature": 0.0352513,
      "transformers_version": "4.18.0",
      "type_vocab_size": [
        3,
        256,
        256,
        2,
        256,
        256,
        10
      ],
      "type_vocab_sizes": [
        3,
        256,
        256,
        2,
        256,
        256,
        10
      ],
      "use_answer_as_supervision": true,
      "use_gumbel_for_aggregation": false,
      "use_gumbel_for_cells": false,
      "use_normalized_answer_loss": false,
      "vocab_size": 30522
    }
```

------

**Get the link to the Hugging Face model card**


```python
print(f"https://huggingface.co/{table_qa.model.config._name_or_path}")
```
```text
    https://huggingface.co/google/tapas-base-finetuned-wtq
```

------


```python
pd.DataFrame(table_qa.tokenizer.vocab.keys()).head(1500).T
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
      <th>26</th>
      <th>27</th>
      <th>28</th>
      <th>29</th>
      <th>30</th>
      <th>31</th>
      <th>32</th>
      <th>33</th>
      <th>34</th>
      <th>35</th>
      <th>36</th>
      <th>37</th>
      <th>38</th>
      <th>39</th>
      <th>40</th>
      <th>41</th>
      <th>42</th>
      <th>43</th>
      <th>44</th>
      <th>45</th>
      <th>46</th>
      <th>47</th>
      <th>48</th>
      <th>49</th>
      <th>50</th>
      <th>51</th>
      <th>52</th>
      <th>53</th>
      <th>54</th>
      <th>55</th>
      <th>56</th>
      <th>57</th>
      <th>58</th>
      <th>59</th>
      <th>60</th>
      <th>61</th>
      <th>62</th>
      <th>63</th>
      <th>64</th>
      <th>65</th>
      <th>66</th>
      <th>67</th>
      <th>68</th>
      <th>69</th>
      <th>70</th>
      <th>71</th>
      <th>72</th>
      <th>73</th>
      <th>74</th>
      <th>75</th>
      <th>76</th>
      <th>77</th>
      <th>78</th>
      <th>79</th>
      <th>80</th>
      <th>81</th>
      <th>82</th>
      <th>83</th>
      <th>84</th>
      <th>85</th>
      <th>86</th>
      <th>87</th>
      <th>88</th>
      <th>89</th>
      <th>90</th>
      <th>91</th>
      <th>92</th>
      <th>93</th>
      <th>94</th>
      <th>95</th>
      <th>96</th>
      <th>97</th>
      <th>98</th>
      <th>99</th>
      <th>100</th>
      <th>101</th>
      <th>102</th>
      <th>103</th>
      <th>104</th>
      <th>105</th>
      <th>106</th>
      <th>107</th>
      <th>108</th>
      <th>109</th>
      <th>110</th>
      <th>111</th>
      <th>112</th>
      <th>113</th>
      <th>114</th>
      <th>115</th>
      <th>116</th>
      <th>117</th>
      <th>118</th>
      <th>119</th>
      <th>120</th>
      <th>121</th>
      <th>122</th>
      <th>123</th>
      <th>124</th>
      <th>125</th>
      <th>126</th>
      <th>127</th>
      <th>128</th>
      <th>129</th>
      <th>130</th>
      <th>131</th>
      <th>132</th>
      <th>133</th>
      <th>134</th>
      <th>135</th>
      <th>136</th>
      <th>137</th>
      <th>138</th>
      <th>139</th>
      <th>140</th>
      <th>141</th>
      <th>142</th>
      <th>143</th>
      <th>144</th>
      <th>145</th>
      <th>146</th>
      <th>147</th>
      <th>148</th>
      <th>149</th>
      <th>150</th>
      <th>151</th>
      <th>152</th>
      <th>153</th>
      <th>154</th>
      <th>155</th>
      <th>156</th>
      <th>157</th>
      <th>158</th>
      <th>159</th>
      <th>160</th>
      <th>161</th>
      <th>162</th>
      <th>163</th>
      <th>164</th>
      <th>165</th>
      <th>166</th>
      <th>167</th>
      <th>168</th>
      <th>169</th>
      <th>170</th>
      <th>171</th>
      <th>172</th>
      <th>173</th>
      <th>174</th>
      <th>175</th>
      <th>176</th>
      <th>177</th>
      <th>178</th>
      <th>179</th>
      <th>180</th>
      <th>181</th>
      <th>182</th>
      <th>183</th>
      <th>184</th>
      <th>185</th>
      <th>186</th>
      <th>187</th>
      <th>188</th>
      <th>189</th>
      <th>190</th>
      <th>191</th>
      <th>192</th>
      <th>193</th>
      <th>194</th>
      <th>195</th>
      <th>196</th>
      <th>197</th>
      <th>198</th>
      <th>199</th>
      <th>200</th>
      <th>201</th>
      <th>202</th>
      <th>203</th>
      <th>204</th>
      <th>205</th>
      <th>206</th>
      <th>207</th>
      <th>208</th>
      <th>209</th>
      <th>210</th>
      <th>211</th>
      <th>212</th>
      <th>213</th>
      <th>214</th>
      <th>215</th>
      <th>216</th>
      <th>217</th>
      <th>218</th>
      <th>219</th>
      <th>220</th>
      <th>221</th>
      <th>222</th>
      <th>223</th>
      <th>224</th>
      <th>225</th>
      <th>226</th>
      <th>227</th>
      <th>228</th>
      <th>229</th>
      <th>230</th>
      <th>231</th>
      <th>232</th>
      <th>233</th>
      <th>234</th>
      <th>235</th>
      <th>236</th>
      <th>237</th>
      <th>238</th>
      <th>239</th>
      <th>240</th>
      <th>241</th>
      <th>242</th>
      <th>243</th>
      <th>244</th>
      <th>245</th>
      <th>246</th>
      <th>247</th>
      <th>248</th>
      <th>249</th>
      <th>250</th>
      <th>251</th>
      <th>252</th>
      <th>253</th>
      <th>254</th>
      <th>255</th>
      <th>256</th>
      <th>257</th>
      <th>258</th>
      <th>259</th>
      <th>260</th>
      <th>261</th>
      <th>262</th>
      <th>263</th>
      <th>264</th>
      <th>265</th>
      <th>266</th>
      <th>267</th>
      <th>268</th>
      <th>269</th>
      <th>270</th>
      <th>271</th>
      <th>272</th>
      <th>273</th>
      <th>274</th>
      <th>275</th>
      <th>276</th>
      <th>277</th>
      <th>278</th>
      <th>279</th>
      <th>280</th>
      <th>281</th>
      <th>282</th>
      <th>283</th>
      <th>284</th>
      <th>285</th>
      <th>286</th>
      <th>287</th>
      <th>288</th>
      <th>289</th>
      <th>290</th>
      <th>291</th>
      <th>292</th>
      <th>293</th>
      <th>294</th>
      <th>295</th>
      <th>296</th>
      <th>297</th>
      <th>298</th>
      <th>299</th>
      <th>300</th>
      <th>301</th>
      <th>302</th>
      <th>303</th>
      <th>304</th>
      <th>305</th>
      <th>306</th>
      <th>307</th>
      <th>308</th>
      <th>309</th>
      <th>310</th>
      <th>311</th>
      <th>312</th>
      <th>313</th>
      <th>314</th>
      <th>315</th>
      <th>316</th>
      <th>317</th>
      <th>318</th>
      <th>319</th>
      <th>320</th>
      <th>321</th>
      <th>322</th>
      <th>323</th>
      <th>324</th>
      <th>325</th>
      <th>326</th>
      <th>327</th>
      <th>328</th>
      <th>329</th>
      <th>330</th>
      <th>331</th>
      <th>332</th>
      <th>333</th>
      <th>334</th>
      <th>335</th>
      <th>336</th>
      <th>337</th>
      <th>338</th>
      <th>339</th>
      <th>340</th>
      <th>341</th>
      <th>342</th>
      <th>343</th>
      <th>344</th>
      <th>345</th>
      <th>346</th>
      <th>347</th>
      <th>348</th>
      <th>349</th>
      <th>350</th>
      <th>351</th>
      <th>352</th>
      <th>353</th>
      <th>354</th>
      <th>355</th>
      <th>356</th>
      <th>357</th>
      <th>358</th>
      <th>359</th>
      <th>360</th>
      <th>361</th>
      <th>362</th>
      <th>363</th>
      <th>364</th>
      <th>365</th>
      <th>366</th>
      <th>367</th>
      <th>368</th>
      <th>369</th>
      <th>370</th>
      <th>371</th>
      <th>372</th>
      <th>373</th>
      <th>374</th>
      <th>375</th>
      <th>376</th>
      <th>377</th>
      <th>378</th>
      <th>379</th>
      <th>380</th>
      <th>381</th>
      <th>382</th>
      <th>383</th>
      <th>384</th>
      <th>385</th>
      <th>386</th>
      <th>387</th>
      <th>388</th>
      <th>389</th>
      <th>390</th>
      <th>391</th>
      <th>392</th>
      <th>393</th>
      <th>394</th>
      <th>395</th>
      <th>396</th>
      <th>397</th>
      <th>398</th>
      <th>399</th>
      <th>400</th>
      <th>401</th>
      <th>402</th>
      <th>403</th>
      <th>404</th>
      <th>405</th>
      <th>406</th>
      <th>407</th>
      <th>408</th>
      <th>409</th>
      <th>410</th>
      <th>411</th>
      <th>412</th>
      <th>413</th>
      <th>414</th>
      <th>415</th>
      <th>416</th>
      <th>417</th>
      <th>418</th>
      <th>419</th>
      <th>420</th>
      <th>421</th>
      <th>422</th>
      <th>423</th>
      <th>424</th>
      <th>425</th>
      <th>426</th>
      <th>427</th>
      <th>428</th>
      <th>429</th>
      <th>430</th>
      <th>431</th>
      <th>432</th>
      <th>433</th>
      <th>434</th>
      <th>435</th>
      <th>436</th>
      <th>437</th>
      <th>438</th>
      <th>439</th>
      <th>440</th>
      <th>441</th>
      <th>442</th>
      <th>443</th>
      <th>444</th>
      <th>445</th>
      <th>446</th>
      <th>447</th>
      <th>448</th>
      <th>449</th>
      <th>450</th>
      <th>451</th>
      <th>452</th>
      <th>453</th>
      <th>454</th>
      <th>455</th>
      <th>456</th>
      <th>457</th>
      <th>458</th>
      <th>459</th>
      <th>460</th>
      <th>461</th>
      <th>462</th>
      <th>463</th>
      <th>464</th>
      <th>465</th>
      <th>466</th>
      <th>467</th>
      <th>468</th>
      <th>469</th>
      <th>470</th>
      <th>471</th>
      <th>472</th>
      <th>473</th>
      <th>474</th>
      <th>475</th>
      <th>476</th>
      <th>477</th>
      <th>478</th>
      <th>479</th>
      <th>480</th>
      <th>481</th>
      <th>482</th>
      <th>483</th>
      <th>484</th>
      <th>485</th>
      <th>486</th>
      <th>487</th>
      <th>488</th>
      <th>489</th>
      <th>490</th>
      <th>491</th>
      <th>492</th>
      <th>493</th>
      <th>494</th>
      <th>495</th>
      <th>496</th>
      <th>497</th>
      <th>498</th>
      <th>499</th>
      <th>500</th>
      <th>501</th>
      <th>502</th>
      <th>503</th>
      <th>504</th>
      <th>505</th>
      <th>506</th>
      <th>507</th>
      <th>508</th>
      <th>509</th>
      <th>510</th>
      <th>511</th>
      <th>512</th>
      <th>513</th>
      <th>514</th>
      <th>515</th>
      <th>516</th>
      <th>517</th>
      <th>518</th>
      <th>519</th>
      <th>520</th>
      <th>521</th>
      <th>522</th>
      <th>523</th>
      <th>524</th>
      <th>525</th>
      <th>526</th>
      <th>527</th>
      <th>528</th>
      <th>529</th>
      <th>530</th>
      <th>531</th>
      <th>532</th>
      <th>533</th>
      <th>534</th>
      <th>535</th>
      <th>536</th>
      <th>537</th>
      <th>538</th>
      <th>539</th>
      <th>540</th>
      <th>541</th>
      <th>542</th>
      <th>543</th>
      <th>544</th>
      <th>545</th>
      <th>546</th>
      <th>547</th>
      <th>548</th>
      <th>549</th>
      <th>550</th>
      <th>551</th>
      <th>552</th>
      <th>553</th>
      <th>554</th>
      <th>555</th>
      <th>556</th>
      <th>557</th>
      <th>558</th>
      <th>559</th>
      <th>560</th>
      <th>561</th>
      <th>562</th>
      <th>563</th>
      <th>564</th>
      <th>565</th>
      <th>566</th>
      <th>567</th>
      <th>568</th>
      <th>569</th>
      <th>570</th>
      <th>571</th>
      <th>572</th>
      <th>573</th>
      <th>574</th>
      <th>575</th>
      <th>576</th>
      <th>577</th>
      <th>578</th>
      <th>579</th>
      <th>580</th>
      <th>581</th>
      <th>582</th>
      <th>583</th>
      <th>584</th>
      <th>585</th>
      <th>586</th>
      <th>587</th>
      <th>588</th>
      <th>589</th>
      <th>590</th>
      <th>591</th>
      <th>592</th>
      <th>593</th>
      <th>594</th>
      <th>595</th>
      <th>596</th>
      <th>597</th>
      <th>598</th>
      <th>599</th>
      <th>600</th>
      <th>601</th>
      <th>602</th>
      <th>603</th>
      <th>604</th>
      <th>605</th>
      <th>606</th>
      <th>607</th>
      <th>608</th>
      <th>609</th>
      <th>610</th>
      <th>611</th>
      <th>612</th>
      <th>613</th>
      <th>614</th>
      <th>615</th>
      <th>616</th>
      <th>617</th>
      <th>618</th>
      <th>619</th>
      <th>620</th>
      <th>621</th>
      <th>622</th>
      <th>623</th>
      <th>624</th>
      <th>625</th>
      <th>626</th>
      <th>627</th>
      <th>628</th>
      <th>629</th>
      <th>630</th>
      <th>631</th>
      <th>632</th>
      <th>633</th>
      <th>634</th>
      <th>635</th>
      <th>636</th>
      <th>637</th>
      <th>638</th>
      <th>639</th>
      <th>640</th>
      <th>641</th>
      <th>642</th>
      <th>643</th>
      <th>644</th>
      <th>645</th>
      <th>646</th>
      <th>647</th>
      <th>648</th>
      <th>649</th>
      <th>650</th>
      <th>651</th>
      <th>652</th>
      <th>653</th>
      <th>654</th>
      <th>655</th>
      <th>656</th>
      <th>657</th>
      <th>658</th>
      <th>659</th>
      <th>660</th>
      <th>661</th>
      <th>662</th>
      <th>663</th>
      <th>664</th>
      <th>665</th>
      <th>666</th>
      <th>667</th>
      <th>668</th>
      <th>669</th>
      <th>670</th>
      <th>671</th>
      <th>672</th>
      <th>673</th>
      <th>674</th>
      <th>675</th>
      <th>676</th>
      <th>677</th>
      <th>678</th>
      <th>679</th>
      <th>680</th>
      <th>681</th>
      <th>682</th>
      <th>683</th>
      <th>684</th>
      <th>685</th>
      <th>686</th>
      <th>687</th>
      <th>688</th>
      <th>689</th>
      <th>690</th>
      <th>691</th>
      <th>692</th>
      <th>693</th>
      <th>694</th>
      <th>695</th>
      <th>696</th>
      <th>697</th>
      <th>698</th>
      <th>699</th>
      <th>700</th>
      <th>701</th>
      <th>702</th>
      <th>703</th>
      <th>704</th>
      <th>705</th>
      <th>706</th>
      <th>707</th>
      <th>708</th>
      <th>709</th>
      <th>710</th>
      <th>711</th>
      <th>712</th>
      <th>713</th>
      <th>714</th>
      <th>715</th>
      <th>716</th>
      <th>717</th>
      <th>718</th>
      <th>719</th>
      <th>720</th>
      <th>721</th>
      <th>722</th>
      <th>723</th>
      <th>724</th>
      <th>725</th>
      <th>726</th>
      <th>727</th>
      <th>728</th>
      <th>729</th>
      <th>730</th>
      <th>731</th>
      <th>732</th>
      <th>733</th>
      <th>734</th>
      <th>735</th>
      <th>736</th>
      <th>737</th>
      <th>738</th>
      <th>739</th>
      <th>740</th>
      <th>741</th>
      <th>742</th>
      <th>743</th>
      <th>744</th>
      <th>745</th>
      <th>746</th>
      <th>747</th>
      <th>748</th>
      <th>749</th>
      <th>750</th>
      <th>751</th>
      <th>752</th>
      <th>753</th>
      <th>754</th>
      <th>755</th>
      <th>756</th>
      <th>757</th>
      <th>758</th>
      <th>759</th>
      <th>760</th>
      <th>761</th>
      <th>762</th>
      <th>763</th>
      <th>764</th>
      <th>765</th>
      <th>766</th>
      <th>767</th>
      <th>768</th>
      <th>769</th>
      <th>770</th>
      <th>771</th>
      <th>772</th>
      <th>773</th>
      <th>774</th>
      <th>775</th>
      <th>776</th>
      <th>777</th>
      <th>778</th>
      <th>779</th>
      <th>780</th>
      <th>781</th>
      <th>782</th>
      <th>783</th>
      <th>784</th>
      <th>785</th>
      <th>786</th>
      <th>787</th>
      <th>788</th>
      <th>789</th>
      <th>790</th>
      <th>791</th>
      <th>792</th>
      <th>793</th>
      <th>794</th>
      <th>795</th>
      <th>796</th>
      <th>797</th>
      <th>798</th>
      <th>799</th>
      <th>800</th>
      <th>801</th>
      <th>802</th>
      <th>803</th>
      <th>804</th>
      <th>805</th>
      <th>806</th>
      <th>807</th>
      <th>808</th>
      <th>809</th>
      <th>810</th>
      <th>811</th>
      <th>812</th>
      <th>813</th>
      <th>814</th>
      <th>815</th>
      <th>816</th>
      <th>817</th>
      <th>818</th>
      <th>819</th>
      <th>820</th>
      <th>821</th>
      <th>822</th>
      <th>823</th>
      <th>824</th>
      <th>825</th>
      <th>826</th>
      <th>827</th>
      <th>828</th>
      <th>829</th>
      <th>830</th>
      <th>831</th>
      <th>832</th>
      <th>833</th>
      <th>834</th>
      <th>835</th>
      <th>836</th>
      <th>837</th>
      <th>838</th>
      <th>839</th>
      <th>840</th>
      <th>841</th>
      <th>842</th>
      <th>843</th>
      <th>844</th>
      <th>845</th>
      <th>846</th>
      <th>847</th>
      <th>848</th>
      <th>849</th>
      <th>850</th>
      <th>851</th>
      <th>852</th>
      <th>853</th>
      <th>854</th>
      <th>855</th>
      <th>856</th>
      <th>857</th>
      <th>858</th>
      <th>859</th>
      <th>860</th>
      <th>861</th>
      <th>862</th>
      <th>863</th>
      <th>864</th>
      <th>865</th>
      <th>866</th>
      <th>867</th>
      <th>868</th>
      <th>869</th>
      <th>870</th>
      <th>871</th>
      <th>872</th>
      <th>873</th>
      <th>874</th>
      <th>875</th>
      <th>876</th>
      <th>877</th>
      <th>878</th>
      <th>879</th>
      <th>880</th>
      <th>881</th>
      <th>882</th>
      <th>883</th>
      <th>884</th>
      <th>885</th>
      <th>886</th>
      <th>887</th>
      <th>888</th>
      <th>889</th>
      <th>890</th>
      <th>891</th>
      <th>892</th>
      <th>893</th>
      <th>894</th>
      <th>895</th>
      <th>896</th>
      <th>897</th>
      <th>898</th>
      <th>899</th>
      <th>900</th>
      <th>901</th>
      <th>902</th>
      <th>903</th>
      <th>904</th>
      <th>905</th>
      <th>906</th>
      <th>907</th>
      <th>908</th>
      <th>909</th>
      <th>910</th>
      <th>911</th>
      <th>912</th>
      <th>913</th>
      <th>914</th>
      <th>915</th>
      <th>916</th>
      <th>917</th>
      <th>918</th>
      <th>919</th>
      <th>920</th>
      <th>921</th>
      <th>922</th>
      <th>923</th>
      <th>924</th>
      <th>925</th>
      <th>926</th>
      <th>927</th>
      <th>928</th>
      <th>929</th>
      <th>930</th>
      <th>931</th>
      <th>932</th>
      <th>933</th>
      <th>934</th>
      <th>935</th>
      <th>936</th>
      <th>937</th>
      <th>938</th>
      <th>939</th>
      <th>940</th>
      <th>941</th>
      <th>942</th>
      <th>943</th>
      <th>944</th>
      <th>945</th>
      <th>946</th>
      <th>947</th>
      <th>948</th>
      <th>949</th>
      <th>950</th>
      <th>951</th>
      <th>952</th>
      <th>953</th>
      <th>954</th>
      <th>955</th>
      <th>956</th>
      <th>957</th>
      <th>958</th>
      <th>959</th>
      <th>960</th>
      <th>961</th>
      <th>962</th>
      <th>963</th>
      <th>964</th>
      <th>965</th>
      <th>966</th>
      <th>967</th>
      <th>968</th>
      <th>969</th>
      <th>970</th>
      <th>971</th>
      <th>972</th>
      <th>973</th>
      <th>974</th>
      <th>975</th>
      <th>976</th>
      <th>977</th>
      <th>978</th>
      <th>979</th>
      <th>980</th>
      <th>981</th>
      <th>982</th>
      <th>983</th>
      <th>984</th>
      <th>985</th>
      <th>986</th>
      <th>987</th>
      <th>988</th>
      <th>989</th>
      <th>990</th>
      <th>991</th>
      <th>992</th>
      <th>993</th>
      <th>994</th>
      <th>995</th>
      <th>996</th>
      <th>997</th>
      <th>998</th>
      <th>999</th>
      <th>1000</th>
      <th>1001</th>
      <th>1002</th>
      <th>1003</th>
      <th>1004</th>
      <th>1005</th>
      <th>1006</th>
      <th>1007</th>
      <th>1008</th>
      <th>1009</th>
      <th>1010</th>
      <th>1011</th>
      <th>1012</th>
      <th>1013</th>
      <th>1014</th>
      <th>1015</th>
      <th>1016</th>
      <th>1017</th>
      <th>1018</th>
      <th>1019</th>
      <th>1020</th>
      <th>1021</th>
      <th>1022</th>
      <th>1023</th>
      <th>1024</th>
      <th>1025</th>
      <th>1026</th>
      <th>1027</th>
      <th>1028</th>
      <th>1029</th>
      <th>1030</th>
      <th>1031</th>
      <th>1032</th>
      <th>1033</th>
      <th>1034</th>
      <th>1035</th>
      <th>1036</th>
      <th>1037</th>
      <th>1038</th>
      <th>1039</th>
      <th>1040</th>
      <th>1041</th>
      <th>1042</th>
      <th>1043</th>
      <th>1044</th>
      <th>1045</th>
      <th>1046</th>
      <th>1047</th>
      <th>1048</th>
      <th>1049</th>
      <th>1050</th>
      <th>1051</th>
      <th>1052</th>
      <th>1053</th>
      <th>1054</th>
      <th>1055</th>
      <th>1056</th>
      <th>1057</th>
      <th>1058</th>
      <th>1059</th>
      <th>1060</th>
      <th>1061</th>
      <th>1062</th>
      <th>1063</th>
      <th>1064</th>
      <th>1065</th>
      <th>1066</th>
      <th>1067</th>
      <th>1068</th>
      <th>1069</th>
      <th>1070</th>
      <th>1071</th>
      <th>1072</th>
      <th>1073</th>
      <th>1074</th>
      <th>1075</th>
      <th>1076</th>
      <th>1077</th>
      <th>1078</th>
      <th>1079</th>
      <th>1080</th>
      <th>1081</th>
      <th>1082</th>
      <th>1083</th>
      <th>1084</th>
      <th>1085</th>
      <th>1086</th>
      <th>1087</th>
      <th>1088</th>
      <th>1089</th>
      <th>1090</th>
      <th>1091</th>
      <th>1092</th>
      <th>1093</th>
      <th>1094</th>
      <th>1095</th>
      <th>1096</th>
      <th>1097</th>
      <th>1098</th>
      <th>1099</th>
      <th>1100</th>
      <th>1101</th>
      <th>1102</th>
      <th>1103</th>
      <th>1104</th>
      <th>1105</th>
      <th>1106</th>
      <th>1107</th>
      <th>1108</th>
      <th>1109</th>
      <th>1110</th>
      <th>1111</th>
      <th>1112</th>
      <th>1113</th>
      <th>1114</th>
      <th>1115</th>
      <th>1116</th>
      <th>1117</th>
      <th>1118</th>
      <th>1119</th>
      <th>1120</th>
      <th>1121</th>
      <th>1122</th>
      <th>1123</th>
      <th>1124</th>
      <th>1125</th>
      <th>1126</th>
      <th>1127</th>
      <th>1128</th>
      <th>1129</th>
      <th>1130</th>
      <th>1131</th>
      <th>1132</th>
      <th>1133</th>
      <th>1134</th>
      <th>1135</th>
      <th>1136</th>
      <th>1137</th>
      <th>1138</th>
      <th>1139</th>
      <th>1140</th>
      <th>1141</th>
      <th>1142</th>
      <th>1143</th>
      <th>1144</th>
      <th>1145</th>
      <th>1146</th>
      <th>1147</th>
      <th>1148</th>
      <th>1149</th>
      <th>1150</th>
      <th>1151</th>
      <th>1152</th>
      <th>1153</th>
      <th>1154</th>
      <th>1155</th>
      <th>1156</th>
      <th>1157</th>
      <th>1158</th>
      <th>1159</th>
      <th>1160</th>
      <th>1161</th>
      <th>1162</th>
      <th>1163</th>
      <th>1164</th>
      <th>1165</th>
      <th>1166</th>
      <th>1167</th>
      <th>1168</th>
      <th>1169</th>
      <th>1170</th>
      <th>1171</th>
      <th>1172</th>
      <th>1173</th>
      <th>1174</th>
      <th>1175</th>
      <th>1176</th>
      <th>1177</th>
      <th>1178</th>
      <th>1179</th>
      <th>1180</th>
      <th>1181</th>
      <th>1182</th>
      <th>1183</th>
      <th>1184</th>
      <th>1185</th>
      <th>1186</th>
      <th>1187</th>
      <th>1188</th>
      <th>1189</th>
      <th>1190</th>
      <th>1191</th>
      <th>1192</th>
      <th>1193</th>
      <th>1194</th>
      <th>1195</th>
      <th>1196</th>
      <th>1197</th>
      <th>1198</th>
      <th>1199</th>
      <th>1200</th>
      <th>1201</th>
      <th>1202</th>
      <th>1203</th>
      <th>1204</th>
      <th>1205</th>
      <th>1206</th>
      <th>1207</th>
      <th>1208</th>
      <th>1209</th>
      <th>1210</th>
      <th>1211</th>
      <th>1212</th>
      <th>1213</th>
      <th>1214</th>
      <th>1215</th>
      <th>1216</th>
      <th>1217</th>
      <th>1218</th>
      <th>1219</th>
      <th>1220</th>
      <th>1221</th>
      <th>1222</th>
      <th>1223</th>
      <th>1224</th>
      <th>1225</th>
      <th>1226</th>
      <th>1227</th>
      <th>1228</th>
      <th>1229</th>
      <th>1230</th>
      <th>1231</th>
      <th>1232</th>
      <th>1233</th>
      <th>1234</th>
      <th>1235</th>
      <th>1236</th>
      <th>1237</th>
      <th>1238</th>
      <th>1239</th>
      <th>1240</th>
      <th>1241</th>
      <th>1242</th>
      <th>1243</th>
      <th>1244</th>
      <th>1245</th>
      <th>1246</th>
      <th>1247</th>
      <th>1248</th>
      <th>1249</th>
      <th>1250</th>
      <th>1251</th>
      <th>1252</th>
      <th>1253</th>
      <th>1254</th>
      <th>1255</th>
      <th>1256</th>
      <th>1257</th>
      <th>1258</th>
      <th>1259</th>
      <th>1260</th>
      <th>1261</th>
      <th>1262</th>
      <th>1263</th>
      <th>1264</th>
      <th>1265</th>
      <th>1266</th>
      <th>1267</th>
      <th>1268</th>
      <th>1269</th>
      <th>1270</th>
      <th>1271</th>
      <th>1272</th>
      <th>1273</th>
      <th>1274</th>
      <th>1275</th>
      <th>1276</th>
      <th>1277</th>
      <th>1278</th>
      <th>1279</th>
      <th>1280</th>
      <th>1281</th>
      <th>1282</th>
      <th>1283</th>
      <th>1284</th>
      <th>1285</th>
      <th>1286</th>
      <th>1287</th>
      <th>1288</th>
      <th>1289</th>
      <th>1290</th>
      <th>1291</th>
      <th>1292</th>
      <th>1293</th>
      <th>1294</th>
      <th>1295</th>
      <th>1296</th>
      <th>1297</th>
      <th>1298</th>
      <th>1299</th>
      <th>1300</th>
      <th>1301</th>
      <th>1302</th>
      <th>1303</th>
      <th>1304</th>
      <th>1305</th>
      <th>1306</th>
      <th>1307</th>
      <th>1308</th>
      <th>1309</th>
      <th>1310</th>
      <th>1311</th>
      <th>1312</th>
      <th>1313</th>
      <th>1314</th>
      <th>1315</th>
      <th>1316</th>
      <th>1317</th>
      <th>1318</th>
      <th>1319</th>
      <th>1320</th>
      <th>1321</th>
      <th>1322</th>
      <th>1323</th>
      <th>1324</th>
      <th>1325</th>
      <th>1326</th>
      <th>1327</th>
      <th>1328</th>
      <th>1329</th>
      <th>1330</th>
      <th>1331</th>
      <th>1332</th>
      <th>1333</th>
      <th>1334</th>
      <th>1335</th>
      <th>1336</th>
      <th>1337</th>
      <th>1338</th>
      <th>1339</th>
      <th>1340</th>
      <th>1341</th>
      <th>1342</th>
      <th>1343</th>
      <th>1344</th>
      <th>1345</th>
      <th>1346</th>
      <th>1347</th>
      <th>1348</th>
      <th>1349</th>
      <th>1350</th>
      <th>1351</th>
      <th>1352</th>
      <th>1353</th>
      <th>1354</th>
      <th>1355</th>
      <th>1356</th>
      <th>1357</th>
      <th>1358</th>
      <th>1359</th>
      <th>1360</th>
      <th>1361</th>
      <th>1362</th>
      <th>1363</th>
      <th>1364</th>
      <th>1365</th>
      <th>1366</th>
      <th>1367</th>
      <th>1368</th>
      <th>1369</th>
      <th>1370</th>
      <th>1371</th>
      <th>1372</th>
      <th>1373</th>
      <th>1374</th>
      <th>1375</th>
      <th>1376</th>
      <th>1377</th>
      <th>1378</th>
      <th>1379</th>
      <th>1380</th>
      <th>1381</th>
      <th>1382</th>
      <th>1383</th>
      <th>1384</th>
      <th>1385</th>
      <th>1386</th>
      <th>1387</th>
      <th>1388</th>
      <th>1389</th>
      <th>1390</th>
      <th>1391</th>
      <th>1392</th>
      <th>1393</th>
      <th>1394</th>
      <th>1395</th>
      <th>1396</th>
      <th>1397</th>
      <th>1398</th>
      <th>1399</th>
      <th>1400</th>
      <th>1401</th>
      <th>1402</th>
      <th>1403</th>
      <th>1404</th>
      <th>1405</th>
      <th>1406</th>
      <th>1407</th>
      <th>1408</th>
      <th>1409</th>
      <th>1410</th>
      <th>1411</th>
      <th>1412</th>
      <th>1413</th>
      <th>1414</th>
      <th>1415</th>
      <th>1416</th>
      <th>1417</th>
      <th>1418</th>
      <th>1419</th>
      <th>1420</th>
      <th>1421</th>
      <th>1422</th>
      <th>1423</th>
      <th>1424</th>
      <th>1425</th>
      <th>1426</th>
      <th>1427</th>
      <th>1428</th>
      <th>1429</th>
      <th>1430</th>
      <th>1431</th>
      <th>1432</th>
      <th>1433</th>
      <th>1434</th>
      <th>1435</th>
      <th>1436</th>
      <th>1437</th>
      <th>1438</th>
      <th>1439</th>
      <th>1440</th>
      <th>1441</th>
      <th>1442</th>
      <th>1443</th>
      <th>1444</th>
      <th>1445</th>
      <th>1446</th>
      <th>1447</th>
      <th>1448</th>
      <th>1449</th>
      <th>1450</th>
      <th>1451</th>
      <th>1452</th>
      <th>1453</th>
      <th>1454</th>
      <th>1455</th>
      <th>1456</th>
      <th>1457</th>
      <th>1458</th>
      <th>1459</th>
      <th>1460</th>
      <th>1461</th>
      <th>1462</th>
      <th>1463</th>
      <th>1464</th>
      <th>1465</th>
      <th>1466</th>
      <th>1467</th>
      <th>1468</th>
      <th>1469</th>
      <th>1470</th>
      <th>1471</th>
      <th>1472</th>
      <th>1473</th>
      <th>1474</th>
      <th>1475</th>
      <th>1476</th>
      <th>1477</th>
      <th>1478</th>
      <th>1479</th>
      <th>1480</th>
      <th>1481</th>
      <th>1482</th>
      <th>1483</th>
      <th>1484</th>
      <th>1485</th>
      <th>1486</th>
      <th>1487</th>
      <th>1488</th>
      <th>1489</th>
      <th>1490</th>
      <th>1491</th>
      <th>1492</th>
      <th>1493</th>
      <th>1494</th>
      <th>1495</th>
      <th>1496</th>
      <th>1497</th>
      <th>1498</th>
      <th>1499</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[PAD]</td>
      <td>[EMPTY]</td>
      <td>[unused1]</td>
      <td>[unused2]</td>
      <td>[unused3]</td>
      <td>[unused4]</td>
      <td>[unused5]</td>
      <td>[unused6]</td>
      <td>[unused7]</td>
      <td>[unused8]</td>
      <td>[unused9]</td>
      <td>[unused10]</td>
      <td>[unused11]</td>
      <td>[unused12]</td>
      <td>[unused13]</td>
      <td>[unused14]</td>
      <td>[unused15]</td>
      <td>[unused16]</td>
      <td>[unused17]</td>
      <td>[unused18]</td>
      <td>[unused19]</td>
      <td>[unused20]</td>
      <td>[unused21]</td>
      <td>[unused22]</td>
      <td>[unused23]</td>
      <td>[unused24]</td>
      <td>[unused25]</td>
      <td>[unused26]</td>
      <td>[unused27]</td>
      <td>[unused28]</td>
      <td>[unused29]</td>
      <td>[unused30]</td>
      <td>[unused31]</td>
      <td>[unused32]</td>
      <td>[unused33]</td>
      <td>[unused34]</td>
      <td>[unused35]</td>
      <td>[unused36]</td>
      <td>[unused37]</td>
      <td>[unused38]</td>
      <td>[unused39]</td>
      <td>[unused40]</td>
      <td>[unused41]</td>
      <td>[unused42]</td>
      <td>[unused43]</td>
      <td>[unused44]</td>
      <td>[unused45]</td>
      <td>[unused46]</td>
      <td>[unused47]</td>
      <td>[unused48]</td>
      <td>[unused49]</td>
      <td>[unused50]</td>
      <td>[unused51]</td>
      <td>[unused52]</td>
      <td>[unused53]</td>
      <td>[unused54]</td>
      <td>[unused55]</td>
      <td>[unused56]</td>
      <td>[unused57]</td>
      <td>[unused58]</td>
      <td>[unused59]</td>
      <td>[unused60]</td>
      <td>[unused61]</td>
      <td>[unused62]</td>
      <td>[unused63]</td>
      <td>[unused64]</td>
      <td>[unused65]</td>
      <td>[unused66]</td>
      <td>[unused67]</td>
      <td>[unused68]</td>
      <td>[unused69]</td>
      <td>[unused70]</td>
      <td>[unused71]</td>
      <td>[unused72]</td>
      <td>[unused73]</td>
      <td>[unused74]</td>
      <td>[unused75]</td>
      <td>[unused76]</td>
      <td>[unused77]</td>
      <td>[unused78]</td>
      <td>[unused79]</td>
      <td>[unused80]</td>
      <td>[unused81]</td>
      <td>[unused82]</td>
      <td>[unused83]</td>
      <td>[unused84]</td>
      <td>[unused85]</td>
      <td>[unused86]</td>
      <td>[unused87]</td>
      <td>[unused88]</td>
      <td>[unused89]</td>
      <td>[unused90]</td>
      <td>[unused91]</td>
      <td>[unused92]</td>
      <td>[unused93]</td>
      <td>[unused94]</td>
      <td>[unused95]</td>
      <td>[unused96]</td>
      <td>[unused97]</td>
      <td>[unused98]</td>
      <td>[UNK]</td>
      <td>[CLS]</td>
      <td>[SEP]</td>
      <td>[MASK]</td>
      <td>[unused99]</td>
      <td>[unused100]</td>
      <td>[unused101]</td>
      <td>[unused102]</td>
      <td>[unused103]</td>
      <td>[unused104]</td>
      <td>[unused105]</td>
      <td>[unused106]</td>
      <td>[unused107]</td>
      <td>[unused108]</td>
      <td>[unused109]</td>
      <td>[unused110]</td>
      <td>[unused111]</td>
      <td>[unused112]</td>
      <td>[unused113]</td>
      <td>[unused114]</td>
      <td>[unused115]</td>
      <td>[unused116]</td>
      <td>[unused117]</td>
      <td>[unused118]</td>
      <td>[unused119]</td>
      <td>[unused120]</td>
      <td>[unused121]</td>
      <td>[unused122]</td>
      <td>[unused123]</td>
      <td>[unused124]</td>
      <td>[unused125]</td>
      <td>[unused126]</td>
      <td>[unused127]</td>
      <td>[unused128]</td>
      <td>[unused129]</td>
      <td>[unused130]</td>
      <td>[unused131]</td>
      <td>[unused132]</td>
      <td>[unused133]</td>
      <td>[unused134]</td>
      <td>[unused135]</td>
      <td>[unused136]</td>
      <td>[unused137]</td>
      <td>[unused138]</td>
      <td>[unused139]</td>
      <td>[unused140]</td>
      <td>[unused141]</td>
      <td>[unused142]</td>
      <td>[unused143]</td>
      <td>[unused144]</td>
      <td>[unused145]</td>
      <td>[unused146]</td>
      <td>[unused147]</td>
      <td>[unused148]</td>
      <td>[unused149]</td>
      <td>[unused150]</td>
      <td>[unused151]</td>
      <td>[unused152]</td>
      <td>[unused153]</td>
      <td>[unused154]</td>
      <td>[unused155]</td>
      <td>[unused156]</td>
      <td>[unused157]</td>
      <td>[unused158]</td>
      <td>[unused159]</td>
      <td>[unused160]</td>
      <td>[unused161]</td>
      <td>[unused162]</td>
      <td>[unused163]</td>
      <td>[unused164]</td>
      <td>[unused165]</td>
      <td>[unused166]</td>
      <td>[unused167]</td>
      <td>[unused168]</td>
      <td>[unused169]</td>
      <td>[unused170]</td>
      <td>[unused171]</td>
      <td>[unused172]</td>
      <td>[unused173]</td>
      <td>[unused174]</td>
      <td>[unused175]</td>
      <td>[unused176]</td>
      <td>[unused177]</td>
      <td>[unused178]</td>
      <td>[unused179]</td>
      <td>[unused180]</td>
      <td>[unused181]</td>
      <td>[unused182]</td>
      <td>[unused183]</td>
      <td>[unused184]</td>
      <td>[unused185]</td>
      <td>[unused186]</td>
      <td>[unused187]</td>
      <td>[unused188]</td>
      <td>[unused189]</td>
      <td>[unused190]</td>
      <td>[unused191]</td>
      <td>[unused192]</td>
      <td>[unused193]</td>
      <td>[unused194]</td>
      <td>[unused195]</td>
      <td>[unused196]</td>
      <td>[unused197]</td>
      <td>[unused198]</td>
      <td>[unused199]</td>
      <td>[unused200]</td>
      <td>[unused201]</td>
      <td>[unused202]</td>
      <td>[unused203]</td>
      <td>[unused204]</td>
      <td>[unused205]</td>
      <td>[unused206]</td>
      <td>[unused207]</td>
      <td>[unused208]</td>
      <td>[unused209]</td>
      <td>[unused210]</td>
      <td>[unused211]</td>
      <td>[unused212]</td>
      <td>[unused213]</td>
      <td>[unused214]</td>
      <td>[unused215]</td>
      <td>[unused216]</td>
      <td>[unused217]</td>
      <td>[unused218]</td>
      <td>[unused219]</td>
      <td>[unused220]</td>
      <td>[unused221]</td>
      <td>[unused222]</td>
      <td>[unused223]</td>
      <td>[unused224]</td>
      <td>[unused225]</td>
      <td>[unused226]</td>
      <td>[unused227]</td>
      <td>[unused228]</td>
      <td>[unused229]</td>
      <td>[unused230]</td>
      <td>[unused231]</td>
      <td>[unused232]</td>
      <td>[unused233]</td>
      <td>[unused234]</td>
      <td>[unused235]</td>
      <td>[unused236]</td>
      <td>[unused237]</td>
      <td>[unused238]</td>
      <td>[unused239]</td>
      <td>[unused240]</td>
      <td>[unused241]</td>
      <td>[unused242]</td>
      <td>[unused243]</td>
      <td>[unused244]</td>
      <td>[unused245]</td>
      <td>[unused246]</td>
      <td>[unused247]</td>
      <td>[unused248]</td>
      <td>[unused249]</td>
      <td>[unused250]</td>
      <td>[unused251]</td>
      <td>[unused252]</td>
      <td>[unused253]</td>
      <td>[unused254]</td>
      <td>[unused255]</td>
      <td>[unused256]</td>
      <td>[unused257]</td>
      <td>[unused258]</td>
      <td>[unused259]</td>
      <td>[unused260]</td>
      <td>[unused261]</td>
      <td>[unused262]</td>
      <td>[unused263]</td>
      <td>[unused264]</td>
      <td>[unused265]</td>
      <td>[unused266]</td>
      <td>[unused267]</td>
      <td>[unused268]</td>
      <td>[unused269]</td>
      <td>[unused270]</td>
      <td>[unused271]</td>
      <td>[unused272]</td>
      <td>[unused273]</td>
      <td>[unused274]</td>
      <td>[unused275]</td>
      <td>[unused276]</td>
      <td>[unused277]</td>
      <td>[unused278]</td>
      <td>[unused279]</td>
      <td>[unused280]</td>
      <td>[unused281]</td>
      <td>[unused282]</td>
      <td>[unused283]</td>
      <td>[unused284]</td>
      <td>[unused285]</td>
      <td>[unused286]</td>
      <td>[unused287]</td>
      <td>[unused288]</td>
      <td>[unused289]</td>
      <td>[unused290]</td>
      <td>[unused291]</td>
      <td>[unused292]</td>
      <td>[unused293]</td>
      <td>[unused294]</td>
      <td>[unused295]</td>
      <td>[unused296]</td>
      <td>[unused297]</td>
      <td>[unused298]</td>
      <td>[unused299]</td>
      <td>[unused300]</td>
      <td>[unused301]</td>
      <td>[unused302]</td>
      <td>[unused303]</td>
      <td>[unused304]</td>
      <td>[unused305]</td>
      <td>[unused306]</td>
      <td>[unused307]</td>
      <td>[unused308]</td>
      <td>[unused309]</td>
      <td>[unused310]</td>
      <td>[unused311]</td>
      <td>[unused312]</td>
      <td>[unused313]</td>
      <td>[unused314]</td>
      <td>[unused315]</td>
      <td>[unused316]</td>
      <td>[unused317]</td>
      <td>[unused318]</td>
      <td>[unused319]</td>
      <td>[unused320]</td>
      <td>[unused321]</td>
      <td>[unused322]</td>
      <td>[unused323]</td>
      <td>[unused324]</td>
      <td>[unused325]</td>
      <td>[unused326]</td>
      <td>[unused327]</td>
      <td>[unused328]</td>
      <td>[unused329]</td>
      <td>[unused330]</td>
      <td>[unused331]</td>
      <td>[unused332]</td>
      <td>[unused333]</td>
      <td>[unused334]</td>
      <td>[unused335]</td>
      <td>[unused336]</td>
      <td>[unused337]</td>
      <td>[unused338]</td>
      <td>[unused339]</td>
      <td>[unused340]</td>
      <td>[unused341]</td>
      <td>[unused342]</td>
      <td>[unused343]</td>
      <td>[unused344]</td>
      <td>[unused345]</td>
      <td>[unused346]</td>
      <td>[unused347]</td>
      <td>[unused348]</td>
      <td>[unused349]</td>
      <td>[unused350]</td>
      <td>[unused351]</td>
      <td>[unused352]</td>
      <td>[unused353]</td>
      <td>[unused354]</td>
      <td>[unused355]</td>
      <td>[unused356]</td>
      <td>[unused357]</td>
      <td>[unused358]</td>
      <td>[unused359]</td>
      <td>[unused360]</td>
      <td>[unused361]</td>
      <td>[unused362]</td>
      <td>[unused363]</td>
      <td>[unused364]</td>
      <td>[unused365]</td>
      <td>[unused366]</td>
      <td>[unused367]</td>
      <td>[unused368]</td>
      <td>[unused369]</td>
      <td>[unused370]</td>
      <td>[unused371]</td>
      <td>[unused372]</td>
      <td>[unused373]</td>
      <td>[unused374]</td>
      <td>[unused375]</td>
      <td>[unused376]</td>
      <td>[unused377]</td>
      <td>[unused378]</td>
      <td>[unused379]</td>
      <td>[unused380]</td>
      <td>[unused381]</td>
      <td>[unused382]</td>
      <td>[unused383]</td>
      <td>[unused384]</td>
      <td>[unused385]</td>
      <td>[unused386]</td>
      <td>[unused387]</td>
      <td>[unused388]</td>
      <td>[unused389]</td>
      <td>[unused390]</td>
      <td>[unused391]</td>
      <td>[unused392]</td>
      <td>[unused393]</td>
      <td>[unused394]</td>
      <td>[unused395]</td>
      <td>[unused396]</td>
      <td>[unused397]</td>
      <td>[unused398]</td>
      <td>[unused399]</td>
      <td>[unused400]</td>
      <td>[unused401]</td>
      <td>[unused402]</td>
      <td>[unused403]</td>
      <td>[unused404]</td>
      <td>[unused405]</td>
      <td>[unused406]</td>
      <td>[unused407]</td>
      <td>[unused408]</td>
      <td>[unused409]</td>
      <td>[unused410]</td>
      <td>[unused411]</td>
      <td>[unused412]</td>
      <td>[unused413]</td>
      <td>[unused414]</td>
      <td>[unused415]</td>
      <td>[unused416]</td>
      <td>[unused417]</td>
      <td>[unused418]</td>
      <td>[unused419]</td>
      <td>[unused420]</td>
      <td>[unused421]</td>
      <td>[unused422]</td>
      <td>[unused423]</td>
      <td>[unused424]</td>
      <td>[unused425]</td>
      <td>[unused426]</td>
      <td>[unused427]</td>
      <td>[unused428]</td>
      <td>[unused429]</td>
      <td>[unused430]</td>
      <td>[unused431]</td>
      <td>[unused432]</td>
      <td>[unused433]</td>
      <td>[unused434]</td>
      <td>[unused435]</td>
      <td>[unused436]</td>
      <td>[unused437]</td>
      <td>[unused438]</td>
      <td>[unused439]</td>
      <td>[unused440]</td>
      <td>[unused441]</td>
      <td>[unused442]</td>
      <td>[unused443]</td>
      <td>[unused444]</td>
      <td>[unused445]</td>
      <td>[unused446]</td>
      <td>[unused447]</td>
      <td>[unused448]</td>
      <td>[unused449]</td>
      <td>[unused450]</td>
      <td>[unused451]</td>
      <td>[unused452]</td>
      <td>[unused453]</td>
      <td>[unused454]</td>
      <td>[unused455]</td>
      <td>[unused456]</td>
      <td>[unused457]</td>
      <td>[unused458]</td>
      <td>[unused459]</td>
      <td>[unused460]</td>
      <td>[unused461]</td>
      <td>[unused462]</td>
      <td>[unused463]</td>
      <td>[unused464]</td>
      <td>[unused465]</td>
      <td>[unused466]</td>
      <td>[unused467]</td>
      <td>[unused468]</td>
      <td>[unused469]</td>
      <td>[unused470]</td>
      <td>[unused471]</td>
      <td>[unused472]</td>
      <td>[unused473]</td>
      <td>[unused474]</td>
      <td>[unused475]</td>
      <td>[unused476]</td>
      <td>[unused477]</td>
      <td>[unused478]</td>
      <td>[unused479]</td>
      <td>[unused480]</td>
      <td>[unused481]</td>
      <td>[unused482]</td>
      <td>[unused483]</td>
      <td>[unused484]</td>
      <td>[unused485]</td>
      <td>[unused486]</td>
      <td>[unused487]</td>
      <td>[unused488]</td>
      <td>[unused489]</td>
      <td>[unused490]</td>
      <td>[unused491]</td>
      <td>[unused492]</td>
      <td>[unused493]</td>
      <td>[unused494]</td>
      <td>[unused495]</td>
      <td>[unused496]</td>
      <td>[unused497]</td>
      <td>[unused498]</td>
      <td>[unused499]</td>
      <td>[unused500]</td>
      <td>[unused501]</td>
      <td>[unused502]</td>
      <td>[unused503]</td>
      <td>[unused504]</td>
      <td>[unused505]</td>
      <td>[unused506]</td>
      <td>[unused507]</td>
      <td>[unused508]</td>
      <td>[unused509]</td>
      <td>[unused510]</td>
      <td>[unused511]</td>
      <td>[unused512]</td>
      <td>[unused513]</td>
      <td>[unused514]</td>
      <td>[unused515]</td>
      <td>[unused516]</td>
      <td>[unused517]</td>
      <td>[unused518]</td>
      <td>[unused519]</td>
      <td>[unused520]</td>
      <td>[unused521]</td>
      <td>[unused522]</td>
      <td>[unused523]</td>
      <td>[unused524]</td>
      <td>[unused525]</td>
      <td>[unused526]</td>
      <td>[unused527]</td>
      <td>[unused528]</td>
      <td>[unused529]</td>
      <td>[unused530]</td>
      <td>[unused531]</td>
      <td>[unused532]</td>
      <td>[unused533]</td>
      <td>[unused534]</td>
      <td>[unused535]</td>
      <td>[unused536]</td>
      <td>[unused537]</td>
      <td>[unused538]</td>
      <td>[unused539]</td>
      <td>[unused540]</td>
      <td>[unused541]</td>
      <td>[unused542]</td>
      <td>[unused543]</td>
      <td>[unused544]</td>
      <td>[unused545]</td>
      <td>[unused546]</td>
      <td>[unused547]</td>
      <td>[unused548]</td>
      <td>[unused549]</td>
      <td>[unused550]</td>
      <td>[unused551]</td>
      <td>[unused552]</td>
      <td>[unused553]</td>
      <td>[unused554]</td>
      <td>[unused555]</td>
      <td>[unused556]</td>
      <td>[unused557]</td>
      <td>[unused558]</td>
      <td>[unused559]</td>
      <td>[unused560]</td>
      <td>[unused561]</td>
      <td>[unused562]</td>
      <td>[unused563]</td>
      <td>[unused564]</td>
      <td>[unused565]</td>
      <td>[unused566]</td>
      <td>[unused567]</td>
      <td>[unused568]</td>
      <td>[unused569]</td>
      <td>[unused570]</td>
      <td>[unused571]</td>
      <td>[unused572]</td>
      <td>[unused573]</td>
      <td>[unused574]</td>
      <td>[unused575]</td>
      <td>[unused576]</td>
      <td>[unused577]</td>
      <td>[unused578]</td>
      <td>[unused579]</td>
      <td>[unused580]</td>
      <td>[unused581]</td>
      <td>[unused582]</td>
      <td>[unused583]</td>
      <td>[unused584]</td>
      <td>[unused585]</td>
      <td>[unused586]</td>
      <td>[unused587]</td>
      <td>[unused588]</td>
      <td>[unused589]</td>
      <td>[unused590]</td>
      <td>[unused591]</td>
      <td>[unused592]</td>
      <td>[unused593]</td>
      <td>[unused594]</td>
      <td>[unused595]</td>
      <td>[unused596]</td>
      <td>[unused597]</td>
      <td>[unused598]</td>
      <td>[unused599]</td>
      <td>[unused600]</td>
      <td>[unused601]</td>
      <td>[unused602]</td>
      <td>[unused603]</td>
      <td>[unused604]</td>
      <td>[unused605]</td>
      <td>[unused606]</td>
      <td>[unused607]</td>
      <td>[unused608]</td>
      <td>[unused609]</td>
      <td>[unused610]</td>
      <td>[unused611]</td>
      <td>[unused612]</td>
      <td>[unused613]</td>
      <td>[unused614]</td>
      <td>[unused615]</td>
      <td>[unused616]</td>
      <td>[unused617]</td>
      <td>[unused618]</td>
      <td>[unused619]</td>
      <td>[unused620]</td>
      <td>[unused621]</td>
      <td>[unused622]</td>
      <td>[unused623]</td>
      <td>[unused624]</td>
      <td>[unused625]</td>
      <td>[unused626]</td>
      <td>[unused627]</td>
      <td>[unused628]</td>
      <td>[unused629]</td>
      <td>[unused630]</td>
      <td>[unused631]</td>
      <td>[unused632]</td>
      <td>[unused633]</td>
      <td>[unused634]</td>
      <td>[unused635]</td>
      <td>[unused636]</td>
      <td>[unused637]</td>
      <td>[unused638]</td>
      <td>[unused639]</td>
      <td>[unused640]</td>
      <td>[unused641]</td>
      <td>[unused642]</td>
      <td>[unused643]</td>
      <td>[unused644]</td>
      <td>[unused645]</td>
      <td>[unused646]</td>
      <td>[unused647]</td>
      <td>[unused648]</td>
      <td>[unused649]</td>
      <td>[unused650]</td>
      <td>[unused651]</td>
      <td>[unused652]</td>
      <td>[unused653]</td>
      <td>[unused654]</td>
      <td>[unused655]</td>
      <td>[unused656]</td>
      <td>[unused657]</td>
      <td>[unused658]</td>
      <td>[unused659]</td>
      <td>[unused660]</td>
      <td>[unused661]</td>
      <td>[unused662]</td>
      <td>[unused663]</td>
      <td>[unused664]</td>
      <td>[unused665]</td>
      <td>[unused666]</td>
      <td>[unused667]</td>
      <td>[unused668]</td>
      <td>[unused669]</td>
      <td>[unused670]</td>
      <td>[unused671]</td>
      <td>[unused672]</td>
      <td>[unused673]</td>
      <td>[unused674]</td>
      <td>[unused675]</td>
      <td>[unused676]</td>
      <td>[unused677]</td>
      <td>[unused678]</td>
      <td>[unused679]</td>
      <td>[unused680]</td>
      <td>[unused681]</td>
      <td>[unused682]</td>
      <td>[unused683]</td>
      <td>[unused684]</td>
      <td>[unused685]</td>
      <td>[unused686]</td>
      <td>[unused687]</td>
      <td>[unused688]</td>
      <td>[unused689]</td>
      <td>[unused690]</td>
      <td>[unused691]</td>
      <td>[unused692]</td>
      <td>[unused693]</td>
      <td>[unused694]</td>
      <td>[unused695]</td>
      <td>[unused696]</td>
      <td>[unused697]</td>
      <td>[unused698]</td>
      <td>[unused699]</td>
      <td>[unused700]</td>
      <td>[unused701]</td>
      <td>[unused702]</td>
      <td>[unused703]</td>
      <td>[unused704]</td>
      <td>[unused705]</td>
      <td>[unused706]</td>
      <td>[unused707]</td>
      <td>[unused708]</td>
      <td>[unused709]</td>
      <td>[unused710]</td>
      <td>[unused711]</td>
      <td>[unused712]</td>
      <td>[unused713]</td>
      <td>[unused714]</td>
      <td>[unused715]</td>
      <td>[unused716]</td>
      <td>[unused717]</td>
      <td>[unused718]</td>
      <td>[unused719]</td>
      <td>[unused720]</td>
      <td>[unused721]</td>
      <td>[unused722]</td>
      <td>[unused723]</td>
      <td>[unused724]</td>
      <td>[unused725]</td>
      <td>[unused726]</td>
      <td>[unused727]</td>
      <td>[unused728]</td>
      <td>[unused729]</td>
      <td>[unused730]</td>
      <td>[unused731]</td>
      <td>[unused732]</td>
      <td>[unused733]</td>
      <td>[unused734]</td>
      <td>[unused735]</td>
      <td>[unused736]</td>
      <td>[unused737]</td>
      <td>[unused738]</td>
      <td>[unused739]</td>
      <td>[unused740]</td>
      <td>[unused741]</td>
      <td>[unused742]</td>
      <td>[unused743]</td>
      <td>[unused744]</td>
      <td>[unused745]</td>
      <td>[unused746]</td>
      <td>[unused747]</td>
      <td>[unused748]</td>
      <td>[unused749]</td>
      <td>[unused750]</td>
      <td>[unused751]</td>
      <td>[unused752]</td>
      <td>[unused753]</td>
      <td>[unused754]</td>
      <td>[unused755]</td>
      <td>[unused756]</td>
      <td>[unused757]</td>
      <td>[unused758]</td>
      <td>[unused759]</td>
      <td>[unused760]</td>
      <td>[unused761]</td>
      <td>[unused762]</td>
      <td>[unused763]</td>
      <td>[unused764]</td>
      <td>[unused765]</td>
      <td>[unused766]</td>
      <td>[unused767]</td>
      <td>[unused768]</td>
      <td>[unused769]</td>
      <td>[unused770]</td>
      <td>[unused771]</td>
      <td>[unused772]</td>
      <td>[unused773]</td>
      <td>[unused774]</td>
      <td>[unused775]</td>
      <td>[unused776]</td>
      <td>[unused777]</td>
      <td>[unused778]</td>
      <td>[unused779]</td>
      <td>[unused780]</td>
      <td>[unused781]</td>
      <td>[unused782]</td>
      <td>[unused783]</td>
      <td>[unused784]</td>
      <td>[unused785]</td>
      <td>[unused786]</td>
      <td>[unused787]</td>
      <td>[unused788]</td>
      <td>[unused789]</td>
      <td>[unused790]</td>
      <td>[unused791]</td>
      <td>[unused792]</td>
      <td>[unused793]</td>
      <td>[unused794]</td>
      <td>[unused795]</td>
      <td>[unused796]</td>
      <td>[unused797]</td>
      <td>[unused798]</td>
      <td>[unused799]</td>
      <td>[unused800]</td>
      <td>[unused801]</td>
      <td>[unused802]</td>
      <td>[unused803]</td>
      <td>[unused804]</td>
      <td>[unused805]</td>
      <td>[unused806]</td>
      <td>[unused807]</td>
      <td>[unused808]</td>
      <td>[unused809]</td>
      <td>[unused810]</td>
      <td>[unused811]</td>
      <td>[unused812]</td>
      <td>[unused813]</td>
      <td>[unused814]</td>
      <td>[unused815]</td>
      <td>[unused816]</td>
      <td>[unused817]</td>
      <td>[unused818]</td>
      <td>[unused819]</td>
      <td>[unused820]</td>
      <td>[unused821]</td>
      <td>[unused822]</td>
      <td>[unused823]</td>
      <td>[unused824]</td>
      <td>[unused825]</td>
      <td>[unused826]</td>
      <td>[unused827]</td>
      <td>[unused828]</td>
      <td>[unused829]</td>
      <td>[unused830]</td>
      <td>[unused831]</td>
      <td>[unused832]</td>
      <td>[unused833]</td>
      <td>[unused834]</td>
      <td>[unused835]</td>
      <td>[unused836]</td>
      <td>[unused837]</td>
      <td>[unused838]</td>
      <td>[unused839]</td>
      <td>[unused840]</td>
      <td>[unused841]</td>
      <td>[unused842]</td>
      <td>[unused843]</td>
      <td>[unused844]</td>
      <td>[unused845]</td>
      <td>[unused846]</td>
      <td>[unused847]</td>
      <td>[unused848]</td>
      <td>[unused849]</td>
      <td>[unused850]</td>
      <td>[unused851]</td>
      <td>[unused852]</td>
      <td>[unused853]</td>
      <td>[unused854]</td>
      <td>[unused855]</td>
      <td>[unused856]</td>
      <td>[unused857]</td>
      <td>[unused858]</td>
      <td>[unused859]</td>
      <td>[unused860]</td>
      <td>[unused861]</td>
      <td>[unused862]</td>
      <td>[unused863]</td>
      <td>[unused864]</td>
      <td>[unused865]</td>
      <td>[unused866]</td>
      <td>[unused867]</td>
      <td>[unused868]</td>
      <td>[unused869]</td>
      <td>[unused870]</td>
      <td>[unused871]</td>
      <td>[unused872]</td>
      <td>[unused873]</td>
      <td>[unused874]</td>
      <td>[unused875]</td>
      <td>[unused876]</td>
      <td>[unused877]</td>
      <td>[unused878]</td>
      <td>[unused879]</td>
      <td>[unused880]</td>
      <td>[unused881]</td>
      <td>[unused882]</td>
      <td>[unused883]</td>
      <td>[unused884]</td>
      <td>[unused885]</td>
      <td>[unused886]</td>
      <td>[unused887]</td>
      <td>[unused888]</td>
      <td>[unused889]</td>
      <td>[unused890]</td>
      <td>[unused891]</td>
      <td>[unused892]</td>
      <td>[unused893]</td>
      <td>[unused894]</td>
      <td>[unused895]</td>
      <td>[unused896]</td>
      <td>[unused897]</td>
      <td>[unused898]</td>
      <td>[unused899]</td>
      <td>[unused900]</td>
      <td>[unused901]</td>
      <td>[unused902]</td>
      <td>[unused903]</td>
      <td>[unused904]</td>
      <td>[unused905]</td>
      <td>[unused906]</td>
      <td>[unused907]</td>
      <td>[unused908]</td>
      <td>[unused909]</td>
      <td>[unused910]</td>
      <td>[unused911]</td>
      <td>[unused912]</td>
      <td>[unused913]</td>
      <td>[unused914]</td>
      <td>[unused915]</td>
      <td>[unused916]</td>
      <td>[unused917]</td>
      <td>[unused918]</td>
      <td>[unused919]</td>
      <td>[unused920]</td>
      <td>[unused921]</td>
      <td>[unused922]</td>
      <td>[unused923]</td>
      <td>[unused924]</td>
      <td>[unused925]</td>
      <td>[unused926]</td>
      <td>[unused927]</td>
      <td>[unused928]</td>
      <td>[unused929]</td>
      <td>[unused930]</td>
      <td>[unused931]</td>
      <td>[unused932]</td>
      <td>[unused933]</td>
      <td>[unused934]</td>
      <td>[unused935]</td>
      <td>[unused936]</td>
      <td>[unused937]</td>
      <td>[unused938]</td>
      <td>[unused939]</td>
      <td>[unused940]</td>
      <td>[unused941]</td>
      <td>[unused942]</td>
      <td>[unused943]</td>
      <td>[unused944]</td>
      <td>[unused945]</td>
      <td>[unused946]</td>
      <td>[unused947]</td>
      <td>[unused948]</td>
      <td>[unused949]</td>
      <td>[unused950]</td>
      <td>[unused951]</td>
      <td>[unused952]</td>
      <td>[unused953]</td>
      <td>[unused954]</td>
      <td>[unused955]</td>
      <td>[unused956]</td>
      <td>[unused957]</td>
      <td>[unused958]</td>
      <td>[unused959]</td>
      <td>[unused960]</td>
      <td>[unused961]</td>
      <td>[unused962]</td>
      <td>[unused963]</td>
      <td>[unused964]</td>
      <td>[unused965]</td>
      <td>[unused966]</td>
      <td>[unused967]</td>
      <td>[unused968]</td>
      <td>[unused969]</td>
      <td>[unused970]</td>
      <td>[unused971]</td>
      <td>[unused972]</td>
      <td>[unused973]</td>
      <td>[unused974]</td>
      <td>[unused975]</td>
      <td>[unused976]</td>
      <td>[unused977]</td>
      <td>[unused978]</td>
      <td>[unused979]</td>
      <td>[unused980]</td>
      <td>[unused981]</td>
      <td>[unused982]</td>
      <td>[unused983]</td>
      <td>[unused984]</td>
      <td>[unused985]</td>
      <td>[unused986]</td>
      <td>[unused987]</td>
      <td>[unused988]</td>
      <td>[unused989]</td>
      <td>[unused990]</td>
      <td>[unused991]</td>
      <td>[unused992]</td>
      <td>[unused993]</td>
      <td>!</td>
      <td>"</td>
      <td>#</td>
      <td>$</td>
      <td>%</td>
      <td>&amp;</td>
      <td>'</td>
      <td>(</td>
      <td>)</td>
      <td>*</td>
      <td>+</td>
      <td>,</td>
      <td>-</td>
      <td>.</td>
      <td>/</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>4</td>
      <td>5</td>
      <td>6</td>
      <td>7</td>
      <td>8</td>
      <td>9</td>
      <td>:</td>
      <td>;</td>
      <td>&lt;</td>
      <td>=</td>
      <td>&gt;</td>
      <td>?</td>
      <td>@</td>
      <td>[</td>
      <td>\</td>
      <td>]</td>
      <td>^</td>
      <td>_</td>
      <td>`</td>
      <td>a</td>
      <td>b</td>
      <td>c</td>
      <td>d</td>
      <td>e</td>
      <td>f</td>
      <td>g</td>
      <td>h</td>
      <td>i</td>
      <td>j</td>
      <td>k</td>
      <td>l</td>
      <td>m</td>
      <td>n</td>
      <td>o</td>
      <td>p</td>
      <td>q</td>
      <td>r</td>
      <td>s</td>
      <td>t</td>
      <td>u</td>
      <td>v</td>
      <td>w</td>
      <td>x</td>
      <td>y</td>
      <td>z</td>
      <td>{</td>
      <td>|</td>
      <td>}</td>
      <td>~</td>
      <td>¡</td>
      <td>¢</td>
      <td>£</td>
      <td>¤</td>
      <td>¥</td>
      <td>¦</td>
      <td>§</td>
      <td>¨</td>
      <td>©</td>
      <td>ª</td>
      <td>«</td>
      <td>¬</td>
      <td>®</td>
      <td>°</td>
      <td>±</td>
      <td>²</td>
      <td>³</td>
      <td>´</td>
      <td>µ</td>
      <td>¶</td>
      <td>·</td>
      <td>¹</td>
      <td>º</td>
      <td>»</td>
      <td>¼</td>
      <td>½</td>
      <td>¾</td>
      <td>¿</td>
      <td>×</td>
      <td>ß</td>
      <td>æ</td>
      <td>ð</td>
      <td>÷</td>
      <td>ø</td>
      <td>þ</td>
      <td>đ</td>
      <td>ħ</td>
      <td>ı</td>
      <td>ł</td>
      <td>ŋ</td>
      <td>œ</td>
      <td>ƒ</td>
      <td>ɐ</td>
      <td>ɑ</td>
      <td>ɒ</td>
      <td>ɔ</td>
      <td>ɕ</td>
      <td>ə</td>
      <td>ɛ</td>
      <td>ɡ</td>
      <td>ɣ</td>
      <td>ɨ</td>
      <td>ɪ</td>
      <td>ɫ</td>
      <td>ɬ</td>
      <td>ɯ</td>
      <td>ɲ</td>
      <td>ɴ</td>
      <td>ɹ</td>
      <td>ɾ</td>
      <td>ʀ</td>
      <td>ʁ</td>
      <td>ʂ</td>
      <td>ʃ</td>
      <td>ʉ</td>
      <td>ʊ</td>
      <td>ʋ</td>
      <td>ʌ</td>
      <td>ʎ</td>
      <td>ʐ</td>
      <td>ʑ</td>
      <td>ʒ</td>
      <td>ʔ</td>
      <td>ʰ</td>
      <td>ʲ</td>
      <td>ʳ</td>
      <td>ʷ</td>
      <td>ʸ</td>
      <td>ʻ</td>
      <td>ʼ</td>
      <td>ʾ</td>
      <td>ʿ</td>
      <td>ˈ</td>
      <td>ː</td>
      <td>ˡ</td>
      <td>ˢ</td>
      <td>ˣ</td>
      <td>ˤ</td>
      <td>α</td>
      <td>β</td>
      <td>γ</td>
      <td>δ</td>
      <td>ε</td>
      <td>ζ</td>
      <td>η</td>
      <td>θ</td>
      <td>ι</td>
      <td>κ</td>
      <td>λ</td>
      <td>μ</td>
      <td>ν</td>
      <td>ξ</td>
      <td>ο</td>
      <td>π</td>
      <td>ρ</td>
      <td>ς</td>
      <td>σ</td>
      <td>τ</td>
      <td>υ</td>
      <td>φ</td>
      <td>χ</td>
      <td>ψ</td>
      <td>ω</td>
      <td>а</td>
      <td>б</td>
      <td>в</td>
      <td>г</td>
      <td>д</td>
      <td>е</td>
      <td>ж</td>
      <td>з</td>
      <td>и</td>
      <td>к</td>
      <td>л</td>
      <td>м</td>
      <td>н</td>
      <td>о</td>
      <td>п</td>
      <td>р</td>
      <td>с</td>
      <td>т</td>
      <td>у</td>
      <td>ф</td>
      <td>х</td>
      <td>ц</td>
      <td>ч</td>
      <td>ш</td>
      <td>щ</td>
      <td>ъ</td>
      <td>ы</td>
      <td>ь</td>
      <td>э</td>
      <td>ю</td>
      <td>я</td>
      <td>ђ</td>
      <td>є</td>
      <td>і</td>
      <td>ј</td>
      <td>љ</td>
      <td>њ</td>
      <td>ћ</td>
      <td>ӏ</td>
      <td>ա</td>
      <td>բ</td>
      <td>գ</td>
      <td>դ</td>
      <td>ե</td>
      <td>թ</td>
      <td>ի</td>
      <td>լ</td>
      <td>կ</td>
      <td>հ</td>
      <td>մ</td>
      <td>յ</td>
      <td>ն</td>
      <td>ո</td>
      <td>պ</td>
      <td>ս</td>
      <td>վ</td>
      <td>տ</td>
      <td>ր</td>
      <td>ւ</td>
      <td>ք</td>
      <td>־</td>
      <td>א</td>
      <td>ב</td>
      <td>ג</td>
      <td>ד</td>
      <td>ה</td>
      <td>ו</td>
      <td>ז</td>
      <td>ח</td>
      <td>ט</td>
      <td>י</td>
      <td>ך</td>
      <td>כ</td>
      <td>ל</td>
      <td>ם</td>
      <td>מ</td>
      <td>ן</td>
      <td>נ</td>
      <td>ס</td>
      <td>ע</td>
      <td>ף</td>
      <td>פ</td>
      <td>ץ</td>
      <td>צ</td>
      <td>ק</td>
      <td>ר</td>
      <td>ש</td>
      <td>ת</td>
      <td>،</td>
      <td>ء</td>
      <td>ا</td>
      <td>ب</td>
      <td>ة</td>
      <td>ت</td>
      <td>ث</td>
      <td>ج</td>
      <td>ح</td>
      <td>خ</td>
      <td>د</td>
      <td>ذ</td>
      <td>ر</td>
      <td>ز</td>
      <td>س</td>
      <td>ش</td>
      <td>ص</td>
      <td>ض</td>
      <td>ط</td>
      <td>ظ</td>
      <td>ع</td>
      <td>غ</td>
      <td>ـ</td>
      <td>ف</td>
      <td>ق</td>
      <td>ك</td>
      <td>ل</td>
      <td>م</td>
      <td>ن</td>
      <td>ه</td>
      <td>و</td>
      <td>ى</td>
      <td>ي</td>
      <td>ٹ</td>
      <td>پ</td>
      <td>چ</td>
      <td>ک</td>
      <td>گ</td>
      <td>ں</td>
      <td>ھ</td>
      <td>ہ</td>
      <td>ی</td>
      <td>ے</td>
      <td>अ</td>
      <td>आ</td>
      <td>उ</td>
      <td>ए</td>
      <td>क</td>
      <td>ख</td>
      <td>ग</td>
      <td>च</td>
      <td>ज</td>
      <td>ट</td>
      <td>ड</td>
      <td>ण</td>
      <td>त</td>
      <td>थ</td>
      <td>द</td>
      <td>ध</td>
      <td>न</td>
      <td>प</td>
      <td>ब</td>
      <td>भ</td>
      <td>म</td>
      <td>य</td>
      <td>र</td>
      <td>ल</td>
      <td>व</td>
      <td>श</td>
      <td>ष</td>
      <td>स</td>
      <td>ह</td>
      <td>ा</td>
      <td>ि</td>
      <td>ी</td>
      <td>ो</td>
      <td>।</td>
      <td>॥</td>
      <td>ং</td>
      <td>অ</td>
      <td>আ</td>
      <td>ই</td>
      <td>উ</td>
      <td>এ</td>
      <td>ও</td>
      <td>ক</td>
      <td>খ</td>
      <td>গ</td>
      <td>চ</td>
      <td>ছ</td>
      <td>জ</td>
      <td>ট</td>
      <td>ড</td>
      <td>ণ</td>
      <td>ত</td>
      <td>থ</td>
      <td>দ</td>
      <td>ধ</td>
      <td>ন</td>
      <td>প</td>
      <td>ব</td>
      <td>ভ</td>
      <td>ম</td>
      <td>য</td>
      <td>র</td>
      <td>ল</td>
      <td>শ</td>
      <td>ষ</td>
      <td>স</td>
      <td>হ</td>
      <td>া</td>
      <td>ি</td>
      <td>ী</td>
      <td>ে</td>
      <td>க</td>
      <td>ச</td>
      <td>ட</td>
      <td>த</td>
      <td>ந</td>
      <td>ன</td>
      <td>ப</td>
      <td>ம</td>
      <td>ய</td>
      <td>ர</td>
      <td>ல</td>
      <td>ள</td>
      <td>வ</td>
      <td>ா</td>
      <td>ி</td>
      <td>ு</td>
      <td>ே</td>
      <td>ை</td>
      <td>ನ</td>
      <td>ರ</td>
      <td>ಾ</td>
      <td>ක</td>
      <td>ය</td>
      <td>ර</td>
      <td>ල</td>
      <td>ව</td>
      <td>ා</td>
      <td>ก</td>
      <td>ง</td>
      <td>ต</td>
      <td>ท</td>
      <td>น</td>
      <td>พ</td>
      <td>ม</td>
      <td>ย</td>
      <td>ร</td>
      <td>ล</td>
      <td>ว</td>
      <td>ส</td>
      <td>อ</td>
      <td>า</td>
      <td>เ</td>
      <td>་</td>
      <td>།</td>
      <td>ག</td>
      <td>ང</td>
      <td>ད</td>
      <td>ན</td>
      <td>པ</td>
      <td>བ</td>
      <td>མ</td>
      <td>འ</td>
      <td>ར</td>
      <td>ལ</td>
      <td>ས</td>
      <td>မ</td>
      <td>ა</td>
      <td>ბ</td>
      <td>გ</td>
      <td>დ</td>
      <td>ე</td>
      <td>ვ</td>
      <td>თ</td>
      <td>ი</td>
      <td>კ</td>
      <td>ლ</td>
      <td>მ</td>
      <td>ნ</td>
      <td>ო</td>
      <td>რ</td>
      <td>ს</td>
      <td>ტ</td>
      <td>უ</td>
      <td>ᄀ</td>
      <td>ᄂ</td>
      <td>ᄃ</td>
      <td>ᄅ</td>
      <td>ᄆ</td>
      <td>ᄇ</td>
      <td>ᄉ</td>
      <td>ᄊ</td>
      <td>ᄋ</td>
      <td>ᄌ</td>
      <td>ᄎ</td>
      <td>ᄏ</td>
      <td>ᄐ</td>
      <td>ᄑ</td>
      <td>ᄒ</td>
      <td>ᅡ</td>
      <td>ᅢ</td>
      <td>ᅥ</td>
      <td>ᅦ</td>
      <td>ᅧ</td>
      <td>ᅩ</td>
      <td>ᅪ</td>
      <td>ᅭ</td>
      <td>ᅮ</td>
      <td>ᅯ</td>
      <td>ᅲ</td>
      <td>ᅳ</td>
      <td>ᅴ</td>
      <td>ᅵ</td>
      <td>ᆨ</td>
      <td>ᆫ</td>
      <td>ᆯ</td>
      <td>ᆷ</td>
      <td>ᆸ</td>
      <td>ᆼ</td>
      <td>ᴬ</td>
      <td>ᴮ</td>
      <td>ᴰ</td>
      <td>ᴵ</td>
      <td>ᴺ</td>
      <td>ᵀ</td>
      <td>ᵃ</td>
      <td>ᵇ</td>
      <td>ᵈ</td>
      <td>ᵉ</td>
    </tr>
  </tbody>
</table>
</div>
------


#### `TapasForQuestionAnswering`
* [Documentation](https://huggingface.co/docs/transformers/main/en/model_doc/tapas#transformers.TapasForQuestionAnswering)
* Create a Tapas Model with a cell selection head and optional aggregation head for question answering tasks.

**Pass some queries to the model**


```python
queries = ["What's the topic in chapter 4?",
           "What is the total number of pages?",
           "On which page does the chapter about question-answering start?",
           "How many chapters have more than 20 pages?"]
preds = table_qa(table, queries)
```


```python
for query, pred in zip(queries, preds):
    print(query)
    if pred["aggregator"] == "NONE": 
        print("Predicted answer: " + pred["answer"])
    else: 
        print("Predicted answer: " + pred["answer"])
    print('='*50)
```
```text
    What's the topic in chapter 4?
    Predicted answer: Summarization
    ==================================================
    What is the total number of pages?
    Predicted answer: SUM > 10, 36, 24, 46, 19, 3
    ==================================================
    On which page does the chapter about question-answering start?
    Predicted answer: AVERAGE > 74
    ==================================================
    How many chapters have more than 20 pages?
    Predicted answer: COUNT > 1, 2, 3
    ==================================================
```

**Note:**
* The model predicted exactly one cell with no aggregation for the first query, and the answer is correct.
* For the second query, the model correctly predicted that we need to sum the individual page counts for each chapter to determine the total number of pages.
* The model correctly answered question three but included an unnecessary average aggregation.
* The model correctly determined that chapters 1, 2, and 3 have more than 20 pages.
* The ability to ask questions in natural language instead of Python code allows a much wider audience to query the data to answer specific questions.

------

## Multimodal Transformers

### Speech-to-Text
* Speaking is more convenient than reading and writing for a significant portion of the population.
* Automatic speech recognition (ASR) involves converting spoken words to text and enables voice technologies like Siri to answer questions like "What is the weather like today?".
* The [wave2vec 2.0](https://huggingface.co/models?search=wav2vec2+facebook) family of models is one of the most recent developments in ASR and uses a transformer layer in combination with a CNN.
    * [wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations](https://arxiv.org/abs/2006.11477)
* These models leverage unlabeled data to achieve competitive results with only a few minutes of labeled data.
* The Hugging Face Transformers library includes wave2vec 2.0 models.

**Create an automatic speech recognition pipeline**


```python
asr = pipeline("automatic-speech-recognition")
asr.model.config
```
```text
    Wav2Vec2Config {
      "_name_or_path": "facebook/wav2vec2-base-960h",
      "activation_dropout": 0.1,
      "adapter_kernel_size": 3,
      "adapter_stride": 2,
      "add_adapter": false,
      "apply_spec_augment": true,
      "architectures": [
        "Wav2Vec2ForCTC"
      ],
      "attention_dropout": 0.1,
      "bos_token_id": 1,
      "classifier_proj_size": 256,
      "codevector_dim": 256,
      "contrastive_logits_temperature": 0.1,
      "conv_bias": false,
      "conv_dim": [
        512,
        512,
        512,
        512,
        512,
        512,
        512
      ],
      "conv_kernel": [
        10,
        3,
        3,
        3,
        3,
        2,
        2
      ],
      "conv_stride": [
        5,
        2,
        2,
        2,
        2,
        2,
        2
      ],
      "ctc_loss_reduction": "sum",
      "ctc_zero_infinity": false,
      "diversity_loss_weight": 0.1,
      "do_stable_layer_norm": false,
      "eos_token_id": 2,
      "feat_extract_activation": "gelu",
      "feat_extract_dropout": 0.0,
      "feat_extract_norm": "group",
      "feat_proj_dropout": 0.1,
      "feat_quantizer_dropout": 0.0,
      "final_dropout": 0.1,
      "gradient_checkpointing": false,
      "hidden_act": "gelu",
      "hidden_dropout": 0.1,
      "hidden_dropout_prob": 0.1,
      "hidden_size": 768,
      "initializer_range": 0.02,
      "intermediate_size": 3072,
      "layer_norm_eps": 1e-05,
      "layerdrop": 0.1,
      "mask_feature_length": 10,
      "mask_feature_min_masks": 0,
      "mask_feature_prob": 0.0,
      "mask_time_length": 10,
      "mask_time_min_masks": 2,
      "mask_time_prob": 0.05,
      "model_type": "wav2vec2",
      "num_adapter_layers": 3,
      "num_attention_heads": 12,
      "num_codevector_groups": 2,
      "num_codevectors_per_group": 320,
      "num_conv_pos_embedding_groups": 16,
      "num_conv_pos_embeddings": 128,
      "num_feat_extract_layers": 7,
      "num_hidden_layers": 12,
      "num_negatives": 100,
      "output_hidden_size": 768,
      "pad_token_id": 0,
      "proj_codevector_dim": 256,
      "tdnn_dilation": [
        1,
        2,
        3,
        1,
        1
      ],
      "tdnn_dim": [
        512,
        512,
        512,
        512,
        1500
      ],
      "tdnn_kernel": [
        5,
        3,
        3,
        1,
        1
      ],
      "transformers_version": "4.18.0",
      "use_weighted_layer_sum": false,
      "vocab_size": 32,
      "xvector_output_dim": 512
    }
```

------

**Get the link to the Hugging Face model card**


```python
print(f"https://huggingface.co/{asr.model.config._name_or_path}")
```
```text
    https://huggingface.co/facebook/wav2vec2-base-960h
```

**Note:** The model trained on 960 hours of speech audio.

------

#### `Wav2Vec2ForCTC`
* [Documentation](https://huggingface.co/docs/transformers/main/en/model_doc/wav2vec2#transformers.Wav2Vec2ForCTC)
* Create a Wav2Vec2 model with a language modeling head for Connectionist Temporal Classification (CTC).

------


```python
from datasets import load_dataset
```

------

#### The SUPERB Dataset

* [Hugging Face Dataset Card](https://huggingface.co/datasets/superb)
* SUPERB is a leaderboard to benchmark the performance of a shared model across a wide range of speech processing tasks with minimal architecture changes and labeled data.

**Load the ASR subset of the SUPERB dataset**


```python
ds = load_dataset("superb", "asr", split="validation[:1]")
pd.DataFrame(ds[0])
```
<div style="overflow-x:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>file</th>
      <th>audio</th>
      <th>text</th>
      <th>speaker_id</th>
      <th>chapter_id</th>
      <th>id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>array</th>
      <td>/home/innom-dt/.cache/huggingface/datasets/downloads/extracted/aa91addd71e85ab524e5b5b56fa3d0de777838850cb76ec55ad066e969fd5144/LibriSpeech/dev-clean/1272/128104/1272-128104-0000.flac</td>
      <td>[0.002380371, 0.0020751953, 0.0019836426, 0.002105713, 0.0016174316, 0.00030517578, 9.1552734e-05, 0.00033569336, 0.0009765625, 0.0018310547, 0.0020141602, 0.002105713, 0.001739502, 0.00045776367, -0.00039672852, 0.00045776367, 0.0010070801, 9.1552734e-05, 0.00048828125, 0.001159668, 0.0007324219, 0.0009460449, 0.0018005371, 0.0018310547, 0.00088500977, 0.0004272461, 0.00048828125, 0.0007324219, 0.0010986328, 0.002105713, 0.0025634766, 0.002532959, 0.0025634766, 0.0022888184, 0.0018005371, 0.0010681152, 0.00064086914, 0.00012207031, 0.0002746582, 0.001159668, 0.0015258789, 0.0015563965, 0.0019226074, 0.0012207031, -3.0517578e-05, -0.00036621094, -0.00039672852, -0.00039672852, -0.00015258789, 0.0006713867, 0.0012817383, 0.0018615723, 0.0015869141, 0.0012817383, 0.0007324219, 9.1552734e-05, -0.000579834, -0.00045776367, 9.1552734e-05, 0.00033569336, 0.00024414062, 0.0011291504, 0.001373291, 0.0012817383, 0.00088500977, 0.00030517578, -0.00088500977, -0.0014648438, -0.0008239746, 0.00012207031, 0.0011901855, 0.0019226074, 0.0016479492, 0.00088500977, 0.00076293945, 0.0004272461, -0.0005187988, -0.0005493164, -0.00036621094, -0.0004272461, -0.00018310547, 0.000579834, 0.0009460449, 0.0007324219, 0.0010070801, 0.0007019043, 0.00024414062, -0.00018310547, -0.00064086914, -0.00088500977, -0.00048828125, 0.0002746582, 0.0007324219, 0.0018310547, 0.0018005371, 0.0012512207, 0.00061035156, -0.00036621094, -0.0012817383, -0.00091552734, ...]</td>
      <td>MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES AND WE ARE GLAD TO WELCOME HIS GOSPEL</td>
      <td>1272</td>
      <td>128104</td>
      <td>1272-128104-0000</td>
    </tr>
    <tr>
      <th>path</th>
      <td>/home/innom-dt/.cache/huggingface/datasets/downloads/extracted/aa91addd71e85ab524e5b5b56fa3d0de777838850cb76ec55ad066e969fd5144/LibriSpeech/dev-clean/1272/128104/1272-128104-0000.flac</td>
      <td>/home/innom-dt/.cache/huggingface/datasets/downloads/extracted/aa91addd71e85ab524e5b5b56fa3d0de777838850cb76ec55ad066e969fd5144/LibriSpeech/dev-clean/1272/128104/1272-128104-0000.flac</td>
      <td>MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES AND WE ARE GLAD TO WELCOME HIS GOSPEL</td>
      <td>1272</td>
      <td>128104</td>
      <td>1272-128104-0000</td>
    </tr>
    <tr>
      <th>sampling_rate</th>
      <td>/home/innom-dt/.cache/huggingface/datasets/downloads/extracted/aa91addd71e85ab524e5b5b56fa3d0de777838850cb76ec55ad066e969fd5144/LibriSpeech/dev-clean/1272/128104/1272-128104-0000.flac</td>
      <td>16000</td>
      <td>MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES AND WE ARE GLAD TO WELCOME HIS GOSPEL</td>
      <td>1272</td>
      <td>128104</td>
      <td>1272-128104-0000</td>
    </tr>
  </tbody>
</table>
</div>
**Note:**

* The file column contains the path to the audio sample, and the text column contains the expected transcription.
* We can use the [SoundFile library](https://pysoundfile.readthedocs.io/en/latest/) to read each audio file and convert the audio to an array of floats.

------


```python
import soundfile as sf
```

------

**Add a new column storing each audio sample as an array of floats**


```python
def map_to_array(batch):
    speech, _ = sf.read(batch["file"])
    batch["speech"] = speech
    return batch

ds = ds.map(map_to_array)
```

------

**Play a sample from the dataset**


```python
from IPython.display import Audio

display(Audio(ds[0]['speech'], rate=16000))
```

------


```python
ds.set_format("numpy")
```

------

**Pass the audio sample the pipeline**


```python
pred = asr(ds[0]["speech"])
print(pred)
```
```text
    {'text': 'MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES AND WE ARE GLAD TO WELCOME HIS GOSPEL'}
```

**Note:** 
* The words in the transcription are correct, but the punctuation is missing.
* It is hard to infer punctuation from audio alone, and we could add it in a post-processing step.
* Building a model for a new language still requires a minimum amount of labeled data, which can be challenging to obtain.
* A new method named wav2vec-U combines clever clustering and GAN training to build a speech-to-text model using only independent unlabeled speech and unlabeled text data.
    * This method requires not aligned speech and text data, enabling the training of highly performant speech-to-text models for a much larger spectrum of languages.
    * [Unsupervised Speech Recognition](https://arxiv.org/abs/2105.11084)

------

<img alt="wav2vec-u" width="800" caption="Training scheme for wav2vec-U (courtesy of Alexsei Baevski)" src="../images/notes-transformers-book/chapter-11/chapter11_wav2vec-u.png" id="wav2vec-u"/>

### Vision and Text
* There have been several developments in combining visual and textual information.

#### VQA
* [Making the V in VQA Matter: Elevating the Role of Image Understanding in Visual Question Answering](https://arxiv.org/abs/1612.00837)
* Models such as LXMERT and VisualBERT use vision models like ResNets to extract features from images and then use transformer encoders to combine them with the natural questions and predict and answer.
    * [LXMERT: Learning Cross-Modality Encoder Representations from Transformers](https://arxiv.org/abs/1908.07490)
    * [VisualBERT: A Simple and Performant Baseline for Vision and Language](https://arxiv.org/abs/1908.03557)

#### LayoutLM
* The [LayoutLM](https://huggingface.co/models?search=microsoft+layoutlm) family of models uses an enhanced Transformer architecture that receives a text sequence, an image, and a layout as input.
* There are embedding layers associated with each modality, a spatially-aware self-attention mechanism, and a mix of image and text/image pretraining objectives to align the different modalities.
* LayoutLM models pre-train on millions of scanned documents and can transfer to various downstream tasks, similar to BERT for NLP.
* LayoutLM models are the current state of the art for analyzing scanned business documents like receipts, invoices, or reports.

<img alt="layoutlm" width="500" caption="The model architecture and pretraining strategies for LayoutLMv2 (courtesy of Yang Xu)" src="../images/notes-transformers-book/chapter-11/chapter11_layoutlm.png" id="layoutlm"/> 

#### DALL·E
* DALLE uses the GPT architecture and autoregressive modeling to generate images from text.
* It regards the words and pixels as one sequence of tokens and can, therefore, continue generating an image from a text prompt.
* [Zero-Shot Text-to-Image Generation](https://arxiv.org/abs/2102.12092)

#### CLIP
* [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)
* We can use the pretrained model for classification by embedding the possible classes with the text encoder and comparing the class embeddings to the image embedding that we want to classify.
* We select the class with the highest similarity. 
* CLIP has remarkable zero-shot image classification performance and is competitive with fully supervised-trained vision models while being more flexible.
* We need to instantiate a processor that contains a feature extractor and a tokenizer for image-to-text tasks.
* The feature extractor converts the image into a form suitable for the model, while the tokenizer decodes the model predictions into text.

<img alt="clip-arch" width="800" caption="Architecture of CLIP (courtesy of Alec Radford)" src="../images/notes-transformers-book/chapter-11/chapter11_clip-arch.png" id="clip-arch"/>

------


```python
from transformers import CLIPProcessor, CLIPModel
```

------

#### `CLIPProcessor`

* [Documentation](https://huggingface.co/docs/transformers/main/en/model_doc/clip#transformers.CLIPProcessor)
* Create a CLIP processor which wraps a CLIP feaure extractor and a CLIP tokenizer into a single processor.

#### `CLIPModel`
* [Documentation](https://huggingface.co/docs/transformers/main/en/model_doc/clip#transformers.CLIPModel)

**Instantiate a CLIPModel and processor**


```python
clip_ckpt = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(clip_ckpt)
processor = CLIPProcessor.from_pretrained(clip_ckpt)
```

------

```python
print(f"https://huggingface.co/{clip_ckpt}")
```
```text
    https://huggingface.co/openai/clip-vit-base-patch32
```

------


```python
processor
```
```text
    CLIPProcessor:
    - feature_extractor: CLIPFeatureExtractor {
      "crop_size": 224,
      "do_center_crop": true,
      "do_normalize": true,
      "do_resize": true,
      "feature_extractor_type": "CLIPFeatureExtractor",
      "image_mean": [
        0.48145466,
        0.4578275,
        0.40821073
      ],
      "image_std": [
        0.26862954,
        0.26130258,
        0.27577711
      ],
      "resample": 3,
      "size": 224
    }
    
    - tokenizer: PreTrainedTokenizerFast(name_or_path='openai/clip-vit-base-patch32', vocab_size=49408, model_max_len=77, is_fast=True, padding_side='right', truncation_side='right', special_tokens={'bos_token': AddedToken("<|startoftext|>", rstrip=False, lstrip=False, single_word=False, normalized=True), 'eos_token': AddedToken("<|endoftext|>", rstrip=False, lstrip=False, single_word=False, normalized=True), 'unk_token': AddedToken("<|endoftext|>", rstrip=False, lstrip=False, single_word=False, normalized=True), 'pad_token': '<|endoftext|>'})
```

------

**Load a test image**


```python
image = Image.open("dog.jpg")
plt.imshow(image)
plt.axis("off")
plt.show()
```
![png](../images/notes-transformers-book/chapter-11/output_86_0.png)

------

```python
import torch
```

------

**Create some sample image captions**


```python
texts = ["a photo of a golden retriever", "a photo of a dog", "a photo of agi"]
```

------

**Compare the image to the captions**


```python
inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)
with torch.no_grad():
    outputs = model(**inputs)
logits_per_image = outputs.logits_per_image
probs = logits_per_image.softmax(dim=1)
pd.DataFrame(zip(texts, probs[0].numpy()), columns=['Text', "Probability"])
```


<div style="overflow-x:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Text</th>
      <th>Probability</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>a photo of a golden retriever</td>
      <td>0.868025</td>
    </tr>
    <tr>
      <th>1</th>
      <td>a photo of a dog</td>
      <td>0.131801</td>
    </tr>
    <tr>
      <th>2</th>
      <td>a photo of agi</td>
      <td>0.000174</td>
    </tr>
  </tbody>
</table>
</div>
------



## References

* [Natural Language Processing with Transformers Book](https://transformersbook.com/)
* [The Transformers book GitHub Repository](https://github.com/nlp-with-transformers/notebooks)
