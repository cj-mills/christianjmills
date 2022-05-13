---
title: Notes on fastai Book Ch. 12
layout: post
toc: false
comments: true
description: Chapter 12 covers building and training an LSTM from scratch.
categories: [ai, fastai, notes, pytorch]
hide: false
permalink: /:title/
search_exclude: false
---



* [The Data](#the-data)
* [Our First Language Model from Scratch](#our-first-language-model-from-scratch)
* [Improving the RNN](#improving-the-rnn)
* [Multilayer RNNs](#multilayer-rnns)
* [LSTM](#lstm)
* [Regularizing an LSTM](#regularizing-an-lstm)
* [References](#references)

-----

```python
#hide
# !pip install -Uqq fastbook
import fastbook
fastbook.setup_book()
```


```python
from fastbook import *
```


```python
import inspect
def print_source(obj):
    for line in inspect.getsource(obj).split("\n"):
        print(line)
```

## A Language Model from Scratch

## The Data
* try to think of the simplest useable dataset when starting on a new problem
* the starter dataset should allow you to quickly and easily try out methods and interpret the results
* one of the most common practical mistakes is failing to use appropriate datasets at appropriate times during the analysis process
    * most people tend to start with datasets that are too big and too complicated


```python
from fastai.text.all import *
```

#### fastai Human Numbers Dataset
* A synthetic dataset consisting of human number counts in text such as one, two, three, four.. 
* Useful for experimenting with Language Models


```python
URLs.HUMAN_NUMBERS
```
```text
'https://s3.amazonaws.com/fast-ai-sample/human_numbers.tgz'
```




```python
path = untar_data(URLs.HUMAN_NUMBERS)
path
```
```text
Path('/home/innom-dt/.fastai/data/human_numbers')
```




```python
path.ls()
```
```text
(#2) [Path('/home/innom-dt/.fastai/data/human_numbers/train.txt'),Path('/home/innom-dt/.fastai/data/human_numbers/valid.txt')]
```




```python
train_file = path.ls()[0]
```


```python
cat $train_file | head -5
```
```text
one 
two 
three 
four 
five 
cat: write error: Broken pipe
```



```python
valid_file = path.ls()[1]
```


```python
cat $valid_file | head -5
```
```text
eight thousand one 
eight thousand two 
eight thousand three 
eight thousand four 
eight thousand five 
cat: write error: Broken pipe
```



```python
lines = L()
# Combine the training and validation sets into a single List
with open(path/'train.txt') as f: lines += L(*f.readlines())
with open(path/'valid.txt') as f: lines += L(*f.readlines())
lines
```
```text
(#9998) ['one \n','two \n','three \n','four \n','five \n','six \n','seven \n','eight \n','nine \n','ten \n'...]
```




```python
# Remove the '\n' new line characters and separate the words with a '.'
text = ' . '.join([l.strip() for l in lines])
text[:100]
```
```text
'one . two . three . four . five . six . seven . eight . nine . ten . eleven . twelve . thirteen . fo'
```




```python
# Separate the words into a list
tokens = text.split(' ')
tokens[:10]
```
```text
['one', '.', 'two', '.', 'three', '.', 'four', '.', 'five', '.']
```




```python
# Generate unique vocab
vocab = L(*tokens).unique()
vocab
```
```text
(#30) ['one','.','two','three','four','five','six','seven','eight','nine'...]
```




```python
pd.DataFrame(list(vocab))
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
      <td>one</td>
    </tr>
    <tr>
      <th>1</th>
      <td>.</td>
    </tr>
    <tr>
      <th>2</th>
      <td>two</td>
    </tr>
    <tr>
      <th>3</th>
      <td>three</td>
    </tr>
    <tr>
      <th>4</th>
      <td>four</td>
    </tr>
    <tr>
      <th>5</th>
      <td>five</td>
    </tr>
    <tr>
      <th>6</th>
      <td>six</td>
    </tr>
    <tr>
      <th>7</th>
      <td>seven</td>
    </tr>
    <tr>
      <th>8</th>
      <td>eight</td>
    </tr>
    <tr>
      <th>9</th>
      <td>nine</td>
    </tr>
    <tr>
      <th>10</th>
      <td>ten</td>
    </tr>
    <tr>
      <th>11</th>
      <td>eleven</td>
    </tr>
    <tr>
      <th>12</th>
      <td>twelve</td>
    </tr>
    <tr>
      <th>13</th>
      <td>thirteen</td>
    </tr>
    <tr>
      <th>14</th>
      <td>fourteen</td>
    </tr>
    <tr>
      <th>15</th>
      <td>fifteen</td>
    </tr>
    <tr>
      <th>16</th>
      <td>sixteen</td>
    </tr>
    <tr>
      <th>17</th>
      <td>seventeen</td>
    </tr>
    <tr>
      <th>18</th>
      <td>eighteen</td>
    </tr>
    <tr>
      <th>19</th>
      <td>nineteen</td>
    </tr>
    <tr>
      <th>20</th>
      <td>twenty</td>
    </tr>
    <tr>
      <th>21</th>
      <td>thirty</td>
    </tr>
    <tr>
      <th>22</th>
      <td>forty</td>
    </tr>
    <tr>
      <th>23</th>
      <td>fifty</td>
    </tr>
    <tr>
      <th>24</th>
      <td>sixty</td>
    </tr>
    <tr>
      <th>25</th>
      <td>seventy</td>
    </tr>
    <tr>
      <th>26</th>
      <td>eighty</td>
    </tr>
    <tr>
      <th>27</th>
      <td>ninety</td>
    </tr>
    <tr>
      <th>28</th>
      <td>hundred</td>
    </tr>
    <tr>
      <th>29</th>
      <td>thousand</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Map words to their vocab indices
word2idx = {w:i for i,w in enumerate(vocab)}
# Numericalize dataset
nums = L(word2idx[i] for i in tokens)
nums
```
```text
(#63095) [0,1,2,1,3,1,4,1,5,1...]
```



## Our First Language Model from Scratch


```python
# Create a list of (input, target) tuples
# input: the previous three words
# target: the next word
L((tokens[i:i+3], tokens[i+3]) for i in range(0,len(tokens)-4,3))
```
```text
(#21031) [(['one', '.', 'two'], '.'),(['.', 'three', '.'], 'four'),(['four', '.', 'five'], '.'),(['.', 'six', '.'], 'seven'),(['seven', '.', 'eight'], '.'),(['.', 'nine', '.'], 'ten'),(['ten', '.', 'eleven'], '.'),(['.', 'twelve', '.'], 'thirteen'),(['thirteen', '.', 'fourteen'], '.'),(['.', 'fifteen', '.'], 'sixteen')...]
```




```python
# # Create a list of (input, target) tuples
# input: a tensor containing the numericalized forms of previous three words
# target: the numericalized form of the next word
seqs = L((tensor(nums[i:i+3]), nums[i+3]) for i in range(0,len(nums)-4,3))
seqs
```
```text
(#21031) [(tensor([0, 1, 2]), 1),(tensor([1, 3, 1]), 4),(tensor([4, 1, 5]), 1),(tensor([1, 6, 1]), 7),(tensor([7, 1, 8]), 1),(tensor([1, 9, 1]), 10),(tensor([10,  1, 11]), 1),(tensor([ 1, 12,  1]), 13),(tensor([13,  1, 14]), 1),(tensor([ 1, 15,  1]), 16)...]
```




```python
DataLoaders.from_dsets
```
```text
<bound method DataLoaders.from_dsets of <class 'fastai.data.core.DataLoaders'>>
```




```python
print_source(DataLoaders.from_dsets)
```
```text
    @classmethod
    def from_dsets(cls, *ds, path='.',  bs=64, device=None, dl_type=TfmdDL, **kwargs):
        default = (True,) + (False,) * (len(ds)-1)
        defaults = {'shuffle': default, 'drop_last': default}
        tfms = {k:tuple(Pipeline(kwargs[k]) for i in range_of(ds)) for k in _batch_tfms if k in kwargs}
        kwargs = merge(defaults, {k: tuplify(v, match=ds) for k,v in kwargs.items() if k not in _batch_tfms}, tfms)
        kwargs = [{k: v[i] for k,v in kwargs.items()} for i in range_of(ds)]
        return cls(*[dl_type(d, bs=bs, **k) for d,k in zip(ds, kwargs)], path=path, device=device)
```




```python
bs = 64
# Split data between train and valid 80/20
cut = int(len(seqs) * 0.8)
dls = DataLoaders.from_dsets(seqs[:cut], seqs[cut:], bs=64, shuffle=False)
```


```python
dls.one_batch()[0].shape, dls.one_batch()[1].shape
```
```text
(torch.Size([64, 3]), torch.Size([64]))
```




```python
dls.one_batch()[0][0], dls.one_batch()[1][0]
```
```text
(tensor([0, 1, 2]), tensor(1))
```



### Our Language Model in PyTorch
* Every word is interpreted in the information context of any words preceding it


```python
class LMModel1(Module):
    def __init__(self, vocab_sz, n_hidden):
        # Input to hidden
        self.i_h = nn.Embedding(vocab_sz, n_hidden)
        # Hidden to hidden
        self.h_h = nn.Linear(n_hidden, n_hidden)     
        # Hidden to output
        self.h_o = nn.Linear(n_hidden,vocab_sz)
        
    def forward(self, x):
        # First input word
        # Pass embedding for first word to first linear layer
        h = F.relu(self.h_h(self.i_h(x[:,0])))
        # Second input word
        # Add embedding for second word to previous output
        h = h + self.i_h(x[:,1])
        # Pass to first linear layer
        h = F.relu(self.h_h(h))
        # Third input word
        # Add embeddingfor third word to previous output
        h = h + self.i_h(x[:,2])
        # Pass to first linear layer
        h = F.relu(self.h_h(h))
        # Pass output to second linear layer
        return self.h_o(h)
```


```python
learn = Learner(dls, LMModel1(len(vocab), 64), loss_func=F.cross_entropy, 
                metrics=accuracy)
learn.fit_one_cycle(4, 1e-3)
```
<div style="overflow-x:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1.813438</td>
      <td>1.944979</td>
      <td>0.466603</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1.405106</td>
      <td>1.702907</td>
      <td>0.474447</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1.427549</td>
      <td>1.650981</td>
      <td>0.489898</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>3</td>
      <td>1.380016</td>
      <td>1.685956</td>
      <td>0.470882</td>
      <td>00:01</td>
    </tr>
  </tbody>
</table>
</div>


```python
range_of
```
```text
<function fastcore.basics.range_of(a, b=None, step=None)>
```




```python
print_source(range_of)
```
```text
def range_of(a, b=None, step=None):
    "All indices of collection `a`, if `a` is a collection, otherwise `range`"
    if is_coll(a): a = len(a)
    return list(range(a,b,step) if step is not None else range(a,b) if b is not None else range(a))
```




```python
# Get the number of occurrences of each unique vocab item in the validation set
n,counts = 0,torch.zeros(len(vocab))
n, counts
for x,y in dls.valid:
    n += y.shape[0]
    # Keep track of 
    for i in range_of(vocab): counts[i] += (y==i).long().sum()
```


```python
# Get the index for the most common token in the validation set
idx = torch.argmax(counts)
# Print the most common index
(idx, 
# Print the corresponding word for the index
vocab[idx.item()], 
# Calculate the likelihood of randomly picking the most common word
counts[idx].item()/n)
```
```text
(tensor(29), 'thousand', 0.15165200855716662)
```



**Note:** This indicates the model is performing much better than picking a word at random.

### Our First Recurrent Neural Network (a.k.a A Looping Network)
* replace the hardcoded forward function in the LMModel1 with a for loop

#### Hidden State:
* the activations that are updated at each step of a recurrent neural network


```python
class LMModel2(Module):
    def __init__(self, vocab_sz, n_hidden):
        self.i_h = nn.Embedding(vocab_sz, n_hidden)  
        self.h_h = nn.Linear(n_hidden, n_hidden)     
        self.h_o = nn.Linear(n_hidden,vocab_sz)
        
    def forward(self, x):
        # Initialize the hidden state
        h = 0
        for i in range(3):
            # Update the hidden state
            h = h + self.i_h(x[:,i])
            h = F.relu(self.h_h(h))
        return self.h_o(h)
```


```python
learn = Learner(dls, LMModel2(len(vocab), 64), loss_func=F.cross_entropy, 
                metrics=accuracy)
learn.fit_one_cycle(4, 1e-3)
```
<div style="overflow-x:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1.790029</td>
      <td>1.993387</td>
      <td>0.463038</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1.389832</td>
      <td>1.836371</td>
      <td>0.466603</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1.422808</td>
      <td>1.669952</td>
      <td>0.487045</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>3</td>
      <td>1.380381</td>
      <td>1.681706</td>
      <td>0.459472</td>
      <td>00:01</td>
    </tr>
  </tbody>
</table>
</div>


## Improving the RNN

* the above LMModel2 version resets the hidden state for every new input sequence
    * throwing away all the information we have about the sentences we have seen so far
* the above LMModel2 version only tries to predict the fourth word

### Maintaining the State of an RNN


```python
class LMModel3(Module):
    def __init__(self, vocab_sz, n_hidden):
        self.i_h = nn.Embedding(vocab_sz, n_hidden)  
        self.h_h = nn.Linear(n_hidden, n_hidden)     
        self.h_o = nn.Linear(n_hidden,vocab_sz)
        # Maintain the same hidden state across input sequences
        self.h = 0
        
    def forward(self, x):
        for i in range(3):
            self.h = self.h + self.i_h(x[:,i])
            self.h = F.relu(self.h_h(self.h))
        out = self.h_o(self.h)
        # Detach the hidden state from the pytorch computation graph
        self.h = self.h.detach()
        return out
    
    def reset(self): self.h = 0
```

#### Backpropogation Through Time (BPTT)
* Treating a neural net with effectively one layer per time step (usually refactored using a loop) as one big model, and calculating gradients on it in the usual way
* usually use Truncated BPTT which detaches the history of computation steps in the hidden state every few time steps.


```python
m = len(seqs)//bs
m,bs,len(seqs)
```
```text
(328, 64, 21031)
```




```python
def group_chunks(ds, bs):
    # Calculate the number of groups
    m = len(ds) // bs
    # Initialize new dataset container
    new_ds = L()
    # Group dataset into chunks
    for i in range(m): new_ds += L(ds[i + m*j] for j in range(bs))
    return new_ds
```


```python
# Split dataset 80/20 into training and validation
cut = int(len(seqs) * 0.8)
dls = DataLoaders.from_dsets(
    group_chunks(seqs[:cut], bs), 
    group_chunks(seqs[cut:], bs), 
    bs=bs, 
    # Drop the last batch that does not have the shape of bs
    drop_last=True, 
    shuffle=False)
```


```python
ModelResetter
```
```text
fastai.callback.rnn.ModelResetter
```




```python
print_source(ModelResetter)
```
```text
@docs
class ModelResetter(Callback):
    "`Callback` that resets the model at each validation/training step"
    def before_train(self):    self.model.reset()
    def before_validate(self): self.model.reset()
    def after_fit(self):       self.model.reset()
    _docs = dict(before_train="Reset the model before training",
                 before_validate="Reset the model before validation",
                 after_fit="Reset the model after fitting")
```



#### fastai Callbacks
* [Documentation](https://docs.fast.ai/callback.core.html#Callback)
* `after_create:` called after the Learner is created
* `before_fit:` called before starting training or inference, ideal for initial setup.
* `before_epoch:` called at the beginning of each epoch, useful for any behavior you need to reset at each epoch.
* `before_train:` called at the beginning of the training part of an epoch.
* `before_batch:` called at the beginning of each batch, just after drawing said batch. It can be used to do any setup necessary for the batch (like hyper-parameter scheduling) or to change the input/target before it goes in the model (change of the input with techniques like mixup for instance).
* `after_pred:` called after computing the output of the model on the batch. It can be used to change that output before it's fed to the loss.
* `after_loss:` called after the loss has been computed, but before the backward pass. It can be used to add any penalty to the loss (AR or TAR in RNN training for instance).
* `before_backward:` called after the loss has been computed, but only in training mode (i.e. when the backward pass will be used)
* `before_step:` called after the backward pass, but before the update of the parameters. It can be used to do any change to the gradients before said update (gradient clipping for instance).
* `after_step:` called after the step and before the gradients are zeroed.
* `after_batch:` called at the end of a batch, for any clean-up before the next one.
* `after_train:` called at the end of the training phase of an epoch.
* `before_validate:` called at the beginning of the validation phase of an epoch, useful for any setup needed specifically for validation.
* `after_validate:` called at the end of the validation part of an epoch.
* `after_epoch:` called at the end of an epoch, for any clean-up before the next one.
* `after_fit:` called at the end of training, for final clean-up.



```python
Callback
```
```text
fastai.callback.core.Callback
```




```python
print_source(Callback)
```
```text
@funcs_kwargs(as_method=True)
class Callback(Stateful,GetAttr):
    "Basic class handling tweaks of the training loop by changing a `Learner` in various events"
    order,_default,learn,run,run_train,run_valid = 0,'learn',None,True,True,True
    _methods = _events

    def __init__(self, **kwargs): assert not kwargs, f'Passed unknown events: {kwargs}'
    def __repr__(self): return type(self).__name__

    def __call__(self, event_name):
        "Call `self.{event_name}` if it's defined"
        _run = (event_name not in _inner_loop or (self.run_train and getattr(self, 'training', True)) or
               (self.run_valid and not getattr(self, 'training', False)))
        res = None
        if self.run and _run: res = getattr(self, event_name, noop)()
        if event_name=='after_fit': self.run=True #Reset self.run to True at each end of fit
        return res

    def __setattr__(self, name, value):
        if hasattr(self.learn,name):
            warn(f"You are shadowing an attribute ({name}) that exists in the learner. Use `self.learn.{name}` to avoid this")
        super().__setattr__(name, value)

    @property
    def name(self):
        "Name of the `Callback`, camel-cased and with '*Callback*' removed"
        return class2attr(self, 'Callback')
```




```python
learn = Learner(dls, 
                LMModel3(len(vocab), 64), 
                loss_func=F.cross_entropy,
                metrics=accuracy, 
                # reset the model at the beginning of each epoch and before each validation phase
                cbs=ModelResetter)
learn.fit_one_cycle(10, 3e-3)
```
<div style="overflow-x:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1.695570</td>
      <td>1.837262</td>
      <td>0.474519</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1.316114</td>
      <td>1.939660</td>
      <td>0.366346</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1.102734</td>
      <td>1.578932</td>
      <td>0.469471</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>3</td>
      <td>1.017313</td>
      <td>1.470766</td>
      <td>0.552163</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.961458</td>
      <td>1.568437</td>
      <td>0.551923</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.920572</td>
      <td>1.632755</td>
      <td>0.574519</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.932616</td>
      <td>1.634864</td>
      <td>0.588221</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.848161</td>
      <td>1.668468</td>
      <td>0.587500</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.802442</td>
      <td>1.698610</td>
      <td>0.591827</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.794550</td>
      <td>1.716233</td>
      <td>0.594952</td>
      <td>00:01</td>
    </tr>
  </tbody>
</table>
</div>

### Creating More Signal
* we can increase the amount of signal for updating the model weights by predicting the next word after every single word, rather than every three words


```python
# Define the sequence length
sl = 16
# Update the dependent variable to include each of the words 
# that follow each of the words in the independent variable
seqs = L((tensor(nums[i:i+sl]), tensor(nums[i+1:i+sl+1]))
         for i in range(0,len(nums)-sl-1,sl))
# Define the split for the training and validation set
cut = int(len(seqs) * 0.8)
dls = DataLoaders.from_dsets(group_chunks(seqs[:cut], bs),
                             group_chunks(seqs[cut:], bs),
                             bs=bs, drop_last=True, shuffle=False)
```


```python
[L(vocab[o] for o in s) for s in seqs[0]]
```
```text
[(#16) ['one','.','two','.','three','.','four','.','five','.'...],
 (#16) ['.','two','.','three','.','four','.','five','.','six'...]]
```




```python
class LMModel4(Module):
    def __init__(self, vocab_sz, n_hidden):
        self.i_h = nn.Embedding(vocab_sz, n_hidden)  
        self.h_h = nn.Linear(n_hidden, n_hidden)     
        self.h_o = nn.Linear(n_hidden,vocab_sz)
        self.h = 0
        
    def forward(self, x):
        outs = []
        for i in range(sl):
            self.h = self.h + self.i_h(x[:,i])
            self.h = F.relu(self.h_h(self.h))
            # Store the output for each word in the current sequence
            outs.append(self.h_o(self.h))
        self.h = self.h.detach()
        # stack the output for each word in the current sequence
        return torch.stack(outs, dim=1)
    
    def reset(self): self.h = 0
```


```python
# Define custom loss function that flattens the output before calculating cross entropy
def loss_func(inp, targ):
    return F.cross_entropy(inp.view(-1, len(vocab)), targ.view(-1))
```


```python
learn = Learner(dls, LMModel4(len(vocab), 64), loss_func=loss_func,
                metrics=accuracy, cbs=ModelResetter)
learn.fit_one_cycle(15, 3e-3)
```
<div style="overflow-x:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>3.229987</td>
      <td>3.069768</td>
      <td>0.249756</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2.291759</td>
      <td>1.903835</td>
      <td>0.468018</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1.719411</td>
      <td>1.769336</td>
      <td>0.469157</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>3</td>
      <td>1.444394</td>
      <td>1.729377</td>
      <td>0.459554</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>4</td>
      <td>1.273674</td>
      <td>1.625678</td>
      <td>0.531169</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>5</td>
      <td>1.141202</td>
      <td>1.762818</td>
      <td>0.545898</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>6</td>
      <td>1.037926</td>
      <td>1.575556</td>
      <td>0.573812</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.939284</td>
      <td>1.470020</td>
      <td>0.614095</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.858596</td>
      <td>1.532887</td>
      <td>0.628255</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.784250</td>
      <td>1.495697</td>
      <td>0.655843</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>10</td>
      <td>0.739764</td>
      <td>1.539676</td>
      <td>0.666423</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>11</td>
      <td>0.693413</td>
      <td>1.550242</td>
      <td>0.662191</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>12</td>
      <td>0.661127</td>
      <td>1.519285</td>
      <td>0.680908</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>13</td>
      <td>0.635551</td>
      <td>1.523878</td>
      <td>0.676921</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>14</td>
      <td>0.621697</td>
      <td>1.531653</td>
      <td>0.684408</td>
      <td>00:00</td>
    </tr>
  </tbody>
</table>
</div>


## Multilayer RNNs

* pass the activations from one RNN into another RNN

### The Model


```python
class LMModel5(Module):
    def __init__(self, vocab_sz, n_hidden, n_layers):
        self.i_h = nn.Embedding(vocab_sz, n_hidden)
        self.rnn = nn.RNN(n_hidden, n_hidden, n_layers, batch_first=True)
        self.h_o = nn.Linear(n_hidden, vocab_sz)
        self.h = torch.zeros(n_layers, bs, n_hidden)
        
    def forward(self, x):
        res,h = self.rnn(self.i_h(x), self.h)
        self.h = h.detach()
        return self.h_o(res)
    
    def reset(self): self.h.zero_()
```


```python
LMModel5(len(vocab), 64, 2)
```
```text
LMModel5(
  (i_h): Embedding(30, 64)
  (rnn): RNN(64, 64, num_layers=2, batch_first=True)
  (h_o): Linear(in_features=64, out_features=30, bias=True)
)
```




```python
learn = Learner(dls, LMModel5(len(vocab), 64, 2), 
                loss_func=CrossEntropyLossFlat(), 
                metrics=accuracy, cbs=ModelResetter)
learn.fit_one_cycle(15, 3e-3)
```
<div style="overflow-x:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>3.070420</td>
      <td>2.586252</td>
      <td>0.460775</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2.154392</td>
      <td>1.760734</td>
      <td>0.471680</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1.709090</td>
      <td>1.851027</td>
      <td>0.327311</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>3</td>
      <td>1.523287</td>
      <td>1.790196</td>
      <td>0.412028</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>4</td>
      <td>1.364664</td>
      <td>1.816422</td>
      <td>0.468262</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>5</td>
      <td>1.247051</td>
      <td>1.796951</td>
      <td>0.493001</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>6</td>
      <td>1.156087</td>
      <td>1.907447</td>
      <td>0.489095</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>7</td>
      <td>1.073325</td>
      <td>2.014389</td>
      <td>0.499268</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.995001</td>
      <td>2.056770</td>
      <td>0.501139</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.927453</td>
      <td>2.080244</td>
      <td>0.503743</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>10</td>
      <td>0.874861</td>
      <td>2.084781</td>
      <td>0.502441</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>11</td>
      <td>0.837194</td>
      <td>2.102611</td>
      <td>0.514974</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>12</td>
      <td>0.812340</td>
      <td>2.111124</td>
      <td>0.512126</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>13</td>
      <td>0.797198</td>
      <td>2.110253</td>
      <td>0.513346</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>14</td>
      <td>0.789102</td>
      <td>2.108808</td>
      <td>0.513997</td>
      <td>00:00</td>
    </tr>
  </tbody>
</table>
</div>

**Note:** The multi-layer RNN performs worse than the single-layer RNN

### Exploding or Disappearing Activations
* deeper models are more difficult to train
    * performing matrix multiplication so many times can cause numbers to get extremely big or extremely small
    * floating point numbers get less accurate the further away they get from zero
* [What you never wanted to know about floating point but will be forced to find out](https://www.volkerschatz.com/science/float.html)

* Two types of layers are frequently used to avoid exploding activations in RNNs
    1. Gated Recurrent Units (GRUs)
    2. Long short-term memory (LSTM)



## LSTM

* introduced in 1997 by [Jürgen Schmidhuber](https://en.wikipedia.org/wiki/J%C3%BCrgen_Schmidhuber) and [Sepp Hochreiter](https://en.wikipedia.org/wiki/Sepp_Hochreiter)
* Normal RNNs are realy bad at retaining memory of what happened much earlier in a sentence
* LSTMs maintain two hidden states to address this
    * cell state: responsible for keeping long short-term memory
    * hidden state: focuses on the next token to predict

![lstm-cell](https://upload.wikimedia.org/wikipedia/commons/1/17/The_LSTM_Cell.svg)

* $x_{t}$ the current input
* $(h_{t-1})$: the previous hidden state
* $(c_{t-1})$: the previous hidden state
* $\sigma$: sigmoid function
* $tanh$: a sigmoid function rescaled to the range $[-1,1]$
* $tanh(x) = \frac{e^{x}+e^{-x}}{e^{x}-e^{-x}} = 2\sigma(2x)-1$
* four neural nets (orange) called gates (left to right):
    1. **forget gate:** a linear layer followed by a sigmoid (i.e. output will be scalars [0,1])
        * multipy output by cell state to determine which information to keep
        * gives the LSTM the ability to forget things about its long-term state
    2. **input gate:** works with the third gate (`tanh`) to update the cell state
        * decided which elements of the cell state to updates (values close to 1)
    3. **cell gate:** determines what the updated values are for the cell state
    4. **output gate:** determines which information from the cell state to use to generate the new hidden state


```python
2*torch.sigmoid(2*tensor(0.5)) - 1
```
```text
tensor(0.4621)
```




```python
torch.tanh(tensor(0.5))
```
```text
tensor(0.4621)
```



### Building an LSTM from Scratch


```python
class LSTMCell(Module):
    def __init__(self, ni, nh):
        self.forget_gate = nn.Linear(ni + nh, nh)
        self.input_gate  = nn.Linear(ni + nh, nh)
        self.cell_gate   = nn.Linear(ni + nh, nh)
        self.output_gate = nn.Linear(ni + nh, nh)

    def forward(self, input, state):
        h,c = state
        h = torch.cat([h, input], dim=1)
        forget = torch.sigmoid(self.forget_gate(h))
        c = c * forget
        inp = torch.sigmoid(self.input_gate(h))
        cell = torch.tanh(self.cell_gate(h))
        c = c + inp * cell
        out = torch.sigmoid(self.output_gate(h))
        h = out * torch.tanh(c)
        return h, (h,c)
```

**Note:** It is better for performance reasons to do one big matrix multiplication than four smaller ones
    * launch the special fast kernel on the GPU only once
    * give the GPU more work to do in parallel


```python
class LSTMCell(Module):
    def __init__(self, ni, nh):
        self.ih = nn.Linear(ni,4*nh)
        self.hh = nn.Linear(nh,4*nh)

    def forward(self, input, state):
        h,c = state
        # One big multiplication for all the gates is better than 4 smaller ones
        gates = (self.ih(input) + self.hh(h)).chunk(4, 1)
        ingate,forgetgate,outgate = map(torch.sigmoid, gates[:3])
        cellgate = gates[3].tanh()

        c = (forgetgate*c) + (ingate*cellgate)
        h = outgate * c.tanh()
        return h, (h,c)
```

#### torch chunk
* [Documentation](https://pytorch.org/docs/stable/generated/torch.chunk.html)


```python
help(torch.chunk)
```
```text
Help on built-in function chunk:

chunk(...)
    chunk(input, chunks, dim=0) -> List of Tensors
    
    Attempts to split a tensor into the specified number of chunks. Each chunk is a view of
    the input tensor.
        
    .. note::
        This function may return less then the specified number of chunks!
    
    .. seealso::
    
        :func:`torch.tensor_split` a function that always returns exactly the specified number of chunks
    
    If the tensor size along the given dimesion :attr:`dim` is divisible by :attr:`chunks`,
    all returned chunks will be the same size.
    If the tensor size along the given dimension :attr:`dim` is not divisible by :attr:`chunks`,
    all returned chunks will be the same size, except the last one.
    If such division is not possible, this function may return less
    than the specified number of chunks.
    
    Arguments:
        input (Tensor): the tensor to split
        chunks (int): number of chunks to return
        dim (int): dimension along which to split the tensor
    
    Example::
        >>> torch.arange(11).chunk(6)
        (tensor([0, 1]),
         tensor([2, 3]),
         tensor([4, 5]),
         tensor([6, 7]),
         tensor([8, 9]),
         tensor([10]))
        >>> torch.arange(12).chunk(6)
        (tensor([0, 1]),
         tensor([2, 3]),
         tensor([4, 5]),
         tensor([6, 7]),
         tensor([8, 9]),
         tensor([10, 11]))
        >>> torch.arange(13).chunk(6)
        (tensor([0, 1, 2]),
         tensor([3, 4, 5]),
         tensor([6, 7, 8]),
         tensor([ 9, 10, 11]),
         tensor([12]))
```




```python
t = torch.arange(0,10); t
```
```text
tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
```




```python
t.chunk(2)
```
```text
(tensor([0, 1, 2, 3, 4]), tensor([5, 6, 7, 8, 9]))
```



### Training a Language Model Using LSTMs


```python
class LMModel6(Module):
    def __init__(self, vocab_sz, n_hidden, n_layers):
        self.i_h = nn.Embedding(vocab_sz, n_hidden)
        self.rnn = nn.LSTM(n_hidden, n_hidden, n_layers, batch_first=True)
        self.h_o = nn.Linear(n_hidden, vocab_sz)
        self.h = [torch.zeros(n_layers, bs, n_hidden) for _ in range(2)]
        
    def forward(self, x):
        res,h = self.rnn(self.i_h(x), self.h)
        self.h = [h_.detach() for h_ in h]
        return self.h_o(res)
    
    def reset(self): 
        for h in self.h: h.zero_()
```


```python
# Using a two-layer LSTM
learn = Learner(dls, LMModel6(len(vocab), 64, 2), 
                loss_func=CrossEntropyLossFlat(), 
                metrics=accuracy, cbs=ModelResetter)
learn.model
```
```text
LMModel6(
  (i_h): Embedding(30, 64)
  (rnn): LSTM(64, 64, num_layers=2, batch_first=True)
  (h_o): Linear(in_features=64, out_features=30, bias=True)
)
```




```python
learn.fit_one_cycle(15, 1e-2)
```
<div style="overflow-x:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>3.013088</td>
      <td>2.705310</td>
      <td>0.417074</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2.215323</td>
      <td>1.904673</td>
      <td>0.406657</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1.622977</td>
      <td>1.772446</td>
      <td>0.438232</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>3</td>
      <td>1.319893</td>
      <td>1.853711</td>
      <td>0.519613</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>4</td>
      <td>1.096065</td>
      <td>1.868788</td>
      <td>0.554118</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.872888</td>
      <td>1.679482</td>
      <td>0.609375</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.590291</td>
      <td>1.355017</td>
      <td>0.661458</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.385917</td>
      <td>1.319989</td>
      <td>0.667887</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.284691</td>
      <td>1.221118</td>
      <td>0.689290</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.228731</td>
      <td>1.181922</td>
      <td>0.730632</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>10</td>
      <td>0.172228</td>
      <td>1.250237</td>
      <td>0.727946</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>11</td>
      <td>0.124468</td>
      <td>1.155407</td>
      <td>0.754720</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>12</td>
      <td>0.090831</td>
      <td>1.183195</td>
      <td>0.749674</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>13</td>
      <td>0.071399</td>
      <td>1.179867</td>
      <td>0.750081</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>14</td>
      <td>0.061995</td>
      <td>1.168421</td>
      <td>0.753499</td>
      <td>00:01</td>
    </tr>
  </tbody>
</table>
</div>

**Note:** We were able to use a higher learning rate and achieve a much higher accuracy than the multi-layer RNN.
**Note:** There is still some overfitting.



## Regularizing an LSTM

* [Regularizing and Optimizing LSTM Language Models](https://arxiv.org/abs/1708.02182)
    * used an LSTM with dropout, activation regularization, and temporal activation regularization to beat state-of-the-art results that previously required much more complicated models
    * called the combination an **AWD-LSTM**

### Dropout
* [Improving neural networks by preventing co-adaptation of feature detectors](https://arxiv.org/abs/1207.0580)
* [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](https://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf)
* randomly change some activations to zero at training time
* makes activations more noisy
* makes sure all neurons actively work toward the output
* makes the model more robust
* need to rescale activations after applying dropout
    * divide activations by $1-p$ where p is the probability to keep an activation
* using dropout before passing the output of our LSTM to the final layer will help reduce overfitting
* make sure to turn off dropout during inference


```python
class Dropout(Module):
    def __init__(self, p): self.p = p
    def forward(self, x):
        if not self.training: return x
        mask = x.new(*x.shape).bernoulli_(1-p)
        return x * mask.div_(1-p)
```


```python
help(torch.bernoulli)
```
```text
Help on built-in function bernoulli:

bernoulli(...)
    bernoulli(input, *, generator=None, out=None) -> Tensor
    
    Draws binary random numbers (0 or 1) from a Bernoulli distribution.
    
    The :attr:`input` tensor should be a tensor containing probabilities
    to be used for drawing the binary random number.
    Hence, all values in :attr:`input` have to be in the range:
    :math:`0 \leq \text{input}_i \leq 1`.
    
    The :math:`\text{i}^{th}` element of the output tensor will draw a
    value :math:`1` according to the :math:`\text{i}^{th}` probability value given
    in :attr:`input`.
    
    .. math::
        \text{out}_{i} \sim \mathrm{Bernoulli}(p = \text{input}_{i})
    
    The returned :attr:`out` tensor only has values 0 or 1 and is of the same
    shape as :attr:`input`.
    
    :attr:`out` can have integral ``dtype``, but :attr:`input` must have floating
    point ``dtype``.
    
    Args:
        input (Tensor): the input tensor of probability values for the Bernoulli distribution
    
    Keyword args:
        generator (:class:`torch.Generator`, optional): a pseudorandom number generator for sampling
        out (Tensor, optional): the output tensor.
    
    Example::
    
        >>> a = torch.empty(3, 3).uniform_(0, 1)  # generate a uniform random matrix with range [0, 1]
        >>> a
        tensor([[ 0.1737,  0.0950,  0.3609],
                [ 0.7148,  0.0289,  0.2676],
                [ 0.9456,  0.8937,  0.7202]])
        >>> torch.bernoulli(a)
        tensor([[ 1.,  0.,  0.],
                [ 0.,  0.,  0.],
                [ 1.,  1.,  1.]])
    
        >>> a = torch.ones(3, 3) # probability of drawing "1" is 1
        >>> torch.bernoulli(a)
        tensor([[ 1.,  1.,  1.],
                [ 1.,  1.,  1.],
                [ 1.,  1.,  1.]])
        >>> a = torch.zeros(3, 3) # probability of drawing "1" is 0
        >>> torch.bernoulli(a)
        tensor([[ 0.,  0.,  0.],
                [ 0.,  0.,  0.],
                [ 0.,  0.,  0.]])
```



### Activation Regularization and Temporal Activation Regularization
* both are similar to weight decay (AR)
* activation regularization: try to make the final activations produced by the LSTM as small as possible
    * `loss += alpha * activations.pow(2).mean()`
    * often applied on dropped-out activations to not penalize the activations set to zero
* temporal activation regularization (TAR)
    * linked to the fact we are predicting tokens in a sentence
    * the outputs of our LSTMs should somewhat make sense when we read them in order
    * TAR encourages that behavior by adding a penalty to the loss to make the difference between two consecutive activations as small as possible
    * `loss += beta * (activations[:,1:] - activations[:,:-1]).pow(2).mean()`
        * alpha and beta are tunable hyperparameters
    * applied to non-dropped-out activations (because the zeros in the dropped-out activations create big differences)

### Training a Weight-Tied Regularized LSTM
* need to return the normal output from the LSTM, the dropped-out activations, and the activations from the LSTMs

#### Weight Tying
* in a language model, the input embeddings represent a mapping from English words to activations and the output hidden layer represents a mapping from activations to English words
    * these mappings could be the same
* introduced in AWD-LSTM paper
* `self.h_o.weight = self.i_h.weight`


```python
class LMModel7(Module):
    def __init__(self, vocab_sz, n_hidden, n_layers, p):
        self.i_h = nn.Embedding(vocab_sz, n_hidden)
        self.rnn = nn.LSTM(n_hidden, n_hidden, n_layers, batch_first=True)
        self.drop = nn.Dropout(p)
        self.h_o = nn.Linear(n_hidden, vocab_sz)
        self.h_o.weight = self.i_h.weight
        self.h = [torch.zeros(n_layers, bs, n_hidden) for _ in range(2)]
        
    def forward(self, x):
        raw,h = self.rnn(self.i_h(x), self.h)
        out = self.drop(raw)
        self.h = [h_.detach() for h_ in h]
        return self.h_o(out),raw,out
    
    def reset(self): 
        for h in self.h: h.zero_()
```


```python
learn = Learner(dls, LMModel7(len(vocab), 64, 2, 0.5),
                loss_func=CrossEntropyLossFlat(), metrics=accuracy,
                cbs=[ModelResetter, RNNRegularizer(alpha=2, beta=1)])
```


```python
RNNRegularizer
```
```text
fastai.callback.rnn.RNNRegularizer
```




```python
print_source(RNNRegularizer)
```
```text
class RNNRegularizer(Callback):
    "Add AR and TAR regularization"
    order,run_valid = RNNCallback.order+1,False
    def __init__(self, alpha=0., beta=0.): store_attr()
    def after_loss(self):
        if not self.training: return
        if self.alpha: self.learn.loss_grad += self.alpha * self.rnn.out.float().pow(2).mean()
        if self.beta:
            h = self.rnn.raw_out
            if len(h)>1: self.learn.loss_grad += self.beta * (h[:,1:] - h[:,:-1]).float().pow(2).mean()
```




```python
learn = TextLearner(dls, LMModel7(len(vocab), 64, 2, 0.4),
                    loss_func=CrossEntropyLossFlat(), metrics=accuracy)
```


```python
learn.model
```
```text
LMModel7(
  (i_h): Embedding(30, 64)
  (rnn): LSTM(64, 64, num_layers=2, batch_first=True)
  (drop): Dropout(p=0.4, inplace=False)
  (h_o): Linear(in_features=64, out_features=30, bias=True)
)
```




```python
learn.fit_one_cycle(15, 1e-2, wd=0.1)
```
<div style="overflow-x:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>2.620218</td>
      <td>1.797085</td>
      <td>0.484294</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1.622718</td>
      <td>1.452620</td>
      <td>0.652181</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.864787</td>
      <td>0.726230</td>
      <td>0.773275</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.434755</td>
      <td>0.699705</td>
      <td>0.828613</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.225359</td>
      <td>0.579946</td>
      <td>0.842855</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.126518</td>
      <td>0.571510</td>
      <td>0.850911</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.076041</td>
      <td>0.444107</td>
      <td>0.874349</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.051340</td>
      <td>0.366569</td>
      <td>0.882487</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.037389</td>
      <td>0.547799</td>
      <td>0.854818</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.027291</td>
      <td>0.392787</td>
      <td>0.880615</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>10</td>
      <td>0.022100</td>
      <td>0.354383</td>
      <td>0.889648</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>11</td>
      <td>0.018304</td>
      <td>0.380172</td>
      <td>0.885417</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>12</td>
      <td>0.015668</td>
      <td>0.384031</td>
      <td>0.885010</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>13</td>
      <td>0.013562</td>
      <td>0.389092</td>
      <td>0.884033</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>14</td>
      <td>0.012376</td>
      <td>0.383106</td>
      <td>0.885254</td>
      <td>00:01</td>
    </tr>
  </tbody>
</table>
</div>

**Note:** This performance is significantly better than the regular LSTM.




## References

* [Deep Learning for Coders with fastai & PyTorch](https://www.oreilly.com/library/view/deep-learning-for/9781492045519/)
* [The fastai book GitHub Repository](https://github.com/fastai/fastbook)