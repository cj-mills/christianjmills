---
title: Notes on fastai Book Ch. 7
date: 2022-3-14
image: /images/empty.gif
title-block-categories: true
hide: false
search_exclude: false
comments:
  utterances:
    repo: cj-mills/christianjmills
description: Chapter 7 covers data normalization, progressive resizing, test-time
  augmentation, mixup, and label smoothing.
categories: [ai, fastai, notes, pytorch]

aliases:
- /Notes-on-Fastai-Book-07/
---


::: {.callout-tip}
## This post is part of the following series:
* [**Deep Learning for Coders with fastai & PyTorch**](/series/notes/fastai-book-notes.html)
:::


* [Training a State-of-the-Art Model](#training-a-state-of-the-art-model)
* [Imagenette](#imagenette)
* [Normalization](#normalization)
* [Progressive Resizing](#progressive-resizing)
* [Test Time Augmentation](#test-time-augmentation)
* [Mixup](#mixup)
* [Label Smoothing](#Label Smoothing)
* [Papers and Math](#papers-and-math)
* [References](#references)


## Training a State-of-the-Art Model
- the dataset you are given is not necessarily the dataset you want.
- aim to have an iteration speed of no more than a couple of minutes
    - think about how you can cut down your dataset, or simplify your model to improve your experimentation speed
- the more experiments your can do the better

## Imagenette
* [https://docs.fast.ai/data.external.html](https://docs.fast.ai/data.external.html)
* A smaller version of the [imagenet dataset](https://image-net.org/)
* Useful for quick experimentation and iteration

-----


```python
from fastai.vision.all import *
```

-----

```python
path = untar_data(URLs.IMAGENETTE)
path
```

```text
Path('/home/innom-dt/.fastai/data/imagenette2')
```

#### parent_label

* [https://docs.fast.ai/data.transforms.html#parent_label](https://docs.fast.ai/data.transforms.html#parent_label)
* Label item with the parent folder name.

-----

```python
parent_label
```

```text
<function fastai.data.transforms.parent_label(o)>
```

-----

```python
dblock = DataBlock(blocks=(
    # TransformBlock for images
    ImageBlock(), 
    # TransformBlock for single-label categorical target
    CategoryBlock()),
                   # recursively load image files from path
                   get_items=get_image_files,
                   # label images using the parent folder name
                   get_y=parent_label,
                   # presize images to 460px
                   item_tfms=Resize(460),
                   # Batch resize to 224 and perform data augmentations
                   batch_tfms=aug_transforms(size=224, min_scale=0.75))
dls = dblock.dataloaders(path, bs=64, num_workers=8)
```

-----

```python
xresnet50
```

```text
<function fastai.vision.models.xresnet.xresnet50(pretrained=False, **kwargs)>
```

#### CrossEntropyLossFlat

* [https://docs.fast.ai/losses.html#CrossEntropyLossFlat](https://docs.fast.ai/losses.html#CrossEntropyLossFlat)
* Same as `nn.CrossEntropyLoss`, but flattens input and target.

-----

```python
CrossEntropyLossFlat
```

```text
fastai.losses.CrossEntropyLossFlat
```

-----

```python
# Initialize the model without pretrained weights
model = xresnet50(n_out=dls.c)
learn = Learner(dls, model, loss_func=CrossEntropyLossFlat(), metrics=accuracy)
learn.fit_one_cycle(5, 3e-3)
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
      <td>1.672769</td>
      <td>3.459394</td>
      <td>0.301718</td>
      <td>00:59</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1.224001</td>
      <td>1.404229</td>
      <td>0.552651</td>
      <td>01:00</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.968035</td>
      <td>0.996460</td>
      <td>0.660941</td>
      <td>01:00</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.699550</td>
      <td>0.709341</td>
      <td>0.771471</td>
      <td>01:00</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.578120</td>
      <td>0.571692</td>
      <td>0.820388</td>
      <td>01:00</td>
    </tr>
  </tbody>
</table>
</div>



-----

```python
# Initialize the model without pretrained weights
model = xresnet50(n_out=dls.c)
# Use mixed precision
learn = Learner(dls, model, loss_func=CrossEntropyLossFlat(), metrics=accuracy).to_fp16()
learn.fit_one_cycle(5, 3e-3)
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
      <td>1.569645</td>
      <td>3.962554</td>
      <td>0.329724</td>
      <td>00:33</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1.239950</td>
      <td>2.608771</td>
      <td>0.355489</td>
      <td>00:33</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.964794</td>
      <td>0.982138</td>
      <td>0.688200</td>
      <td>00:34</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.721289</td>
      <td>0.681677</td>
      <td>0.791636</td>
      <td>00:33</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.606473</td>
      <td>0.581621</td>
      <td>0.824122</td>
      <td>00:33</td>
    </tr>
  </tbody>
</table>
</div>



## Normalization

- normalized data: has a mean value of `0` and a standard deviation of `1`
- it is easier to train models with normalized data
- normalization is especially important when using pretrained models
    - make sure to use the same normalization stats the pretrained model was trained on

-----

```python
x,y = dls.one_batch()
x.mean(dim=[0,2,3]),x.std(dim=[0,2,3])
```

```text
(TensorImage([0.4498, 0.4448, 0.4141], device='cuda:0'),
 TensorImage([0.2893, 0.2792, 0.3022], device='cuda:0'))
```

#### Normalize

* [https://docs.fast.ai/data.transforms.html#Normalize](https://docs.fast.ai/data.transforms.html#Normalize)
* Normalize/denormalize a bath of [TensorImage](https://docs.fast.ai/torch_core.html#TensorImage)

-----


```python
Normalize
```

```text
fastai.data.transforms.Normalize
```

-----

```python
Normalize.from_stats
```

```text
<bound method Normalize.from_stats of <class 'fastai.data.transforms.Normalize'>>
```

-----

```python
def get_dls(bs, size):
    dblock = DataBlock(blocks=(ImageBlock, CategoryBlock),
                   get_items=get_image_files,
                   get_y=parent_label,
                   item_tfms=Resize(460),
                   batch_tfms=[*aug_transforms(size=size, min_scale=0.75),
                               Normalize.from_stats(*imagenet_stats)])
    return dblock.dataloaders(path, bs=bs)
```

-----

```python
dls = get_dls(64, 224)
```

-----

```python
x,y = dls.one_batch()
x.mean(dim=[0,2,3]),x.std(dim=[0,2,3])
```

```text
(TensorImage([-0.2055, -0.0843,  0.0192], device='cuda:0'),
 TensorImage([1.1835, 1.1913, 1.2377], device='cuda:0'))
```

-----

```python
model = xresnet50(n_out=dls.c)
learn = Learner(dls, model, loss_func=CrossEntropyLossFlat(), metrics=accuracy).to_fp16()
learn.fit_one_cycle(5, 3e-3)
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
      <td>1.545518</td>
      <td>3.255928</td>
      <td>0.342046</td>
      <td>00:35</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1.234556</td>
      <td>1.449043</td>
      <td>0.560866</td>
      <td>00:35</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.970857</td>
      <td>1.310043</td>
      <td>0.617252</td>
      <td>00:35</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.736170</td>
      <td>0.770678</td>
      <td>0.758402</td>
      <td>00:36</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.619965</td>
      <td>0.575979</td>
      <td>0.822629</td>
      <td>00:36</td>
    </tr>
  </tbody>
</table>
</div>



## Progressive Resizing

- start training with smaller images and end training with larger images
    - gradually using larger and larger images as you train
- used by a team of [fast.ai](http://fast.ai) students to [win the DAWNBench competition in 2018](https://www.theverge.com/2018/5/7/17316010/fast-ai-speed-test-stanford-dawnbench-google-intel)
- smaller images helps training complete much faster
- larger images helps makes accuracy much higher
- progressive resizing serves as another form of data augmentation
    - should result in better generalization
- progressive resizing might hurt performance when using transfer learning
    - most likely to happen if your pretrained model was very similar to your target task and the dataset it was trained on had similar-sized images

-----


```python
dls = get_dls(128, 128)
learn = Learner(dls, xresnet50(n_out=dls.c), loss_func=CrossEntropyLossFlat(), 
                metrics=accuracy).to_fp16()
learn.fit_one_cycle(4, 3e-3)
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
      <td>1.627504</td>
      <td>2.495554</td>
      <td>0.393951</td>
      <td>00:21</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1.264693</td>
      <td>1.233987</td>
      <td>0.613518</td>
      <td>00:21</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.970736</td>
      <td>0.958903</td>
      <td>0.707618</td>
      <td>00:21</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.740324</td>
      <td>0.659166</td>
      <td>0.794996</td>
      <td>00:21</td>
    </tr>
  </tbody>
</table>
</div>


-----

```python
learn.dls = get_dls(64, 224)
learn.fine_tune(5, 1e-3)
```
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
      <td>0.828744</td>
      <td>1.024683</td>
      <td>0.669529</td>
      <td>00:36</td>
    </tr>
  </tbody>
</table>
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
      <td>0.670041</td>
      <td>0.716627</td>
      <td>0.776326</td>
      <td>00:36</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.689798</td>
      <td>0.706051</td>
      <td>0.768857</td>
      <td>00:36</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.589789</td>
      <td>0.519608</td>
      <td>0.831217</td>
      <td>00:35</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.506784</td>
      <td>0.436529</td>
      <td>0.870426</td>
      <td>00:36</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.453270</td>
      <td>0.401451</td>
      <td>0.877147</td>
      <td>00:36</td>
    </tr>
  </tbody>
</table>


## Test Time Augmentation

- during inference or validation, creating multiple versions of each image using augmentation, and then taking the average or maximum of the predictions for each augmented version of the image
- can result in dramatic improvements in accuracy, depending on the dataset
- does not change the time required to train
- will increase the amount of time required for validation or inference

#### Learner.tta
* [https://docs.fast.ai/learner.html#Learner.tta](https://docs.fast.ai/learner.html#Learner.tta)
* returns predictions using Test Time Augmentation

-----


```python
learn.tta
```

```text
<bound method Learner.tta of <fastai.learner.Learner object at 0x7f75b4be5f40>>
```

-----

```python
preds,targs = learn.tta()
accuracy(preds, targs).item()
```

```text
0.882001519203186
```


## Mixup

- a powerful data augmentation technique that can provide dramatically higher accuracy, especially when you don’t have much data and don’t have a pretrained model
- introduced in the 2017 paper [mixup: Beyond Empirical Risk Minimization](https://arxiv.org/abs/1710.09412)
    - “While data augmentation consistently leads to improved generalization, the procedure is dataset-dependent, and thus requires the use of expert knowledge
- Mixup steps
    1. Select another image from your dataset at random
    2. Pick a weight at random
    3. Take a weighted average of the selected image with your image, to serve as your independent variable
    4. Take a weighted average of this image’s labels with your image’s labels, to server as your dependent variable
- target needs to be one-hot encoded
- $\tilde{x} = \lambda x_{i} + (1 - \lambda) x_{j} \text{, where } x_{i} \text{ and } x_{j} \text{ are raw input vectors}$
- $\tilde{y} = \lambda y_{i} + (1 - \lambda) y_{j} \text{, where } y_{i} \text{ and } y_{j} \text{ are one-hot label encodings}$
- more difficult to train
- less prone to overfitting
- requires far more epochs to to train to get better accuracy
- can be applied to types of data other than photos
- can even be used on activations inside of model
- resolves the issue where it is not typically possible to achieve a perfect loss score
    - our labels are 1s and 0s, but the outputs of softmax and sigmoid can never equal 1 or 0
    - with Mixup our labels will only be exactly 1 or 0 if two images from the same class are mixed
- Mixup is “accidentally” making the labels bigger than 0 or smaller than 1
    - can be resolved with Label Smoothing

-----


```python
# Get two images from different classes
church = PILImage.create(get_image_files_sorted(path/'train'/'n03028079')[0])
gas = PILImage.create(get_image_files_sorted(path/'train'/'n03425413')[0])
# Resize images
church = church.resize((256,256))
gas = gas.resize((256,256))

# Scale pixel values to the range [0,1]
tchurch = tensor(church).float() / 255.
tgas = tensor(gas).float() / 255.

_,axs = plt.subplots(1, 3, figsize=(12,4))
# Show the first image
show_image(tchurch, ax=axs[0]);
# Show the second image
show_image(tgas, ax=axs[1]);
# Take the weighted average of the two images
show_image((0.3*tchurch + 0.7*tgas), ax=axs[2]);
```

![](./images/output_31_0.png){fig-align="center"}

-----

```python
model = xresnet50()
learn = Learner(dls, model, loss_func=CrossEntropyLossFlat(), metrics=accuracy, cbs=MixUp).to_fp16()
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
      <td>2.332906</td>
      <td>1.680691</td>
      <td>0.431292</td>
      <td>00:21</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1.823880</td>
      <td>1.699880</td>
      <td>0.481329</td>
      <td>00:21</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1.660909</td>
      <td>1.162998</td>
      <td>0.650112</td>
      <td>00:21</td>
    </tr>
    <tr>
      <td>3</td>
      <td>1.520751</td>
      <td>1.302749</td>
      <td>0.582524</td>
      <td>00:21</td>
    </tr>
    <tr>
      <td>4</td>
      <td>1.391567</td>
      <td>1.256566</td>
      <td>0.595967</td>
      <td>00:21</td>
    </tr>
    <tr>
      <td>5</td>
      <td>1.308175</td>
      <td>1.193670</td>
      <td>0.638163</td>
      <td>00:21</td>
    </tr>
    <tr>
      <td>6</td>
      <td>1.224825</td>
      <td>0.921357</td>
      <td>0.706871</td>
      <td>00:21</td>
    </tr>
    <tr>
      <td>7</td>
      <td>1.190292</td>
      <td>0.846658</td>
      <td>0.733383</td>
      <td>00:21</td>
    </tr>
    <tr>
      <td>8</td>
      <td>1.124314</td>
      <td>0.707856</td>
      <td>0.780807</td>
      <td>00:21</td>
    </tr>
    <tr>
      <td>9</td>
      <td>1.085013</td>
      <td>0.701829</td>
      <td>0.778193</td>
      <td>00:21</td>
    </tr>
    <tr>
      <td>10</td>
      <td>1.028223</td>
      <td>0.509176</td>
      <td>0.851008</td>
      <td>00:21</td>
    </tr>
    <tr>
      <td>11</td>
      <td>0.992827</td>
      <td>0.518169</td>
      <td>0.845780</td>
      <td>00:21</td>
    </tr>
    <tr>
      <td>12</td>
      <td>0.945492</td>
      <td>0.458248</td>
      <td>0.864078</td>
      <td>00:21</td>
    </tr>
    <tr>
      <td>13</td>
      <td>0.923450</td>
      <td>0.418989</td>
      <td>0.871546</td>
      <td>00:21</td>
    </tr>
    <tr>
      <td>14</td>
      <td>0.904607</td>
      <td>0.416422</td>
      <td>0.876400</td>
      <td>00:21</td>
    </tr>
  </tbody>
</table>
</div>



## Label Smoothing

- **[Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567)**
- in the theoretical expression of loss, in Classification problems, our targets are one-hot encoded
    - the model is trained to return 0 for all  categories but one, for which it is trained to return 1
    - this encourages  overfitting and gives your a model at inference time that is not going to give meaningful probabilities
    - this can be harmful if your data is not perfectly labeled
- label smoothing: replace all our 1s with a number that is a bit less than 1, and our 0s with a number that is a bit more then 0
    - encourages your model to be less confident
    - makes your training more robust, even if there is mislabeled data
    - results in a model that generalizes better at inference
- Steps
    1. start with one-hot encoded labels
    2. replace all 0s with $\frac{\epsilon}{N}$ where $N$ is the number of classes and $\epsilon$ is a parameter (usually 0.1)
    3. replace all 1s with $1 - \epsilon + \frac{\epsilon}{N}$ to make sure the labels add up to 1

-----

```python
model = xresnet50()
learn = Learner(dls, model, loss_func=LabelSmoothingCrossEntropy(), metrics=accuracy).to_fp16()
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
      <td>2.796061</td>
      <td>2.399328</td>
      <td>0.513443</td>
      <td>00:21</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2.335293</td>
      <td>2.222970</td>
      <td>0.584391</td>
      <td>00:21</td>
    </tr>
    <tr>
      <td>2</td>
      <td>2.125152</td>
      <td>2.478721</td>
      <td>0.490291</td>
      <td>00:21</td>
    </tr>
    <tr>
      <td>3</td>
      <td>1.967522</td>
      <td>1.977260</td>
      <td>0.690441</td>
      <td>00:21</td>
    </tr>
    <tr>
      <td>4</td>
      <td>1.853788</td>
      <td>1.861635</td>
      <td>0.715459</td>
      <td>00:21</td>
    </tr>
    <tr>
      <td>5</td>
      <td>1.747451</td>
      <td>1.889759</td>
      <td>0.699776</td>
      <td>00:21</td>
    </tr>
    <tr>
      <td>6</td>
      <td>1.683000</td>
      <td>1.710128</td>
      <td>0.770351</td>
      <td>00:21</td>
    </tr>
    <tr>
      <td>7</td>
      <td>1.610975</td>
      <td>1.672254</td>
      <td>0.780807</td>
      <td>00:21</td>
    </tr>
    <tr>
      <td>8</td>
      <td>1.534964</td>
      <td>1.691175</td>
      <td>0.769231</td>
      <td>00:21</td>
    </tr>
    <tr>
      <td>9</td>
      <td>1.480721</td>
      <td>1.490685</td>
      <td>0.842420</td>
      <td>00:21</td>
    </tr>
    <tr>
      <td>10</td>
      <td>1.417200</td>
      <td>1.463211</td>
      <td>0.852502</td>
      <td>00:21</td>
    </tr>
    <tr>
      <td>11</td>
      <td>1.360376</td>
      <td>1.395671</td>
      <td>0.867812</td>
      <td>00:21</td>
    </tr>
    <tr>
      <td>12</td>
      <td>1.312882</td>
      <td>1.360292</td>
      <td>0.887603</td>
      <td>00:21</td>
    </tr>
    <tr>
      <td>13</td>
      <td>1.283740</td>
      <td>1.346170</td>
      <td>0.890217</td>
      <td>00:21</td>
    </tr>
    <tr>
      <td>14</td>
      <td>1.264030</td>
      <td>1.339298</td>
      <td>0.892830</td>
      <td>00:21</td>
    </tr>
  </tbody>
</table>
</div>



## Label Smoothing, Mixup and Progressive Resizing

```python
dls = get_dls(128, 128)
model = xresnet50()
learn = Learner(dls, model, loss_func=LabelSmoothingCrossEntropy(), metrics=accuracy, cbs=MixUp).to_fp16()
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
      <td>3.045166</td>
      <td>2.561215</td>
      <td>0.449589</td>
      <td>00:21</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2.642317</td>
      <td>2.906508</td>
      <td>0.405900</td>
      <td>00:21</td>
    </tr>
    <tr>
      <td>2</td>
      <td>2.473271</td>
      <td>2.389416</td>
      <td>0.516804</td>
      <td>00:21</td>
    </tr>
    <tr>
      <td>3</td>
      <td>2.356234</td>
      <td>2.263084</td>
      <td>0.557506</td>
      <td>00:21</td>
    </tr>
    <tr>
      <td>4</td>
      <td>2.268788</td>
      <td>2.401770</td>
      <td>0.544436</td>
      <td>00:21</td>
    </tr>
    <tr>
      <td>5</td>
      <td>2.181318</td>
      <td>2.040797</td>
      <td>0.650485</td>
      <td>00:21</td>
    </tr>
    <tr>
      <td>6</td>
      <td>2.122742</td>
      <td>1.711615</td>
      <td>0.761762</td>
      <td>00:21</td>
    </tr>
    <tr>
      <td>7</td>
      <td>2.068317</td>
      <td>1.961520</td>
      <td>0.688200</td>
      <td>00:21</td>
    </tr>
    <tr>
      <td>8</td>
      <td>2.022716</td>
      <td>1.751058</td>
      <td>0.743839</td>
      <td>00:21</td>
    </tr>
    <tr>
      <td>9</td>
      <td>1.980203</td>
      <td>1.635354</td>
      <td>0.792009</td>
      <td>00:21</td>
    </tr>
    <tr>
      <td>10</td>
      <td>1.943118</td>
      <td>1.711313</td>
      <td>0.758028</td>
      <td>00:21</td>
    </tr>
    <tr>
      <td>11</td>
      <td>1.889408</td>
      <td>1.454949</td>
      <td>0.854742</td>
      <td>00:21</td>
    </tr>
    <tr>
      <td>12</td>
      <td>1.853412</td>
      <td>1.433971</td>
      <td>0.862584</td>
      <td>00:21</td>
    </tr>
    <tr>
      <td>13</td>
      <td>1.847395</td>
      <td>1.412596</td>
      <td>0.867438</td>
      <td>00:22</td>
    </tr>
    <tr>
      <td>14</td>
      <td>1.817760</td>
      <td>1.409608</td>
      <td>0.875280</td>
      <td>00:23</td>
    </tr>
  </tbody>
</table>
</div>

-----

```python
learn.dls = get_dls(64, 224)
learn.fine_tune(10, 1e-3)
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
      <td>1.951753</td>
      <td>1.672776</td>
      <td>0.789395</td>
      <td>00:36</td>
    </tr>
  </tbody>
</table>
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
      <td>1.872399</td>
      <td>1.384301</td>
      <td>0.892457</td>
      <td>00:36</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1.860005</td>
      <td>1.441491</td>
      <td>0.864078</td>
      <td>00:36</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1.876859</td>
      <td>1.425859</td>
      <td>0.867438</td>
      <td>00:36</td>
    </tr>
    <tr>
      <td>3</td>
      <td>1.851872</td>
      <td>1.460640</td>
      <td>0.863331</td>
      <td>00:36</td>
    </tr>
    <tr>
      <td>4</td>
      <td>1.840423</td>
      <td>1.413441</td>
      <td>0.880508</td>
      <td>00:36</td>
    </tr>
    <tr>
      <td>5</td>
      <td>1.808990</td>
      <td>1.444332</td>
      <td>0.863704</td>
      <td>00:36</td>
    </tr>
    <tr>
      <td>6</td>
      <td>1.777755</td>
      <td>1.321098</td>
      <td>0.910754</td>
      <td>00:36</td>
    </tr>
    <tr>
      <td>7</td>
      <td>1.761589</td>
      <td>1.312523</td>
      <td>0.912621</td>
      <td>00:36</td>
    </tr>
    <tr>
      <td>8</td>
      <td>1.756679</td>
      <td>1.302988</td>
      <td>0.919716</td>
      <td>00:36</td>
    </tr>
    <tr>
      <td>9</td>
      <td>1.745481</td>
      <td>1.304583</td>
      <td>0.918969</td>
      <td>00:36</td>
    </tr>
  </tbody>
</table>
</div>





## Papers and Math

- **[Greek letters used in mathematics, science, and engineering](https://en.wikipedia.org/wiki/Greek_letters_used_in_mathematics,_science,_and_engineering)**
- **[Glossary of mathematical symbols](https://en.wikipedia.org/wiki/Glossary_of_mathematical_symbols)**
- **[Detexify](https://detexify.kirelabs.org/classify.html)**
    - draw a mathematical symbol and get the latex code


## References

* [Deep Learning for Coders with fastai & PyTorch](https://www.oreilly.com/library/view/deep-learning-for/9781492045519/)
* [The fastai book GitHub Repository](https://github.com/fastai/fastbook)



**Previous:** [Notes on fastai Book Ch. 6](../chapter-6/)

**Next:** [Notes on fastai Book Ch. 8](../chapter-8/)

