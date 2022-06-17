---
title: Notes on fastai Book Ch. 15
layout: post
toc: false
comments: true
description: Chapter 15 provides a deep dive into different application architectures in the fast.ai library.
categories: [ai, fastai, notes, pytorch]
hide: false
permalink: /:title/
search_exclude: false
---



* [Application Architectures Deep Dive](#application-architectures-deep-dive)
* [Computer Vision](#computer-vision)
* [Natural Language Processing](#natural-language-processing)
* [Tabular](#tabular)
* [Conclusion](#conclusion)
* [References](#references)



-----

```python
#hide
# !pip install -Uqq fastbook
import fastbook
fastbook.setup_book()
```


```python
#hide
from fastbook import *
```


```python
import inspect
def print_source(obj):
    for line in inspect.getsource(obj).split("\n"):
        print(line)
```




## Application Architectures Deep Dive



## Computer Vision

### cnn_learner

#### Transfer Learning
* the head (the final layers) of the pretrained model needs to be cut off  and replaced
* fastai stores where to cut the included pretrained models in the [model_meta](https://github.com/fastai/fastai/blob/01d7f879d3efe14530243e1074c1c5efbd717195/fastai/vision/learner.py#L120) dictionary

#### Head
* the part that is specialized for a particular task
* generally the part after the adaptive average pooling layer

#### Body
* everything other than the head
* includes the stem


```python
pd.DataFrame(model_meta)
```
<div style="overflow-x:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>&lt;function xresnet18 at 0x7f8668ee5310&gt;</th>
      <th>&lt;function xresnet34 at 0x7f8668ee53a0&gt;</th>
      <th>&lt;function xresnet50 at 0x7f8668ee5430&gt;</th>
      <th>&lt;function xresnet101 at 0x7f8668ee54c0&gt;</th>
      <th>&lt;function xresnet152 at 0x7f8668ee5550&gt;</th>
      <th>&lt;function resnet18 at 0x7f866b0d0670&gt;</th>
      <th>&lt;function resnet34 at 0x7f866b0d0700&gt;</th>
      <th>&lt;function resnet50 at 0x7f866b0d0790&gt;</th>
      <th>&lt;function resnet101 at 0x7f866b0d0820&gt;</th>
      <th>&lt;function resnet152 at 0x7f866b0d08b0&gt;</th>
      <th>&lt;function squeezenet1_0 at 0x7f866b0d7790&gt;</th>
      <th>&lt;function squeezenet1_1 at 0x7f866b0d7820&gt;</th>
      <th>&lt;function densenet121 at 0x7f866a7b78b0&gt;</th>
      <th>&lt;function densenet169 at 0x7f866a7b79d0&gt;</th>
      <th>&lt;function densenet201 at 0x7f866a7b7a60&gt;</th>
      <th>&lt;function densenet161 at 0x7f866a7b7940&gt;</th>
      <th>&lt;function vgg11_bn at 0x7f866b0d7040&gt;</th>
      <th>&lt;function vgg13_bn at 0x7f866b0d7160&gt;</th>
      <th>&lt;function vgg16_bn at 0x7f866b0d7280&gt;</th>
      <th>&lt;function vgg19_bn at 0x7f866b0d73a0&gt;</th>
      <th>&lt;function alexnet at 0x7f866b0c2d30&gt;</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>cut</th>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-2</td>
      <td>-2</td>
      <td>-2</td>
      <td>-2</td>
      <td>-2</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-2</td>
      <td>-2</td>
      <td>-2</td>
      <td>-2</td>
      <td>-2</td>
    </tr>
    <tr>
      <th>split</th>
      <td>&lt;function _xresnet_split at 0x7f8662906ee0&gt;</td>
      <td>&lt;function _xresnet_split at 0x7f8662906ee0&gt;</td>
      <td>&lt;function _xresnet_split at 0x7f8662906ee0&gt;</td>
      <td>&lt;function _xresnet_split at 0x7f8662906ee0&gt;</td>
      <td>&lt;function _xresnet_split at 0x7f8662906ee0&gt;</td>
      <td>&lt;function _resnet_split at 0x7f8662906f70&gt;</td>
      <td>&lt;function _resnet_split at 0x7f8662906f70&gt;</td>
      <td>&lt;function _resnet_split at 0x7f8662906f70&gt;</td>
      <td>&lt;function _resnet_split at 0x7f8662906f70&gt;</td>
      <td>&lt;function _resnet_split at 0x7f8662906f70&gt;</td>
      <td>&lt;function _squeezenet_split at 0x7f866290e040&gt;</td>
      <td>&lt;function _squeezenet_split at 0x7f866290e040&gt;</td>
      <td>&lt;function _densenet_split at 0x7f866290e0d0&gt;</td>
      <td>&lt;function _densenet_split at 0x7f866290e0d0&gt;</td>
      <td>&lt;function _densenet_split at 0x7f866290e0d0&gt;</td>
      <td>&lt;function _densenet_split at 0x7f866290e0d0&gt;</td>
      <td>&lt;function _vgg_split at 0x7f866290e160&gt;</td>
      <td>&lt;function _vgg_split at 0x7f866290e160&gt;</td>
      <td>&lt;function _vgg_split at 0x7f866290e160&gt;</td>
      <td>&lt;function _vgg_split at 0x7f866290e160&gt;</td>
      <td>&lt;function _alexnet_split at 0x7f866290e1f0&gt;</td>
    </tr>
    <tr>
      <th>stats</th>
      <td>([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])</td>
      <td>([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])</td>
      <td>([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])</td>
      <td>([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])</td>
      <td>([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])</td>
      <td>([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])</td>
      <td>([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])</td>
      <td>([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])</td>
      <td>([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])</td>
      <td>([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])</td>
      <td>([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])</td>
      <td>([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])</td>
      <td>([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])</td>
      <td>([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])</td>
      <td>([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])</td>
      <td>([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])</td>
      <td>([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])</td>
      <td>([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])</td>
      <td>([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])</td>
      <td>([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])</td>
      <td>([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])</td>
    </tr>
  </tbody>
</table>
</div>




```python
model_meta[resnet50]
```
```text
{'cut': -2,
 'split': <function fastai.vision.learner._resnet_split(m)>,
 'stats': ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])}
```




```python
print_source(model_meta[resnet50]['split'])
```
```text
def  _resnet_split(m): return L(m[0][:6], m[0][6:], m[1:]).map(params)
```




```python
create_head(20,2)
```
```text
Sequential(
  (0): AdaptiveConcatPool2d(
    (ap): AdaptiveAvgPool2d(output_size=1)
    (mp): AdaptiveMaxPool2d(output_size=1)
  )
  (1): Flatten(full=False)
  (2): BatchNorm1d(40, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (3): Dropout(p=0.25, inplace=False)
  (4): Linear(in_features=40, out_features=512, bias=False)
  (5): ReLU(inplace=True)
  (6): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (7): Dropout(p=0.5, inplace=False)
  (8): Linear(in_features=512, out_features=2, bias=False)
)
```



**Note:** fastai add two linear layers by default for transfer learning
* using just one linear layer is unlikely to be enough when transferring a pretrained model to very different domains


```python
create_head
```
```text
<function fastai.vision.learner.create_head(nf, n_out, lin_ftrs=None, ps=0.5, concat_pool=True, first_bn=True, bn_final=False, lin_first=False, y_range=None)>
```




```python
print_source(create_head)
```
```text
def create_head(nf, n_out, lin_ftrs=None, ps=0.5, concat_pool=True, first_bn=True, bn_final=False,
                lin_first=False, y_range=None):
    "Model head that takes `nf` features, runs through `lin_ftrs`, and out `n_out` classes."
    if concat_pool: nf *= 2
    lin_ftrs = [nf, 512, n_out] if lin_ftrs is None else [nf] + lin_ftrs + [n_out]
    bns = [first_bn] + [True]*len(lin_ftrs[1:])
    ps = L(ps)
    if len(ps) == 1: ps = [ps[0]/2] * (len(lin_ftrs)-2) + ps
    actns = [nn.ReLU(inplace=True)] * (len(lin_ftrs)-2) + [None]
    pool = AdaptiveConcatPool2d() if concat_pool else nn.AdaptiveAvgPool2d(1)
    layers = [pool, Flatten()]
    if lin_first: layers.append(nn.Dropout(ps.pop(0)))
    for ni,no,bn,p,actn in zip(lin_ftrs[:-1], lin_ftrs[1:], bns, ps, actns):
        layers += LinBnDrop(ni, no, bn=bn, p=p, act=actn, lin_first=lin_first)
    if lin_first: layers.append(nn.Linear(lin_ftrs[-2], n_out))
    if bn_final: layers.append(nn.BatchNorm1d(lin_ftrs[-1], momentum=0.01))
    if y_range is not None: layers.append(SigmoidRange(*y_range))
    return nn.Sequential(*layers)
```



#### One Last Batchnorm
* bn_final: setting this to True will cause a batchnorm layher to be added as the final layer
* can be useful in helping your model scale appropriately for your output activations


```python
AdaptiveConcatPool2d
```
```text
fastai.layers.AdaptiveConcatPool2d
```




```python
print_source(AdaptiveConcatPool2d)
```
```text
class AdaptiveConcatPool2d(Module):
    "Layer that concats `AdaptiveAvgPool2d` and `AdaptiveMaxPool2d`"
    def __init__(self, size=None):
        self.size = size or 1
        self.ap = nn.AdaptiveAvgPool2d(self.size)
        self.mp = nn.AdaptiveMaxPool2d(self.size)
    def forward(self, x): return torch.cat([self.mp(x), self.ap(x)], 1)
```



### unet_learner
* used for generative vision models
* use a custom head which progressively increases the dimensions back to the same as the source image
    * can use nearest neighbor interpolation
    * can use a transposed convolution
        * zero padding is inserted between all the pixels in the input before performing a convolution
        * also known as a stride half convolution
        * fastai `ConvLayer(transpose=True)`
* unets use skip connections pass information from the encoding layers in the body to the decoding layers in the head
* [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)

#### Tasks
* segmentation
* super resolution
* colorization
* style transfer

### A Siamese Network


```python
#hide
from fastai.vision.all import *
```


```python
path = untar_data(URLs.PETS)
path
```
```text
Path('/home/innom-dt/.fastai/data/oxford-iiit-pet')
```




```python
files = get_image_files(path/"images")
files
```
```text
(#7390) [Path('/home/innom-dt/.fastai/data/oxford-iiit-pet/images/Birman_121.jpg'),Path('/home/innom-dt/.fastai/data/oxford-iiit-pet/images/shiba_inu_131.jpg'),Path('/home/innom-dt/.fastai/data/oxford-iiit-pet/images/Bombay_176.jpg'),Path('/home/innom-dt/.fastai/data/oxford-iiit-pet/images/Bengal_199.jpg'),Path('/home/innom-dt/.fastai/data/oxford-iiit-pet/images/beagle_41.jpg'),Path('/home/innom-dt/.fastai/data/oxford-iiit-pet/images/beagle_27.jpg'),Path('/home/innom-dt/.fastai/data/oxford-iiit-pet/images/great_pyrenees_181.jpg'),Path('/home/innom-dt/.fastai/data/oxford-iiit-pet/images/Bengal_100.jpg'),Path('/home/innom-dt/.fastai/data/oxford-iiit-pet/images/keeshond_124.jpg'),Path('/home/innom-dt/.fastai/data/oxford-iiit-pet/images/havanese_115.jpg')...]
```




```python
# Custom type to allow us to show siamese image pairs
# Tracks whether the two images belong to the same class
class SiameseImage(fastuple):
    def show(self, ctx=None, **kwargs): 
        img1,img2,same_breed = self
        if not isinstance(img1, Tensor):
            if img2.size != img1.size: img2 = img2.resize(img1.size)
            t1,t2 = tensor(img1),tensor(img2)
            t1,t2 = t1.permute(2,0,1),t2.permute(2,0,1)
        else: t1,t2 = img1,img2
        line = t1.new_zeros(t1.shape[0], t1.shape[1], 10)
        return show_image(torch.cat([t1,line,t2], dim=2), 
                          title=same_breed, ctx=ctx)
```


```python
def label_func(fname):
    return re.match(r'^(.*)_\d+.jpg$', fname.name).groups()[0]
```


```python
class SiameseTransform(Transform):
    def __init__(self, files, label_func, splits):
        # Generate list of unique labels
        self.labels = files.map(label_func).unique()
        # Create a dictionary to match labels to filenames
        self.lbl2files = {l: L(f for f in files if label_func(f) == l) 
                          for l in self.labels}
        self.label_func = label_func
        self.valid = {f: self._draw(f) for f in files[splits[1]]}
        
    def encodes(self, f):
        f2,t = self.valid.get(f, self._draw(f))
        img1,img2 = PILImage.create(f),PILImage.create(f2)
        # Create siamese image pair
        return SiameseImage(img1, img2, t)
    
    def _draw(self, f):
        # 50/50 chance of generating a pair of the same class
        same = random.random() < 0.5
        cls = self.label_func(f)
        if not same: 
            cls = random.choice(L(l for l in self.labels if l != cls)) 
        return random.choice(self.lbl2files[cls]),same
```


```python
splits = RandomSplitter()(files)
tfm = SiameseTransform(files, label_func, splits)
tls = TfmdLists(files, tfm, splits=splits)
dls = tls.dataloaders(after_item=[Resize(224), ToTensor], 
    after_batch=[IntToFloatTensor, Normalize.from_stats(*imagenet_stats)])
```


```python
class SiameseModel(Module):
    def __init__(self, encoder, head):
        self.encoder,self.head = encoder,head
    
    def forward(self, x1, x2):
        ftrs = torch.cat([self.encoder(x1), self.encoder(x2)], dim=1)
        return self.head(ftrs)
```


```python
encoder = create_body(resnet34, cut=-2)
```


```python
create_body
```
```text
<function fastai.vision.learner.create_body(arch, n_in=3, pretrained=True, cut=None)>
```




```python
print_source(create_body)
```
```text
def create_body(arch, n_in=3, pretrained=True, cut=None):
    "Cut off the body of a typically pretrained `arch` as determined by `cut`"
    model = arch(pretrained=pretrained)
    _update_first_layer(model, n_in, pretrained)
    #cut = ifnone(cut, cnn_config(arch)['cut'])
    if cut is None:
        ll = list(enumerate(model.children()))
        cut = next(i for i,o in reversed(ll) if has_pool_type(o))
    if   isinstance(cut, int): return nn.Sequential(*list(model.children())[:cut])
    elif callable(cut): return cut(model)
    else: raise NameError("cut must be either integer or a function")
```




```python
head = create_head(512*2, 2, ps=0.5)
head
```
```text
Sequential(
  (0): AdaptiveConcatPool2d(
    (ap): AdaptiveAvgPool2d(output_size=1)
    (mp): AdaptiveMaxPool2d(output_size=1)
  )
  (1): Flatten(full=False)
  (2): BatchNorm1d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (3): Dropout(p=0.25, inplace=False)
  (4): Linear(in_features=2048, out_features=512, bias=False)
  (5): ReLU(inplace=True)
  (6): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (7): Dropout(p=0.5, inplace=False)
  (8): Linear(in_features=512, out_features=2, bias=False)
)
```




```python
model = SiameseModel(encoder, head)
```


```python
def loss_func(out, targ):
    return nn.CrossEntropyLoss()(out, targ.long())
```


```python
# Tell fastai how to split the model into parameter groups
def siamese_splitter(model):
    return [params(model.encoder), params(model.head)]
```


```python
learn = Learner(dls, model, loss_func=loss_func, 
                splitter=siamese_splitter, metrics=accuracy)
learn.freeze()
```


```python
Learner.freeze
```
```text
<function fastai.learner.Learner.freeze(self: fastai.learner.Learner)>
```




```python
print_source(Learner.freeze)
```
```text
@patch
def freeze(self:Learner): self.freeze_to(-1)
```




```python
print_source(Learner.freeze_to)
```
```text
@patch
def freeze_to(self:Learner, n):
    if self.opt is None: self.create_opt()
    self.opt.freeze_to(n)
    self.opt.clear_state()
```




```python
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
      <td>0.530529</td>
      <td>0.281408</td>
      <td>0.887686</td>
      <td>00:29</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.377506</td>
      <td>0.224826</td>
      <td>0.912043</td>
      <td>00:29</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.276916</td>
      <td>0.195273</td>
      <td>0.928958</td>
      <td>00:29</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.242797</td>
      <td>0.170715</td>
      <td>0.933018</td>
      <td>00:29</td>
    </tr>
  </tbody>
</table>
</div>



```python
learn.unfreeze()
learn.fit_one_cycle(4, slice(1e-6,1e-4))
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
      <td>0.249987</td>
      <td>0.160208</td>
      <td>0.939784</td>
      <td>00:38</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.236774</td>
      <td>0.157880</td>
      <td>0.941137</td>
      <td>00:38</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.222469</td>
      <td>0.151024</td>
      <td>0.945196</td>
      <td>00:38</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.218679</td>
      <td>0.160581</td>
      <td>0.939107</td>
      <td>00:38</td>
    </tr>
  </tbody>
</table>
</div>


## Natural Language Processing
* We can convert an AWD-LSTM language model into a transfer learning classifier by selecting stack RNN for the encoder
* [Universal Language Model Fine-tuning for Text Classification](https://arxiv.org/abs/1801.06146)
    * divide the document into fixed-length batches of size b
    * the model is initialized at the beginning of each batch with the final state of the previous batch
    * keep track of the hidden states for mean and max-pooling
    * gradients are backpropogated to the batches whose hidden states contributed to the final prediction
        * use variable lenght backpropogation sequences
* the classifier contains a for-loop, which loops over each batch of a sequence
    * need to gather data in batches
    * each text needs to be treated separately as they have their own labels
    * it is likely the texts will not all be the same length
        * we won't be able to put them all in the same array
        * need to use padding
            * when grabbing a bunch of texts, determine which one has the greatest length
            * fill the ones that are shorter with the special character `xxpad`.
            * make sure texts of similar sizes are put together to minimize excess padding
* the state is maintained across batches
* the activations of each batch are stored
* at the end, we use the same average and max concatenated pooling trick used for computer vision models



## Tabular


```python
from fastai.tabular.all import *
```


```python
TabularModel
```
```text
fastai.tabular.model.TabularModel
```




```python
print_source(TabularModel)
```
```text
class TabularModel(Module):
    "Basic model for tabular data."
    def __init__(self, emb_szs, n_cont, out_sz, layers, ps=None, embed_p=0.,
                 y_range=None, use_bn=True, bn_final=False, bn_cont=True, act_cls=nn.ReLU(inplace=True),
                 lin_first=True):
        ps = ifnone(ps, [0]*len(layers))
        if not is_listy(ps): ps = [ps]*len(layers)
        self.embeds = nn.ModuleList([Embedding(ni, nf) for ni,nf in emb_szs])
        self.emb_drop = nn.Dropout(embed_p)
        self.bn_cont = nn.BatchNorm1d(n_cont) if bn_cont else None
        n_emb = sum(e.embedding_dim for e in self.embeds)
        self.n_emb,self.n_cont = n_emb,n_cont
        sizes = [n_emb + n_cont] + layers + [out_sz]
        actns = [act_cls for _ in range(len(sizes)-2)] + [None]
        _layers = [LinBnDrop(sizes[i], sizes[i+1], bn=use_bn and (i!=len(actns)-1 or bn_final), p=p, act=a, lin_first=lin_first)
                       for i,(p,a) in enumerate(zip(ps+[0.],actns))]
        if y_range is not None: _layers.append(SigmoidRange(*y_range))
        self.layers = nn.Sequential(*_layers)

    def forward(self, x_cat, x_cont=None):
        if self.n_emb != 0:
            x = [e(x_cat[:,i]) for i,e in enumerate(self.embeds)]
            x = torch.cat(x, 1)
            x = self.emb_drop(x)
        if self.n_cont != 0:
            if self.bn_cont is not None: x_cont = self.bn_cont(x_cont)
            x = torch.cat([x, x_cont], 1) if self.n_emb != 0 else x_cont
        return self.layers(x)
```



```python
def forward(self, x_cat, x_cont=None):
    # Check if there are any embeddings to deal with
    if self.n_emb != 0:
        # Get the activations of each embedding matrix
        x = [e(x_cat[:,i]) for i,e in enumerate(self.embeds)]
        # Concatenate embeddings to a single tensor
        x = torch.cat(x, 1)
        # Apply dropout
        x = self.emb_drop(x)
    # Check if there are any continuous variables to deal with 
    if self.n_cont != 0:
        # Pass continuous variables through batch normalization layer
        if self.bn_cont is not None: x_cont = self.bn_cont(x_cont)
        # Concatenate continuous variables with embedding activations
        x = torch.cat([x, x_cont], 1) if self.n_emb != 0 else x_cont
    # Pass concatenated input through linear layers
    return self.layers(x)
```



## Conclusion

* deep learning can be challenging because your data, memory, and time are typically limited
* train a smaller model when memory is limited
* if you are not able to overfit your model to your data, you are not taking advantage of the capacity of your model
* You should first get to a point where you can overfit
* Steps to reduce overfitting in order of priority
    1. More data
     * add more labels to data you already have
     * find additional tasks your model could be asked to solve
     * create additional synthetic data by using more or different augmentation techniques
    2. Data augmentation
        * Mixup
    3. Generalizable architecture
        * Add batch normalization
    4. Regularization
        * Adding dropout to the last layer or two is often sufficient
        * Adding dropout of different types throughout your model can help even more
    5. Reduce architecture complexity
        * Should be the last thing you try




## References

* [Deep Learning for Coders with fastai & PyTorch](https://www.oreilly.com/library/view/deep-learning-for/9781492045519/)
* [The fastai book GitHub Repository](https://github.com/fastai/fastbook)



**Previous:** [Notes on fastai Book Ch. 14](https://christianjmills.com/Notes-on-Fastai-Book-14/)

**Next:** [Notes on fastai Book Ch. 16](https://christianjmills.com/Notes-on-Fastai-Book-16/)