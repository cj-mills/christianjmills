---
title: How to Create a LibTorch Plugin for Unity on Windows Pt. 1
date: 2022-6-28
image: /images/empty.gif
title-block-categories: true
layout: post
toc: false
hide: false
search_exclude: false
description: This follow-up to the fastai-to-unity tutorial covers creating a LibTorch
  plugin for the Unity game engine. Part 1 covers the required modifications to the
  original training code.
categories: [fastai, libtorch, unity]

aliases:
- /Fastai-to-LibTorch-to-Unity-Tutorial-Windows-1/

twitter-card:
  creator: "@cdotjdotmills"
  site: "@cdotjdotmills"
open-graph:
---

* [Introduction](#introduction)
* [Overview](#overview)
* [Install Dependencies](#install-dependencies)
* [Select a Model](#select-a-model)
* [Modify Transforms](#modify-transforms)
* [Define Learner](#define-learner)
* [Export the Model](#export-the-model)
* [Summary](#summary)



## Introduction

The previous [fastai-to-unity](../../fastai-to-unity-tutorial/part-1/) tutorial series implemented a [ResNet](https://arxiv.org/abs/1512.03385)-based image classifier in [Unity](https://unity.com/) with the [Barracuda](https://docs.unity3d.com/Packages/com.unity.barracuda@3.0/manual/index.html) inference library. The Barracuda library works well with the older ResNet architecture but does not support more recent ones like [ConvNeXt](https://arxiv.org/abs/2201.03545) and [MobileViT](https://arxiv.org/abs/2110.02178) at the time of writing. 

This follow-up series covers using [LibTorch](https://pytorch.org/cppdocs/installing.html), the C++ distribution of [PyTorch](https://pytorch.org/), to perform inference with these newer model architectures. We'll modify the original tutorial code and create a dynamic link library ([DLL](https://docs.microsoft.com/en-us/troubleshoot/windows-client/deployment/dynamic-link-library)) file to access the LibTorch functionality in Unity.



![libtorch-plugin-demo](./videos/libtorch-plugin-demo.mp4)





## Overview

This post covers the required modifications to the [original training code](https://github.com/cj-mills/fastai-to-unity-tutorial#training-code). We'll finetune models from the [Timm library](https://github.com/rwightman/pytorch-image-models) on the same [ASL dataset](https://www.kaggle.com/datasets/belalelwikel/asl-and-some-words) as the original tutorial. The Timm library provides access to a wide range of pretrained computer vision models and integrates with the [fastai library](https://docs.fast.ai/). Below is a link to the complete modified training code, along with links for running the notebook on Google Colab and Kaggle.

| GitHub Repository                                            | Colab                                                        | Kaggle                                                       |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| [Jupyter Notebook](https://github.com/cj-mills/fastai-to-libtorch-to-unity-tutorial/blob/main/notebooks/Fastai-timm-to-Torchscript-Tutorial.ipynb) | [Open in Colab](https://colab.research.google.com/github/cj-mills/fastai-to-libtorch-to-unity-tutorial/blob/main/notebooks/Fastai-timm-to-Torchscript-Tutorial.ipynb) | [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/cj-mills/fastai-to-libtorch-to-unity-tutorial/blob/main/notebooks/Fastai-timm-to-Torchscript-Tutorial.ipynb) |





## Install Dependencies

The [pip package](https://pypi.org/project/timm/) for the Timm library is more stable than the GitHub repository but has fewer model types and pretrained weights. For example, the pip package has [pretrained ConvNeXt models](https://github.com/rwightman/pytorch-image-models/blob/0.5.x/timm/models/convnext.py) but no MobileViT models. However, the latest GitHub version had some issues running the MobileNetV3 models at the time of writing.

Recent [updates](https://github.com/fastai/fastai/releases/tag/2.7.0) to the fastai library resolve some [performance issues](https://benjaminwarner.dev/2022/06/14/debugging-pytorch-performance-decrease) with PyTorch so let's update that too. They also provide a new `ChannelsLast` (beta) callback that further [improves performance](https://pytorch.org/tutorials/intermediate/memory_format_tutorial.html#performance-gains) on modern GPUs.

**Uncomment the cell below if running on Google Colab or Kaggle**


```python
# %%capture
# !pip3 install -U torch torchvision torchaudio
# !pip3 install -U fastai==2.7.2
# !pip3 install -U kaggle==1.5.12
# !pip3 install -U Pillow==9.1.0
# !pip3 install -U timm==0.5.4 # more stable fewer models
# !pip3 install -U git+https://github.com/rwightman/pytorch-image-models.git # more models less stable
```

**Note for Colab:**  You must restart the runtime in order to use newly installed version of Pillow.

**Import all fastai computer vision functionality**


```python
from fastai.vision.all import *
```

**Disable max rows and columns for pandas**


```python
import pandas as pd
pd.set_option('max_colwidth', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
```



## Select a Model

Let's start by selecting a model from the Timm library to finetune. The available pretrained models depend on the version of the Timm library installed.

**Import the Timm library**


```python
import timm
```


```python
timm.__version__
```


```text
'0.6.2.dev0'
```



**Check available pretrained model types**

We can check which model types have pretrained weights using the `timm.list_models()` function. 


```python
model_types = list(set([model.split('_')[0] for model in timm.list_models(pretrained=True)]))
model_types.sort()
pd.DataFrame(model_types)
```




<div style="overflow-x:auto; max-height:500px">
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
      <td>adv</td>
    </tr>
    <tr>
      <th>1</th>
      <td>bat</td>
    </tr>
    <tr>
      <th>2</th>
      <td>beit</td>
    </tr>
    <tr>
      <th>3</th>
      <td>botnet26t</td>
    </tr>
    <tr>
      <th>4</th>
      <td>cait</td>
    </tr>
    <tr>
      <th>5</th>
      <td>coat</td>
    </tr>
    <tr>
      <th>6</th>
      <td>convit</td>
    </tr>
    <tr>
      <th>7</th>
      <td>convmixer</td>
    </tr>
    <tr>
      <th>8</th>
      <td>convnext</td>
    </tr>
    <tr>
      <th>9</th>
      <td>crossvit</td>
    </tr>
    <tr>
      <th>10</th>
      <td>cspdarknet53</td>
    </tr>
    <tr>
      <th>11</th>
      <td>cspresnet50</td>
    </tr>
    <tr>
      <th>12</th>
      <td>cspresnext50</td>
    </tr>
    <tr>
      <th>13</th>
      <td>deit</td>
    </tr>
    <tr>
      <th>14</th>
      <td>densenet121</td>
    </tr>
    <tr>
      <th>15</th>
      <td>densenet161</td>
    </tr>
    <tr>
      <th>16</th>
      <td>densenet169</td>
    </tr>
    <tr>
      <th>17</th>
      <td>densenet201</td>
    </tr>
    <tr>
      <th>18</th>
      <td>densenetblur121d</td>
    </tr>
    <tr>
      <th>19</th>
      <td>dla102</td>
    </tr>
    <tr>
      <th>20</th>
      <td>dla102x</td>
    </tr>
    <tr>
      <th>21</th>
      <td>dla102x2</td>
    </tr>
    <tr>
      <th>22</th>
      <td>dla169</td>
    </tr>
    <tr>
      <th>23</th>
      <td>dla34</td>
    </tr>
    <tr>
      <th>24</th>
      <td>dla46</td>
    </tr>
    <tr>
      <th>25</th>
      <td>dla46x</td>
    </tr>
    <tr>
      <th>26</th>
      <td>dla60</td>
    </tr>
    <tr>
      <th>27</th>
      <td>dla60x</td>
    </tr>
    <tr>
      <th>28</th>
      <td>dm</td>
    </tr>
    <tr>
      <th>29</th>
      <td>dpn107</td>
    </tr>
    <tr>
      <th>30</th>
      <td>dpn131</td>
    </tr>
    <tr>
      <th>31</th>
      <td>dpn68</td>
    </tr>
    <tr>
      <th>32</th>
      <td>dpn68b</td>
    </tr>
    <tr>
      <th>33</th>
      <td>dpn92</td>
    </tr>
    <tr>
      <th>34</th>
      <td>dpn98</td>
    </tr>
    <tr>
      <th>35</th>
      <td>eca</td>
    </tr>
    <tr>
      <th>36</th>
      <td>ecaresnet101d</td>
    </tr>
    <tr>
      <th>37</th>
      <td>ecaresnet269d</td>
    </tr>
    <tr>
      <th>38</th>
      <td>ecaresnet26t</td>
    </tr>
    <tr>
      <th>39</th>
      <td>ecaresnet50d</td>
    </tr>
    <tr>
      <th>40</th>
      <td>ecaresnet50t</td>
    </tr>
    <tr>
      <th>41</th>
      <td>ecaresnetlight</td>
    </tr>
    <tr>
      <th>42</th>
      <td>efficientnet</td>
    </tr>
    <tr>
      <th>43</th>
      <td>efficientnetv2</td>
    </tr>
    <tr>
      <th>44</th>
      <td>ens</td>
    </tr>
    <tr>
      <th>45</th>
      <td>ese</td>
    </tr>
    <tr>
      <th>46</th>
      <td>fbnetc</td>
    </tr>
    <tr>
      <th>47</th>
      <td>fbnetv3</td>
    </tr>
    <tr>
      <th>48</th>
      <td>gc</td>
    </tr>
    <tr>
      <th>49</th>
      <td>gcresnet33ts</td>
    </tr>
    <tr>
      <th>50</th>
      <td>gcresnet50t</td>
    </tr>
    <tr>
      <th>51</th>
      <td>gcresnext26ts</td>
    </tr>
    <tr>
      <th>52</th>
      <td>gcresnext50ts</td>
    </tr>
    <tr>
      <th>53</th>
      <td>gernet</td>
    </tr>
    <tr>
      <th>54</th>
      <td>ghostnet</td>
    </tr>
    <tr>
      <th>55</th>
      <td>gluon</td>
    </tr>
    <tr>
      <th>56</th>
      <td>gmixer</td>
    </tr>
    <tr>
      <th>57</th>
      <td>gmlp</td>
    </tr>
    <tr>
      <th>58</th>
      <td>halo2botnet50ts</td>
    </tr>
    <tr>
      <th>59</th>
      <td>halonet26t</td>
    </tr>
    <tr>
      <th>60</th>
      <td>halonet50ts</td>
    </tr>
    <tr>
      <th>61</th>
      <td>haloregnetz</td>
    </tr>
    <tr>
      <th>62</th>
      <td>hardcorenas</td>
    </tr>
    <tr>
      <th>63</th>
      <td>hrnet</td>
    </tr>
    <tr>
      <th>64</th>
      <td>ig</td>
    </tr>
    <tr>
      <th>65</th>
      <td>inception</td>
    </tr>
    <tr>
      <th>66</th>
      <td>jx</td>
    </tr>
    <tr>
      <th>67</th>
      <td>lambda</td>
    </tr>
    <tr>
      <th>68</th>
      <td>lamhalobotnet50ts</td>
    </tr>
    <tr>
      <th>69</th>
      <td>lcnet</td>
    </tr>
    <tr>
      <th>70</th>
      <td>legacy</td>
    </tr>
    <tr>
      <th>71</th>
      <td>levit</td>
    </tr>
    <tr>
      <th>72</th>
      <td>mixer</td>
    </tr>
    <tr>
      <th>73</th>
      <td>mixnet</td>
    </tr>
    <tr>
      <th>74</th>
      <td>mnasnet</td>
    </tr>
    <tr>
      <th>75</th>
      <td>mobilenetv2</td>
    </tr>
    <tr>
      <th>76</th>
      <td>mobilenetv3</td>
    </tr>
    <tr>
      <th>77</th>
      <td>mobilevit</td>
    </tr>
    <tr>
      <th>78</th>
      <td>nasnetalarge</td>
    </tr>
    <tr>
      <th>79</th>
      <td>nf</td>
    </tr>
    <tr>
      <th>80</th>
      <td>nfnet</td>
    </tr>
    <tr>
      <th>81</th>
      <td>pit</td>
    </tr>
    <tr>
      <th>82</th>
      <td>pnasnet5large</td>
    </tr>
    <tr>
      <th>83</th>
      <td>poolformer</td>
    </tr>
    <tr>
      <th>84</th>
      <td>regnetv</td>
    </tr>
    <tr>
      <th>85</th>
      <td>regnetx</td>
    </tr>
    <tr>
      <th>86</th>
      <td>regnety</td>
    </tr>
    <tr>
      <th>87</th>
      <td>regnetz</td>
    </tr>
    <tr>
      <th>88</th>
      <td>repvgg</td>
    </tr>
    <tr>
      <th>89</th>
      <td>res2net101</td>
    </tr>
    <tr>
      <th>90</th>
      <td>res2net50</td>
    </tr>
    <tr>
      <th>91</th>
      <td>res2next50</td>
    </tr>
    <tr>
      <th>92</th>
      <td>resmlp</td>
    </tr>
    <tr>
      <th>93</th>
      <td>resnest101e</td>
    </tr>
    <tr>
      <th>94</th>
      <td>resnest14d</td>
    </tr>
    <tr>
      <th>95</th>
      <td>resnest200e</td>
    </tr>
    <tr>
      <th>96</th>
      <td>resnest269e</td>
    </tr>
    <tr>
      <th>97</th>
      <td>resnest26d</td>
    </tr>
    <tr>
      <th>98</th>
      <td>resnest50d</td>
    </tr>
    <tr>
      <th>99</th>
      <td>resnet101</td>
    </tr>
    <tr>
      <th>100</th>
      <td>resnet101d</td>
    </tr>
    <tr>
      <th>101</th>
      <td>resnet152</td>
    </tr>
    <tr>
      <th>102</th>
      <td>resnet152d</td>
    </tr>
    <tr>
      <th>103</th>
      <td>resnet18</td>
    </tr>
    <tr>
      <th>104</th>
      <td>resnet18d</td>
    </tr>
    <tr>
      <th>105</th>
      <td>resnet200d</td>
    </tr>
    <tr>
      <th>106</th>
      <td>resnet26</td>
    </tr>
    <tr>
      <th>107</th>
      <td>resnet26d</td>
    </tr>
    <tr>
      <th>108</th>
      <td>resnet26t</td>
    </tr>
    <tr>
      <th>109</th>
      <td>resnet32ts</td>
    </tr>
    <tr>
      <th>110</th>
      <td>resnet33ts</td>
    </tr>
    <tr>
      <th>111</th>
      <td>resnet34</td>
    </tr>
    <tr>
      <th>112</th>
      <td>resnet34d</td>
    </tr>
    <tr>
      <th>113</th>
      <td>resnet50</td>
    </tr>
    <tr>
      <th>114</th>
      <td>resnet50d</td>
    </tr>
    <tr>
      <th>115</th>
      <td>resnet51q</td>
    </tr>
    <tr>
      <th>116</th>
      <td>resnet61q</td>
    </tr>
    <tr>
      <th>117</th>
      <td>resnetblur50</td>
    </tr>
    <tr>
      <th>118</th>
      <td>resnetrs101</td>
    </tr>
    <tr>
      <th>119</th>
      <td>resnetrs152</td>
    </tr>
    <tr>
      <th>120</th>
      <td>resnetrs200</td>
    </tr>
    <tr>
      <th>121</th>
      <td>resnetrs270</td>
    </tr>
    <tr>
      <th>122</th>
      <td>resnetrs350</td>
    </tr>
    <tr>
      <th>123</th>
      <td>resnetrs420</td>
    </tr>
    <tr>
      <th>124</th>
      <td>resnetrs50</td>
    </tr>
    <tr>
      <th>125</th>
      <td>resnetv2</td>
    </tr>
    <tr>
      <th>126</th>
      <td>resnext101</td>
    </tr>
    <tr>
      <th>127</th>
      <td>resnext26ts</td>
    </tr>
    <tr>
      <th>128</th>
      <td>resnext50</td>
    </tr>
    <tr>
      <th>129</th>
      <td>resnext50d</td>
    </tr>
    <tr>
      <th>130</th>
      <td>rexnet</td>
    </tr>
    <tr>
      <th>131</th>
      <td>sebotnet33ts</td>
    </tr>
    <tr>
      <th>132</th>
      <td>sehalonet33ts</td>
    </tr>
    <tr>
      <th>133</th>
      <td>selecsls42b</td>
    </tr>
    <tr>
      <th>134</th>
      <td>selecsls60</td>
    </tr>
    <tr>
      <th>135</th>
      <td>selecsls60b</td>
    </tr>
    <tr>
      <th>136</th>
      <td>semnasnet</td>
    </tr>
    <tr>
      <th>137</th>
      <td>sequencer2d</td>
    </tr>
    <tr>
      <th>138</th>
      <td>seresnet152d</td>
    </tr>
    <tr>
      <th>139</th>
      <td>seresnet33ts</td>
    </tr>
    <tr>
      <th>140</th>
      <td>seresnet50</td>
    </tr>
    <tr>
      <th>141</th>
      <td>seresnext101</td>
    </tr>
    <tr>
      <th>142</th>
      <td>seresnext101d</td>
    </tr>
    <tr>
      <th>143</th>
      <td>seresnext26d</td>
    </tr>
    <tr>
      <th>144</th>
      <td>seresnext26t</td>
    </tr>
    <tr>
      <th>145</th>
      <td>seresnext26ts</td>
    </tr>
    <tr>
      <th>146</th>
      <td>seresnext50</td>
    </tr>
    <tr>
      <th>147</th>
      <td>seresnextaa101d</td>
    </tr>
    <tr>
      <th>148</th>
      <td>skresnet18</td>
    </tr>
    <tr>
      <th>149</th>
      <td>skresnet34</td>
    </tr>
    <tr>
      <th>150</th>
      <td>skresnext50</td>
    </tr>
    <tr>
      <th>151</th>
      <td>spnasnet</td>
    </tr>
    <tr>
      <th>152</th>
      <td>ssl</td>
    </tr>
    <tr>
      <th>153</th>
      <td>swin</td>
    </tr>
    <tr>
      <th>154</th>
      <td>swinv2</td>
    </tr>
    <tr>
      <th>155</th>
      <td>swsl</td>
    </tr>
    <tr>
      <th>156</th>
      <td>tf</td>
    </tr>
    <tr>
      <th>157</th>
      <td>tinynet</td>
    </tr>
    <tr>
      <th>158</th>
      <td>tnt</td>
    </tr>
    <tr>
      <th>159</th>
      <td>tresnet</td>
    </tr>
    <tr>
      <th>160</th>
      <td>tv</td>
    </tr>
    <tr>
      <th>161</th>
      <td>twins</td>
    </tr>
    <tr>
      <th>162</th>
      <td>vgg11</td>
    </tr>
    <tr>
      <th>163</th>
      <td>vgg13</td>
    </tr>
    <tr>
      <th>164</th>
      <td>vgg16</td>
    </tr>
    <tr>
      <th>165</th>
      <td>vgg19</td>
    </tr>
    <tr>
      <th>166</th>
      <td>visformer</td>
    </tr>
    <tr>
      <th>167</th>
      <td>vit</td>
    </tr>
    <tr>
      <th>168</th>
      <td>volo</td>
    </tr>
    <tr>
      <th>169</th>
      <td>wide</td>
    </tr>
    <tr>
      <th>170</th>
      <td>xception</td>
    </tr>
    <tr>
      <th>171</th>
      <td>xception41</td>
    </tr>
    <tr>
      <th>172</th>
      <td>xception41p</td>
    </tr>
    <tr>
      <th>173</th>
      <td>xception65</td>
    </tr>
    <tr>
      <th>174</th>
      <td>xception65p</td>
    </tr>
    <tr>
      <th>175</th>
      <td>xception71</td>
    </tr>
    <tr>
      <th>176</th>
      <td>xcit</td>
    </tr>
  </tbody>
</table>
</div>


Timm provides many pretrained models, but not all of them are fast enough for real-time applications. We can filter the results by providing a full or partial model name.



**Check available pretrained [ConvNeXt](https://arxiv.org/abs/2201.03545) models**


```python
pd.DataFrame(timm.list_models('convnext*', pretrained=True))
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
      <td>convnext_base</td>
    </tr>
    <tr>
      <th>1</th>
      <td>convnext_base_384_in22ft1k</td>
    </tr>
    <tr>
      <th>2</th>
      <td>convnext_base_in22ft1k</td>
    </tr>
    <tr>
      <th>3</th>
      <td>convnext_base_in22k</td>
    </tr>
    <tr>
      <th>4</th>
      <td>convnext_large</td>
    </tr>
    <tr>
      <th>5</th>
      <td>convnext_large_384_in22ft1k</td>
    </tr>
    <tr>
      <th>6</th>
      <td>convnext_large_in22ft1k</td>
    </tr>
    <tr>
      <th>7</th>
      <td>convnext_large_in22k</td>
    </tr>
    <tr>
      <th>8</th>
      <td>convnext_small</td>
    </tr>
    <tr>
      <th>9</th>
      <td>convnext_small_384_in22ft1k</td>
    </tr>
    <tr>
      <th>10</th>
      <td>convnext_small_in22ft1k</td>
    </tr>
    <tr>
      <th>11</th>
      <td>convnext_small_in22k</td>
    </tr>
    <tr>
      <th>12</th>
      <td>convnext_tiny</td>
    </tr>
    <tr>
      <th>13</th>
      <td>convnext_tiny_384_in22ft1k</td>
    </tr>
    <tr>
      <th>14</th>
      <td>convnext_tiny_hnf</td>
    </tr>
    <tr>
      <th>15</th>
      <td>convnext_tiny_in22ft1k</td>
    </tr>
    <tr>
      <th>16</th>
      <td>convnext_tiny_in22k</td>
    </tr>
    <tr>
      <th>17</th>
      <td>convnext_xlarge_384_in22ft1k</td>
    </tr>
    <tr>
      <th>18</th>
      <td>convnext_xlarge_in22ft1k</td>
    </tr>
    <tr>
      <th>19</th>
      <td>convnext_xlarge_in22k</td>
    </tr>
  </tbody>
</table>
</div>


Let's go with the `convnext_tiny` model since we want higher framerates. Each model comes with a set of default configuration parameters. We must keep track of the mean and std values used to normalize the model input. Many pretrained models use the ImageNet normalization stats, but others like MobileViT do not.



**Inspect the default configuration for the `convnext_tiny` model**

```python
from timm.models import convnext
convnext_model = 'convnext_tiny'
pd.DataFrame.from_dict(convnext.default_cfgs[convnext_model], orient='index')
```

<div style="overflow-x: auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>url</th>
      <td>https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth</td>
    </tr>
    <tr>
      <th>num_classes</th>
      <td>1000</td>
    </tr>
    <tr>
      <th>input_size</th>
      <td>(3, 224, 224)</td>
    </tr>
    <tr>
      <th>pool_size</th>
      <td>(7, 7)</td>
    </tr>
    <tr>
      <th>crop_pct</th>
      <td>0.875</td>
    </tr>
    <tr>
      <th>interpolation</th>
      <td>bicubic</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>(0.485, 0.456, 0.406)</td>
    </tr>
    <tr>
      <th>std</th>
      <td>(0.229, 0.224, 0.225)</td>
    </tr>
    <tr>
      <th>first_conv</th>
      <td>stem.0</td>
    </tr>
    <tr>
      <th>classifier</th>
      <td>head.fc</td>
    </tr>
  </tbody>
</table>
</div>


**Check available pretrained [MobileNetV2](https://arxiv.org/abs/1801.04381) models**


```python
pd.DataFrame(timm.list_models('mobilenetv2*', pretrained=True))
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
      <td>mobilenetv2_050</td>
    </tr>
    <tr>
      <th>1</th>
      <td>mobilenetv2_100</td>
    </tr>
    <tr>
      <th>2</th>
      <td>mobilenetv2_110d</td>
    </tr>
    <tr>
      <th>3</th>
      <td>mobilenetv2_120d</td>
    </tr>
    <tr>
      <th>4</th>
      <td>mobilenetv2_140</td>
    </tr>
  </tbody>
</table>
</div>


**Inspect the default configuration for the `mobilenetv2_050` model**

```python
from timm.models import efficientnet
mobilenetv2_model = 'mobilenetv2_050'
pd.DataFrame.from_dict(efficientnet.default_cfgs[mobilenetv2_model], orient='index')
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
      <th>url</th>
      <td>https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mobilenetv2_050-3d30d450.pth</td>
    </tr>
    <tr>
      <th>num_classes</th>
      <td>1000</td>
    </tr>
    <tr>
      <th>input_size</th>
      <td>(3, 224, 224)</td>
    </tr>
    <tr>
      <th>pool_size</th>
      <td>(7, 7)</td>
    </tr>
    <tr>
      <th>crop_pct</th>
      <td>0.875</td>
    </tr>
    <tr>
      <th>interpolation</th>
      <td>bicubic</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>(0.485, 0.456, 0.406)</td>
    </tr>
    <tr>
      <th>std</th>
      <td>(0.229, 0.224, 0.225)</td>
    </tr>
    <tr>
      <th>first_conv</th>
      <td>conv_stem</td>
    </tr>
    <tr>
      <th>classifier</th>
      <td>classifier</td>
    </tr>
  </tbody>
</table>
</div>



**Check available pretrained [MobileNetV3](https://arxiv.org/abs/1905.02244) models**


```python
pd.DataFrame(timm.list_models('mobilenetv3*', pretrained=True))
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
      <td>mobilenetv3_large_100</td>
    </tr>
    <tr>
      <th>1</th>
      <td>mobilenetv3_large_100_miil</td>
    </tr>
    <tr>
      <th>2</th>
      <td>mobilenetv3_large_100_miil_in21k</td>
    </tr>
    <tr>
      <th>3</th>
      <td>mobilenetv3_rw</td>
    </tr>
    <tr>
      <th>4</th>
      <td>mobilenetv3_small_050</td>
    </tr>
    <tr>
      <th>5</th>
      <td>mobilenetv3_small_075</td>
    </tr>
    <tr>
      <th>6</th>
      <td>mobilenetv3_small_100</td>
    </tr>
  </tbody>
</table>
</div>


**Inspect the default configuration for the `mobilenetv3_small_050` model**

```python
from timm.models import mobilenetv3
mobilenetv3_model = 'mobilenetv3_small_050'
pd.DataFrame.from_dict(mobilenetv3.default_cfgs[mobilenetv3_model], orient='index')
```


<div style="overflow-x:auto">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>url</th>
      <td>https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mobilenetv3_small_050_lambc-4b7bbe87.pth</td>
    </tr>
    <tr>
      <th>num_classes</th>
      <td>1000</td>
    </tr>
    <tr>
      <th>input_size</th>
      <td>(3, 224, 224)</td>
    </tr>
    <tr>
      <th>pool_size</th>
      <td>(7, 7)</td>
    </tr>
    <tr>
      <th>crop_pct</th>
      <td>0.875</td>
    </tr>
    <tr>
      <th>interpolation</th>
      <td>bicubic</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>(0.485, 0.456, 0.406)</td>
    </tr>
    <tr>
      <th>std</th>
      <td>(0.229, 0.224, 0.225)</td>
    </tr>
    <tr>
      <th>first_conv</th>
      <td>conv_stem</td>
    </tr>
    <tr>
      <th>classifier</th>
      <td>classifier</td>
    </tr>
  </tbody>
</table>
</div>



**Check available pretrained [MobileViT](https://arxiv.org/abs/2110.02178) models**
* **Note:** MobileViT models are not available in timm `0.5.4`


```python
pd.DataFrame(timm.list_models('mobilevit*', pretrained=True))
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
      <td>mobilevit_s</td>
    </tr>
    <tr>
      <th>1</th>
      <td>mobilevit_xs</td>
    </tr>
    <tr>
      <th>2</th>
      <td>mobilevit_xxs</td>
    </tr>
  </tbody>
</table>
</div>


**Inspect the default configuration for the `mobilevit_xxs` model**

```python
from timm.models import mobilevit
mobilevit_model = 'mobilevit_xxs'
pd.DataFrame.from_dict(mobilevit.default_cfgs[mobilevit_model], orient='index')
```

<div style="overflow-x:auto">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>url</th>
      <td>https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-mvit-weights/mobilevit_xxs-ad385b40.pth</td>
    </tr>
    <tr>
      <th>num_classes</th>
      <td>1000</td>
    </tr>
    <tr>
      <th>input_size</th>
      <td>(3, 256, 256)</td>
    </tr>
    <tr>
      <th>pool_size</th>
      <td>(8, 8)</td>
    </tr>
    <tr>
      <th>crop_pct</th>
      <td>0.9</td>
    </tr>
    <tr>
      <th>interpolation</th>
      <td>bicubic</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>(0, 0, 0)</td>
    </tr>
    <tr>
      <th>std</th>
      <td>(1, 1, 1)</td>
    </tr>
    <tr>
      <th>first_conv</th>
      <td>stem.conv</td>
    </tr>
    <tr>
      <th>classifier</th>
      <td>head.fc</td>
    </tr>
    <tr>
      <th>fixed_input_size</th>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>



**Select a model**

```python
model_type = convnext
model_name = convnext_model
# model_type = efficientnet
# model_name = mobilenetv2_model
# model_type = mobilenetv3
# model_name = mobilenetv3_model
# model_type = mobilevit
# model_name = mobilevit_model
```



After picking a model, we'll store the related normalization stats for future use.



**Store normalization stats**


```python
mean = model_type.default_cfgs[model_name]['mean']
std = model_type.default_cfgs[model_name]['std']
mean, std
```


```text
((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
```





**Define target input dimensions**


```python
# size_1_1 = (224, 224)
# size_3_2 = (224, 336)
# size_4_3 = (216, 288)
size_16_9 = (216, 384)
# size_16_9_l = (288, 512)
```


```python
input_dims = size_16_9
```



## Modify Transforms

We can apply the normalization stats at the end of the batch transforms.

```python
item_tfms = [FlipItem(p=1.0), Resize(input_dims, method=ResizeMethod.Pad, pad_mode=PadMode.Border)]

batch_tfms = [
    Contrast(max_lighting=0.25),
    Saturation(max_lighting=0.25),
    Hue(max_hue=0.05),
    *aug_transforms(
        size=input_dims, 
        mult=1.0,
        do_flip=False,
        flip_vert=False,
        max_rotate=0.0,
        min_zoom=0.5,
        max_zoom=1.5,
        max_lighting=0.5,
        max_warp=0.2, 
        p_affine=0.0,
        pad_mode=PadMode.Border),
    Normalize.from_stats(mean=mean, std=std)
]
```





## Define Learner

The training process is identical to the original tutorial, and we only need to pass the name of the Timm model to the `vision_learner` object.


```python
learn = vision_learner(dls, model_name, metrics=metrics).to_fp16()
# learn = vision_learner(dls, model_name, metrics=metrics, cbs=[ChannelsLast]).to_fp16()
```







## Export the Model

Once training completes, we need to convert our trained PyTorch model to a [TorchScript](https://pytorch.org/docs/stable/jit.html) module for use in LibTorch. We do so using the [`torch.jit.trace()`](https://pytorch.org/docs/stable/generated/torch.jit.trace.html) method.



**Generate a TorchScript module using the test image**


```python
traced_script_module = torch.jit.trace(learn.model.cpu(), batched_tensor)
```



We can perform inference with the TorchScript module the same way we would a PyTorch model.



**Verify the TorchScript module's accuracy**


```python
with torch.no_grad():
    torchscript_preds = traced_script_module(batched_tensor)
learn.dls.vocab[torch.nn.functional.softmax(torchscript_preds, dim=1).argmax()]
```

```text
'J'
```



**Define TorchScript file name**


```python
module_file_name = f"{dataset_path.name}-{learn.arch}.pt"
module_file_name
```

```text
'asl-and-some-words-convnext_tiny.pt'
```



Some models like MobileViT will require the exact input dimensions in LibTorch as was used in the `torch.jit.trace()` method. Therefore we'll convert the PyTorch model again using the training dimensions before saving the TorchScript module to a file.



**Generate a torchscript module using the target input dimensions and save it to a file**


```python
torch.randn(1, 3, *input_dims).shape
```

```text
torch.Size([1, 3, 216, 384])
```




```python
traced_script_module = torch.jit.trace(learn.model.cpu(), torch.randn(1, 3, *input_dims))
traced_script_module.save(module_file_name)
```





We can export the normalization stats to a JSON file using the same method for the class labels. We'll load the stats in Unity and pass them to the LibTorch plugin.



**Export model normalization stats**


```python
normalization_stats = {"mean": list(mean), "std": list(std)}
normalization_stats_file_name = f"{learn.arch}-normalization_stats.json"

with open(normalization_stats_file_name, "w") as write_file:
    json.dump(normalization_stats, write_file)
```






## Summary

This post covered how to modify the training code from the [fastai-to-unity tutorial ](../../fastai-to-unity-tutorial/part-1/)to finetune models from the Timm library and export them as TorchScript modules. Part 2 will cover creating a dynamic link library ([DLL](https://docs.microsoft.com/en-us/troubleshoot/windows-client/deployment/dynamic-link-library)) file in Visual Studio to perform inference with these TorchScript modules using [LibTorch](https://pytorch.org/cppdocs/installing.html).





**Previous:** [Fastai to Unity Tutorial Pt. 3](../../fastai-to-unity-tutorial/part-3/)

**Next:** [How to Create a LibTorch Plugin for Unity on Windows Pt.2](../part-2/)



**Project Resources:** [GitHub Repository](https://github.com/cj-mills/fastai-to-libtorch-to-unity-tutorial)



<!-- Cloudflare Web Analytics --><script defer src='https://static.cloudflareinsights.com/beacon.min.js' data-cf-beacon='{"token": "56b8d2f624604c4891327b3c0d9f6703"}'></script><!-- End Cloudflare Web Analytics -->



