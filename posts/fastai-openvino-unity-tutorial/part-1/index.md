---
title: How to Create an OpenVINO Plugin for Unity on Windows Pt. 1
date: 2022-7-17
image: /images/empty.gif
title-block-categories: true
layout: post
toc: false
hide: false
search_exclude: false
comments:
  utterances:
    repo: cj-mills/christianjmills
description: This follow-up to the fastai-to-unity tutorial covers creating an OpenVINO
  plugin for the Unity game engine. Part 1 covers the required modifications to the
  original training code.
categories: [fastai, openvino, unity]

aliases:
- /Fastai-to-OpenVINO-to-Unity-Tutorial-Windows-1/

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
* [Benchmark OpenVINO Inference](#benchmark-openvino-inference)
* [Summary](#summary)



## Introduction

This tutorial is a follow-up to the [fastai-to-unity](../../fastai-to-unity-tutorial/part-1) tutorial series and covers using [OpenVINO](https://docs.openvino.ai/latest/index.html), an open-source toolkit for optimizing model inference, instead of Unity's Barracuda library. OpenVINO enables significantly faster CPU inference than Barracuda and supports more model types. It also supports GPU inference for integrated and discrete Intel GPUs and will be able to leverage the AI hardware acceleration available in Intel's upcoming ARC GPUs.

We'll modify the [original tutorial code](https://github.com/cj-mills/fastai-to-unity-tutorial) and create a dynamic link library ([DLL](https://docs.microsoft.com/en-us/troubleshoot/windows-client/deployment/dynamic-link-library)) file to access the OpenVINO functionality in Unity.

![openvino-plugin-demo](./videos/openvino-plugin-demo.mp4)



## Overview

This post covers the required modifications to the [original training code](https://github.com/cj-mills/fastai-to-unity-tutorial#training-code). We'll finetune models from the [Timm library](https://github.com/rwightman/pytorch-image-models) on the same [ASL dataset](https://www.kaggle.com/datasets/belalelwikel/asl-and-some-words) as the original tutorial, just like in this [previous follow-up](../../fastai-libtorch-unity-tutorial/part-1/). Below is a link to the complete modified training code, along with links for running the notebook on Google Colab and Kaggle.

| GitHub Repository                                            | Colab                                                        | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Kaggle&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| [Jupyter Notebook](https://github.com/cj-mills/fastai-to-openvino-to-unity-tutorial/blob/main/notebooks/Fastai-timm-to-OpenVINO-Tutorial.ipynb) | [Open in Colab](https://colab.research.google.com/github/cj-mills/fastai-to-openvino-to-unity-tutorial/blob/main/notebooks/Fastai-timm-to-OpenVINO-Tutorial.ipynb) | [<img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Kaggle"  />](https://kaggle.com/kernels/welcome?src=https://github.com/cj-mills/fastai-to-openvino-to-unity-tutorial/blob/main/notebooks/Fastai-timm-to-OpenVINO-Tutorial.ipynb) |





## Install Dependencies

The [pip package](https://pypi.org/project/timm/) for the Timm library is generally more stable than the GitHub repository but may have fewer model types and pretrained weights. However, the latest pip version had some issues running the MobileNetV3 models at the time of writing. Downgrade to version `0.5.4` to use those models.

Recent [updates](https://github.com/fastai/fastai/releases/tag/2.7.0) to the fastai library resolve some [performance issues](https://benjaminwarner.dev/2022/06/14/debugging-pytorch-performance-decrease) with PyTorch so let's update that too.

We need to install the [`openvino-dev`](https://pypi.org/project/openvino-dev/) pip package to convert trained models to OpenVINO's [Intermediate Representation](https://docs.openvino.ai/latest/openvino_docs_MO_DG_IR_and_opsets.html) (IR) format.



**Uncomment the cell below if running on Google Colab or Kaggle**


```python
# %%capture
# !pip3 install -U torch torchvision torchaudio
# !pip3 install -U fastai==2.7.6
# !pip3 install -U kaggle==1.5.12
# !pip3 install -U Pillow==9.1.0
# !pip3 install -U timm==0.6.5 # more stable fewer models
# # !pip3 install -U git+https://github.com/rwightman/pytorch-image-models.git # more models less stable
# !pip3 install openvino-dev==2022.1.0 
```

**Note for Colab:**  You must restart the runtime in order to use newly installed version of Pillow.

**Import all fastai computer vision functionality**


```python
from fastai.vision.all import *
```


```python
import fastai
```


```python
fastai.__version__
```
```text
'2.7.6'
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
'0.6.5'
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
      <td>cs3darknet</td>
    </tr>
    <tr>
      <th>11</th>
      <td>cspdarknet53</td>
    </tr>
    <tr>
      <th>12</th>
      <td>cspresnet50</td>
    </tr>
    <tr>
      <th>13</th>
      <td>cspresnext50</td>
    </tr>
    <tr>
      <th>14</th>
      <td>darknet53</td>
    </tr>
    <tr>
      <th>15</th>
      <td>deit</td>
    </tr>
    <tr>
      <th>16</th>
      <td>deit3</td>
    </tr>
    <tr>
      <th>17</th>
      <td>densenet121</td>
    </tr>
    <tr>
      <th>18</th>
      <td>densenet161</td>
    </tr>
    <tr>
      <th>19</th>
      <td>densenet169</td>
    </tr>
    <tr>
      <th>20</th>
      <td>densenet201</td>
    </tr>
    <tr>
      <th>21</th>
      <td>densenetblur121d</td>
    </tr>
    <tr>
      <th>22</th>
      <td>dla102</td>
    </tr>
    <tr>
      <th>23</th>
      <td>dla102x</td>
    </tr>
    <tr>
      <th>24</th>
      <td>dla102x2</td>
    </tr>
    <tr>
      <th>25</th>
      <td>dla169</td>
    </tr>
    <tr>
      <th>26</th>
      <td>dla34</td>
    </tr>
    <tr>
      <th>27</th>
      <td>dla46</td>
    </tr>
    <tr>
      <th>28</th>
      <td>dla46x</td>
    </tr>
    <tr>
      <th>29</th>
      <td>dla60</td>
    </tr>
    <tr>
      <th>30</th>
      <td>dla60x</td>
    </tr>
    <tr>
      <th>31</th>
      <td>dm</td>
    </tr>
    <tr>
      <th>32</th>
      <td>dpn107</td>
    </tr>
    <tr>
      <th>33</th>
      <td>dpn131</td>
    </tr>
    <tr>
      <th>34</th>
      <td>dpn68</td>
    </tr>
    <tr>
      <th>35</th>
      <td>dpn68b</td>
    </tr>
    <tr>
      <th>36</th>
      <td>dpn92</td>
    </tr>
    <tr>
      <th>37</th>
      <td>dpn98</td>
    </tr>
    <tr>
      <th>38</th>
      <td>eca</td>
    </tr>
    <tr>
      <th>39</th>
      <td>ecaresnet101d</td>
    </tr>
    <tr>
      <th>40</th>
      <td>ecaresnet269d</td>
    </tr>
    <tr>
      <th>41</th>
      <td>ecaresnet26t</td>
    </tr>
    <tr>
      <th>42</th>
      <td>ecaresnet50d</td>
    </tr>
    <tr>
      <th>43</th>
      <td>ecaresnet50t</td>
    </tr>
    <tr>
      <th>44</th>
      <td>ecaresnetlight</td>
    </tr>
    <tr>
      <th>45</th>
      <td>edgenext</td>
    </tr>
    <tr>
      <th>46</th>
      <td>efficientnet</td>
    </tr>
    <tr>
      <th>47</th>
      <td>efficientnetv2</td>
    </tr>
    <tr>
      <th>48</th>
      <td>ens</td>
    </tr>
    <tr>
      <th>49</th>
      <td>ese</td>
    </tr>
    <tr>
      <th>50</th>
      <td>fbnetc</td>
    </tr>
    <tr>
      <th>51</th>
      <td>fbnetv3</td>
    </tr>
    <tr>
      <th>52</th>
      <td>gc</td>
    </tr>
    <tr>
      <th>53</th>
      <td>gcresnet33ts</td>
    </tr>
    <tr>
      <th>54</th>
      <td>gcresnet50t</td>
    </tr>
    <tr>
      <th>55</th>
      <td>gcresnext26ts</td>
    </tr>
    <tr>
      <th>56</th>
      <td>gcresnext50ts</td>
    </tr>
    <tr>
      <th>57</th>
      <td>gernet</td>
    </tr>
    <tr>
      <th>58</th>
      <td>ghostnet</td>
    </tr>
    <tr>
      <th>59</th>
      <td>gluon</td>
    </tr>
    <tr>
      <th>60</th>
      <td>gmixer</td>
    </tr>
    <tr>
      <th>61</th>
      <td>gmlp</td>
    </tr>
    <tr>
      <th>62</th>
      <td>halo2botnet50ts</td>
    </tr>
    <tr>
      <th>63</th>
      <td>halonet26t</td>
    </tr>
    <tr>
      <th>64</th>
      <td>halonet50ts</td>
    </tr>
    <tr>
      <th>65</th>
      <td>haloregnetz</td>
    </tr>
    <tr>
      <th>66</th>
      <td>hardcorenas</td>
    </tr>
    <tr>
      <th>67</th>
      <td>hrnet</td>
    </tr>
    <tr>
      <th>68</th>
      <td>ig</td>
    </tr>
    <tr>
      <th>69</th>
      <td>inception</td>
    </tr>
    <tr>
      <th>70</th>
      <td>jx</td>
    </tr>
    <tr>
      <th>71</th>
      <td>lambda</td>
    </tr>
    <tr>
      <th>72</th>
      <td>lamhalobotnet50ts</td>
    </tr>
    <tr>
      <th>73</th>
      <td>lcnet</td>
    </tr>
    <tr>
      <th>74</th>
      <td>legacy</td>
    </tr>
    <tr>
      <th>75</th>
      <td>levit</td>
    </tr>
    <tr>
      <th>76</th>
      <td>mixer</td>
    </tr>
    <tr>
      <th>77</th>
      <td>mixnet</td>
    </tr>
    <tr>
      <th>78</th>
      <td>mnasnet</td>
    </tr>
    <tr>
      <th>79</th>
      <td>mobilenetv2</td>
    </tr>
    <tr>
      <th>80</th>
      <td>mobilenetv3</td>
    </tr>
    <tr>
      <th>81</th>
      <td>mobilevit</td>
    </tr>
    <tr>
      <th>82</th>
      <td>mobilevitv2</td>
    </tr>
    <tr>
      <th>83</th>
      <td>nasnetalarge</td>
    </tr>
    <tr>
      <th>84</th>
      <td>nf</td>
    </tr>
    <tr>
      <th>85</th>
      <td>nfnet</td>
    </tr>
    <tr>
      <th>86</th>
      <td>pit</td>
    </tr>
    <tr>
      <th>87</th>
      <td>pnasnet5large</td>
    </tr>
    <tr>
      <th>88</th>
      <td>poolformer</td>
    </tr>
    <tr>
      <th>89</th>
      <td>regnetv</td>
    </tr>
    <tr>
      <th>90</th>
      <td>regnetx</td>
    </tr>
    <tr>
      <th>91</th>
      <td>regnety</td>
    </tr>
    <tr>
      <th>92</th>
      <td>regnetz</td>
    </tr>
    <tr>
      <th>93</th>
      <td>repvgg</td>
    </tr>
    <tr>
      <th>94</th>
      <td>res2net101</td>
    </tr>
    <tr>
      <th>95</th>
      <td>res2net50</td>
    </tr>
    <tr>
      <th>96</th>
      <td>res2next50</td>
    </tr>
    <tr>
      <th>97</th>
      <td>resmlp</td>
    </tr>
    <tr>
      <th>98</th>
      <td>resnest101e</td>
    </tr>
    <tr>
      <th>99</th>
      <td>resnest14d</td>
    </tr>
    <tr>
      <th>100</th>
      <td>resnest200e</td>
    </tr>
    <tr>
      <th>101</th>
      <td>resnest269e</td>
    </tr>
    <tr>
      <th>102</th>
      <td>resnest26d</td>
    </tr>
    <tr>
      <th>103</th>
      <td>resnest50d</td>
    </tr>
    <tr>
      <th>104</th>
      <td>resnet101</td>
    </tr>
    <tr>
      <th>105</th>
      <td>resnet101d</td>
    </tr>
    <tr>
      <th>106</th>
      <td>resnet10t</td>
    </tr>
    <tr>
      <th>107</th>
      <td>resnet14t</td>
    </tr>
    <tr>
      <th>108</th>
      <td>resnet152</td>
    </tr>
    <tr>
      <th>109</th>
      <td>resnet152d</td>
    </tr>
    <tr>
      <th>110</th>
      <td>resnet18</td>
    </tr>
    <tr>
      <th>111</th>
      <td>resnet18d</td>
    </tr>
    <tr>
      <th>112</th>
      <td>resnet200d</td>
    </tr>
    <tr>
      <th>113</th>
      <td>resnet26</td>
    </tr>
    <tr>
      <th>114</th>
      <td>resnet26d</td>
    </tr>
    <tr>
      <th>115</th>
      <td>resnet26t</td>
    </tr>
    <tr>
      <th>116</th>
      <td>resnet32ts</td>
    </tr>
    <tr>
      <th>117</th>
      <td>resnet33ts</td>
    </tr>
    <tr>
      <th>118</th>
      <td>resnet34</td>
    </tr>
    <tr>
      <th>119</th>
      <td>resnet34d</td>
    </tr>
    <tr>
      <th>120</th>
      <td>resnet50</td>
    </tr>
    <tr>
      <th>121</th>
      <td>resnet50d</td>
    </tr>
    <tr>
      <th>122</th>
      <td>resnet51q</td>
    </tr>
    <tr>
      <th>123</th>
      <td>resnet61q</td>
    </tr>
    <tr>
      <th>124</th>
      <td>resnetaa50</td>
    </tr>
    <tr>
      <th>125</th>
      <td>resnetblur50</td>
    </tr>
    <tr>
      <th>126</th>
      <td>resnetrs101</td>
    </tr>
    <tr>
      <th>127</th>
      <td>resnetrs152</td>
    </tr>
    <tr>
      <th>128</th>
      <td>resnetrs200</td>
    </tr>
    <tr>
      <th>129</th>
      <td>resnetrs270</td>
    </tr>
    <tr>
      <th>130</th>
      <td>resnetrs350</td>
    </tr>
    <tr>
      <th>131</th>
      <td>resnetrs420</td>
    </tr>
    <tr>
      <th>132</th>
      <td>resnetrs50</td>
    </tr>
    <tr>
      <th>133</th>
      <td>resnetv2</td>
    </tr>
    <tr>
      <th>134</th>
      <td>resnext101</td>
    </tr>
    <tr>
      <th>135</th>
      <td>resnext26ts</td>
    </tr>
    <tr>
      <th>136</th>
      <td>resnext50</td>
    </tr>
    <tr>
      <th>137</th>
      <td>resnext50d</td>
    </tr>
    <tr>
      <th>138</th>
      <td>rexnet</td>
    </tr>
    <tr>
      <th>139</th>
      <td>sebotnet33ts</td>
    </tr>
    <tr>
      <th>140</th>
      <td>sehalonet33ts</td>
    </tr>
    <tr>
      <th>141</th>
      <td>selecsls42b</td>
    </tr>
    <tr>
      <th>142</th>
      <td>selecsls60</td>
    </tr>
    <tr>
      <th>143</th>
      <td>selecsls60b</td>
    </tr>
    <tr>
      <th>144</th>
      <td>semnasnet</td>
    </tr>
    <tr>
      <th>145</th>
      <td>sequencer2d</td>
    </tr>
    <tr>
      <th>146</th>
      <td>seresnet152d</td>
    </tr>
    <tr>
      <th>147</th>
      <td>seresnet33ts</td>
    </tr>
    <tr>
      <th>148</th>
      <td>seresnet50</td>
    </tr>
    <tr>
      <th>149</th>
      <td>seresnext101</td>
    </tr>
    <tr>
      <th>150</th>
      <td>seresnext101d</td>
    </tr>
    <tr>
      <th>151</th>
      <td>seresnext26d</td>
    </tr>
    <tr>
      <th>152</th>
      <td>seresnext26t</td>
    </tr>
    <tr>
      <th>153</th>
      <td>seresnext26ts</td>
    </tr>
    <tr>
      <th>154</th>
      <td>seresnext50</td>
    </tr>
    <tr>
      <th>155</th>
      <td>seresnextaa101d</td>
    </tr>
    <tr>
      <th>156</th>
      <td>skresnet18</td>
    </tr>
    <tr>
      <th>157</th>
      <td>skresnet34</td>
    </tr>
    <tr>
      <th>158</th>
      <td>skresnext50</td>
    </tr>
    <tr>
      <th>159</th>
      <td>spnasnet</td>
    </tr>
    <tr>
      <th>160</th>
      <td>ssl</td>
    </tr>
    <tr>
      <th>161</th>
      <td>swin</td>
    </tr>
    <tr>
      <th>162</th>
      <td>swinv2</td>
    </tr>
    <tr>
      <th>163</th>
      <td>swsl</td>
    </tr>
    <tr>
      <th>164</th>
      <td>tf</td>
    </tr>
    <tr>
      <th>165</th>
      <td>tinynet</td>
    </tr>
    <tr>
      <th>166</th>
      <td>tnt</td>
    </tr>
    <tr>
      <th>167</th>
      <td>tresnet</td>
    </tr>
    <tr>
      <th>168</th>
      <td>tv</td>
    </tr>
    <tr>
      <th>169</th>
      <td>twins</td>
    </tr>
    <tr>
      <th>170</th>
      <td>vgg11</td>
    </tr>
    <tr>
      <th>171</th>
      <td>vgg13</td>
    </tr>
    <tr>
      <th>172</th>
      <td>vgg16</td>
    </tr>
    <tr>
      <th>173</th>
      <td>vgg19</td>
    </tr>
    <tr>
      <th>174</th>
      <td>visformer</td>
    </tr>
    <tr>
      <th>175</th>
      <td>vit</td>
    </tr>
    <tr>
      <th>176</th>
      <td>volo</td>
    </tr>
    <tr>
      <th>177</th>
      <td>wide</td>
    </tr>
    <tr>
      <th>178</th>
      <td>xception</td>
    </tr>
    <tr>
      <th>179</th>
      <td>xception41</td>
    </tr>
    <tr>
      <th>180</th>
      <td>xception41p</td>
    </tr>
    <tr>
      <th>181</th>
      <td>xception65</td>
    </tr>
    <tr>
      <th>182</th>
      <td>xception65p</td>
    </tr>
    <tr>
      <th>183</th>
      <td>xception71</td>
    </tr>
    <tr>
      <th>184</th>
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


Let's go with the `convnext_tiny` model since we want higher framerates. Each model comes with a set of default configuration parameters. We must keep track of the mean and std values used to normalize the model input.




**Inspect the default configuration for the `convnext_tiny` model**


```python
from timm.models import convnext
convnext_model = 'convnext_tiny'
pd.DataFrame.from_dict(convnext.default_cfgs[convnext_model], orient='index')
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



**Inspect the default configuration for the `mobilenetv2_100` model**


```python
from timm.models import efficientnet
mobilenetv2_model = 'mobilenetv2_100'
pd.DataFrame.from_dict(efficientnet.default_cfgs[mobilenetv2_model], orient='index')
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
      <th>url</th>
      <td>https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mobilenetv2_100_ra-b33bc2c4.pth</td>
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



**Check available pretrained [ResNet]() models**


```python
pd.DataFrame(timm.list_models('resnet*', pretrained=True))
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
      <td>resnet10t</td>
    </tr>
    <tr>
      <th>1</th>
      <td>resnet14t</td>
    </tr>
    <tr>
      <th>2</th>
      <td>resnet18</td>
    </tr>
    <tr>
      <th>3</th>
      <td>resnet18d</td>
    </tr>
    <tr>
      <th>4</th>
      <td>resnet26</td>
    </tr>
    <tr>
      <th>5</th>
      <td>resnet26d</td>
    </tr>
    <tr>
      <th>6</th>
      <td>resnet26t</td>
    </tr>
    <tr>
      <th>7</th>
      <td>resnet32ts</td>
    </tr>
    <tr>
      <th>8</th>
      <td>resnet33ts</td>
    </tr>
    <tr>
      <th>9</th>
      <td>resnet34</td>
    </tr>
    <tr>
      <th>10</th>
      <td>resnet34d</td>
    </tr>
    <tr>
      <th>11</th>
      <td>resnet50</td>
    </tr>
    <tr>
      <th>12</th>
      <td>resnet50_gn</td>
    </tr>
    <tr>
      <th>13</th>
      <td>resnet50d</td>
    </tr>
    <tr>
      <th>14</th>
      <td>resnet51q</td>
    </tr>
    <tr>
      <th>15</th>
      <td>resnet61q</td>
    </tr>
    <tr>
      <th>16</th>
      <td>resnet101</td>
    </tr>
    <tr>
      <th>17</th>
      <td>resnet101d</td>
    </tr>
    <tr>
      <th>18</th>
      <td>resnet152</td>
    </tr>
    <tr>
      <th>19</th>
      <td>resnet152d</td>
    </tr>
    <tr>
      <th>20</th>
      <td>resnet200d</td>
    </tr>
    <tr>
      <th>21</th>
      <td>resnetaa50</td>
    </tr>
    <tr>
      <th>22</th>
      <td>resnetblur50</td>
    </tr>
    <tr>
      <th>23</th>
      <td>resnetrs50</td>
    </tr>
    <tr>
      <th>24</th>
      <td>resnetrs101</td>
    </tr>
    <tr>
      <th>25</th>
      <td>resnetrs152</td>
    </tr>
    <tr>
      <th>26</th>
      <td>resnetrs200</td>
    </tr>
    <tr>
      <th>27</th>
      <td>resnetrs270</td>
    </tr>
    <tr>
      <th>28</th>
      <td>resnetrs350</td>
    </tr>
    <tr>
      <th>29</th>
      <td>resnetrs420</td>
    </tr>
    <tr>
      <th>30</th>
      <td>resnetv2_50</td>
    </tr>
    <tr>
      <th>31</th>
      <td>resnetv2_50d_evos</td>
    </tr>
    <tr>
      <th>32</th>
      <td>resnetv2_50d_gn</td>
    </tr>
    <tr>
      <th>33</th>
      <td>resnetv2_50x1_bit_distilled</td>
    </tr>
    <tr>
      <th>34</th>
      <td>resnetv2_50x1_bitm</td>
    </tr>
    <tr>
      <th>35</th>
      <td>resnetv2_50x1_bitm_in21k</td>
    </tr>
    <tr>
      <th>36</th>
      <td>resnetv2_50x3_bitm</td>
    </tr>
    <tr>
      <th>37</th>
      <td>resnetv2_50x3_bitm_in21k</td>
    </tr>
    <tr>
      <th>38</th>
      <td>resnetv2_101</td>
    </tr>
    <tr>
      <th>39</th>
      <td>resnetv2_101x1_bitm</td>
    </tr>
    <tr>
      <th>40</th>
      <td>resnetv2_101x1_bitm_in21k</td>
    </tr>
    <tr>
      <th>41</th>
      <td>resnetv2_101x3_bitm</td>
    </tr>
    <tr>
      <th>42</th>
      <td>resnetv2_101x3_bitm_in21k</td>
    </tr>
    <tr>
      <th>43</th>
      <td>resnetv2_152x2_bit_teacher</td>
    </tr>
    <tr>
      <th>44</th>
      <td>resnetv2_152x2_bit_teacher_384</td>
    </tr>
    <tr>
      <th>45</th>
      <td>resnetv2_152x2_bitm</td>
    </tr>
    <tr>
      <th>46</th>
      <td>resnetv2_152x2_bitm_in21k</td>
    </tr>
    <tr>
      <th>47</th>
      <td>resnetv2_152x4_bitm</td>
    </tr>
    <tr>
      <th>48</th>
      <td>resnetv2_152x4_bitm_in21k</td>
    </tr>
  </tbody>
</table>
</div>



**Inspect the default configuration for the `resnet10t` model**


```python
from timm.models import resnet
resnet_model = 'resnet10t'
pd.DataFrame.from_dict(resnet.default_cfgs[resnet_model], orient='index')
```
<div style="overflow-x:auto; max-height:600px">
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
      <td>https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet10t_176_c3-f3215ab1.pth</td>
    </tr>
    <tr>
      <th>num_classes</th>
      <td>1000</td>
    </tr>
    <tr>
      <th>input_size</th>
      <td>(3, 176, 176)</td>
    </tr>
    <tr>
      <th>pool_size</th>
      <td>(6, 6)</td>
    </tr>
    <tr>
      <th>crop_pct</th>
      <td>0.875</td>
    </tr>
    <tr>
      <th>interpolation</th>
      <td>bilinear</td>
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
      <td>conv1.0</td>
    </tr>
    <tr>
      <th>classifier</th>
      <td>fc</td>
    </tr>
    <tr>
      <th>test_crop_pct</th>
      <td>0.95</td>
    </tr>
    <tr>
      <th>test_input_size</th>
      <td>(3, 224, 224)</td>
    </tr>
  </tbody>
</table>
</div>



**Select a model**


```python
# model_type = convnext
# model_name = convnext_model
# model_type = efficientnet
# model_name = mobilenetv2_model
model_type = resnet
model_name = resnet_model
```

**Store normalization stats**


```python
mean = model_type.default_cfgs[model_name]['mean']
std = model_type.default_cfgs[model_name]['std']
mean, std
```
```text
((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
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
```





## Export the Model

The OpenVINO model conversion script does not support PyTorch models, so we need to export the trained model to ONNX. We can then convert the ONNX model to OpenVINO's IR format.



**Define ONNX file name**

```python
onnx_file_name = f"{dataset_path.name}-{learn.arch}.onnx"
onnx_file_name
```
```text
'asl-and-some-words-resnet10t.onnx'
```

**Export trained model to ONNX**


```python
torch.onnx.export(learn.model.cpu(),
                  batched_tensor,
                  onnx_file_name,
                  export_params=True,
                  opset_version=11,
                  do_constant_folding=False,
                  input_names = ['input'],
                  output_names = ['output'],
                  dynamic_axes={'input': {2 : 'height', 3 : 'width'}}
                 )
```

Now we can define the argument for OpenVINO's model conversion script.



**Import OpenVINO Dependencies**


```python
from IPython.display import Markdown, display
```


```python
from openvino.runtime import Core
```

**Define export directory**


```python
output_dir = Path('./')
output_dir
```
```text
Path('.')
```

**Define path for OpenVINO IR xml model file**

The conversion script generates an XML containing information about the model architecture and a BIN file that stores the trained weights. We need both files to perform inference. OpenVINO uses the same name for the BIN file as provided for the XML file.


```python
ir_path = Path(f"{onnx_file_name.split('.')[0]}.xml")
ir_path
```
```text
Path('asl-and-some-words-resnet10t.xml')
```

**Define arguments for model conversion script**

OpenVINO provides the option to include the normalization stats in the IR model. That way, we don't need to account for different normalization stats when performing inference with multiple models. We can also convert the model to FP16 precision to reduce file size and improve inference speed.


```python
# Construct the command for Model Optimizer
mo_command = f"""mo
                 --input_model "{onnx_file_name}"
                 --input_shape "[1,3, {input_dims[0]}, {input_dims[1]}]"
                 --mean_values="{mean}"
                 --scale_values="{std}"
                 --data_type FP16
                 --output_dir "{output_dir}"
                 """
mo_command = " ".join(mo_command.split())
print("Model Optimizer command to convert the ONNX model to OpenVINO:")
display(Markdown(f"`{mo_command}`"))
```
```text
Model Optimizer command to convert the ONNX model to OpenVINO:
```

```bash
mo --input_model "asl-and-some-words-resnet10t.onnx" --input_shape "[1,3, 216, 384]" --mean_values="(0.485, 0.456, 0.406)" --scale_values="(0.229, 0.224, 0.225)" --data_type FP16 --output_dir "."
```





**Convert ONNX model to OpenVINO IR**


```python
if not ir_path.exists():
    print("Exporting ONNX model to IR... This may take a few minutes.")
    mo_result = %sx $mo_command
    print("\n".join(mo_result))
else:
    print(f"IR model {ir_path} already exists.")
```
```text
    Exporting ONNX model to IR... This may take a few minutes.
    Model Optimizer arguments:
    Common parameters:
    	- Path to the Input Model: 	/media/innom-dt/Samsung_T3/My_Environments/jupyter-notebooks/openvino/asl-and-some-words-resnet10t.onnx
    	- Path for generated IR: 	/media/innom-dt/Samsung_T3/My_Environments/jupyter-notebooks/openvino/.
    	- IR output name: 	asl-and-some-words-resnet10t
    	- Log level: 	ERROR
    	- Batch: 	Not specified, inherited from the model
    	- Input layers: 	Not specified, inherited from the model
    	- Output layers: 	Not specified, inherited from the model
    	- Input shapes: 	[1,3, 216, 384]
    	- Source layout: 	Not specified
    	- Target layout: 	Not specified
    	- Layout: 	Not specified
    	- Mean values: 	(0.485, 0.456, 0.406)
    	- Scale values: 	(0.229, 0.224, 0.225)
    	- Scale factor: 	Not specified
    	- Precision of IR: 	FP16
    	- Enable fusing: 	True
    	- User transformations: 	Not specified
    	- Reverse input channels: 	False
    	- Enable IR generation for fixed input shape: 	False
    	- Use the transformations config file: 	None
    Advanced parameters:
    	- Force the usage of legacy Frontend of Model Optimizer for model conversion into IR: 	False
    	- Force the usage of new Frontend of Model Optimizer for model conversion into IR: 	False
    OpenVINO runtime found in: 	/home/innom-dt/mambaforge/envs/fastai-openvino/lib/python3.9/site-packages/openvino
    OpenVINO runtime version: 	2022.1.0-7019-cdb9bec7210-releases/2022/1
    Model Optimizer version: 	2022.1.0-7019-cdb9bec7210-releases/2022/1
    [ WARNING ]  
    Detected not satisfied dependencies:
    	numpy: installed: 1.23.0, required: < 1.20
    
    Please install required versions of components or run pip installation
    pip install openvino-dev
    [ SUCCESS ] Generated IR version 11 model.
    [ SUCCESS ] XML file: /media/innom-dt/Samsung_T3/My_Environments/jupyter-notebooks/openvino/asl-and-some-words-resnet10t.xml
    [ SUCCESS ] BIN file: /media/innom-dt/Samsung_T3/My_Environments/jupyter-notebooks/openvino/asl-and-some-words-resnet10t.bin
    [ SUCCESS ] Total execution time: 0.43 seconds. 
    [ SUCCESS ] Memory consumed: 123 MB. 
    It's been a while, check for a new version of Intel(R) Distribution of OpenVINO(TM) toolkit here https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit/download.html?cid=other&source=prod&campid=ww_2022_bu_IOTG_OpenVINO-2022-1&content=upg_all&medium=organic or on the GitHub*
    [ INFO ] The model was converted to IR v11, the latest model format that corresponds to the source DL framework input/output format. While IR v11 is backwards compatible with OpenVINO Inference Engine API v1.0, please use API v2.0 (as of 2022.1) to take advantage of the latest improvements in IR v11.
    Find more information about API v2.0 and IR v11 at https://docs.openvino.ai
```





## Benchmark OpenVINO Inference

Now we can compare inference speed between OpenVINO and PyTorch. OpenVINO supports inference with ONNX models in addition to its IR format.

**Get available OpenVINO compute devices**

OpenVINO does not support GPU inference with non-Intel GPUs.


```python
devices = ie.available_devices
for device in devices:
    device_name = ie.get_property(device_name=device, name="FULL_DEVICE_NAME")
    print(f"{device}: {device_name}")
```

```text
CPU: 11th Gen Intel(R) Core(TM) i7-11700K @ 3.60GHz
```



**Create normalized input for ONNX model**


```python
normalized_input_image = batched_tensor.cpu().detach().numpy()
normalized_input_image.shape
```

```text
(1, 3, 224, 224)
```

**Test ONNX model using OpenVINO**


```python
# Load network to Inference Engine
ie = Core()
model_onnx = ie.read_model(model=onnx_file_name)
compiled_model_onnx = ie.compile_model(model=model_onnx, device_name="CPU")

input_layer_onnx = next(iter(compiled_model_onnx.inputs))
output_layer_onnx = next(iter(compiled_model_onnx.outputs))

# Run inference on the input image
res_onnx = compiled_model_onnx(inputs=[normalized_input_image])[output_layer_onnx]
learn.dls.vocab[np.argmax(res_onnx)]
```

```text
'J'
```

**Benchmark ONNX model CPU inference speed**


```python
%%timeit
compiled_model_onnx(inputs=[normalized_input_image])[output_layer_onnx]
```

```text
3.62 ms ± 61.8 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
```



**Prepare input image for OpenVINO IR model**


```python
input_image = scaled_tensor.unsqueeze(dim=0)
input_image.shape
```
```text
torch.Size([1, 3, 224, 224])
```


**Test OpenVINO IR model**


```python
# Load the network in Inference Engine
ie = Core()
model_ir = ie.read_model(model=ir_path)
model_ir.reshape(input_image.shape)
compiled_model_ir = ie.compile_model(model=model_ir, device_name="CPU")

# Get input and output layers
input_layer_ir = next(iter(compiled_model_ir.inputs))
output_layer_ir = next(iter(compiled_model_ir.outputs))

# Run inference on the input image
res_ir = compiled_model_ir([input_image])[output_layer_ir]
learn.dls.vocab[np.argmax(res_ir)]
```
```text
'J'
```


**Benchmark OpenVINO IR model CPU inference speed**


```python
%%timeit
compiled_model_ir([input_image])[output_layer_ir]
```
```text
3.39 ms ± 84.3 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
```

**Note:** The IR model is slightly faster than the ONNX model and half the file size.



**Benchmark PyTorch model GPU inference speed**


```python
%%timeit
with torch.no_grad(): preds = learn.model.cuda()(batched_tensor.cuda())
```

```text
1.81 ms ± 5.52 µs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)
```

PyTorch inference with a Titan RTX is still faster than OpenVINO inference with an i7-11700K for a ResNet10 model. However, OpenVINO CPU inference is often faster when using models optimized for mobile devices, like MobileNet.



**Benchmark PyTorch model CPU inference speed**


```python
%%timeit
with torch.no_grad(): preds = learn.model.cpu()(batched_tensor.cpu())
```

```text
8.94 ms ± 52.2 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
```

OpenVINO is easily faster than PyTorch for CPU inference.




## Summary

This post covered how to modify the training code from the [fastai-to-unity tutorial ](../../fastai-to-unity-tutorial/part-1)to finetune models from the Timm library and export them as OpenVINO IR models. Part 2 will cover creating a dynamic link library ([DLL](https://docs.microsoft.com/en-us/troubleshoot/windows-client/deployment/dynamic-link-library)) file in Visual Studio to perform inference with these models using [OpenVINO](https://docs.openvino.ai/latest/index.html).





**Previous:** [Fastai to Unity Tutorial Pt. 3](../../fastai-to-unity-tutorial/part-3/)

**Next:** [How to Create an OpenVINO Plugin for Unity on Windows Pt. 2](../part-2/)



**Project Resources:** [GitHub Repository](https://github.com/cj-mills/fastai-to-openvino-to-unity-tutorial)





<!-- Cloudflare Web Analytics --><script defer src='https://static.cloudflareinsights.com/beacon.min.js' data-cf-beacon='{"token": "56b8d2f624604c4891327b3c0d9f6703"}'></script><!-- End Cloudflare Web Analytics -->

