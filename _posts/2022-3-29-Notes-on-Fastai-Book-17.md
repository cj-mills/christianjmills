---
title: Notes on fastai Book Ch. 17
layout: post
toc: false
comments: true
description: My full notes for chapter 17 of Deep Learning for Coders with fastai & PyTorch
categories: [ai, fastai, notes, pytorch]
hide: false
permalink: /:title/
search_exclude: false
---



* [A Neural Net from the Foundations](#a-neural-net-from-the-foundations)
* [Building a Neural Net Layer from Scratch](#building-a-neural-net-layer-from-scratch)
* [The Forward and Backward Passes](#the-forward-and-backward-passes)
* [Conclusion](#conclusion)
* [References](#references)

-----

```python
import fastbook
fastbook.setup_book()
```


```python
import inspect
def print_source(obj):
    for line in inspect.getsource(obj).split("\n"):
        print(line)
```



## A Neural Net from the Foundations

## Building a Neural Net Layer from Scratch

### Modeling a Neuron
* a neuron receives a given number of inputs and has an internal weight for each of them
* the neuron sums the weighted inputs to produce an output and adds an inner bias
* $out = \sum_{i=1}^{n}{x_{i}w_{i}+b}$, where $(x_{1},\ldots,x_{n})$ are inputs, $(w_{1},\ldots,w_{n})$ are the weights, and $b$ is the bias
* ```python
  output = sum([x*w for x,w in zip(inputs, weights)]) + bias
  ```
* the output of the neuron is fed to a nonlinear function called an activation function
* the output of the nonlinear activation function is fed as input to another neuron
* Rectified Linear Unit (ReLU) activation function:
* ```python
  def relu(x): return x if x >= 0 else 0
  ```
* a deep learning model is build by stacking a lot of neurons in successive layers
* a linear layer: all inputs are linked to each neuron in the layer
    * need to compute the dot product for each input and each neuron with a given weight
    * ```python
      sum([x*w for x,w in zip(input,weight)]
      ```
      
#### The output of a fully connected layer
* $y_{i,j} = \sum_{k=1}^{n}{x_{i,k}w_{k,j}+b_{j}}$
* ```python
  y[i,j] = sum([a*b for a,b in zip(x[i,:],w[j,:])]) + b[j]
  ```
* ```python
  y = x @ w.t() + b
  ```
* `x`: a matrix containing the inputs with a size of `batch_size` by `n_inputs`
* `w`: a matrix containing the weights for the neurons with a size of `n_neurons` by `n_inputs`
* `b`: a vector containing the biases for the neurons with a size of `n_neurons`
* `y`: the output of the fully connected layer of size `batch_size` by `n_neurons`
* `@`: a matrix multiplication
* `w.t()`: the transpose matrix of `w`

### Matrix Multiplication from Scratch
* Need three nested loops
    1. for the row indices
    2. for the column indices
    3. for the inner sum


```python
import torch
from torch import tensor
```


```python
def matmul(a,b):
    # Get the number of rows and columns for the two matrices
    ar,ac = a.shape
    br,bc = b.shape
    # The number of columns in the first matrix need to be
    # the same as the number of rows in the second matrix
    assert ac==br
    # Initialize the output matrix
    c = torch.zeros(ar, bc)
    # For each row in the first matrix
    for i in range(ar):
        # For each column in the second matrix
        for j in range(bc):
            # For each column in the first matrix
            # Element-wise multiplication
            # Sum the products
            for k in range(ac): c[i,j] += a[i,k] * b[k,j]
    return c
```


```python
m1 = torch.randn(5,28*28)
m2 = torch.randn(784,10)
```

#### Using nested for-loops


```python
%time t1=matmul(m1, m2)
```
```text
    CPU times: user 329 ms, sys: 0 ns, total: 329 ms
    Wall time: 328 ms
```


```python
%timeit -n 20 t1=matmul(m1, m2)
```
```text
    325 ms ± 801 µs per loop (mean ± std. dev. of 7 runs, 20 loops each)
```

**Note:** Using loops is extremely inefficient!!! Avoid loops whenever possible.

#### Using PyTorch's built-in matrix multiplication operator
* written in C++ to make it fast
* need to vectorize operations on tensors to take advantage of speed of PyTorch
    * use element-wise arithmetic and broadcasting


```python
%time t2=m1@m2
```
```text
    CPU times: user 190 µs, sys: 0 ns, total: 190 µs
    Wall time: 132 µs
```


```python
%timeit -n 20 t2=m1@m2
```
```text
    The slowest run took 9.84 times longer than the fastest. This could mean that an intermediate result is being cached.
    6.42 µs ± 8.4 µs per loop (mean ± std. dev. of 7 runs, 20 loops each)
```

### Elementwise Arithmetic
* addition: `+`
* subtraction: `-`
* multiplication: `*`
* division: `/`
* greater than: `>`
* less than: `<`
* equal to: `==`

**Note:** Both tensors need to have the same shape to perform element-wise arithmetic.


```python
a = tensor([10., 6, -4])
b = tensor([2., 8, 7])
a + b
```
```text
    tensor([12., 14.,  3.])
```



```python
a < b
```
```text
    tensor([False,  True,  True])
```


#### Reduction Operators
* return tensors with only one element
* `all`: Tests if all elements evaluate to `True`.
* `sum`: Returns the sum of all elements in the tensor.
* `mean`: Returns the mean value of all elements in the tensor.


```python
# Check if every element in matrix a is less than the corresponding element in matrix b
((a < b).all(), 
 # Check if every element in matrix a is equal to the corresponding element in matrix b
 (a==b).all())
```
```text
    (tensor(False), tensor(False))
```



```python
# Convert tensor to a plain python number or boolean
(a + b).mean().item()
```
```text
    9.666666984558105
```



```python
m = tensor([[1., 2, 3], [4,5,6], [7,8,9]])
m*m
```
```text
    tensor([[ 1.,  4.,  9.],
            [16., 25., 36.],
            [49., 64., 81.]])
```



```python
# Attempt to perform element-wise arithmetic on tensors with different shapes
n = tensor([[1., 2, 3], [4,5,6]])
m*n
```
```text
    ---------------------------------------------------------------------------
    
    RuntimeError                              Traceback (most recent call last)
    
    /tmp/ipykernel_38356/3763285369.py in <module>
          1 n = tensor([[1., 2, 3], [4,5,6]])
    ----> 2 m*n


    RuntimeError: The size of tensor a (3) must match the size of tensor b (2) at non-singleton dimension 0
```


```python
# Replace the inner-most for-loop with element-wise arithmetic
def matmul(a,b):
    ar,ac = a.shape
    br,bc = b.shape
    assert ac==br
    c = torch.zeros(ar, bc)
    for i in range(ar):
        for j in range(bc): c[i,j] = (a[i] * b[:,j]).sum()
    return c
```


```python
%timeit -n 20 t3 = matmul(m1,m2)
```
```text
    488 µs ± 159 µs per loop (mean ± std. dev. of 7 runs, 20 loops each)
```

**Note:** Just replacing one of the for loops with PyTorch element-wise arithmetic dramatically improved performance.

### Broadcasting
* describes how tensors of different ranks are treated during arithmetic operations
* gives specific rules to codify when shapes are compatible when trying to do an element-wise operation, and how the tensor of the smaller shape is expanded to match the tensor of bigger shape

#### Broadcasting with a scalar
* the scalar is "virtually" expanded to the same shape as the tensor where every element contains the original scalar value


```python
a = tensor([10., 6, -4])
a > 0
```
```text
    tensor([ True,  True, False])
```


**Note:** Broadcasting with a scalar is useful when normalizing a dataset by subtracting the mean and dividing by the standard deviation.


```python
m = tensor([[1., 2, 3], [4,5,6], [7,8,9]])
(m - 5) / 2.73
```
```text
    tensor([[-1.4652, -1.0989, -0.7326],
            [-0.3663,  0.0000,  0.3663],
            [ 0.7326,  1.0989,  1.4652]])
```


#### Broadcasting a vector to a matrix
* the vector is virtually expanded to the same shape as the tensor, by duplicating the rows/columns as needed
* PyTorch uses the [expand_as](https://pytorch.org/docs/stable/generated/torch.Tensor.expand_as.html) method to expand the vector to the same size as the higher-rank tensor
    * creates a new view on the existing vector tensor without allocating new memory
* It is only possible to broadcast a vector of size `n` by a matrix of size `m` by `n`.


```python
c = tensor([10.,20,30])
m = tensor([[1., 2, 3], [4,5,6], [7,8,9]])
m.shape,c.shape
```
```text
    (torch.Size([3, 3]), torch.Size([3]))
```



```python
m + c
```
```text
    tensor([[11., 22., 33.],
            [14., 25., 36.],
            [17., 28., 39.]])
```



```python
c.expand_as(m)
```
```text
    tensor([[10., 20., 30.],
            [10., 20., 30.],
            [10., 20., 30.]])

```


```python
help(torch.Tensor.expand_as)
```
```text
    Help on method_descriptor:
    
    expand_as(...)
        expand_as(other) -> Tensor
        
        Expand this tensor to the same size as :attr:`other`.
        ``self.expand_as(other)`` is equivalent to ``self.expand(other.size())``.
        
        Please see :meth:`~Tensor.expand` for more information about ``expand``.
        
        Args:
            other (:class:`torch.Tensor`): The result tensor has the same size
                as :attr:`other`.
```



```python
help(torch.Tensor.expand)
```
```text
    Help on method_descriptor:
    
    expand(...)
        expand(*sizes) -> Tensor
        
        Returns a new view of the :attr:`self` tensor with singleton dimensions expanded
        to a larger size.
        
        Passing -1 as the size for a dimension means not changing the size of
        that dimension.
        
        Tensor can be also expanded to a larger number of dimensions, and the
        new ones will be appended at the front. For the new dimensions, the
        size cannot be set to -1.
        
        Expanding a tensor does not allocate new memory, but only creates a
        new view on the existing tensor where a dimension of size one is
        expanded to a larger size by setting the ``stride`` to 0. Any dimension
        of size 1 can be expanded to an arbitrary value without allocating new
        memory.
        
        Args:
            *sizes (torch.Size or int...): the desired expanded size
        
        .. warning::
        
            More than one element of an expanded tensor may refer to a single
            memory location. As a result, in-place operations (especially ones that
            are vectorized) may result in incorrect behavior. If you need to write
            to the tensors, please clone them first.
        
        Example::
        
            >>> x = torch.tensor([[1], [2], [3]])
            >>> x.size()
            torch.Size([3, 1])
            >>> x.expand(3, 4)
            tensor([[ 1,  1,  1,  1],
                    [ 2,  2,  2,  2],
                    [ 3,  3,  3,  3]])
            >>> x.expand(-1, 4)   # -1 means not changing the size of that dimension
            tensor([[ 1,  1,  1,  1],
                    [ 2,  2,  2,  2],
                    [ 3,  3,  3,  3]])
```


**Note:** Expanding the vector does not increase the amount of data stored.


```python
t = c.expand_as(m)
t.storage()
```
```text
     10.0
     20.0
     30.0
    [torch.FloatStorage of size 3]
```



```python
help(torch.Tensor.storage)
```
```text
    Help on method_descriptor:
    
    storage(...)
        storage() -> torch.Storage
        
        Returns the underlying storage.
```


**Note:** PyTorch accomplishes this by giving the new dimension a stride of `0`
* When PyTorch looks for the next row by adding the stride, it will stay at the same row


```python
t.stride(), t.shape
```
```text
    ((0, 1), torch.Size([3, 3]))
```



```python
c + m
```
```text
    tensor([[11., 22., 33.],
            [14., 25., 36.],
            [17., 28., 39.]])
```


```python
c = tensor([10.,20,30])
m = tensor([[1., 2, 3], [4,5,6]])
c+m
```
```text
    tensor([[11., 22., 33.],
            [14., 25., 36.]])
```



```python
# Attempt to broadcast a vector with an incompatible matrix
c = tensor([10.,20])
m = tensor([[1., 2, 3], [4,5,6]])
c+m
```
```text

    ---------------------------------------------------------------------------
    
    RuntimeError                              Traceback (most recent call last)
    
    /tmp/ipykernel_38356/3928136702.py in <module>
          1 c = tensor([10.,20])
          2 m = tensor([[1., 2, 3], [4,5,6]])
    ----> 3 c+m


    RuntimeError: The size of tensor a (2) must match the size of tensor b (3) at non-singleton dimension 1
```


```python
c = tensor([10.,20,30])
m = tensor([[1., 2, 3], [4,5,6], [7,8,9]])
# Expand the vector to broadcast across a different dimension
c = c.unsqueeze(1)
m.shape,c.shape, c
```
```text
    (torch.Size([3, 3]),
     torch.Size([3, 1]),
     tensor([[10.],
             [20.],
             [30.]]))
```



```python
c.expand_as(m)
```
```text
    tensor([[10., 10., 10.],
            [20., 20., 20.],
            [30., 30., 30.]])
```



```python
c+m
```
```text
    tensor([[11., 12., 13.],
            [24., 25., 26.],
            [37., 38., 39.]])
```



```python
t = c.expand_as(m)
t.storage()
```
```text
     10.0
     20.0
     30.0
    [torch.FloatStorage of size 3]
```



```python
t.stride(), t.shape
```
```text
    ((1, 0), torch.Size([3, 3]))
```


**Note:** By default, new broadcast dimensions are added at the beginning using `c.unsqueeze(0)` behind the scenes.


```python
c = tensor([10.,20,30])
c.shape, c.unsqueeze(0).shape,c.unsqueeze(1).shape
```
```text
    (torch.Size([3]), torch.Size([1, 3]), torch.Size([3, 1]))
```


**Note:** The unsqueeze command can be replaced by None indexing.


```python
c.shape, c[None,:].shape,c[:,None].shape
```
```text
    (torch.Size([3]), torch.Size([1, 3]), torch.Size([3, 1]))
```


**Note:**
* You can omit training columns when indexing
* `...` means all preceding dimensions


```python
c[None].shape,c[...,None].shape
```
```text
    (torch.Size([1, 3]), torch.Size([3, 1]))
```



```python
c,c.unsqueeze(-1)
```
```text
    (tensor([10., 20., 30.]),
     tensor([[10.],
             [20.],
             [30.]]))
```



```python
# Replace the second for loop with broadcasting
def matmul(a,b):
    ar,ac = a.shape
    br,bc = b.shape
    assert ac==br
    c = torch.zeros(ar, bc)
    for i in range(ar):
#       c[i,j] = (a[i,:]          * b[:,j]).sum() # previous
        c[i]   = (a[i  ].unsqueeze(-1) * b).sum(dim=0)
    return c
```


```python
m1.shape, m1.unsqueeze(-1).shape
```
```text
    (torch.Size([5, 784]), torch.Size([5, 784, 1]))
```



```python
m1[0].unsqueeze(-1).expand_as(m2).shape
```
```text
    torch.Size([784, 10])
```



```python
%timeit -n 20 t4 = matmul(m1,m2)
```
```text
    414 µs ± 18 µs per loop (mean ± std. dev. of 7 runs, 20 loops each)
```

**Note:** Even faster still, though the improvement is not as dramatic.

#### Broadcasting rules
* when operating on two tensors, PyTorch compares their shapes element-wise
    * starts with the trailing dimensions and works with its way backward
    * adds `1` when it meets and empty dimension 
* two dimensions are compatible when one of the following is true
    1. They are equal
    2. One of them is 1, in which case that dimension is broadcast to make it the same as the other
* arrays do not need to have the same number of dimensions

### Einstein Summation
* a compact representation for combining products and sums in a general way
* $ik,kj \rightarrow ij$
* lefthand side represents the operands dimensions, separated by commas
* righthand side represents the result dimensions
* a practical way of expressing operations involving indexing and sum of products

#### Notaion Rules
1. Repeated indices are implicitly summed over.
2. Each index can appear at most twice in any term.
3. Each term must contain identical nonrepeated indices.


```python
def matmul(a,b): return torch.einsum('ik,kj->ij', a, b)
```


```python
%timeit -n 20 t5 = matmul(m1,m2)
```
```text
    The slowest run took 10.35 times longer than the fastest. This could mean that an intermediate result is being cached.
    26.9 µs ± 37.3 µs per loop (mean ± std. dev. of 7 runs, 20 loops each)
```

**Note:** It is extremely fast even compared to the earlier broadcast implementation.
* einsum is often the fastest way to do custom operations in PyTorch, without diving into C++ and CUDA
* still not as fast as carefully optimizes CUDA code

**Additional Einsum Notation Examples**


```python
x = torch.randn(2, 2)
print(x)
# Transpose
torch.einsum('ij->ji', x)
```
```text
    tensor([[ 0.7541, -0.8633],
            [ 2.2312,  0.0933]])





    tensor([[ 0.7541,  2.2312],
            [-0.8633,  0.0933]])
```



```python
x = torch.randn(2, 2)
y = torch.randn(2, 2)
z = torch.randn(2, 2)
print(x)
print(y)
print(z)
# Return a vector of size b where the k-th coordinate is the sum of x[k,i] y[i,j] z[k,j] 
torch.einsum('bi,ij,bj->b', x, y, z)
```
```text
    tensor([[-0.2458, -0.7571],
            [ 0.0921,  0.5496]])
    tensor([[-1.2792, -0.0648],
            [-0.2263, -0.1153]])
    tensor([[-0.2433,  0.4558],
            [ 0.8155,  0.5406]])





    tensor([-0.0711, -0.2349])
```



```python
# trace
x = torch.randn(2, 2)
x, torch.einsum('ii', x)
```
```text
    (tensor([[ 1.4828, -0.7057],
             [-0.6288,  1.3791]]),
     tensor(2.8619))
```



```python
# diagonal
x = torch.randn(2, 2)
x, torch.einsum('ii->i', x)
```
```text
    (tensor([[-1.0796,  1.1161],
             [ 2.2944,  0.6225]]),
     tensor([-1.0796,  0.6225]))
```



```python
# outer product
x = torch.randn(3)
y = torch.randn(2)
f"x: {x}", f"y: {y}", torch.einsum('i,j->ij', x, y)
```
```text
    ('x: tensor([ 0.1439, -1.8456, -1.5355])',
     'y: tensor([-0.7276, -0.5566])',
     tensor([[-0.1047, -0.0801],
             [ 1.3429,  1.0273],
             [ 1.1172,  0.8547]]))
```



```python
# batch matrix multiplication
As = torch.randn(3,2,5)
Bs = torch.randn(3,5,4)
torch.einsum('bij,bjk->bik', As, Bs)
```
```text
    tensor([[[ 1.9657,  0.5904,  2.8094, -2.2607],
             [ 0.7610, -2.0402,  0.7331, -2.2257]],
    
            [[-1.5433, -2.9716,  1.3589,  0.1664],
             [ 2.7327,  4.4207, -1.1955,  0.5618]],
    
            [[-1.7859, -0.8143, -0.8410, -0.2257],
             [-3.4942, -1.9947,  0.7098,  0.5964]]])
```



```python
# with sublist format and ellipsis
torch.einsum(As, [..., 0, 1], Bs, [..., 1, 2], [..., 0, 2])
```
```text
    tensor([[[ 1.9657,  0.5904,  2.8094, -2.2607],
             [ 0.7610, -2.0402,  0.7331, -2.2257]],
    
            [[-1.5433, -2.9716,  1.3589,  0.1664],
             [ 2.7327,  4.4207, -1.1955,  0.5618]],
    
            [[-1.7859, -0.8143, -0.8410, -0.2257],
             [-3.4942, -1.9947,  0.7098,  0.5964]]])
```



```python
# batch permute
A = torch.randn(2, 3, 4, 5)
torch.einsum('...ij->...ji', A).shape
```
```text
    torch.Size([2, 3, 5, 4])
```



```python
# equivalent to torch.nn.functional.bilinear
A = torch.randn(3,5,4)
l = torch.randn(2,5)
r = torch.randn(2,4)
torch.einsum('bn,anm,bm->ba', l, A, r)
```
```text
    tensor([[ 1.1410, -1.7888,  4.7315],
            [ 3.8092,  3.0976,  2.2764]])
```



## The Forward and Backward Passes

### Defining and Initializing a Layer


```python
# Linear layer
def lin(x, w, b): return x @ w + b
```


```python
# Initialize random input values
x = torch.randn(200, 100)
# Initialize random target values
y = torch.randn(200)
```


```python
# Initialize layer 1 weights
w1 = torch.randn(100,50)
# Initialize layer 1 biases
b1 = torch.zeros(50)
# Initialize layer 2 weights
w2 = torch.randn(50,1)
# Initialize layer 2 biases
b2 = torch.zeros(1)
```


```python
# Get a batch of hidden state
l1 = lin(x, w1, b1)
l1.shape
```
```text
    torch.Size([200, 50])
```



```python
l1.mean(), l1.std()
```
```text
    (tensor(-0.0385), tensor(10.0544))
```


**Note:** Having with activations with a high standard deviation is a problem since the values can scale to numbers that can't be represented by a computer by the end of the model.


```python
x = torch.randn(200, 100)
for i in range(50): x = x @ torch.randn(100,100)
x[0:5,0:5]
```
```text
    tensor([[nan, nan, nan, nan, nan],
            [nan, nan, nan, nan, nan],
            [nan, nan, nan, nan, nan],
            [nan, nan, nan, nan, nan],
            [nan, nan, nan, nan, nan]])
```


**Note:** Having activations that are too small can cause all the activations at the end of the model to go to zero.


```python
x = torch.randn(200, 100)
for i in range(50): x = x @ (torch.randn(100,100) * 0.01)
x[0:5,0:5]
```
```text
    tensor([[0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.]])
```


**Note:** Need to scale the weight matrices so the standard deviation of the activations stays at 1
* [Understanding the difficulty of training deep feedforward neural networks](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf)
    * the right scale for a given layer is $\frac{1}{\sqrt{n_{in}}}$, where $n_{in}"$ represents the number of inputs.

**Note:** For a layer with 100 inputs, $\frac{1}{\sqrt{100}}=0.1$


```python
x = torch.randn(200, 100)
for i in range(50): x = x @ (torch.randn(100,100) * 0.1)
x[0:5,0:5]
```
```text
    tensor([[-1.7695,  0.5923,  0.3357, -0.7702, -0.8877],
            [ 0.6093, -0.8594, -0.5679,  0.4050,  0.2279],
            [ 0.4312,  0.0497,  0.1795,  0.3184, -1.7031],
            [-0.7370,  0.0251, -0.8574,  0.6826,  2.0615],
            [-0.2335,  0.0042, -0.1503, -0.2087, -0.0405]])
```



```python
x.std()
```
```text
    tensor(1.0150)
``


**Note:** Even a slight variation from $0.1$ will dramatically change the values


```python
# Redefine inputs and targets
x = torch.randn(200, 100)
y = torch.randn(200)
```


```python
from math import sqrt
# Scale the weights based on the number of inputs to the layers
w1 = torch.randn(100,50) / sqrt(100)
b1 = torch.zeros(50)
w2 = torch.randn(50,1) / sqrt(50)
b2 = torch.zeros(1)
```


```python
l1 = lin(x, w1, b1)
l1.mean(),l1.std()
```
```text
    (tensor(-0.0062), tensor(1.0231))
```



```python
# Define non-linear activation function
def relu(x): return x.clamp_min(0.)
```


```python
l2 = relu(l1)
l2.mean(),l2.std()
```
```text
    (tensor(0.3758), tensor(0.6150))
```


**Note:** The activation function ruined the mean and standard deviation.
* The $\frac{1}{\sqrt{n_{in}}}$ weight initialization method used not account for the ReLU function.


```python
x = torch.randn(200, 100)
for i in range(50): x = relu(x @ (torch.randn(100,100) * 0.1))
x[0:5,0:5]
```
```text
    tensor([[1.2172e-08, 0.0000e+00, 0.0000e+00, 7.1241e-09, 5.9308e-09],
            [1.9647e-08, 0.0000e+00, 0.0000e+00, 9.2189e-09, 7.1026e-09],
            [1.8266e-08, 0.0000e+00, 0.0000e+00, 1.1150e-08, 7.0774e-09],
            [1.8673e-08, 0.0000e+00, 0.0000e+00, 1.0574e-08, 7.3020e-09],
            [2.1829e-08, 0.0000e+00, 0.0000e+00, 1.1662e-08, 1.0466e-08]])
```


#### [Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](https://arxiv.org/abs/1502.01852)
* the article that introduced ResNet
* Introduced Kaiming initialization:
    * $\sqrt{\frac{2}{n_{in}}}$, where $n_{in}$ is the number of inputs of our model


```python
x = torch.randn(200, 100)
for i in range(50): x = relu(x @ (torch.randn(100,100) * sqrt(2/100)))
x[0:5,0:5]
```
```text
tensor([[0.0000, 0.0000, 0.1001, 0.0358, 0.0000],
        [0.0000, 0.0000, 0.1612, 0.0164, 0.0000],
        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.1764, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.1331, 0.0000, 0.0000]])
```




```python
x = torch.randn(200, 100)
y = torch.randn(200)
```


```python
w1 = torch.randn(100,50) * sqrt(2 / 100)
b1 = torch.zeros(50)
w2 = torch.randn(50,1) * sqrt(2 / 50)
b2 = torch.zeros(1)
```


```python
l1 = lin(x, w1, b1)
l2 = relu(l1)
l2.mean(), l2.std()
```
```text
(tensor(0.5720), tensor(0.8259))
```




```python
def model(x):
    l1 = lin(x, w1, b1)
    l2 = relu(l1)
    l3 = lin(l2, w2, b2)
    return l3
```


```python
out = model(x)
out.shape
```
```text
torch.Size([200, 1])
```




```python
# Squeeze the output to get rid of the trailing dimension
def mse(output, targ): return (output.squeeze(-1) - targ).pow(2).mean()
```


```python
loss = mse(out, y)
```

### Gradients and the Backward Pass
* the gradients are computed in the backward pass using the chain rule from calculus
* chain rule: $(g \circ f)'(x) = g'(f(x)) f'(x)$
* our loss if a big composition of different functions
* ```python
  loss = mse(out,y) = mse(lin(l2, w2, b2), y)
  ```
* chain rule:
$$\frac{\text{d} loss}{\text{d} b_{2}} = \frac{\text{d} loss}{\text{d} out} \times \frac{\text{d} out}{\text{d} b_{2}} = \frac{\text{d}}{\text{d} out} mse(out, y) \times \frac{\text{d}}{\text{d} b_{2}} lin(l_{2}, w_{2}, b_{2})$$
* To compute all the gradients we need for the update, we need to begin from the output of the model and work our way backward, one layer after the other.
* We can automate this process by having each function we implemented provided its backward step

#### Gradient of the loss function
1. undo the squeeze in the mse function
2. calculate the derivative of $x^{2}$: $2x$
3. calculate the derivative of the mean: $\frac{1}{n}$ where $n$ is the number of elements in the input


```python
def mse_grad(inp, targ): 
    # grad of loss with respect to output of previous layer
    inp.g = 2. * (inp.squeeze() - targ).unsqueeze(-1) / inp.shape[0]
```

#### Gradient of the ReLU activation function


```python
def relu_grad(inp, out):
    # grad of relu with respect to input activations
    inp.g = (inp>0).float() * out.g
```

#### Gradient of a linear layer


```python
def lin_grad(inp, out, w, b):
    # grad of matmul with respect to input
    inp.g = out.g @ w.t()
    w.g = inp.t() @ out.g
    b.g = out.g.sum(0)
```

### [SymPy](https://docs.sympy.org/latest/tutorial/intro.html)
* a library for symbolic computation that is extremely useful when working with calculus
* Symbolic computation deals with the computation of mathematical objects symbolically
    * the mathematical objects are represented exactly, not approximately, and mathematical expressions with unevaluated variables are left in symbolic form
    
```python
from sympy import symbols,diff
sx,sy = symbols('sx sy')
# Calculate the derivative of sx**2
diff(sx**2, sx)

2*sx
```

#### Define Forward and Backward Pass


```python
def forward_and_backward(inp, targ):
    # forward pass:
    l1 = inp @ w1 + b1
    l2 = relu(l1)
    out = l2 @ w2 + b2
    # we don't actually need the loss in backward!
    loss = mse(out, targ)
    
    # backward pass:
    mse_grad(out, targ)
    lin_grad(l2, out, w2, b2)
    relu_grad(l1, l2)
    lin_grad(inp, l1, w1, b1)
```

### Refactoring the Model
* define classes for each function that include their own forward and backward pass functions


```python
class Relu():
    def __call__(self, inp):
        self.inp = inp
        self.out = inp.clamp_min(0.)
        return self.out
    
    def backward(self): self.inp.g = (self.inp>0).float() * self.out.g
```


```python
class Lin():
    def __init__(self, w, b): self.w,self.b = w,b
        
    def __call__(self, inp):
        self.inp = inp
        self.out = inp@self.w + self.b
        return self.out
    
    def backward(self):
        self.inp.g = self.out.g @ self.w.t()
        self.w.g = self.inp.t() @ self.out.g
        self.b.g = self.out.g.sum(0)
```


```python
class Mse():
    def __call__(self, inp, targ):
        self.inp = inp
        self.targ = targ
        self.out = (inp.squeeze() - targ).pow(2).mean()
        return self.out
    
    def backward(self):
        x = (self.inp.squeeze()-self.targ).unsqueeze(-1)
        self.inp.g = 2.*x/self.targ.shape[0]
```


```python
class Model():
    def __init__(self, w1, b1, w2, b2):
        self.layers = [Lin(w1,b1), Relu(), Lin(w2,b2)]
        self.loss = Mse()
        
    def __call__(self, x, targ):
        for l in self.layers: x = l(x)
        return self.loss(x, targ)
    
    def backward(self):
        self.loss.backward()
        for l in reversed(self.layers): l.backward()
```


```python
model = Model(w1, b1, w2, b2)
```


```python
loss = model(x, y)
```


```python
model.backward()
```

### Going to PyTorch


```python
# Define a base class for all functions in the model
class LayerFunction():
    def __call__(self, *args):
        self.args = args
        self.out = self.forward(*args)
        return self.out
    
    def forward(self):  raise Exception('not implemented')
    def bwd(self):      raise Exception('not implemented')
    def backward(self): self.bwd(self.out, *self.args)
```


```python
class Relu(LayerFunction):
    def forward(self, inp): return inp.clamp_min(0.)
    def bwd(self, out, inp): inp.g = (inp>0).float() * out.g
```


```python
class Lin(LayerFunction):
    def __init__(self, w, b): self.w,self.b = w,b
        
    def forward(self, inp): return inp@self.w + self.b
    
    def bwd(self, out, inp):
        inp.g = out.g @ self.w.t()
        self.w.g = inp.t() @ self.out.g
        self.b.g = out.g.sum(0)
```


```python
class Mse(LayerFunction):
    def forward (self, inp, targ): return (inp.squeeze() - targ).pow(2).mean()
    def bwd(self, out, inp, targ): 
        inp.g = 2*(inp.squeeze()-targ).unsqueeze(-1) / targ.shape[0]
```

#### [torch.autograd.Function](https://pytorch.org/docs/stable/autograd.html#function)
* In PyTorch, each basic function we need to differentiate is written as a [torch.autograd.Function](https://pytorch.org/docs/stable/autograd.html#function) that has a forward and backward method


```python
from torch.autograd import Function
```


```python
class MyRelu(Function):
    # Performs the operation
    @staticmethod
    def forward(ctx, i):
        result = i.clamp_min(0.)
        ctx.save_for_backward(i)
        return result
    
    # Defines a formula for differentiating the operation with 
    # backward mode automatic differentiation
    @staticmethod
    def backward(ctx, grad_output):
        i, = ctx.saved_tensors
        return grad_output * (i>0).float()
```


```python
help(staticmethod)
```
```text
    Help on class staticmethod in module builtins:
    
    class staticmethod(object)
     |  staticmethod(function) -> method
     |  
     |  Convert a function to be a static method.
     |  
     |  A static method does not receive an implicit first argument.
     |  To declare a static method, use this idiom:
     |  
     |       class C:
     |           @staticmethod
     |           def f(arg1, arg2, ...):
     |               ...
     |  
     |  It can be called either on the class (e.g. C.f()) or on an instance
     |  (e.g. C().f()). Both the class and the instance are ignored, and
     |  neither is passed implicitly as the first argument to the method.
     |  
     |  Static methods in Python are similar to those found in Java or C++.
     |  For a more advanced concept, see the classmethod builtin.
     |  
     |  Methods defined here:
     |  
     |  __get__(self, instance, owner, /)
     |      Return an attribute of instance, which is of type owner.
     |  
     |  __init__(self, /, *args, **kwargs)
     |      Initialize self.  See help(type(self)) for accurate signature.
     |  
     |  ----------------------------------------------------------------------
     |  Static methods defined here:
     |  
     |  __new__(*args, **kwargs) from builtins.type
     |      Create and return a new object.  See help(type) for accurate signature.
     |  
     |  ----------------------------------------------------------------------
     |  Data descriptors defined here:
     |  
     |  __dict__
     |  
     |  __func__
     |  
     |  __isabstractmethod__
```


#### [torch.nn.Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#module)
* the base structure for all models in PyTorch

**Implementation Steps**
1. Make sure the superclass `__init__` is called first when you initialize it.
2. Define any parameters of the model as attributes with `nn.Parameter`.
3. Define a forward function that returns the output of your model.


```python
import torch.nn as nn

class LinearLayer(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_out, n_in) * sqrt(2/n_in))
        self.bias = nn.Parameter(torch.zeros(n_out))
    
    def forward(self, x): return x @ self.weight.t() + self.bias
```


```python
lin = LinearLayer(10,2)
p1,p2 = lin.parameters()
p1.shape,p2.shape
```
```text
(torch.Size([2, 10]), torch.Size([2]))
```




```python
class Model(nn.Module):
    def __init__(self, n_in, nh, n_out):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_in,nh), nn.ReLU(), nn.Linear(nh,n_out))
        self.loss = mse
        
    def forward(self, x, targ): return self.loss(self.layers(x).squeeze(), targ)
```

**Note:** fsatai provides its own variant of Module that is identical to `nn.Module`, but automatically calls `super().__init__()`.


```python
class Model(Module):
    def __init__(self, n_in, nh, n_out):
        self.layers = nn.Sequential(
            nn.Linear(n_in,nh), nn.ReLU(), nn.Linear(nh,n_out))
        self.loss = mse
        
    def forward(self, x, targ): return self.loss(self.layers(x).squeeze(), targ)
```



## Conclusion

* A neural net is a bunch of matrix multiplications with nonlinearities in between
* Vectorize and take advantage of techniques such as element-wise arithmetic and broadcasting when possible
* Two tensors are broadcastable if the dimensions starting from the end and going backward match
    * May need to add dimensions of size one to make tensors broadcastable
* Properly initializing a neural net is crucial to get training started
    * Use Kaiming initialization when using ReLU
* The backward pass is the chain rule applied multiple times, computing the gradient from the model output and going back, one layer at a time





## References

* [Deep Learning for Coders with fastai & PyTorch](https://www.oreilly.com/library/view/deep-learning-for/9781492045519/)
* [The fastai book GitHub Repository](https://github.com/fastai/fastbook)