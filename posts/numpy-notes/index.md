---
aliases:
- /Notes-on-NumPy/
categories:
- numpy
- notes
date: '2021-12-29'
description: My notes from Keith Galli's video providing an introduction to NumPy.
hide: false
layout: post
search_exclude: false
title: Notes on NumPy
toc: false

---

* [Overview](#overview)
* [What is NumPy](#what-is-numpy)
* [Applications of NumPy](#applications-of-numpy)
* [Install NumPy](#install-numpy)
* [Import NumPy](#import-numpy)
* [The Basics](#the-basics)
* [Accessing and Changing Arrays](#accessing-and-changing-arrays)
* [Initialize Different Types of Arrays](#initialize-different-types-of-arrays)
* [Mathematics](#mathematics)
* [Reorganizing Arrays](#reorganizing-arrays)
* [Miscellaneous](#miscellaneous)
* [Boolean Masking and Advanced Indexing](#boolean-masking-and-advanced-indexing)



## Overview

Here are some notes I took while watching Keith Galli's [video](https://www.youtube.com/watch?v=QUT1VHiLmmI) providing an introduction to [NumPy](https://numpy.org/).

**Colab Notebook**

* [Google Colaboratory](https://colab.research.google.com/drive/15yXLcByNyZ7rtklG-hVWlxMGp5bNKUlo?usp=sharing)


## What is NumPy

- A multi-dimensional array library

### How are List different from NumPy?

- Lists are very slow
    - Lists are dynamically typed
    - Lists need to store a lot more information to account for unfixed data types
        - Needs to keep track of the following information for single Integer
            - Size: 4 bytes
            - Reference Count: 8 bytes
            - Object Type: 8 bytes
            - Object Value: 8 bytes
        - Does not use contiguous memory
            - Different array elements are scattered in different parts of memory
        - 
- NumPy is very fast
    - NumPy uses fixed types
        - Don’t need to do type checking
    - Default type is Int32 (4 bytes)
    - Faster to read less bytes of memory
    - Can specify specific data types (e.g. Int16, Int8)
    - Uses contiguous memory
        - Data for an array is in the same chunk of memory
        - faster to access
        - lower CPU overhead
        - Can leverage [SIMD](https://en.wikipedia.org/wiki/SIMD) [Vector Processing](https://en.wikipedia.org/wiki/Vector_processor)
            - Single Instruction Multiple Data
                - Can perform operations on all elements simultaneously
        - Effective CPU cache utilization
        - Lot’s more functionality
            - Example: array multiplication `arrayA*arrayB`
            



## Applications of NumPy

- MATLAB replacement
    - SciPy has even more mathematical capability
- Plotting (Matplotlib)
- Backend (Pandas, Digital Photography)
- Machine Learning (Tensors)



## Install NumPy

- `pip install numpy`
- `conda install numpy`



## Import NumPy

```python
import numpy as np
```



## The Basics

**Initialize a 1D array**

```python
# Initialize a 1D array
a = np.array([1,2,3])
a 
```

```python
array([1, 2, 3])
```



**Initialize a 2D array of floats**

```python
# Initialize a 2D array of floats
b = np.array([[9.0,8.0,7.0],[6.0,5.0,4.0]])
b
```

```python
array([[9., 8., 7.],
    [6., 5., 4.]])
```



**Get Dimension**

```python
# Get Dimension
a.ndim
```

```python
1
```



**Get Shape**

```python
# Get Shape
b.shape
```

```python
(2, 3)
```



**Get Type**

```python
# Get Type
a.dtype
```

```python
dtype('int64')
```



**Specify data type**

```python
# Specify data type
a = np.array([1,2,3], dtype='int16')
a.dtype
```

```python
dtype('int16')
```



**Get Size**

```python
# Get Size: the number of bytes per array element
a.itemsize
```

```python
2 (for int16)
```



**Get total size**

```python
# Get total size: number of elements times the number of bytes per element
a.size * a.itemsize
# or
a.nbytes
```

```python
6 (for 3 int16 elements)
```



## Accessing and Changing Arrays

[Indexing - NumPy v1.13 Manual](https://docs.scipy.org/doc/numpy-1.13.0/user/basics.indexing.html)

```python
a = np.array([[1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14]])
print(f'Values: {a}')
print(f'Shape: {a.shape}')

# Get a specific element [r, c]
a[1, 5]
```

```python
Values: [[ 1  2  3  4  5  6  7]
         [ 8  9 10 11 12 13 14]]
Shape: (2, 7)
13
```



**Get a specific row**

```python
# Get a specific row
a[0, :]
```

```python
array([1, 2, 3, 4, 5, 6, 7])
```



**Get a specific column**

```python
# Get a specific column
a[:, 2]
```

```python
array([ 3, 10])
```



```python
# Getting a little more fancy [startindex:endindex:stepsize]
a[0, 1:6:2]
# or
a[0, 1:-1:2]
```

```python
array([2, 4, 6])
```



**Change elements**

```python
# Change elements
a[1,5] = 20
a
```

```python
array([[ 1,  2,  3,  4,  5,  6,  7],
      [ 8,  9, 10, 11, 12, 20, 14]])
```



**Change column index 2**

```python
# Change column index 2 
a[:, 2] = 5
a
```

```python
array([[ 1,  2,  5,  4,  5,  6,  7],
      [ 8,  9,  5, 11, 12, 20, 14]])
```



**Change colum with two different numbers**

```python
# Change colum with two different numbers
# Needs to be the same shape as the part you want to modify
# Two elements in each column means a lenght of 2
a[:, 2] = [1,2]
a
```

```python
array([[ 1,  2,  1,  4,  5,  6,  7],
      [ 8,  9,  2, 11, 12, 20, 14]])
```



**3D Example**

```python
# 3D Example
b = np.array([[[1,2],[3,4]], [[5,6],[7,8]]])
b
```

```python
array([[[1, 2],
       [3, 4]],

      [[5, 6],
       [7, 8]]])
```



**Get specific element**

```python
# Get specific element (work outside in)
# [first_dim, second_dim, third_dim]
b[0, 1, 1]
```

```python
4
```



**Get Specific Element**

```python
# Get specific element (work outside in)
b[:,1,:]
```

```python
array([[3, 4],
      [7, 8]])
```



**Replace values**

```python
# Replace
# New value needs to be the same dimensions as what is being replaced
b[:,1,:] = [[9,9],[8,8]]
b
```

```python
array([[[1, 2],
        [9, 9]],

       [[5, 6],
        [8, 8]]])
```



## Initialize Different Types of Arrays

[Array creation routines - NumPy v1.21 Manual](https://numpy.org/doc/stable/reference/routines.array-creation.html)



**All 0s matrix**

```python
# All 0s matrix
print(f'1D: {np.zeros(5)}')
print(f'2D: {np.zeros((2,3))}')
print(f'3D: {np.zeros((2,3,4))}')
```

```python
1D: [0. 0. 0. 0. 0.]
2D: [[0. 0. 0.]
 [0. 0. 0.]]
3D: [[[0. 0. 0. 0.]
  [0. 0. 0. 0.]
  [0. 0. 0. 0.]]

 [[0. 0. 0. 0.]
  [0. 0. 0. 0.]
  [0. 0. 0. 0.]]]
```



**All 1s matrix**

```python
# All 1s matrix
print(f'1D: {np.ones(5)}')
print(f'2D: {np.ones((2,3))}')
print(f'3D: {np.ones((2,3,4))}')
```

```python
1D: [1. 1. 1. 1. 1.]
2D: [[1. 1. 1.]
 [1. 1. 1.]]
3D: [[[1. 1. 1. 1.]
  [1. 1. 1. 1.]
  [1. 1. 1. 1.]]

 [[1. 1. 1. 1.]
  [1. 1. 1. 1.]
  [1. 1. 1. 1.]]]
```



**Any number**

```python
# Any other number
np.full((2,2), 99)
```

```python
array([[99, 99],
       [99, 99]])
```



**Any other number with the same shape as another array**

```python
# Any other number (full_like)
# Use the same shape as the provided array
np.full_like(a, 55)
```

```python
array([[55, 55, 55, 55, 55, 55, 55],
       [55, 55, 55, 55, 55, 55, 55]])
```



**Random decimal numbers between 0 and 1**

```python
# Random decimal numbers between 0 and 1
# shape of (4,2)
np.random.rand(4,2)
```

```python
array([[0.90796667, 0.18775268],
       [0.36853663, 0.82186396],
       [0.75724737, 0.09608278],
       [0.5953758 , 0.57110868]])
```



**Random decimal number from shape**

```python
# Random decimal number from shape
np.random.random_sample(a.shape)
```

```python
array([[0.96539982, 0.72943229, 0.10863575, 0.84796304, 0.09610215,
        0.88132328, 0.56848496],
       [0.27198747, 0.2295634 , 0.40931032, 0.99669531, 0.90768254,
        0.1626064 , 0.80310083]])
```



**Random integer values**

```python
# Random integer values
# Max value (exclusive) and shape
np.random.randint(7, size=(3,3))
```

```python
array([[6, 2, 0],
       [4, 2, 4],
       [4, 0, 4]])
```



**Random integer values in a range**

```python
# Random integer values
# Range of values (exclusive) and shape
np.random.randint(4,7, size=(3,3))
```

```python
array([[5, 4, 6],
       [4, 5, 6],
       [4, 6, 6]])
```



**Identity matrix**

```python
# Identity matrix
np.identity(3)
```

```python
array([[1., 0., 0.],
       [0., 1., 0.],
       [0., 0., 1.]])
```



**Repeat array**

```python
# Repeat array
arr = np.array([1,2,3])
# Repeat arr 3 times element-wise
r1 = np.repeat(arr,3)
r1
```

```python
array([1, 1, 1, 2, 2, 2, 3, 3, 3])
```



**Repeat 2D array**

```python
# Repeat 2D array
arr = np.array([[1,2,3]])
# Repeat arr 3 times element-wise
r1 = np.repeat(arr,3, axis=0)
r1
```

```python
array([[1, 2, 3],
       [1, 2, 3],
       [1, 2, 3]])
```



**Recreate this array**

```
[1, 1, 1, 1, 1]
[1, 0, 0, 0, 1]
[1, 0, 9, 0, 1]
[1, 0, 0, 0, 1]
[1, 1, 1, 1, 1]
```

```python
c = np.ones((5,5), dtype='int32')
c[1:-1, 1:-1] = 0
c[2,2] = 9
c
```

```python
array([[1, 1, 1, 1, 1],
       [1, 0, 0, 0, 1],
       [1, 0, 9, 0, 1],
       [1, 0, 0, 0, 1],
       [1, 1, 1, 1, 1]], dtype=int32)
```



**Be careful when copying arrays!!!**

```python
# Shallow copy
a = np.array([1,2,3])
b = a
b[0] = 100
a
```

```python
array([100,   2,   3])
```

```python
# Deep copy
a = np.array([1,2,3])
b = a.copy()
b[0] = 100
a
```

```python
array([1, 2, 3])
```



## Mathematics

[Mathematical functions - NumPy v1.21 Manual](https://numpy.org/doc/stable/reference/routines.math.html)

```python
a = np.array([1,2,3,4])
a
```

```python
array([1, 2, 3, 4])
```



**Add**

```python
a + 2
```

```python
array([3, 4, 5, 6])
```



**Subtract**

```python
a - 2
```

```python
array([-1,  0,  1,  2])
```



**Multiply**

```python
a * 2
```

```python
array([2, 4, 6, 8])
```



**Divide**

```python
a / 2
```

```python
array([0.5, 1. , 1.5, 2. ])
```



**Shorthand**

```python
a += 2
a
```

```python
array([3, 4, 5, 6])
```



**Add Arrays**

```python
b = np.array([1,0,1,0])
a + b
```

```python
array([4, 4, 6, 6])
```



**Exponents**

```python
a ** 2
```

```python
array([ 9, 16, 25, 36])
```



**Sine**

```python
# Take the sin
np.sin(a)
```

```python
array([ 0.14112001, -0.7568025 , -0.95892427, -0.2794155 ])
```



**Cosine**

```python
# Take the cosine
np.cos(a)
```

```python
array([-0.9899925 , -0.65364362,  0.28366219,  0.96017029])
```



### Linear Algebra

[Linear algebra (numpy.linalg) - NumPy v1.21 Manual](https://numpy.org/doc/stable/reference/routines.linalg.html)

```python
a = np.ones((2,3))
a
```

```python
array([[1., 1., 1.],
       [1., 1., 1.]])
```

```python
b = np.full((3,2),2)
b
```

```python
array([[2, 2],
       [2, 2],
       [2, 2]])
```



**Matrix multiplication**

```python
# Matrix multiplication
np.matmul(a,b)
```

```python
array([[6., 6.],
       [6., 6.]])
```



**Find the determinant**

```python
# Find the determinant
c = np.identity(3)
np.linalg.det(c)
```

```python
1.0
```



### Statistics

```python
stats = np.array([[1,2,3],[4,5,6]])
stats
```

```python
array([[1, 2, 3],
       [4, 5, 6]])
```



**Get lowest value in array**

```python
# Get lowest value in array
np.min(stats)
```

```python
1
```



**Get lowest value in array along specific axis**

```python
# Get lowest value in array along specific axis
# axis=0: min values in each column
np.min(stats, axis=0)
```

```python
array([1, 2, 3])
```



**Get lowest value in array along specific axis**

```python
# Get lowest value in array along specific axis
# axis=1: min values in each row
np.min(stats, axis=1)
```

```python
array([1, 4])
```



**Get highest value in array**

```python
# Get highest value in array
np.max(stats)
```

```python
6
```



**Sum up values in array**

```python
# Sum up values in array
np.sum(stats)
```

```python
21
```



**Sum up values in array across axis**

```python
# Sum up values in array across axis
# axis=0: sum values in each column
np.sum(stats, axis=0)
```

```python
array([5, 7, 9])
```



**Sum up values in array across axis**

```python
# Sum up values in array across axis
# axis=1: sum values in each row
np.sum(stats, axis=1)
```

```python
array([ 6, 15])
```



## Reorganizing Arrays

**Note:** New shape needs to maintain the same number of values

```python
before = np.array([[1,2,3,4],[5,6,7,8]])
print(before)
print(f'Shape: {before.shape}')
```

```python
[[1 2 3 4]
 [5 6 7 8]]
Shape: (2, 4)
```



**Reshape from (2,4) to (8,1) **

```python
# Reshape array
# Reshape from (2,4) to (8,1) 
after = before.reshape((8,1))
after
```

```python
array([[1],
       [2],
       [3],
       [4],
       [5],
       [6],
       [7],
       [8]])
```



**Reshape from (2,4) to (4,2)**

```python
# Reshape array
# Reshape from (2,4) to (4,2) 
after = before.reshape((4,2))
after
```

```python
array([[1, 2],
       [3, 4],
       [5, 6],
       [7, 8]])
```



**Reshape from (2,4) to (2,2,2)**

```python
# Reshape array
# Reshape from (2,4) to (2,2,2) 
after = before.reshape((2,2,2))
after
```

```python
array([[[1, 2],
        [3, 4]],

       [[5, 6],
        [7, 8]]])
```



**Vertically stacking vectors**

```python
# Vertically stacking vectors
v1 = np.array([1,2,3,4])
v2 = np.array([5,6,7,8])

np.vstack([v1,v2])
```

```python
array([[1, 2, 3, 4],
       [5, 6, 7, 8]])
```



**Stack vectors multiple times**

```python
# Stack vectors multiple times
np.vstack([v1,v2,v2,v1])
```

```python
array([[1, 2, 3, 4],
       [5, 6, 7, 8],
       [5, 6, 7, 8],
       [1, 2, 3, 4]])
```



**Horizontal stacks**

```python
# Horizontal stacks
np.hstack([v1, v2])
```

```python
array([1, 2, 3, 4, 5, 6, 7, 8])
```



**Combining Horizontal and Vertical Stacks**

```python
np.hstack([np.vstack([v1,v2,v2,v1]), np.vstack([v1,v2,v2,v1])])
```

```python
array([[1, 2, 3, 4, 1, 2, 3, 4],
       [5, 6, 7, 8, 5, 6, 7, 8],
       [5, 6, 7, 8, 5, 6, 7, 8],
       [1, 2, 3, 4, 1, 2, 3, 4]])
```



## Miscellaneous

**Load data from text file**

```python
# Load data from text file
# Pass in file path and the delimiter character that separates values
# Casts values to float
filedata = np.genfromtxt('data.txt', delimiter=',')
```



**Cast array values to specific type**

```python
# Cast array values to specific type
filedata = filedata.astype('int32')
```



## Boolean Masking and Advanced Indexing

```python
stats = np.array([[10,2,3],[-4,5,6]])
stats
```

```python
array([[10,  2,  3],
       [-4,  5,  6]])
```



**Boolean mask for values greater than 3**

```python
# Boolean mask for values greater than 3
stats > 3
```

```python
array([[ True, False, False],
       [False,  True,  True]])
```



**Index array using a boolean mask**

```python
# Index array using a boolean mask
stats[stats > 3]
```

```python
array([10,  5,  6])
```



**Index with a list**

```python
# Index with a list
a = np.array([1,2,3,4,5,6,7,8,9])
# List of indices
a[[1,2,8]]
```

```python
array([2, 3, 9])
```



**Check if any values in array return true for a boolean**

```python
# Check if any values in array return true for a boolean
np.any(a > 3, axis=0)
```

```python
True
```



**Check if all values in array return true for a boolean**

```python
# Check if all values in array return true for a boolean
np.all(a > 3, axis=0)
```

```python
False
```



**Use multiple conditions**

```python
# Use multiple conditions
((a > 3) & (a < 7))
```

```python
array([False, False, False,  True,  True,  True, False, False, False])
```



**Use multiple conditions with negation**

```python
# Use multiple conditions with negation
(~((a > 3) & (a < 7)))
```

```python
array([ True,  True,  True, False, False, False,  True,  True,  True])
```



**Test Array**

```python
test_array = np.arange(36).reshape(6, -1)
test_array
```

```python
array([[ 0,  1,  2,  3,  4,  5],
       [ 6,  7,  8,  9, 10, 11],
       [12, 13, 14, 15, 16, 17],
       [18, 19, 20, 21, 22, 23],
       [24, 25, 26, 27, 28, 29],
       [30, 31, 32, 33, 34, 35]])
```



**Range of indices**

```python
# 2:4: range of rows to index
# 0:2: range of columns to index 
test_array[2:4, 0:2]
```

```python
array([[12, 13],
       [18, 19]])
```



**List of indices**

```python
# [0,1,2,3,4]: list of rows
# [1,2,3,4,5]: list of indexes for each row
test_array[[0,1,2,3,4], [1,2,3,4,5]]
```

```python
array([ 1,  8, 15, 22, 29])
```



**Combine range and list of indices**

```python
# [0,4,5]: The list of rows
# Columns 3 and later
test_array[[0,4,5], 3:]
```

```python
array([[ 3,  4,  5],
       [27, 28, 29],
       [33, 34, 35]])
```



**References:**

* [Python NumPy Tutorial for Beginners](https://www.youtube.com/watch?v=QUT1VHiLmmI)

