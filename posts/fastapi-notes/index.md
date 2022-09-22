---
aliases:
- /api/notes/2021/12/29/Notes-on-FastAPI
categories:
- api
- notes
date: '2021-12-29'
description: My notes from Code With Tomi's video providing an introduction to the
  FastAPI framework.
hide: false
layout: post
search_exclude: false
title: Notes on FastAPI Basics
toc: false

---

* [Overview](#overview)
* [What is FastAPI](#what-is-fastapi)
* [Install FastAPI](#install-fastapi)
* [Install Uvicorn Server](#install-uvicorn-server)
* [Import Dependencies](#import-dependencies)
* [Create a FastAPI Instance](#create-a-fastapi-instance)
* [Run a FastAPI Instance](#run-a-fastapi-instance)
* [View Automatic Interactive Documentation](#view-automatic-interactive-documentation)
* [Create an Endpoint Operation](#create-an-endpoint-operation)
* [Path Parameters](#path-parameters)
* [Query Parameters](#query-parameters)
* [POST Method](#post-method)
* [PUT Method](#put-method)
* [DELETE Method](#delete-method)



## Overview

Here are some notes I took while watching Code With Tomi's [video](https://www.youtube.com/watch?v=tLKKmouUams) providing an introduction to the [FastAPI](https://fastapi.tiangolo.com/) framework.

**Replit**

[Intro-to-fastAPI](https://replit.com/@innominate817/Intro-to-FastAPI#main.py)




## What is FastAPI
* A modern, high performance, web framework for building APIs with Python.
* Performance is on par with [Node.js](https://nodejs.org/en/) and [Go](https://go.dev/)
* Automatic interactive documentation
* Fully compatible with the open standards for [OpenAPI](https://github.com/OAI/OpenAPI-Specification) and [JSON Schema](https://json-schema.org/)
* [Documentation](https://fastapi.tiangolo.com/)
* [Source Code](https://github.com/tiangolo/fastapi)



### Core Methods

* GET: get a piece of information
* POST: create something new
* PUT: update something that already exists
* DELETE: delete something



## Install FastAPI

```bash
pip install fastapi
```



## Install Uvicorn Server

```bash
pip install uvicorn
```



## Import Dependencies

```python
from fastapi import FastAPI, Path
from typing import Optional
from pydantic import BaseModel
import uvicorn
```



## Create a FastAPI Instance

```python
app = FastAPI()
```



## Run a FastAPI Instance

```bash
uvicorn <file_name>:<instance_name> --reload
```

```bash
uvicorn main:app --reload
```

* `main`: the file `main.py`
* `app`: the object created inside of `main.py` with the line `app = FastAPI()`
* `--reload`: make the server restart after code changes. Only do this for development



## View Automatic Interactive Documentation

```bash
http://<address>:<port_number>/docs
```

```bash
http://127.0.0.1:8000/docs
```



## Create an Endpoint Operation

**Code**

```python
# Decorator for a GET method
@app.get("/")
def index():
    # return data in JSON format
    return {"name": "First Data"}
```

* `@app.get("/")` decorator
  * a path operation decorator
  * takes the function below and tells FastAPI the function corresponds to the path `/` with an operation `get`
* The `index()` function is in charge of handling the requests that go to
  * The path `/` (the root path)
  * Using an HTTP GET method

**Output**

```bash
{'name': 'First Data'}
```





## Path Parameters

* endpoint parameter: returns a data relating to an input 



**Code**

```python
students = {
    1: {
        "name": "john",
        "age": 17,
        "class": "Year"
    }
}

# A GET method
# Input: an integer greater than 0 that indicates a desired student ID
# Output: student entry
@app.get("/get-student/{student_id}")
def get_student(student_id: int = Path(None, description="The ID of the student you want to view", gt=0)):
    return students[student_id]
```

* `student_id`: an integer indicating the desired key value for the `students` dictionary
* `gt`: requires `student_id` to be greater than the specified value



**Request URL**

```bash
http://127.0.0.1:8000/get-student/1
```

```bash
{"name": "john", "age": 17, "class": "Year"}
```



## Query Parameters

* used to pass a value into a URL

```python
students = {
    1: {
        "name": "john",
        "age": 17,
        "class": "Year"
    }
}

# a GET method
@app.get("/get-by-name")
def get_student(*, name: Optional[str] = None, test: int):
    for student_id in students:
        if students[student_id]["name"] == name:
            return students[student_id]
    return {"Data": "Not found"}
```

**Request URL**

```bash
http://127.0.0.1:8000/get-by-name?name=john
```

```bash
{"name": "john", "age": 17, "class": "Year"}
```





## POST Method

* Data is sent from a client (e.g. a browser) to the API as a **request body**
* Request bodies are declared using [Pydantic models](https://pydantic-docs.helpmanual.io/usage/models/)

```python
students = {
    1: {
        "name": "john",
        "age": 17,
        "class": "Year"
    }
}

class Student(BaseModel):
    name: str
    age: int
    year: str

# POST Method
@app.post("create-student/{student_id")
def create_student(student_id: int, student: Student):
    if student_id in students:
        return {"Error": "Student alread exists"}

    students[student_id] = student
    return students[student_id]
```

* `Student` class: the request body used to send data for a new student to the `students` dictionary

**Request URL**

```bash
http://127.0.0.1:8000/create-student/2
```

**Request Body**

```bash
{"name": "tim", "age": 12, "class": "Year"}
```

**Updated `students`**

```python
{
    1: {
        "name": "john",
        "age": 17,
        "class": "Year"
    },
    2: {
        "name": "time",
        "age": 12,
        "class": "Year"
    }
}
```



## PUT Method

```python
students = {
    1: {
        "name": "john",
        "age": 17,
        "class": "Year"
    }
}

class UpdateStudent(BaseModel):
    name: Optional[str] = None
    age: Optional[int] = None
    year: Optional[str] = None
        
        
# PUT Method
@app.put("/update-student/{student_id}")
def update_student(student_id: int, student: UpdateStudent):
    if student_id not in students:
        return {"Error": "Student does not exist"}

    if student.name != None: students[student_id].name = student.name
    if student.age != None: students[student_id].age = student.age
    if student.year != None: students[student_id].year = student.year
    return students[student_id]
```

* `UpdateStudent` class: the request body used to send updated info for an existing student to the `students` dictionary



**Request URL**

```bash
http://127.0.0.1:8000/update-student/1
```

**Request Body**

```bash
{"name": "john", "age": 21, "class": "Year"}
```

**Updated `students`**

```python
{
    1: {
        "name": "john",
        "age": 21,
        "class": "Year"
    }
}
```





## DELETE Method

```python
students = {
    1: {
        "name": "john",
        "age": 17,
        "year": "Year"
    }
}


# DELETE Method
@app.delete("/delete-student/{student_id}")
def delete_student(student_id: int):
    if student_id not in students:
        return {"Error": "Student does not exist"}
    del students[student_id]
    return {"Message": "Student deleted successfully"}
```

**Request URL**

```bash
http://127.0.0.1:8000/delete-student/1
```

**Updated `students`**

```python
{
    
}
```







**References:**

* [FastAPI Course for Beginners](https://www.youtube.com/watch?v=tLKKmouUams)

