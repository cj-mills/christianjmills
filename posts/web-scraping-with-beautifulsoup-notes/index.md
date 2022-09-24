---
aliases:
- /Notes-on-Web-Scraping-with-Beautiful-Soup/
categories:
- web-scraping
- notes
date: '2021-12-29'
description: My notes from JimShapedCoding's video providing an introduction to web
  scraping with the beautiful soup library.
hide: false
layout: post
search_exclude: false
title: Notes on Web Scraping with Beautiful Soup
toc: false

---

* [Overview](#overview)
* [Package Installation](#package-installation)
* [Import Dependencies](#import-dependencies)
* [Local HTML Scraping](#local-html-scraping)
* [Website Scraping](#website-scraping)



## Overview

Here are some notes I took while watching JimShapedCoding's [video](https://www.youtube.com/watch?v=XVv6mJpFOb0) providing an introduction to web scraping with the beautiful soup library.



## Package Installation

**[Beautiful Soup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/)**

* A Python library for pulling data out of HTML and XML files

```bash
pip install beautifulsoup4
```

**[lxml - XML and HTML with Python](https://lxml.de/)**

* A pythonic binding for the C libraries [libxml2](http://xmlsoft.org/) and [libxslt](http://xmlsoft.org/XSLT/).
* Used as the parser for beautiful soup

```bash
pip install lxml
```

**[Requests: HTTP for Humans](https://docs.python-requests.org/en/latest/)**

* An elegant and simple HTTP library for Python

```bash
pip install requests
```



## Import Dependencies

```python
from bs4 import BeautifulSoup
import requests

import os
```



## Local HTML Scraping

**Sample HTML**

```html
<!doctype html>
<html lang="en">
   <head>
      <meta charset="utf-8">
      <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
      <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css" integrity="sha384-JcKb8q3iqJ61gNV9KGb8thSsNjpSL0n8PARn9HuZOnIxN0hoP+VmmDGMN5t9UJ0Z" crossorigin="anonymous">
      <title>My Courses</title>
   </head>
   <body>
      <h1>Hello, Start Learning!</h1>
      <div class="card" id="card-python-for-beginners">
         <div class="card-header">
            Python
         </div>
         <div class="card-body">
            <h5 class="card-title">Python for beginners</h5>
            <p class="card-text">If you are new to Python, this is the course that you should buy!</p>
            <a href="#" class="btn btn-primary">Start for 20$</a>
         </div>
      </div>
      <div class="card" id="card-python-web-development">
         <div class="card-header">
            Python
         </div>
         <div class="card-body">
            <h5 class="card-title">Python Web Development</h5>
            <p class="card-text">If you feel enough confident with python, you are ready to learn how to create your own website!</p>
            <a href="#" class="btn btn-primary">Start for 50$</a>
         </div>
      </div>
      <div class="card" id="card-python-machine-learning">
         <div class="card-header">
            Python
         </div>
         <div class="card-body">
            <h5 class="card-title">Python Machine Learning</h5>
            <p class="card-text">Become a Python Machine Learning master!</p>
            <a href="#" class="btn btn-primary">Start for 100$</a>
         </div>
      </div>
   </body>
</html>
```



**Find H5 Tags**

**replit:** [https://replit.com/@innominate817/Local-HTML-Scraping#main.py](https://replit.com/@innominate817/Local-HTML-Scraping#main.py)

```python
# Open local html file
with open("home.html", 'r') as html_file:
    # Read in file content
    content = html_file.read()

    # Create an instance of BeautifulSoup
    soup = BeautifulSoup(content, 'lxml')

    # Grab first H5 tag
    tag = soup.find('h5')
    print(f'First H5 tag: {tag}')

    # Grab all H5 tags
    tags = soup.find_all('h5')
    print(f'All h5 tags\n{tags}')
```

```bash
First H5 tag: <h5 class="card-title">Python for beginners</h5>
All h5 tags
[<h5 class="card-title">Python for beginners</h5>, <h5 class="card-title">Python Web Development</h5>, <h5 class="card-title">Python Machine Learning</h5>]
```



**Get a List of Courses in HTML File**

**replit:** [https://replit.com/@innominate817/Local-HTML-Scraping-2#main.py](https://replit.com/@innominate817/Local-HTML-Scraping-2#main.py)

```python
# Open local html file
with open("home.html", 'r') as html_file:
    # Read in file content
    content = html_file.read()

    # Create an instance of BeautifulSoup
    soup = BeautifulSoup(content, 'lxml')
    
    # Grab all H5 tags
    course_tags = soup.find_all('h5')
    for course in course_tags:
        # print text in each tag
        print(course.text)
```

```bash
Python for beginners
Python Web Development
Python Machine Learning
```



**Get Prices for Each Course**

**replit:** [https://replit.com/@innominate817/Local-HTML-Scraping-3#main.py](https://replit.com/@innominate817/Local-HTML-Scraping-3#main.py)

```python
# Open local html file
with open("home.html", 'r') as html_file:
    # Read in file content
    content = html_file.read()

    # Create an instance of BeautifulSoup
    soup = BeautifulSoup(content, 'lxml')
    
    # Grab all course cards
    course_cards = soup.find_all('div', class_='card')
    for course in course_cards:
        # print text in each tag
        course_name = course.h5.text
        course_price = course.a.text.split()[-1]
        print(f'Course: {course_name}\tPrice: {course_price}')
```

```bash
Course: Python for beginners	Price: 20$
Course: Python Web Development	Price: 50$
Course: Python Machine Learning	Price: 100$
```



## Website Scraping

**Get and Parse Raw HTML**

```python
html_text = requests.get('https://www.timesjobs.com/candidate/job-search.html?searchType=personalizedSearch&from=submit&txtKeywords=python').text
soup = BeautifulSoup(html_text, 'lxml')
```



**Get Most Recent Job Listing**

**replit:** [https://replit.com/@innominate817/Website-Scraping-BeautifulSoup-1#main.py](https://replit.com/@innominate817/Website-Scraping-BeautifulSoup-1#main.py)

```python
jobs = soup.find_all('li', class_='clearfix job-bx wht-shd-bx')
job = jobs[0]
company_name = ' '.join(job.find('h3', class_='joblist-comp-name').text.split())
print(f'Company: {company_name}')
skills = ' '.join(job.find('span', class_='srp-skills').text.split()).replace(' ,', ',')
print(f'Skills: {skills}')
published_date = job.find('span', class_='sim-posted').text
print(f'Published: {published_date}')
```

```bash
Company: Pure Tech Codex Private Limited
Skills: rest, python, database, django, debugging, mongodb
Published: 
Posted 4 days ago
```



**Get Jobs Posted a Few Days Ago**

**replit:** [https://replit.com/@innominate817/Website-Scraping-BeautifulSoup-2#main.py](https://replit.com/@innominate817/Website-Scraping-BeautifulSoup-2#main.py)

```python
jobs = soup.find_all('li', class_='clearfix job-bx wht-shd-bx')

for i, job in enumerate(jobs):
    company_name = job.find('h3', class_='joblist-comp-name').text
    company_name = ' '.join(company_name.split())
    skills = ' '.join(job.find('span', class_='srp-skills').text.split()).replace(' ,', ',')
    published_date = ' '.join(job.find('span', class_='sim-posted').text.split())
    if "Posted few days ago" in published_date:
        print(f'Listing #{i+1}:')
        print(f'Company: {company_name}')
        print(f'Skills: {skills}')
        print(f'Published: {published_date}')
        print()
```

```bash
Listing #2:
Company: EAGateway Services India Pvt Ltd (More Jobs)
Skills: python, pandas, numpy, git
Published: Posted few days ago

Listing #3:
Company: Surya Informatics Solutions Pvt. Ltd.
Skills: python, web technologies, linux, mobile, mysql, angularjs, javascript
Published: Posted few days ago

Listing #4:
Company: 3RI Technologies Pvt Ltd
Skills: embedded systems, embedded c, oscilloscope
Published: Posted few days ago

Listing #7:
Company: TEAMPLUS STAFFING SOLUTION PVT. LTD.
Skills: python, python scripting, shell scripting, unix
Published: Posted few days ago

Listing #19:
Company: hk infosoft
Skills: python, django,, framework
Published: Posted few days ago

Listing #22:
Company: CONNECTING 2 WORK
Skills: rest, python, django, mongodb
Published: Posted few days ago
```



**Save Job Listings to Files**

**replit:** [https://replit.com/@innominate817/Website-Scraping-BeautifulSoup-3#main.py](https://replit.com/@innominate817/Website-Scraping-BeautifulSoup-3#main.py)

```python
def find_jobs(unfamiliar_skill='None', path='./posts'):
    jobs = soup.find_all('li', class_='clearfix job-bx wht-shd-bx')

    if not (os.path.exists(path)):
        os.makedirs(path)

    for i, job in enumerate(jobs):
        company_name = job.find('h3', class_='joblist-comp-name').text
        company_name = ' '.join(company_name.split())
        skills = ' '.join(job.find('span', class_='srp-skills').text.split()).replace(' ,', ',')
        more_info = job.header.h2.a['href']
        published_date = ' '.join(job.find('span', class_='sim-posted').text.split())
        if "Posted few days ago" in published_date and unfamiliar_skill not in skills:
            with open(f'posts/{i+1}.txt', 'w') as f:
                f.write(f'Listing #{i+1}:\n')
                f.write(f'Company: {company_name}\n')
                f.write(f'Skills: {skills}\n')
                f.write(f'Published: {published_date}\n')
                f.write(f'More Info: {more_info}\n')
                print(f'File saved: {i}\n')
                
print("Put some skill that your are not familiar with")
unfamiliar_skill = input('>')
print(f'Filtering out {unfamiliar_skill}')

find_jobs(unfamiliar_skill)
```

```bash
Put some skill that your are not familiar with
>django
Filtering out django
File saved: 1

File saved: 2

File saved: 3

File saved: 6
```





**References:**

* [Web Scraping with Python - Beautiful Soup Crash Course](https://www.youtube.com/watch?v=XVv6mJpFOb0)





<!-- Cloudflare Web Analytics --><script defer src='https://static.cloudflareinsights.com/beacon.min.js' data-cf-beacon='{"token": "56b8d2f624604c4891327b3c0d9f6703"}'></script><!-- End Cloudflare Web Analytics -->