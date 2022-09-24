---
categories:
- nats
- cloud
- notes
date: 2021-12-9
description: My notes from the NATS KubeCon + CloudNativeCon NA 2019 Keynote.
hide: false
layout: post
search_exclude: false
title: Notes on NATS Keynote
toc: false

aliases:
- /Notes-on-NATS-Keynote/
---

* [Overview](#overview)
* [What is NATS](#what-is-nats)
* [NATS Ecosystem](#nats-ecosystem)
* [Usage Examples](#usage-examples)
* [NATS Community](#nats-community)



## Overview

I recently heard about [NATS](https://nats.io/), a "message oriented middleware". It is a software infrastructure that provides the exchange of data that is segmented into messages among computer applications and services. I was curious about it's potential in applications that involve streaming real-world sensor data, so I decided to check out there [2019 keynote](https://www.youtube.com/watch?v=lHQXEqyH57U). Below are some notes I took while watching.



## What is NATS

NATS is a simple, reliable, cloud-native messaging system

**Goal of Project:** Be the enabling technology to securely connect all the world's digital systems, services and devices.

## Connect Everything:

- Shared utility of any size
- Decentralized and Federated
    - Mix a shared utility with your own servers and security
- Secure by default, no passwords or keys, powerful authorization
- On-Premise, Multi-Cloud, Multi-Deployment, the Edge, and IoT
- Communicate, Publish, Consume, and Save and Restore, State
- Healthy and thriving Ecosystem, of Services and Streams

* Scalable services and streams

### Quality of service

* Over 30 different client programming language implementations (Nov. 2019)

* Highly resilient and self healing

### Example Use Cases:

- cloud messaging
    - services (micro-services)
    - Event/Data streaming (observability/analytics)
    - Command and control
- IoT and Edge
    - Telemetry/Sensor Data/Command and Control
- Augmenting or Replacing Legacy Messaging

Can support hundreds of thousands of nodes with really fast response times (700ms)

### Some Companies Using NATS

- [CHORIA](https://choria.io/)
- [Netlify](https://www.netlify.com/)
- [Mastercard](https://www.mastercard.us/en-us.html)
- [StorageOS](https://www.ondat.io/) (Now ondat)
- [Tinder](https://tinder.com/)
- [Platform 9](https://platform9.com/)
- [Qlik](https://www.qlik.com/us/)



## NATS Ecosystem

### Integrations

- curl command to install and deploy to [Kubernetes](https://kubernetes.io/)
- [Prometheus](https://prometheus.io/) Exporters, [Fluentd](https://www.fluentd.org/) Plugin and [OpenTracing](https://opentracing.io/)/[Jaeger](https://www.jaegertracing.io/) support
- [Dapr.io](http://Dapr.io) Component Integration
- [Spring Boot](https://spring.io/projects/spring-boot) Starter
- NATS Cloud Stream Binder for [Spring](https://spring.io/)
- NATS/[Kafka](https://kafka.apache.org/) Bridge
- NATS / [MQSeries](https://www.ibm.com/products/mq) Bridge
- [Go-Cloud](https://github.com/google/go-cloud) and [Go-Micro](https://github.com/asim/go-micro) pub/sub integration

### Basic Messaging Patterns

- Services - Request/Reply
    - Scale up and down
    - location transparency
    - observability
- Streams - Events and Data
    - Scalable N:M communications
    - Realtime and persistent
    - playback by time, sequence, all or only last received
    

### Accessing a NATS System

- Free Community Servers
    - [demo.nats.io](http://demo.nats.io) (both open and secure versions)
- Kubernetes
    - `curl [https://nats-io.github.io/k8s/setup.s](https://nats-io.github.io/k8s/setup.sh)| sh`
- Docker
    - `docker run -p 4222:4222 -ti nats:latest`
- Additional Information
    - [Installing a NATS Server](https://docs.nats.io/nats-server/installation)
    



## Usage Examples

### HTTP vs NATS (requestor)

#### HTTP

```jsx
resp, err := http.Get("http://example.com/")
if err != nil {
	// handl error
}

defer resp.Body.Close()
body, err := ioutil.ReadAll(resp.Body)

// decode body
```

#### NATS

```go
nc, err := nats.Connect("demo.nats.io")
if err != nil {
	// handle err
}

resp, err := nc.Request("example.com", nil, 2*time.Second)
// decode resp.Data

```

### HTTP vs NATS (service)

#### HTTP

```go
http.HandlFunc("/bar", func(w http.ResponseWriter, r*http.Request){
	fmt.Fprintf.(w, "Hello World")
})

log.Fatal(http.ListenAndServe(":8080", nil)
```

#### NATS

```go
nc, err := nats.Connect("demo.nats.io")
if err != nil {
	// handle err
}

sub, err := nc.QueueSubscribe("bar", "v0.1", func(m * nats.Msg) {
	m.Respond([]byte("Hello World"))
})

```



## NATS Community

- Over 1,000 contributors
- 33 Client languages (Nov. 2019)
- 75 Public Repos
- 100M NATS Server and Streaming Server Docker Downloads
- ~1,600 Slack Members
- 20+ releases since June 2014
    - https://nats.devstats.cncf.io/d/9/developers-summary
    - [Grafana](https://nats.devstats.cncf.io/d/9/developers-summary)
    

 

**References:**

* [Keynote: NATS: Past, Present and the Future - Derek Collison, Founder and CEO, Synadia](https://www.youtube.com/watch?v=lHQXEqyH57U)





<!-- Cloudflare Web Analytics --><script defer src='https://static.cloudflareinsights.com/beacon.min.js' data-cf-beacon='{"token": "56b8d2f624604c4891327b3c0d9f6703"}'></script><!-- End Cloudflare Web Analytics -->