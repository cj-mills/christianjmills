---
title: "Office Hours 4: Modal with Charles Frye"
date: 2024-7-6
image: /images/empty.gif
hide: false
search_exclude: false
categories: [notes, llms]
description: "This Q&A session covers a wide array of topics related to Modal, a platform designed to simplify the execution of Python code in the cloud."

twitter-card:
  creator: "@cdotjdotmills"
  site: "@cdotjdotmills"
  image: /images/default-preview-image-black.png
open-graph:
  image: /images/default-preview-image-black.png
---



::: {.callout-tip}
## This post is part of the following series:
* [**Mastering LLMs Course Notes**](/series/notes/mastering-llms-course-notes.html): My notes from the course **Mastering LLMs: A Conference For Developers & Data Scientists** by **Hamel Husain** and **Dan Becker**.
:::



## Understanding Modal

* **ELI5:** Modal allows users to run their Python code in the cloud effortlessly.
* **In-depth:** Modal is a remote procedure call (RPC) framework designed for Python, particularly excelling in data-intensive workloads like those involving large models or datasets.
* **Advantages over Traditional Cloud Functions (AWS Lambda, Google Cloud Functions):**
  * **Focus:** Handles compute-heavy and data-intensive tasks more effectively.
  * **Simplicity:** Removes the need for server management.
  * **Scalability:**  Scales resources dynamically based on demand. 
* **Key Features:**
  * Seamless local development with automatic code deployment to the cloud.
  * Simplified parallel processing and scaling.
  * Integration with popular frameworks like [FastAPI](https://fastapi.tiangolo.com/).
* **Founders' Vision:**  Address common infrastructure challenges faced by data scientists and ML practitioners, emphasizing rapid development cycles and feedback loops.



## Startup Times and Optimization

* **Concern:** Modal's startup time compared to traditional server-based solutions.
* **Modal's Performance:**
  * Container startup (Docker run): 1-2 seconds (50th percentile startup time).
  * Slowdown occurs when environment setup requires loading large elements into memory (e.g., language model weights).
* **Solutions for Faster Startup:**
  * **Keep Warm:** Leave applications running for minimal latency, especially crucial for GPU-bound tasks. (Trade-off: potentially higher cost for idle resources).
    * [Keep containers warm for longer with `container_idle_timeout`](https://modal.com/docs/guide/cold-start#keep-containers-warm-for-longer-with-container_idle_timeout)
  * **CUDA Checkpointing:** New feature under integration, expected to accelerate subsequent invocations. 
    * **GitHub Repository:** [cuda-checkpoint](https://github.com/NVIDIA/cuda-checkpoint)
  * **CPU Tasks:** Easily sliced and diced, making them cost-effective in keep-warm mode due to minimal resource consumption during idle periods.
  * **Optimization Potential:**  The LLM fine-tuning repo hasn't been fully optimized for boot times; improvements are possible.
    * **GitHub Repository:** [llm-finetuning](https://github.com/modal-labs/llm-finetuning)



## Local Development and Modal Integration

* **Challenge:** Integrating a complex FastAPI application with local databases and debugging tools, then deploying it seamlessly to Modal.
* **Thin Client Approach:**
  * Modal examples typically employ a thin client architecture for simplicity in dependency management.
  * Local development within the thin client can be limited due to the absence of specific dependencies.
* **Solutions:**
  * **Modal's Remote Development Tools:** Shell access, VS Code integration, and JupyterLab instances within Modal's environment. 
    * [`modal launch`](https://modal.com/docs/reference/cli/launch#modal-launch): Open a serverless app instance on Modal.
      * [`modal launch jupyter`](https://modal.com/docs/reference/cli/launch#modal-launch-jupyter): Start Jupyter Lab on Modal.
      * [`modal launch vscode`](https://modal.com/docs/reference/cli/launch#modal-launch-jupyter): Start Visual Studio Code on Modal.
    * [`modal shell`](https://modal.com/docs/reference/cli/shell#modal-shell): Run an interactive shell inside a Modal image.
  * **Thick Client Architecture:**
    * Build a local environment mirroring the Modal environment.
    * Utilize tools like Dockerfiles, requirements.txt, poetry, or conda for consistent dependency management. 
  * **Resource:** Explore the `awesome-modal` repository on GitHub for production-ready examples, some utilizing a thicker client approach.
    * **GitHub Repository:** [awesome-modal](https://github.com/modal-labs/awesome-modal)



## Iterative Development Workflow

* **Challenge:**  Fine-tuning models locally on a small scale with debugging and then scaling up on Modal with a full dataset and larger models.
* **Recommendations:**
  * **[`modal.Image`](https://modal.com/docs/reference/modal.Image) Class:** 
    * Base class for container images to run functions in.
    * Utilize for environment definition, ensuring consistency between local and remote setups.
    * **Guide:** [Custom containers](https://modal.com/docs/guide/custom-container)
  * **Dependency Management:** Leverage tools like `pip freeze` and poetry for tighter control over environments.
  * **Hardware Considerations:** Be mindful of potential discrepancies between local and Modal GPUs.



## Modal for CPU-Intensive Workloads

* **Question:** Is Modal suitable for parallel CPU-bound tasks rather than just GPU acceleration?
* **Answer:**  Yes, Modal is highly recommended for CPU-intensive and parallelizable tasks.
* **Reasons:**
  * **Cost-Effectiveness:** CPUs are cheaper on Modal due to efficient time-slicing and readily available resources.
  * **Simplified Parallelization:** Modal's architecture and tools streamline the execution of parallel CPU workloads. 



## Cost Comparison and Value Proposition

* **Concern:**  Fine-tuning on Modal appears more expensive than platforms like Jarvis Labs.
* **Modal's Pricing:**
  * Transparent, based on underlying cloud provider costs.
  * No hidden fees or inflated pricing strategies.
* **When Modal Wins:**
  * **High Operational Overhead:**  Modal excels when the effort of managing servers (spinning up, down, utilization tracking) outweighs the raw compute cost.
  * **Unpredictable Workloads:**  Serverless nature shines when demand fluctuates, and predicting utilization is challenging.
  * **Scalability Needs:** Modal simplifies scaling to thousands of GPUs, surpassing the limitations of individual users or smaller organizations.
  * **GPU Accessibility:**  Modal offers readily available GPUs, circumventing the challenges of procurement and allocation.
  * **Developer Experience:**  Streamlined workflow and reduced operational burden can justify a potential price premium for some users.



## Understanding Modal's Cost Structure 

* **Question:**  How can a keep-warm FastAPI app on Modal cost only 30 cents per month when CPU core pricing suggests a much higher cost?
* **Explanation:**
  * **Time-Slicing:** CPUs are shared efficiently, and Modal only charges for actual usage, not idle time.
  * **Low Utilization:** Web apps typically have low average CPU utilization, further reducing costs.
  * **RAM-Based Pricing:** During idle periods, charges are primarily determined by RAM usage, which is often minimal for lightweight apps. 



## Streaming Output from LLMs

* **Question:** Availability of examples showcasing streamed output from LLMs in FastAPI apps.
* **Answer:**
  * Examples for streaming and FastAPI integration are available in the documentation:
    * [Fast inference with vLLM (Mixtral 8x7B)](https://modal.com/docs/examples/vllm_mixtral)
    * [QuiLLMan: Voice Chat with LLMs](https://modal.com/docs/examples/llm-voice-chat)
* **Modal's Async Support:**  Modal simplifies asynchronous programming, making streaming implementations easier.
  * **Guide:** [Asynchronous API usage](https://modal.com/docs/guide/async)




## Code Portability and Modal Dependency

* **Concern:** Modal's decorators might hinder code portability to other environments. 
* **Response:** 
  * While Modal promotes a specific architecture for performance and cost optimization, code can be written to minimize tight coupling.
  * Decorators can be removed or bypassed if needed to port code to different environments.
  * Achieving portability often involves trade-offs in performance and cost-effectiveness.



## Data Privacy

* **Question:** Modal's policy on data privacy and potential use of user data for model training.
* **Answer:**
  * **Commitment to Security:**  Modal is [SOC 2 compliant](https://modal.com/blog/soc2) and working towards SOC 2 Type 2 certification, demonstrating a high standard of data security.
  * **User Data Protection:** Modal treats user application data as confidential. Permission is sought before reviewing data, even for support purposes.
  * **No User Data Training:** Modal, as an infrastructure company, doesn't use customer data for training internal models.



## Running Databases on Modal

* **Question:**  Feasibility of running a key-value store (e.g., [LevelDB](https://github.com/google/leveldb)) on Modal for a development web endpoint.
* **Recommendations:**
  * **Modal's Built-in Solutions:**
    * [`modal.Dict`](https://modal.com/docs/reference/cli/dict):  Offers a persistent, distributed key-value store accessible to all Modal functions. 
    * [`modal.Queue`](https://modal.com/docs/reference/cli/queue): Provides a distributed queue system similar to [Redis](https://redis.io/). 
  * **Alternative Approach for Analytic Databases:**
    * Host databases externally (not ideal on Modal).
    * Mount cloud storage (e.g., S3) containing data in formats like Parquet or Arrow to Modal functions.
    * Utilize libraries like [DuckDB](https://duckdb.org/) for efficient querying within the Modal environment.
    * **Example:** [Analyze NYC yellow taxi data with DuckDB on Parquet files from S3](https://modal.com/docs/examples/s3_bucket_mount#analyze-nyc-yellow-taxi-data-with-duckdb-on-parquet-files-from-s3)



## Balancing Cost and Uptime for GPU Inference

* **Question:**  Finding the sweet spot between cost and uptime for GPU inference when needing varying levels of availability.
* **Rule of Thumb:**
  * Modal tends to be more cost-effective when utilization is 60% or lower.
  * Consider factors like acceptable latency and workload characteristics (batch jobs vs. real-time requests).



## Local vs. Cloud Workload Distribution

* **Question:** Deciding when to utilize a local GPU (e.g., RTX 4090) versus offloading to Modal, considering cost and time efficiency. 
* **Workload Breakdown:**
  * **Inference:** Local GPUs are well-suited due to typically small batch sizes, making VRAM less of a constraint. 
  * **Evaluations:**  Larger eval sets might benefit from cloud GPUs for faster throughput, especially when running multiple evaluations concurrently.
  * **Fine-tuning:**  Often memory-intensive due to gradients and optimizer states. Cloud GPUs provide ample VRAM and simplify the use of techniques like sharding or larger batch sizes.
* **Don't undervalue your time:** Spending a little more on faster cloud compute can save a significant amount of time versus trying to run everything locally on a single GPU.



## Quick Q&A

* **Autoscaling:** Modal supports autoscaling with configurable parameters.
  * **Explanation:** [How does autoscaling work on Modal?](https://modal.com/docs/guide/concurrent-inputs#how-does-autoscaling-work-on-modal)
  * **Auto-scaling LLM inference endpoints:** [Hosting any LLaMA 3 model with Text Generation Inference (TGI)](https://modal.com/docs/examples/text_generation_inference)
* **Docker Image Access:** Downloading built Docker images is not currently supported. Users can build and provide their own images. 
* **Inference Serving:**  
  * [vLLM](https://github.com/vllm-project/vllm) for its ease of use and rapid development
  * [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) is a potentially faster but more involved alternative. 
* **Demo Preparation:**  "Hello World" and "TRT LLM" examples are good starting points.
  * **Example:** [Hello, world!](https://modal.com/docs/examples/hello_world)
  * **Example:** [Serverless TensorRT-LLM (LLaMA 3 8B)](https://modal.com/docs/examples/trtllm_llama)




{{< include /_about-author-cta.qmd >}}
