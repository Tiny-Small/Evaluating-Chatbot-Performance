# Evaluating Chatbot Performance: GPU+CPU Local Configurations vs. CPU on Google Cloud Run

## Project Summary

This study extends [Project Allamak - Retrieval Augmented Generation Chatbot Diagnosis System](https://github.com/danielbdy/allamak), which utilizes retrieval-augmented generation techniques alongside OpenAI's Large Language Models to offer preliminary medical diagnoses based on user-inputted symptoms. The aim here is to evaluate chatbot performance by comparing GPU+CPU configurations used locally with CPU resources on Google Cloud Run.

Our findings show that a local configuration utilizing both a GPU (Nvidia 4060 Laptop version) and a CPU can process approximately 1695 tokens per second, which translates to about 40 seconds per response. In contrast, a cloud configuration with 8 vCPUs and 32 GiB of memory on Google Cloud Run processes around 680 tokens per second, resulting in response times of approximately 91 seconds.

The document is organized into two main sections:
1. **How to Run Locally**: Instructions for setting up and running the chatbot.
2. **Experimental Results**: Analysis of the performance impacts of various configurations on Google Cloud Run.



## How to Run Locally

### 1. Setting up the Requirements
Ensure you have Python and pip already installed on your machine. The Python version used in this study is 3.10.6. Follow the instructions under "GPU Environment" if a GPU is available, or under "Non-GPU Environment" if a GPU is not available.

#### GPU Environment
First, upgrade pip and install the required packages from the `requirements.txt` file:

```
pip install --upgrade pip
pip install --no-cache-dir -r requirements.txt
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121
```

#### Non-GPU Environment
If a GPU is not available, you need to change line 6 in the `requirements.txt` file from `faiss-gpu` to `faiss-cpu` to ensure the chatbot runs on CPU.

After making the change, install the packages with the following commands:
```
pip install --upgrade pip
pip install --no-cache-dir -r requirements.txt
pip install llama-cpp-python
```


### 2. Large Language Model (LLM)
We use the [CapybaraHermes-2.5-Mistral-7B](https://huggingface.co/TheBloke/CapybaraHermes-2.5-Mistral-7B-GGUF) LLM in this study. To set up the model:

1. **Download the Model**: Download the [capybarahermes-2.5-mistral-7b.Q4_K_M.gguf](https://huggingface.co/TheBloke/CapybaraHermes-2.5-Mistral-7B-GGUF/blob/main/capybarahermes-2.5-mistral-7b.Q4_K_M.gguf) file from Hugging Face.
2. **Place the Model in the Correct Directory**: Move the downloaded file to the `LLM` folder within your project directory. If the `LLM` folder does not exist, create it in the root directory of your project.

#### Structure
```
This_folder/
├── 01_embedding_model.py
├── 02_vector_db.py
├── ...
├── LLM/
│ └── MODEL.gguf
```


### 3. Downloading the Embedding Model
The embedding model, [GIST-large-Embedding-v0](https://huggingface.co/avsolatorio/GIST-large-Embedding-v0), is specified in the `config.py` file. To use a different model from Hugging Face, modify line 10 in config.py before proceeding. Download the model with:
```
python3 01_embedding_model.py
```


### 4. Preparing the Vector Store
Prepare your vector database by running the following command. If you would like to use different data, replace the files in the `data` folder before executing this script. A completion message "Done" will confirm the process is finished.
```
python3 02_vector_db.py
```


### 5. Launching the Chatbot
Start the chatbot by running the command below; it should automatically open in your browser. If it doesn’t, please manually enter the *Local URL* specified in the terminal into your browser’s address bar.
```
streamlit run main.py
```



## Experimental Results

For our experiments, we containerized the CPU version of the application using Docker and deployed it to Google Cloud Run. The testing was initiated with the OpenBLAS library, and configured as follows in `setup.py`:

##### Base Configuration
```
llm = LlamaCpp(
    model_path=LLM_PATH,
    temperature=0,              # No randomness in generation
    max_tokens=2048,            # Maximum tokens to generate
    n_ctx=2048,                 # Context window size
    n_batch=512,                # Batch size for processing
    # n_threads=4,              # Number of processing threads. This is commented out.
    callback_manager=callback_manager,  # Callback manager instance
    verbose=True  # Enables debugging to trace computation steps
)
```


### 1. Comparison between using OpenBLAS

We evaluate the impact of using the OpenBLAS library on the response times, measured in tokens per second. OpenBLAS is a software library optimized for enhancing linear algebra operations and improving computational speed across various applications. For these tests, the [Q3_K_M](https://huggingface.co/TheBloke/CapybaraHermes-2.5-Mistral-7B-GGUF/blob/main/capybarahermes-2.5-mistral-7b.Q3_K_M.gguf) model and the base configurations previously detailed were utilized.

| OpenBLAS | Results                                    |
|:---------|:------------------------------------------:|
| Yes      | 150 sec / 1042 tokens (415 tokens per sec) |
| No       | 118 sec / 1042 tokens (528 tokens per sec) |

The results indicate that not using OpenBLAS resulted in a higher number of tokens processed per second. This suggests that, for our specific configuration and workload, OpenBLAS may introduce overhead that affects throughput negatively. This observation prompts further exploration of how different configurations impact performance.


### 2. Comparison between Quantized Model and Embeddings Model Downloaded

In this section, we analyze the response times between two models: the smaller Q3_K_M (3.52 GB) and the larger Q4_K_M (4.37 GB). Despite its larger size, Q4_K_M achieved a higher number of tokens processed per second, which indicates better response times compared to Q3_K_M.

The better response times of Q4_K_M can be attributed to its lower quality loss during the quantization process. Quantization, a technique that reduces the precision of a model’s numerical data from floating-point to lower-bit integers, aims to decrease the model size while increasing computational efficiency.

Due to this lower quality loss, Q4_K_M is capable of generating more concise answers, requiring fewer tokens to convey the same amount of information. Thus, increasing its overall performance.

| Model Size | Embedding Model Pre-Downloaded               | Embedding Model Not Pre-Downloaded        |
|:-----------|:--------------------------------------------:|:-----------------------------------------:|
| Q3_K_M     | 133 sec / 1042 tokens (466 tokens per sec)   | 150 sec / 1042 tokens (415 tokens per sec)|
| Q4_K_M     | *118 sec / 1003 tokens (507 tokens per sec)* | 123 sec / 1003 tokens (487 tokens per sec)|

For subsequent tests, the Q4_K_M model with pre-downloaded embeddings will be utilized, based on its demonstrated efficiency.


### 3. Comparison between the Different Values of n_threads and n_batch with Respect to the Base Configuration

After exploring chatbot performance through LLM quantization and pre-downloaded embedding model, this section examines the impact of adjusting the `n_threads` and `n_batch` parameters within the LlamaCpp function.

`n_threads` determines how many CPU threads are used simultaneously, which enhances the model's capacity to execute multiple tasks concurrently. Conversely, `n_batch` refers to the number of data batches the model processes simultaneously, potentially increasing throughput on systems designed for parallel processing.

Here are the results of testing different configurations of `n_threads` and `n_batch`:

| n_threads   | Results            |
|:------------|:------------------:|
| 2           | 463 tokens per sec |
| 4           | 462 tokens per sec |
| *8*         | 465 tokens per sec |

| n_batch     | Results            |
|:------------|:------------------:|
| 256         | 449 tokens per sec |
| 512         | 487 tokens per sec |
| *1024*      | 505 tokens per sec |
| 2048        | 477 tokens per sec |


While the parameters `n_threads` of 8 and `n_batch` of 1024 delivered the highest tokens per second in our tests, it is crucial to note that these findings might not be universally applicable. The utility of `n_batch` is traditionally significant in scenarios with multiple simultaneous client connections to the LLM, enhancing the processing efficiency of multiple requests and thus reducing wait times and improving throughput. Interestingly, we observed variations in tokens per second with different `n_batch` values, even with only one client in our tests. Additional experiments are necessary to fully explore whether these variations in results are statistically significant.


### 4. GPU+CPU Locally vs CPU Cloud Run

Lastly, we evaluated the chatbot's performance using the final configuration on both local GPU+CPU setups and CPU-based Google Cloud Run. The aim of this test was to compare the chatbot's response times in different environments under the optimal settings previously identified.

#### Final Configuration
```
    llm = LlamaCpp(
        model_path=LLM_PATH,
        # n_gpu_layers=20, # Disable when using CPU version
        temperature=1,
        max_tokens=512,
        n_ctx=2048,
        n_batch=1024,
        n_threads=8, # Disable when using GPU version
        n_threads_batch=8, # Disable when using GPU version
        repeat_penalty=1.2,
        presence_penalty=0.2,
        callback_manager=callback_manager,
        verbose=True
    )
```

| Environment   | Results            |
|:--------------|:------------------:|
| GPU+CPU       | 1695 tokens per sec|
| CPU Cloud Run | 643 tokens per sec |

The GPU+CPU configuration outperformed the CPU-only setup on Google Cloud Run in terms of chatbot performance. While the performance on the CPU-based Cloud Run was lower, significant potential for enhancement exists. Allocating more vCPUs could help bridge the performance gap. Additionally, internet latency is another probable factor contributing to slower performance, as it can delay data transmission between the client and the server, affecting response times and overall throughput. Furthermore, other configurations optimized for CPU usage that have not yet been explored in this study could potentially refine and enhance performance further. Addressing these factors could improve the chatbot's performance on Cloud Run, potentially matching or even surpassing the local GPU+CPU setup.


#### Screenshot of a conversation with the chatbot
![Screenshot 2024-05-14 172918](https://github.com/Tiny-Small/Evaluating-Chatbot-Performance/assets/149786226/d027ba68-0f04-42db-995b-3e8e2f51701c)



## Future Studies

For future studies, including a larger set of test samples for each parameter value is recommended. This expansion will facilitate a more rigorous assessment of the observed differences, utilizing statistical techniques such as the t-test to determine the statistical significance of the variations. These findings are documented here for future reference.
