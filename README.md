- # Enhancing Retrieval-Augmented Generation with Adaptive Chunking 
  
  Contributors: Tanish Upreti, Zihan Gong, Wenlong Zheng, Axel Tang
  
  ## An Applied Deep Learning Project
  
  Tech Used: 
  
  ```ruby
  Concepts:
    - RAG
    - LLM
    - Adaptive RAG
  Tech:
    - Python
    - Qwen-2.5
    - FAISS, BGE-M3
  ```
  
  ## A Core Summary 
  
  Retrieval-Augmented Generation (RAG) - A framework that combines information retrieval (fetching relevant documents) with large language models (LLMs) by introducing adaptive chunking which is a smarter way of splitting text into pieces for retrieval.
  
  Adaptive RAG system is more intelligent by breaking down questions into subquestions, leading to better retrieval quality, stronger context understanding, and higher fidelity AI responses. Core improvement in Adaptive RAG system is the efficiency compared to iterative RAG, as it encourages early stop when all necessary documents are retrieved.
  
  In this project , we will find ways to enhance this process leading to better accuracy and efficiency for Adaptive RAG.
  
  ## Requirements of the Project
  
  1. Implement a Retrieval-Augmented Generation pipeline using fixed-size chunking strategy as
     baseline
  2. Implement Adaptive Retrieval-AugmentedGeneration
  3. Training model with GRPO
  4. Use TriviaQA, NaturalQuestions to compare and analyze the impact of different chunking strategies
  5. Deliver source code for implementation of different chunking strategies, and results
     of each
  
  ## Retrieve Setup
  
  | Embedding Model |      Vector Index       |        Retrieval Source        |
  | :-------------: | :---------------------: | :----------------------------: |
  |     BGE-M3      | FAISS (IVF65546, PQ128) | English Wikipedia Corpus (42M) |
  
  ## Adaptive RAG Implementation
  
  The Adaptive RAG implementation is implemented in `rag_pipeline/` package.
  
  Core Files:
  
  - `rag_pipeline/` : package implementing chunking, embeddings, Retriever, Generator, Router
    an FAISS Index Builder, SQlite vector store, Retriever, Generator, Router and RAG orchestrator.
  - `trainer/` : package implementing GRPO training
  - `scripts/` : package implementing dataset preprocess
  - `eval.py` : script used to evaluate results on benchmark
  - `data/`: directory containing the preprocessed benchmark used to evalute the standard RAG Baseline
  - `faiss_store/`: directory containing the FAISS Index + SQLite Database (you have to manually create this directory and download the FAISS Index + SQLite Database)
  - `out/`: directory containing the trained model weights (will be created after training)
  - `requirements.txt` : dependency suggestions for the baseline.
  
  ### Quickstart:
  
  The Adaptive RAG System is dependent on the FAISS Index + SQLite Database to store Wikipedia corpus. If you don't want to use Adaptive RAG, you can skip this section. The preprocessed benchmark used to evalute the standard RAG Baseline is stored in `data/` directory.  
  
  ### Download FAISS Index + SQLite Database:
  
  ```bash
  mkdir faiss_store
  cd faiss_store
  wget https://drive.google.com/file/d/1KOguF62rpBTnNmNEikyCpYAxO6QvGCFB/view?usp=sharing
  tar -xvf corpus-BAAI.tar.gz
  rm corpus-BAAI.tar.gz
  ```
  
  ### Training:
  
  ```bash
  pip install -r requirements.txt
  cd scripts
  python preprocess_dataset.py
  cd ..
  cd trainer
  deepspeed --num_gpus 4 train_grpo.py \
      --deepspeed \
  ```
  
  #### Training Parameters used for 4090:
  
  ```bash
  deepspeed --num_gpus 1 train_grpo.py \
      --deepspeed \
      --save_dir ../out/full_weights \
      --save_name grpo_qwen_0.5b \
      --model Qwen/Qwen2.5-0.5B-Instruct \
      --data_path ../dataset/RAG_1000.jsonl \
      --num_generations 4 \
      --batch_size 4 \
      --accumulation_steps 4 \
      --learning_rate 8e-8 \
      --stage 2 \
      --dtype bfloat16 \
      --epochs 10 \
      --max_gen_len 256 \
      --use_wandb \
      --num_workers 4 \
      --max_step 10 \
      --from_checkpoint 1 \
      --max_seq_len 1280 \
      --offload 1\
      --chunk_size 6 \
      --wandb_project AdaptiveRAG-GRPO
  ```
  
  ### Convert DeepSpeed Checkpoint to Hugging Face Model:
  
  ```bash
  cd out/full_weights
  python zero_to_fp32.py . .
  ```
  
  ### Train with LoRA on H800:
  
  ```bash
  deepspeed --num_gpus 1 train_grpo_lora.py \
      --deepspeed \
      --save_dir ../out/lora \
      --save_name grpo_qwen_7b_LoRA \
      --model Qwen/Qwen2.5-7B-Instruct \
      --data_path ../dataset/RAG_1000.jsonl \
      --num_generations 8 \
      --batch_size 2 \
      --accumulation_steps 2 \
      --learning_rate 8e-8 \
      --stage 2 \
      --dtype bfloat16 \
      --epochs 3 \
      --max_gen_len 256 \
      --use_wandb \
      --num_workers 4 \
      --max_step 10 \
      --from_checkpoint 1 \
      --max_seq_len 1536 \
      --offload 1 \
      --chunk_size 3 \
      --wandb_project AdaptiveRAG-GRPO-LoRA-7B
  ```
  
  You can find the saved LoRA adapter directory in `out/lora/RAG_lora_adapter`
  
  **Note:**
  
  - If you don't want to use wandb to record and visualize training process, you can remove `--use_wandb` and `--wandb_project` from the command.
  
  - Deepspeed Zero 3 is currently not supported
  
  ### Download the Trained LoRA Adapter Weight
  
  ```bash
  wget https://drive.google.com/file/d/11piyfWI392JYN3LIilm_4xf3olT7lonv/view?usp=sharing
  ```
  
  This is the LoRA Adapter weight trained by us.
  
  ### Evaluation:
  
  ```bash
  python eval.py
  ```
  
  If you don't want to use Adaptive RAG, you can use the following command to evaluate the Standard RAG Baseline only:
  
  ```bash
  python eval.py --no_adaptive
  ```
  
  ### Evaluation on Trained LoRA Adapter:
  
  ```bash
  pthon eval.py --weights_path ../out/lora/RAG_lora_adapter
  ```
  
  **Note:**
  
  - If you want to evaluate Iterative RAG, you can do it by replacing all the prompts in ``rag_pipeline/router.py`` from ADAPTIVE to ITERATIVE
  
  ## Evaluation Result (Accuracy)
  
  Source:
  
  |     Benchmark     |        Model         | vanilla | RAG (Top 1) |
  | :---------------: | :------------------: | :-----: | :---------: |
  | TriviaQA_VAL_1000 |      Qwen-3-8B       |  59.2%  |    63.5%    |
  | TriviaQA_VAL_1000 | Qwen-2.5-7B-Instruct |  54.1%  |    57.3%    |
  |    NQ_VAL_1000    |      Qwen-3-8B       |  25.2%  |    41.7%    |
  |    NQ_VAL_1000    | Qwen-2.5-7B-Instruct |  21.8%  |    37.9%    |
  
  Context Confidence
  
  |  Benchmark   | DPR (Top 20) | BM25(Top 20) | BGEM3(Top 20) |
  | :----------: | :----------: | :----------: | :-----------: |
  | TriviaQA_VAL |    83.4%     |    90.3%     |     90.9%     |
  |    NQ_VAL    |    77.3%     |    62.4%     |     69.2%     |
  
  |             | DPR  | BM25 | BGE-M3 |
  | :---------: | :--: | :--: | :----: |
  | Mmeory Cost | 60GB | 4GB  |  10GB  |
  
  |     Benchmark     |        Model         | BM25  | BGE-M3 |  DPR  |
  | :---------------: | :------------------: | :---: | :----: | :---: |
  | TriviaQA_VAL_1000 | Qwen-2.5-7B-Instruct | 74.3% | 74.1%  | 57.3% |
  |    NQ_VAL_1000    | Qwen-2.5-7B-Instruct | 30.4% | 37.1%  | 37.9% |
  
  Specs for FAISS index of BGE-M3:
  
  - IVF65536,PQ128
  
  ## Adaptive RAG Evaluation
  
  |   Benchmark   |          Router Model          | RAG SYSTEM | BGE-M3 | Avg. STEP | Avg. Doc |   Time   |
  | :-----------: | :----------------------------: | :--------: | :----: | :-------: | :------: | :------: |
  | TriviaQA_1000 |              None              |    None    | 54.1%  |    NA     |    NA    |    NA    |
  | TriviaQA_1000 |      Qwen-2.5-7B-Instruct      |   Native   | 74.1%  |     1     |    1     |    NA    |
  | TriviaQA_1000 |      Qwen-2.5-7B-Instruct      | Iterative  | 79.4%  |    MAX    |   4.84   | 06:08:43 |
  | TriviaQA_1000 |      Qwen-2.5-7B-Instruct      |  Adaptive  | 76.6%  |   1.031   |   1.94   | 01:46:36 |
  | TriviaQA_1000 | Qwen-2.5-7B-Instruct-Finetuned |  Adaptive  | 76.6%  |   1.024   |   1.95   | 02:02:3  |
  |    NQ_1000    |              None              |    None    | 21.8%  |    NA     |    NA    |    NA    |
  |    NQ_1000    |      Qwen-2.5-7B-Instruct      |   Native   | 37.1%  |     1     |    1     |    NA    |
  |    NQ_1000    |      Qwen-2.5-7B-Instruct      | Iterative  | 43.9%  |    MAX    |   5.00   | 04:58:23 |
  |    NQ_1000    |      Qwen-2.5-7B-Instruct      |  Adaptive  | 38.4%  |   1.032   |   1.75   | 01:48:23 |
  |    NQ_1000    | Qwen-2.5-7B-Instruct-Finetuned |  Adaptive  | 38.6%  |   1.053   |   1.77   | 02:14:05 |
  
  
  
  Notes:
  
  Small GPU notes
  
  - See `docs/RAG_small_gpu_instructions.md` for safe instructions on attempting a real-model
    run on GPUs with limited VRAM (for example, RTX 3080 mobile with ~8GB). The demo defaults
    to deterministic fallback embeddings to avoid heavy memory use.
  - The code is implemented with the help of Gemini 3 Pro for debugging purpose
