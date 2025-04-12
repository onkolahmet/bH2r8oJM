# Text to SQL Generator Case Study

This project develops and deploys a fine-tuned language model for converting natural language questions into SQL queries. The goal is to make database querying more accessible to non-technical users by allowing them to ask questions in plain English and receive corresponding SQL code.

## Table of Contents
- [Project Overview](#project-overview)
- [Model Selection](#model-selection)
- [Dataset](#dataset)
- [Implementation Details](#implementation-details)
- [Evaluation Results](#evaluation-results)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [Project Architecture](#project-architecture)
- [Future Work](#future-work)

## Project Overview

In this project, I've developed a system that:
1. Fine-tunes a lightweight but powerful language model (Qwen2-0.5B-Instruct) on text-to-SQL tasks
2. Implements SQL post-processing to improve the quality of generated queries
3. Deploys the model with a user-friendly Gradio web interface

The system enables users to input natural language questions along with optional table schema information and receive properly formatted SQL queries that answer their questions.

## Model Selection

I selected **Qwen/Qwen2-0.5B-Instruct** for this project for several key reasons:

1. **Balance of size and performance**: At 0.5B parameters, the model is small enough to be deployed easily but still has strong reasoning capabilities when fine-tuned.
2. **Instruction-tuned foundation**: The base model already understands how to follow instructions, making it well-suited for the fine-tuning approach.
3. **Resource efficiency**: The smaller size allows for faster training and inference, making it practical for deployment.
4. **Customizability**: Being smaller than models like Llama-3 or Mixtral, it has more room for improvement through fine-tuning on specialized tasks.

The decision focused on finding a balance between model capabilities and practical deployment constraints, prioritizing a model that could be effectively trained and served without requiring excessive computational resources.

## Dataset

For training, I used the **gretelai/synthetic_text_to_sql** dataset from Hugging Face, which contains:
- Natural language questions
- Corresponding SQL queries
- Table context/schema information

This dataset is particularly valuable because:
1. It contains diverse SQL querying scenarios
2. It includes table schema information, which helps the model understand database structure
3. It has high-quality SQL examples that follow best practices
4. It's large enough (10,000 examples used for training) to fine-tune effectively

## Implementation Details

### Fine-tuning Approach

I used the Hugging Face **TRL (Transformer Reinforcement Learning)** library with **SFTTrainer** for supervised fine-tuning. Key implementation decisions include:

1. **Response template masking**: Used a "### The response query is:" template with DataCollatorForCompletionOnlyLM to focus training on generation rather than understanding.
2. **Structured prompt format**: Implemented a consistent prompt structure including table context and questions.
3. **Training parameters**: Used a learning rate of 1e-4, gradient accumulation steps of 4, and mixed precision (fp16) for efficient training.
4. **Epoch-based saving**: Saved model checkpoints after each epoch with a save limit of 2 to conserve space.

### SQL Post-processing

I implemented a thorough SQL post-processing pipeline (in `evaluation.py` and `app.py`) that:
1. Removes SQL comments and markdown formatting
2. Identifies and deduplicates multiple queries
3. Selects the most relevant query based on complexity and SQL structure
4. Formats the SQL consistently with proper indentation and keyword casing

This post-processing significantly improved the quality of the generated SQL, as shown in the evaluation results (BLEU score improved from 0.1452 to 0.5195).

### Deployment

The model is deployed using **Gradio**, which provides:
1. A clean web interface accessible to non-technical users
2. Examples to help users understand what kinds of questions can be asked
3. Schema input capabilities to better contextualize questions
4. Syntax highlighting for the generated SQL
5. No-code deployment that runs on standard hardware

## Evaluation Results

The model was evaluated using standard text generation metrics and SQL-specific metrics:

### Zero-shot Prompting (on `gretelai/synthetic_text_to_sql/test`)

**After Post-processing:**
* **BLEU Score:** 0.5195
* **ROUGE-L F1:** 0.7031
* **CHRF Score:** 70.0409

**Before Post-processing:**
* **BLEU Score:** 0.1452
* **ROUGE-L F1:** 0.3009
* **CHRF Score:** 47.8182

**SQL-Specific Metrics:**
* **Exact Match (case insensitive):** 0.1600
* **Normalized Exact Match:** 0.1500
* **Average Component Match:** 0.4528
* **Average Entity Match:** 0.8807

**Query Quality Distribution:**
* **High Quality (≥80% component match):** 18 (18.0%)
* **Medium Quality (50-79% component match):** 28 (28.0%)
* **Low Quality (<50% component match):** 54 (54.0%)

### Few-shot Prompting (on `gretelai/synthetic_text_to_sql/test`)

**After Post-processing:**
* **BLEU Score:** 0.2680
* **ROUGE-L F1:** 0.4975
* **CHRF Score:** 57.1704

**Before Post-processing:**
* **BLEU Score:** 0.1272
* **ROUGE-L F1:** 0.2816
* **CHRF Score:** 46.1643

**SQL-Specific Metrics:**
* **Exact Match (case insensitive):** 0.0000
* **Normalized Exact Match:** 0.0000
* **Average Component Match:** 0.2140
* **Average Entity Match:** 0.8067

**Query Quality Distribution:**
* **High Quality (≥80% component match):** 4 (4.0%)
* **Medium Quality (50-79% component match):** 17 (17.0%)
* **Low Quality (<50% component match):** 79 (79.0%)

### Key Insights from Evaluation

1. **Post-processing is essential**: The SQL cleaning pipeline dramatically improved results.
2. **Entity recognition is strong**: The model correctly identifies entities (tables, columns) with 88% accuracy.
3. **Zero-shot outperforms few-shot**: Surprisingly, zero-shot prompting performed better than few-shot examples, possibly due to confusion from mixed examples.
4. **Room for improvement**: While 18% of queries are high quality, there's significant room for improvement, particularly in complex query construction.

## Setup Instructions

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended but not required)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/text-to-sql-generator.git
cd text-to-sql-generator
```
2. Run the setup script to install dependencies:
```bash
python setup.py
```
This script will:

- Check for GPU availability
- Install pip if not available
- Install required packages from requirements.txt

## Model Training (Optional)
If you want to train the model yourself:
```bash
python main.py --model_id "Qwen/Qwen2-0.5B-Instruct" --dataset_id "gretelai/synthetic_text_to_sql" --epochs 5 --batch_size 8 --user_name "your_username"
```
## Parameters

- `--model_id`: Hugging Face model to use as base  
- `--dataset_id`: Dataset to use for fine-tuning  
- `--epochs`: Number of training epochs  
- `--batch_size`: Batch size for training  
- `--hf_login`: Whether to login to Hugging Face (set to False if not pushing to Hub)  

## Usage

### Running the Web Interface

Start the Gradio web interface:

```bash
gradio app.py
```
Then open your browser and navigate to [http://localhost:7860](http://localhost:7860) to use the application.

### Using the Interface

1. Enter your question in natural language  
   _(e.g., "Find all products in the Electronics category with a price less than $500")_
2. Optionally provide table context/schema information  
3. Click **"Generate SQL Query"** or press Enter  
4. View and copy the generated SQL query  

---

## Project Architecture

The project consists of several key files:

- `setup.py`: Handles environment setup and dependency installation  
- `get_model.py`: Loads pre-trained model and tokenizer  
- `data.py`: Handles dataset loading and formatting for training  
- `main.py`: Contains training logic and Hugging Face Hub integration  
- `evaluation.py`: Implements comprehensive model evaluation metrics  
- `app.py`: Deploys the model with a Gradio web interface  

---

## Future Work

Based on the evaluation results, several areas could be improved:

- **Enhanced training data**: Incorporating more complex SQL queries and diverse table schemas  
- **Iterative prompting**: Implementing a two-step process that first identifies tables and then constructs queries  
- **SQL validation**: Adding a validation step to ensure generated SQL is syntactically correct  
- **User feedback loop**: Implementing a feedback mechanism to collect corrections that could be used for further fine-tuning  
- **Optimization for specific database dialects**: Tailoring the model for MySQL, PostgreSQL, or SQLite specific syntax  

---
