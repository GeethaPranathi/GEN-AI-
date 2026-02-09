# Introduction to Generative AI and Large Language Models (LLMs)


---

## 1. Introduction to Generative AI

### What is Generative AI?

Generative AI refers to a class of artificial intelligence models that can **create new content** rather than just analyze or classify existing data. The generated content can include:

- Text  
- Images  
- Audio  
- Video  
- Code  
- Structured data  

Traditional AI systems focus on **prediction and classification** based on input data. Generative AI goes a step further by **learning patterns from data** and using them to produce **original outputs**.

### Examples of Generative AI

- Writing human-like text (articles, emails, chat responses)
- Generating images from text descriptions
- Writing, explaining, and debugging code
- Creating summaries, stories, and reports

---

## 2. Evolution and History of AI to Generative AI

### 1. Rule-Based Systems (1950s – 1980s)

Early AI systems worked using **predefined rules** written by humans.


**Example:**
- If–else logic
- Expert systems

---

### 2. Machine Learning Era (1990s – 2000s)

Machine Learning introduced **data-driven learning**, allowing systems to learn patterns from data instead of explicit rules.

**Example Use-Cases:**
- Spam classification
- Recommendation systems (Netflix, Amazon)

---

### 3. Deep Learning (2010s)

Deep learning uses **neural networks with multiple layers** to process large datasets.

**Key Achievements:**
- Image recognition
- Speech recognition
- Natural language processing

---

### 4. Transformers and Generative AI (2017 – Present)

Transformers introduced the **attention mechanism**, enabling models to understand context and long-range dependencies.

This led to the rise of **Large Language Models (LLMs)** and modern Generative AI.

**Examples:**
- GPT
- LLaMA
- Claude
- Gemini

---

## 3. Why Generative AI is Important

- Handles unstructured data like text, images, and audio
- Reduces manual effort in content creation
- Enables natural language interaction with machines
- Forms the foundation for intelligent agents and automation

---

## 4. Use-Cases of Generative AI

### 1. Conversational AI
- Chatbots
- Virtual assistants
- Customer support automation

### 2. Content Generation
- Blogs and articles
- Emails and marketing copy
- Summarization and paraphrasing

### 3. Code Generation
- Writing code snippets
- Explaining complex logic
- Debugging programs

### 4. Search and Knowledge Systems
- Semantic search
- Question answering systems

### 5. Business Automation
- Automated report generation
- Data analysis assistance

### 6. Agentic Systems
- Task planning and execution
- Multi-step autonomous workflows

---

## 5. Generative AI Ecosystem

### 1. Foundation Models

Large pre-trained models trained on massive datasets.

**Examples:**
- GPT family
- LLaMA
- Gemini

---

### 2. Model Access Layer

Ways to access models:

- Cloud-based APIs
- Self-hosted open-source models

---

### 3. Tooling and Frameworks

Libraries and frameworks used to build GenAI applications.

**Examples:**
- LangChain
- LlamaIndex
- Hugging Face

---

### 4. Vector Databases

Used to store and retrieve embeddings for semantic search and RAG systems.

**Examples:**
- FAISS
- Chroma
- Pinecone

---

### 5. Application Layer

End-user applications built on top of Generative AI.

**Examples:**
- Chat applications
- Retrieval-Augmented Generation (RAG)
- AI agents

---

## 6. Limitations of Generative AI

- Can hallucinate incorrect or misleading information
- Highly dependent on training data quality
- Sensitive to prompt phrasing
- Requires guardrails for safe and ethical usage
- Lacks true reasoning and real-world understanding

---

## 7. Introduction to Large Language Models (LLMs)

### What are Large Language Models?

Large Language Models (LLMs) are a subset of Generative AI models designed specifically to **understand, generate, and manipulate human language**.

They are trained on **massive amounts of text data** using deep learning and transformer architectures.

---

### Key Characteristics of LLMs

- Trained on billions or trillions of parameters
- Capable of performing multiple tasks using a single model
- Use attention mechanisms to understand context
- Can be fine-tuned or prompted for specific tasks

---

### Common Examples of LLMs

- GPT (OpenAI)
- LLaMA (Meta)
- Claude (Anthropic)
- Gemini (Google)

---

### Capabilities of LLMs

- Text generation and completion
- Question answering
- Summarization
- Translation
- Code generation
- Reasoning and explanation

---

### How LLMs Work (High-Level)

1. Input text is tokenized
2. Tokens are processed using transformer layers
3. Attention mechanism captures context
4. Model predicts the next token
5. Tokens are combined to generate output

---


### Role of LLMs in Modern GenAI Applications

- Core engine behind chatbots
- Powering RAG systems
- Acting as decision-making agents
- Automating workflows and reasoning tasks

---

## 8. Conclusion

Generative AI and Large Language Models are transforming how humans interact with machines. By enabling systems to generate content, understand context, and perform complex tasks, GenAI and LLMs are shaping the future of software development, business automation, and intelligent systems.

---

# Prompting
---
## What is a Prompt?

A **prompt** is the input given to a generative AI model to guide its response.

- It can be a **question**, **instruction**, or **text** that tells the AI what task to perform.
- The **quality, clarity, and structure** of a prompt directly affect the accuracy and usefulness of the AI’s output.

## Example

**Prompt:**  
 - Explain machine learning in simple terms.

➡️ The AI generates a **beginner-friendly explanation** based on the instruction.

---

# Types of Prompts

## 1. Zero-Shot Prompting

Zero-shot prompting means asking the AI to perform a task **without providing any examples**.

### Characteristics
- Uses the model’s pretrained knowledge  
- Simple and quick  
- Output depends heavily on prompt clarity  

### Example Prompt
- Explain what Natural Language Processing is.

### Use Cases
- Simple explanations  
- General questions  
- Broad reasoning tasks  

---

## 2. One-Shot Prompting

One-shot prompting provides **one example** to guide the AI’s response.

### Characteristics
- Improves accuracy compared to zero-shot  
- Helps define output format  
- Easy to implement  

### Example Prompt
- Translate English to French.  
- **Example:** Hello → Bonjour  
- **Now translate:** Thank you

### Use Cases
- Slightly complex tasks  
- Format-sensitive outputs  

---

## 3. Few-Shot Prompting

Few-shot prompting provides **multiple examples** so the AI can identify patterns.

### Characteristics
- Improves consistency and reliability  
- Helps with classification and pattern recognition  
- Uses more tokens  

### Example Prompt
- Classify sentiment:  
- "I love this movie" → Positive  
- "I hate delays" → Negative  
- "The service was okay" → ?

### Use Cases
- Sentiment analysis  
- Classification tasks  
- Structured outputs  

---

## 4. Instruction-Based Prompting

The AI is given **clear and direct instructions** about the task.

### Characteristics
- Easy to control output  
- Reduces ambiguity  
- Works well for structured tasks  

### Example Prompt
- Summarize the following paragraph in 3 bullet points.

### Use Cases
- Summarization  
- Content generation  
- Step-by-step explanations  

---

## 5. Role-Based Prompting

In role-based prompting, the AI is assigned a **specific role or persona**.

### Characteristics
- Produces domain-specific answers  
- Improves explanation quality  
- Useful for expert-level responses  

### Example Prompt
- You are a data scientist. Explain overfitting to a beginner.

### Use Cases
- Teaching concepts  
- Professional explanations  
- Domain-focused tasks  

---

## 6. Chain-of-Thought Prompting

Chain-of-thought prompting encourages the AI to **reason step by step** before giving the final answer.

### Characteristics
- Improves logical reasoning  
- Makes answers more explainable  
- Useful for multi-step problems  

### Example Prompt
 **Solve step by step:**  
- If a shirt costs 500 and has a 20% discount, what is the final price?

### Use Cases
- Math problems  
- Logical reasoning  
- Complex decision-making  

---

# Components of a Good Prompt

A well-structured prompt usually contains the following components:

## 1. Role
Defines who the AI should act as.  
**Example:**  
- You are a data science instructor.

## 2. Task
Clearly states what the AI should do.  
**Example:**  
- Explain overfitting.

## 3. Context
Provides background information.  
**Example:**  
- Explain to a beginner.

## 4. Constraints
Limits length, style, or format.  
**Example:**  
- Limit the answer to 5 bullet points.

## 5. Output Format
Specifies how the output should look.  
**Example:**  
- Answer in bullet points.

---

# Chatbot Creation Using Google Gemini API and Flask

AI chatbots are built using a request–response lifecycle, which defines how user input is processed and returned as an intelligent response.

## Request–Response Lifecycle 

First, the user sends input from a user interface.The backend server receives this request and extracts the message.The prompt is sent to the Large Language Model (LLM) API.The LLM processes the prompt and generates a response.The response is returned to the backend, which sends it back to the user.  
This lifecycle is the foundation of chatbots, assistants, RAG systems, and AI agents.

## Google GenAI (Gemini) API 
- Google GenAI provides access to Gemini models through a Python SDK.  
- These models generate human-like responses and can be integrated into applications using an API key.  
- Core elements include client initialization, model selection, prompt input, and generated output.

## Flask as Backend Server  
- Flask is a lightweight Python web framework used to build APIs.  
- In this system, Flask acts as the backend server that receives user requests, sends prompts to Gemini, and returns responses using JSON.

## Prompting: API vs Chat UI 
- Chat UI is manual and mainly used for testing.  
- API-based prompting is automated, scalable, and suitable for production systems.

## Important API Parameters  
- The model controls capability and speed.  
- The prompt defines the input.  
- Tokens control output length.  
- Temperature controls creativity versus accuracy.

## Security Best Practices  
- API keys should never be hardcoded.  
- They must be stored using environment variables and never exposed in frontend code.

## Use Cases  
- Web-based chatbots, customer support systems, internal AI tools, and backends for RAG and agent systems.
--- 

# Natural Language Processing (NLP)
---
## 1. What is Natural Language Processing (NLP)?

Natural Language Processing (NLP) is a branch of Artificial Intelligence that enables machines to **understand, interpret, and process human language** in a meaningful way.

NLP is widely used in:
- Chatbots and virtual assistants
- Search engines
- Sentiment analysis
- Text summarization
- Language translation
- Question answering systems

---

## 2. Why NLP is Needed

Human language is:
- Unstructured
- Ambiguous
- Context-dependent

Machines require **structured and numerical data**, so NLP acts as a bridge between human language and machine understanding.

---

## 3. NLP Processing Pipeline (Order)

The typical NLP preprocessing flow is:

1. Text Normalization  
2. Sentence Tokenization  
3. Word Tokenization  
4. Stemming  
5. Lemmatization  

---

## 4. Text Normalization

Text normalization converts raw text into a **standard and consistent format**.

### Normalization Includes:
- Converting text to lowercase
- Removing punctuation
- Removing extra spaces

### Example

**Original Text:** My name is Geetha Pranathi

**Normalized Text:** my name is geetha pranathi

```python
import re

text = "My name is Geetha Pranathi"
normalized_text = text.lower()
normalized_text = re.sub(r'[^\w\s]', '', normalized_text)

print(normalized_text)

```
---

## 5. Sentence Tokenization

Sentence tokenization splits a **paragraph into individual sentences.**
### Example
```python
Text:"NLP is powerful. It is used in chatbots. Tokenization is the first step."
```

### NLTK Code

```python

from nltk.tokenize import sent_tokenize
import nltk

nltk.download('punkt')

text = "NLP is powerful. It is used in chatbots. Tokenization is the first step."
sentences = sent_tokenize(text)

print(sentences)

```

### Output

```css

['NLP is powerful.', 'It is used in chatbots.', 'Tokenization is the first step.']

```
---
## 6. Word Tokenization

Word tokenization splits text into **individual words and punctuation.**

**What is a Token?**

A **token** is the **smallest meaningful unit of text** processed by an NLP system.
In word tokenization, each word or punctuation mark becomes a token.

### Example

```python
Sentence: my name is geetha pranathi
Tokens: ["my", "name", "is", "geetha", "pranathi"]
```

### NLTK Code

```python
from nltk.tokenize import word_tokenize

text = "NLP is powerful. It is used in chatbots."
words = word_tokenize(text)

print(words)
```
### Output

```css
['NLP', 'is', 'powerful', '.', 'It', 'is', 'used', 'in', 'chatbots', '.']
```
---
## 7. Stemming (Porter Stemmer)

Stemming reduces words to their **root form** by removing suffixes using rule-based logic.

It is **fast**, but may not always produce meaningful words.

### Example

```arduino
running → run
runs → run
runner → runner
easily → easili
```

### NLTK Code

```python
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()
words = ["running", "runs", "runner", "easily"]

stemmed_words = [stemmer.stem(word) for word in words]
print(stemmed_words)
```
### Output

```css
['run', 'run', 'runner', 'easili']
```
---
## 8. Lemmatization (WordNet Lemmatizer)
Lemmatization converts words to their **dictionary base form (lemma)** using linguistic knowledge.

It produces **real and meaningful words**, making it more accurate than stemming.

### Example
```ngink
cars → car
feet → foot
better → better
```
### NLTK Code
```python
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download('wordnet')
nltk.download('omw-1.4')

lemmatizer = WordNetLemmatizer()
words = ["running", "cars", "better", "feet"]

lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
print(lemmatized_words)
```
### Output
```css
['running', 'car', 'better', 'foot']
```
---
## 9. Stemming vs Lemmatization

| Feature              | Stemming   | Lemmatization |
| -------------------- | ---------- | ------------- |
| Speed                | Faster     | Slower        |
| Accuracy             | Lower      | Higher        |
| Context aware        | No         | Yes           |
| Produces valid words | Not always | Yes           |
---
## 10. Summary
- **Normalization** standardizes text
- **Sentence Tokenization** splits text into sentences
- **Word Tokenization** splits sentences into tokens
- **Stemming removes** suffixes quickly
- **Lemmatization** produces meaningful base words
---
# ATS (Applicant Tracking System) using Gemini GenAI

---

## Overview

An **Applicant Tracking System (ATS)** is a software system used by companies to automatically analyze resumes and match them with job descriptions.  
It helps recruiters shortlist candidates efficiently by reducing manual effort.

In this project, a **GenAI-powered ATS** is designed using **Google Gemini** to intelligently understand and compare resumes and job descriptions.

### Key Capabilities
- Accept resume in PDF format
- Extract text from resume
- Understand resume content using LLM reasoning
- Parse job description requirements
- Compare resume and job description
- Generate AI-based match score and feedback

This project demonstrates **real-world Generative AI usage in HR and recruitment systems**.

---

## Problem Statement

Manual resume screening suffers from multiple limitations:

- Time-consuming
- Prone to human bias
- Error-prone
- Not scalable for large hiring volumes

The goal of this ATS system is to **automate resume evaluation using GenAI-powered reasoning**, ensuring faster, fairer, and scalable hiring.

---

## System Flow

1. User uploads a resume in PDF format
2. Backend extracts raw text from the PDF
3. Resume text is sent to Gemini GenAI for understanding
4. Job description text is also parsed using Gemini
5. ATS logic compares both outputs
6. AI-generated evaluation is returned as structured JSON

---

## Technology Stack

- **Python** – Backend logic
- **Flask** – API framework
- **PyPDF2** – PDF text extraction
- **Google Gemini GenAI** – LLM reasoning
- **JSON** – Structured output format

---

## Resume Parsing using GenAI

Resume parsing means converting **unstructured resume text** into structured information such as:

- Skills
- Experience
- Education
- Tools & technologies

### Why LLM-Based Parsing?

Traditional parsing techniques (regex, keyword matching) are limited and rigid.  
LLM-based parsing offers:

- Context-aware understanding
- Better handling of varied resume formats
- Industry-aligned interpretation
- Higher accuracy

---

## PDF Text Extraction

LLMs cannot directly process PDF files. Therefore:

- Resume is uploaded as a PDF
- PDF parsing library extracts raw text
- Extracted text is passed to Gemini GenAI

This step bridges the gap between **document formats** and **LLM reasoning**.

---

## Job Description Parsing

The job description is analyzed to extract:

- Required skills
- Responsibilities
- Preferred qualifications
- Role expectations

Parsing both resume and job description ensures **structured and fair comparison**.

---

## ATS Matching Logic

The ATS system compares:

- Parsed resume details
- Parsed job description requirements

Gemini GenAI evaluates:

- Skill overlap
- Experience relevance
- Missing competencies

### AI-Generated Output Includes:
- Match percentage (0–100)
- Matching skills
- Missing skills
- Candidate strengths
- Improvement suggestions

---

## API Design (Conceptual)

The system exposes a single API endpoint:

POST /analyze

The endpoint:

- Accepts resume PDF via multipart/form-data
- Accepts job description as text
- Returns structured ATS evaluation

---

## Sample Input

**Resume:** resume.pdf  
**Job Description:** Backend developer with Python, NLP, and API experience

---

## Sample Output

- Match Percentage: 85–90%
- Matching Skills: Python, NLP, APIs
- Missing Skills: Cloud deployment
- Suggestions: Add cloud-based project experience

---

## Key Concepts Covered

- File upload handling
- PDF parsing
- LLM-based document understanding
- Prompt engineering
- Real-world GenAI workflows
- ATS system design

---

## Learning Outcomes

By the end of this module, learners will be able to:

- Understand how ATS systems work in industry
- Build GenAI-powered document analysis pipelines
- Parse unstructured documents using LLMs
- Design production-style GenAI APIs
- Apply GenAI to real-world business problems

---

## Optional Enhancements

- Convert outputs into strict JSON schema
- Add embeddings for semantic matching
- Build a frontend UI for resume upload
- Store ATS results in a database
- Extend system into a RAG-based ATS

---

## Summary

This ATS system combines **PDF parsing**, **LLM reasoning**, and **backend design** to simulate a real-world recruitment application.  
It demonstrates how Generative AI can transform traditional HR workflows into intelligent, scalable systems.

---
# LLM Evaluation – Theory & Conceptual Practical Examples

---

## 1️. Introduction to LLM Evaluation

**LLM Evaluation** is the process of measuring how well a **Large Language Model (LLM)** performs on a given task.

Unlike traditional machine learning models, LLM outputs are **open-ended**, so evaluation focuses on **quality**, not just correctness.

### Evaluation Focus Areas
- Correctness
- Clarity
- Relevance
- Instruction following
- Safety and bias

---

## 2️. Why LLM Evaluation is Important

Large Language Models may:

- Hallucinate incorrect facts
- Produce fluent but misleading responses
- Give inconsistent answers to similar prompts
- Fail silently in real-world applications

Therefore, LLM evaluation is:
- Continuous
- Multi-dimensional
- Essential for production systems

---

## 3️. Traditional ML vs LLM Evaluation

| Traditional ML | LLM Evaluation |
|---------------|----------------|
| Fixed outputs | Open-ended text |
| Single correct answer | Multiple valid answers |
| Accuracy-based | Quality-based |
| Fully automated | Requires judgment |

---

## 4️. Types of LLM Evaluation

### 4.1 Automatic (Metric-Based) Evaluation

Used when a **reference answer** exists.

#### Common Metrics
- Accuracy
- Precision / Recall / F1
- Exact Match
- BLEU / ROUGE

#### Suitable For
- Text classification
- Named Entity Recognition
- Question answering with known answers

 Limited for creative or reasoning tasks.

---

### 4.2 Human Evaluation

Human evaluators score responses based on:

- Correctness
- Clarity
- Relevance
- Tone

#### Limitations
- Expensive
- Time-consuming
- Subjective

---

### 4.3 LLM-as-a-Judge (Industry Standard)

A powerful LLM evaluates another LLM’s output using a **rubric-based prompt**.

#### Advantages
- Scalable
- Cost-effective
- Suitable for reasoning tasks

#### Limitation
- Potential judge bias

---

## 5️. Evaluation Dimensions

Common dimensions used to evaluate LLM outputs:

- Correctness
- Instruction following
- Clarity & coherence
- Factual accuracy
- Hallucination rate
- Safety & toxicity
- Bias and fairness

---

## 6️ Offline vs Online Evaluation

### Offline Evaluation
- Conducted before deployment
- Uses fixed test datasets
- Used for benchmarking

### Online Evaluation
- Conducted after deployment
- Uses user feedback and logs
- Helps monitor real-world performance

---

## 7️. Conceptual Practical Examples

### 🔹 Practical Example 1: LLM-as-a-Judge

**Scenario:**  
A model is asked:  
> *"What is tokenization in NLP?"*

**Model Response:**  
> "Tokenization is the process of breaking text into smaller units called tokens."

**Evaluation Approach:**  
Another LLM evaluates this answer based on:
- Correctness
- Clarity
- Relevance

**Result:**  
High-quality response with correct and clear explanation.

---

### 🔹 Practical Example 2: Metric-Based Evaluation

**Reference Answer:**  
> "Tokenization splits text into words or sentences."

**Model Output:**  
> "Tokenization breaks text into tokens."

**Evaluation:**  
- Exact Match: ❌  
- Semantic Match: ✅  

This shows metric-based evaluation works well when reference answers exist.

---

### 🔹 Practical Example 3: Human Evaluation

A human evaluator scores the response:

| Criteria | Score (1–5) |
|--------|-------------|
| Correctness | 4 |
| Clarity | 5 |
| Instruction Following | 5 |
| Safety | 5 |

This approach is useful but not scalable.

---

## 8️. Popular LLM Evaluation Tools

| Tool | Purpose |
|-----|--------|
| OpenEvals | LLM-as-a-Judge evaluation |
| DeepEval | Test-driven LLM evaluation |
| lm-evaluation-harness | Benchmarking LLMs |
| Hugging Face Evaluate | BLEU, ROUGE, BERTScore |
| LangSmith / Langfuse | Observability + evaluation |

---

## 9️.. Real-World Applications

- Chatbot response quality monitoring
- RAG system validation
- AI agent reliability checking
- Safety and hallucination detection
- Model benchmarking

---

## 10. Summary

LLM Evaluation is a critical component of modern Generative AI systems.  
By combining:
- Metric-based evaluation
- Human judgment
- LLM-as-a-Judge

we can build **reliable, safe, and production-ready GenAI applications**.

---
# LLM Inference
## 1️. What is LLM Inference?

LLM Inference is the process of using a trained Large Language Model (LLM) to generate outputs such as text, answers, summaries, or code from a given input prompt.

Training → learning model weights ❌  
Inference → using learned weights to generate output ✅  

---

## 2️. Inference Pipeline (High Level)

Input Text → Tokenizer → Model → Generated Tokens → Decoder → Output Text

### Components

- **Tokenizer:** Converts text into numerical tokens that the model can understand.

- **Model:** Predicts the next most probable tokens based on input.

- **Decoder:** Converts generated tokens back into readable text.

---

## 3️. Popular Packages for LLM Inference

| Package        | Use Case |
|----------------|----------|
| transformers   | Most commonly used (Hugging Face) |
| torch          | Model execution backend |
| accelerate     | Multi-GPU and performance optimization |
| vllm           | High-speed, production-level inference |

---

## 4️. Installing Required Packages

```bash
pip install transformers torch
```
---

## 5. Simple LLM Inference (Step‑by‑Step)

**Example Model**
We use GPT‑2 because:
- Small size
- Works on CPU
- No authentication required
---
##  Code: Basic LLM Inference
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load tokenizer and model
model_name = "gpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Input prompt
prompt = "Artificial Intelligence is"

# Tokenize input
inputs = tokenizer(prompt, return_tensors="pt")

# Generate output
outputs = model.generate(
    **inputs,
    max_new_tokens=50,
    temperature=0.7,
    do_sample=True
)

# Decode tokens to text
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)
```
---
## 6. Explanation of Important Parameters
| Parameter      | Meaning                                          |
| -------------- | ------------------------------------------------ |
| max_new_tokens | Maximum number of tokens to generate             |
| temperature    | Controls creativity (0.2 = safe, 1.0 = creative) |
| do_sample      | Enables randomness in output                     |

---
## 7. Simplified Inference using Pipeline API
Best for quick **demos, teaching, and seminars.**
```python
from transformers import pipeline

generator = pipeline("text-generation", model="gpt2")

output = generator(
    "Artificial Intelligence is",
    max_new_tokens=50
)

print(output[0]["generated_text"])
```
---
## 8. CPU vs GPU Inference
**CPU Inference (Default)**
```python
device = -1
```

**GPU Inference (CUDA)**
```python
device = 0

generator = pipeline(
    "text-generation",
    model="gpt2",
    device=0
)
```
GPU requires:
- NVIDIA GPU
- CUDA installed
- Compatible PyTorch version
---

## 9. Real‑World Use Cases of LLM Inference
- Chatbots (ChatGPT-like systems)
- Resume parsing (ATS systems)
- Code generation
- Question answering systems
- RAG (Retrieval Augmented Generation)
- AI Agents
- Content generation

---
## 10. Key Takeaways
- Inference ≠ Training
- Tokenizer + Model + Decoder = LLM Inference
- Transformers is the industry standard
- Pipeline API is best for beginners and demos
- GPT-2 is ideal for classroom and seminar usage
---

# Transformers

Transformers are the backbone of today’s advanced AI systems such as:
- Chatbots
- Machine translation systems
- Large Language Models (LLMs)

They have revolutionized Natural Language Processing by enabling efficient handling of long-range dependencies using attention mechanisms.

---

## Evolution of Neural Networks Over Time
---

### 1. Artificial Neural Networks (ANN)

Artificial Neural Networks are the earliest form of neural network models, inspired by the working of the human brain.

---

![WhatsApp Image 2026-02-09 at 12 58 36 PM](https://github.com/user-attachments/assets/8bdefd10-5272-4432-b5de-9b8be9553501)

---

#### Key Characteristics

- Designed to learn and process information similar to the human brain
- Consists of:
  - Input layer
  - One or more hidden layers
  - Output layer
- Each connection has:
  - Weights
  - Biases (initially assigned randomly)
- Uses **activation functions** to decide how much information should be passed to the next layer
- Training involves:
  - **Forward Propagation**: data flows from input to output
  - **Backward Propagation**: errors are propagated back to update weights
- Optimizers are used to minimize loss by updating weights and biases
- Input and output data remain constant during training; only weights and biases are updated
---

#### Limitation of ANN

- Cannot handle sequential data
- Does not retain past information (no memory mechanism)
- Unsuitable for tasks such as:
  - Language modeling
  - Time-series prediction
  - Speech recognition

---
## Recurrent Neural Networks (RNN)

## Overview
Recurrent Neural Networks (RNNs) are a class of neural networks designed to process **sequential data**.  
Unlike traditional neural networks, RNNs have a **memory component** that allows them to consider previous inputs while processing new inputs.

---
![WhatsApp Image 2026-02-09 at 1 30 51 PM](https://github.com/user-attachments/assets/6a0cc237-49aa-4c94-99f4-5eb6690a777e)

---

## Example
**Sentence:**  
> I love AI  

Each State depens on the previous step 

I -> love-> AI
---
## Core Idea
- Each output depends on the **current input** and the **previous hidden state**.
- This structure allows RNNs to capture **temporal dependencies** in sequential data.

---

## Applications of RNNs
- Language modeling
- Sentiment analysis
- Speech recognition
- Music generation
- Time series forecasting

---

## Problems with RNNs

### Vanishing Gradient Problem

- Difficult to learn **long-term dependencies**
- Gradients become very small during backpropagation
---

### Exploding Gradient Problem

- Gradients grow too large
- Makes training unstable
---

### Computational Limitation

- Sequential processing makes RNNs **slow to train**

---

## Solution to RNN Limitations
- Advanced architectures like **LSTM (Long Short-Term Memory)** and **GRU (Gated Recurrent Unit)** were introduced
- These models handle **long-term dependencies** more effectively

---
## LSTM (Long Short-Term Memory)

- Designed to overcome the **vanishing gradient problem**
- Learns **long-term dependencies**
- Uses a **cell state (Cₜ)** and **hidden state (hₜ)**
- **Three gates:** Forget, Input, Output 
.Forget gate:- Dcises what to forget from the previous cell 
- ⁠Input gate:-decides what new info I have to store in the cell 
- ⁠Output gate:-it is going to decide what to output as a new hidden state
---
## GRU (Gated Recurrent Unit)

- Simplified version of LSTM
- ⁠GRUs have fewer parameters than LSTMs,leading to simple architecture and optimisation.
- **Two gates:** Update, Reset 
- ⁠update gate=Forget+input
---
## Usage
- **LSTM:** Better for complex long-term dependencies  
- **GRU:** Better for speed and simplicity
---

# Transformers

Transformers are the backbone of modern AI models such as GPT, BERT, Gemini, and LLaMA.  
They replaced older sequence models by using **attention instead of recurrence**, enabling efficient understanding of **context, meaning, and relationships** in text.

---

## PART 1: THEORY

## Why Do We Need Transformers?

### Limitations of Earlier Models (RNNs & LSTMs)
- Process text **word by word**
- Slow when handling **long sequences**
- Struggle to capture **long-range dependencies**

### How Transformers Solve These Issues
- Process all words **at the same time**
- Use **attention** to focus on important words
- Scale efficiently on **GPUs**
- Handle long sequences more effectively

---

## What is Attention?

Attention helps the model decide:
> **Which words should I focus on while understanding this word?**

### Example
Sentence:Attention allows the model to understand that:
- **"it" refers to "animal"**, not "road"

---

## Self-Attention (Simple Explanation)

- Each word looks at **every other word** in the sentence
- Decides how important other words are for its understanding

Each word generates:
- **Query (Q)** – What am I looking for?
- **Key (K)** – What information do I contain?
- **Value (V)** – What information do I provide?

The model compares **Query with Key** to determine which **Values** to focus on.

---

## Multi-Head Attention

- Uses **multiple attention heads** instead of a single one
- Each head learns different relationships such as:
  - Grammar
  - Semantic meaning
  - Subject–object relationships
- This improves the model’s overall understanding

---

## Transformer Architecture (High Level)

A Transformer block consists of:
- Embedding Layer
- Positional Encoding
- Multi-Head Self-Attention
- Feed Forward Neural Network
- Residual Connections
- Layer Normalization

Multiple Transformer blocks are **stacked together** to form the full model.

---

## Why Positional Encoding?

- Transformers process words **in parallel**
- They do not naturally understand **word order**
- Positional Encoding adds sequence information to embeddings
- Helps the model learn **position and order of words**

---
![WhatsApp Image 2026-02-09 at 1 49 34 PM](https://github.com/user-attachments/assets/9c2b177d-7189-43c8-a5d6-da01c3802dbc)

---
## Transformer Example (Language Translation)

### Input

- Sentence: `The big red dog`

### Tokenization

- Tokens: `["The", "big", "red", "dog"]`

### Embedding + Positional Encoding

- Convert tokens into vectors
- Add position information

### Encoder

- Applies self-attention (Q, K, V)
- Generates contextual representations

### Decoder

- Receives encoder output
- Uses masked self-attention

### Output Layer

- Linear layer + Softmax
- Generates translated words

### Output

- French: `Le grand chien rouge`
- Hindi: `बड़ा लाल कुत्ता`
---
## Transformer Model Types

### Encoder-Only Models

Encoder-only models focus on understanding input text by learning rich contextual representations.  
They are mainly used for tasks like text classification, sentiment analysis, and semantic understanding.  
**Example:** BERT

---

### Decoder-Only Models

Decoder-only models generate text one token at a time using masked self-attention.  
They are designed for text generation tasks such as chatbots and language modeling.  
**Example:** GPT

---

### Encoder–Decoder Models

Encoder–Decoder models combine both components to handle input–output sequence tasks.  
The encoder processes the input sequence, and the decoder generates the output sequence.  
They are commonly used for machine translation and summarization.

---




