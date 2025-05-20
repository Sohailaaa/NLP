# 💡 NLP Projects Portfolio

A collection of advanced Natural Language Processing (NLP) projects focused on Arabic text analysis, extractive question answering, and medical domain QA systems. Each project is crafted with a tailored pipeline, integrating state-of-the-art tools, models, and visualizations.

---

## 📘 Project 1: Advanced Arabic Text Processing and Analysis

<details>
<summary>Click to expand</summary>

### 📝 Summary
A comprehensive Arabic NLP pipeline with specialized tools for preprocessing, stopword filtering, NER, and topic modeling using TF-IDF and LDA. Designed for dialect-aware Arabic language applications.

### 🔧 Features

#### 📦 Library Setup
- `farasapy` for Arabic segmentation, stemming, tagging, and NER
- `bertopic` for topic modeling
- `arabic-stopwords` and custom dialectal stopwords
- Visualization tools (matplotlib, seaborn, pyvis)

#### 🧹 Text Processing Functions
- `segment()`: Tokenize using `FarasaSegmenter`
- `stem()`: Apply `FarasaStemmer`
- `tag()`: POS tagging via `FarasaPOSTagger`
- `recognize()`: Named entity recognition
- `removeStopWords()`: Eliminate common/EGY stopwords
- `removeNonArabic()`: Filter non-Arabic content
- `process_text()`: Full pipeline for cleaned, rich-text input

#### 🚫 Stopwords Management
- Standard Arabic stopwords
- Egyptian dialect expressions
- Fillers, pronouns, and particles

#### 🧪 Data Processing Pipeline
- YouTube text extraction
- File-wise named entity extraction
- Batch processing of raw Arabic data

#### 📊 Advanced Analysis Components
- **TF-IDF Scoring**: Weighted term importance with visual charts
- **Vector Space Model**: Document retrieval using cosine similarity
- **Topic Modeling**:
  - Standard LDA
  - NER-only & NER-boosted LDA
  - Combined frequent word & NER topic modeling
- **Visualizations**:
  - Word relation graphs
  - TF-IDF bar charts
  - Topic distribution in Arabic (with proper font support)

</details>

---

## 📗 Project 2: Extractive Question Answering System

<details>
<summary>Click to expand</summary>

### 📝 Summary
A SQuAD-style QA pipeline using a GRU-based neural network with GloVe embeddings and SpaCy-enhanced heuristics for fine-grained answer extraction.

### 🔧 Features

#### 🧼 Tokenization & Preprocessing
- BERT tokenizer for question/context
- Character-token span mapping
- Answer span alignment with verification

#### 📊 Dataset Management
- SQuAD format parsing and validation
- Visual stats: context length, question types
- Vocabulary building

#### 🧠 Model Architecture
- Bidirectional GRU model (ExtractiveQARNN)
- Pre-trained GloVe (300d) embeddings
- Independent start/end position heads

#### 📍 Advanced Answer Extraction
- Softmax span scoring
- SpaCy-based linguistic heuristics:
  - "Who" → PERSON entities
  - "How much/many" → NUM entities
  - "When" → DATE/TIME detection
  - "What happened" → Verb phrase identification

#### 🏋️ Training Pipeline
- Metrics: Loss, Exact Match, F1 score
- Early stopping & validation monitoring
- Training visualization

#### 🔎 Inference System
- Token-to-char mapping
- Visual answer comparison (true vs. predicted)

</details>

---

## 📕 Project 3: Medical Question Answering System chatbot

<details>
<summary>Click to expand</summary>

### 📝 Summary
A medical QA pipeline using transformer-based models fine-tuned on the MedQuad dataset, with integration of RAG and dynamic document retrieval for better coverage and factual grounding.

### 🧪 Experiment 1: Qwen2-0.5B Fine-tuning
- Model: Qwen2-0.5B + LoRA + 4-bit quantization
- Dataset: MedQuad (custom format for CLM)
- Training: 2 epochs, BLEU/ROUGE evaluation
- Visualization: BLEU & ROUGE score tracking
- Sample prediction testing

#### 🤖 Chatbot Interface (Qwen2)
- LangChain-based
- LoRA model loading
- Context buffer memory
- Live medical QA responses

---

### 🧪 Experiment 2: Flan-T5-Small Fine-tuning
- Model: Flan-T5-Small
- Dataset: MedQuad
- Includes question type classification
- Selective layer freezing for performance
- Training metrics & model archiving

#### 🤖 Chatbot Interface (Flan-T5)
- Beam search for answer generation
- Heuristic classification for question types
- Persistent conversation context

---

### 🔍 Retrieval-Augmented Generation (RAG)
- FAISS vector database of MedQuad
- Wikipedia retrieval for OOD questions
- Semantically matches user input
- Sources shown for attribution and reliability

</details>

---

## 📞 Contact

If you have any questions or want to collaborate, feel free to reach out via [LinkedIn](https://www.linkedin.com/in/sohaila-hakeem-819801221/)

---
