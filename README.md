# AI-Powered Bug RCA Assistant

## Overview

The **AI-powered Bug RCA (Root Cause Analysis) Assistant** is designed to accelerate the root cause analysis process for newly reported software issues. By leveraging historical Jira bug data and semantic search, this system empowers support engineers and solution teams to identify and resolve issues faster.

---

## Features

- **Fast RCA Retrieval**  
  Find similar historical bugs using natural language and retrieve RCA/resolution metadata.

- **AI-Powered Matching**  
  Uses sentence-transformer models and FAISS vector search for high-accuracy similarity detection.

- **Continuous Learning**  
  Enhances the knowledge base over time by ingesting newly resolved issues.

- **Fallback RCA Capture**  
  Supports manual RCA input when no match is found, enabling future match potential.

---

## System Architecture

### 1. Data Ingestion Layer
- Connects to Jira via REST APIs
- Extracts relevant fields: `summary`, `RCA`, `resolution`, `comments`, etc.
- Preprocesses and cleans data for semantic embedding

### 2. Embedding Layer
- Generates dense vector representations using **sentence-transformers**
- Stores embeddings and metadata in a **FAISS** vector database

### 3. Query Interface
- Accepts issue prompts via a UI or API endpoint
- Performs semantic similarity search against vector database

### 4. Response Engine
- Retrieves the most relevant historical issues and associated RCA/resolution
- Optionally integrates a **RAG (Retrieval-Augmented Generation)** pipeline for enhanced explanations

### 5. Feedback Loop
- Accepts feedback or manual RCAs when no match is found
- Automatically updates the semantic index and vector store

---

## Benefits

- **Reduces MTTR (Mean Time to Resolution)**  
  Eliminates repetitive investigations by leveraging previous solutions.

- **Promotes RCA Reusability**  
  Encourages the reuse of validated root causes and resolutions.

- **Empowers Engineering Teams**  
  Equips support engineers with contextual insights for faster and more confident debugging.

---

## Getting Started

### Prerequisites
- Python 3.8+
- Access to Jira REST API
- Sentence-transformers (`sentence-transformers` package)
- FAISS (`faiss-cpu` or `faiss-gpu`)
- Vector store (e.g., FAISS or other compatible DB)

### Installation

```bash
git clone https://github.com/your-org/bug-rca-assistant.git
cd bug-rca-assistant
pip install -r requirements.txt
