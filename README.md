# LangChain Models Collection

A comprehensive collection of LangChain model implementations demonstrating various AI/ML capabilities including Large Language Models (LLMs), Chat Models, and Embedding Models.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Setup](#setup)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)

## 🚀 Overview

This repository contains practical examples and implementations using LangChain framework with various AI models. It's designed to help developers understand and implement different types of AI models for various use cases including text generation, chat interactions, and document similarity.

## ✨ Features

- **Large Language Models (LLMs)**: OpenAI GPT models for text generation
- **Chat Models**: Interactive chat implementations with multiple providers
  - OpenAI GPT-4
  - Anthropic Claude
  - Google Gemini
  - Hugging Face (API and Local)
- **Embedding Models**: Text embedding and similarity search
  - OpenAI embeddings
  - Hugging Face embeddings
  - Document similarity analysis

## 🛠 Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd langchain-models-master
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## ⚙️ Setup

1. Create a `.env` file in the root directory with your API keys:
```env
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
HUGGINGFACE_API_KEY=your_huggingface_api_key_here
```

2. Make sure you have Python 3.8+ installed

## 📁 Project Structure

```
langchain-models-master/
├── 1.LLMs/                    # Large Language Model examples
│   └── 1_llm_demo.py         # Basic LLM implementation
├── 2.ChatModels/              # Chat model implementations
│   ├── 1_chatmodel_openai.py # OpenAI chat model
│   ├── 2_chatmodel_anthropic.py # Anthropic Claude model
│   ├── 3_chatmodel_google.py # Google Gemini model
│   ├── 4_chatmodel_hf_api.py # Hugging Face API model
│   └── 5_chatmodel_hf_local.py # Local Hugging Face model
├── 3.EmbeddingModels/         # Embedding model examples
│   ├── 1_embedding_openai_query.py # OpenAI query embedding
│   ├── 2_embedding_openai_docs.py # OpenAI document embedding
│   ├── 3_embedding_hf_local.py # Local Hugging Face embedding
│   └── 4_document_similarity.py # Document similarity analysis
├── requirements.txt           # Python dependencies
├── test.py                   # Basic LangChain version test
└── README.md                 # This file
```

## 🎯 Usage

### Running Individual Examples

Each script can be run independently. For example:

```bash
# Test LangChain installation
python test.py

# Run LLM demo
python 1.LLMs/1_llm_demo.py

# Run OpenAI chat model
python 2.ChatModels/1_chatmodel_openai.py

# Run document similarity analysis
python 3.EmbeddingModels/4_document_similarity.py
```

### Example: Document Similarity

The document similarity example demonstrates how to:
1. Create embeddings for multiple documents
2. Generate an embedding for a query
3. Calculate cosine similarity scores
4. Find the most relevant document

```python
# Example from 4_document_similarity.py
documents = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting...",
    "MS Dhoni is a former Indian captain famous for his calm demeanor...",
    # ... more documents
]

query = 'tell me about bumrah'
# Returns the most similar document with similarity score
```

## 📚 Examples

### 1. Basic LLM Usage
```python
from langchain_openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
llm = OpenAI(model='gpt-3.5-turbo-instruct')
result = llm.invoke("What is the capital of India")
print(result)
```

### 2. Chat Model with Parameters
```python
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model='gpt-4', temperature=1.5, max_completion_tokens=10)
result = model.invoke("Write a 5 line poem on cricket")
print(result.content)
```

### 3. Embedding and Similarity
```python
from langchain_openai import OpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

embedding = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=300)
doc_embeddings = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)
scores = cosine_similarity([query_embedding], doc_embeddings)[0]
```

## 🔧 Dependencies

- **LangChain Core**: `langchain`, `langchain-core`
- **OpenAI Integration**: `langchain-openai`, `openai`
- **Anthropic Integration**: `langchain-anthropic`
- **Google Integration**: `langchain-google-genai`, `google-generativeai`
- **Hugging Face Integration**: `langchain-huggingface`, `transformers`, `huggingface-hub`
- **Utilities**: `python-dotenv`, `numpy`, `scikit-learn`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 📞 Support

If you encounter any issues or have questions, please open an issue in the repository.

---

**Note**: Make sure to keep your API keys secure and never commit them to version control. Use the `.env` file for local development and environment variables for production deployments.
