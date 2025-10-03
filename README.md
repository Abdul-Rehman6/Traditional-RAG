# RAG System with LangChain & LangGraph

A sophisticated Retrieval-Augmented Generation (RAG) system built with LangChain and LangGraph for intelligent document querying and conversation.

## Overview

This project implements a traditional RAG pipeline that enables users to query research papers and video transcripts using natural language. The system retrieves relevant context from documents and generates accurate, context-aware responses using OpenAI's GPT models.

## Features

- **Document Processing**: Load and process PDF research papers and video transcripts
- **Intelligent Chunking**: Automatically splits documents into optimal chunks for retrieval
- **Vector Search**: FAISS-based semantic search for finding relevant context
- **Context-Aware Responses**: GPT-4 powered answers based on retrieved information
- **Chain Architecture**: Modular pipeline using LangChain's LCEL (LangChain Expression Language)
- **YouTube Transcript Support**: Extract and query video transcripts

## Research Papers Included

The project includes several influential papers on transformer architectures and language models:

- **Attention Is All You Need** - Original transformer architecture
- **BERT** - Bidirectional Encoder Representations from Transformers
- **RoBERTa** - Robustly Optimized BERT Approach
- **ALBERT** - A Lite BERT for Self-supervised Learning
- **DistilBERT** - Distilled version of BERT
- **RAG** - Retrieval-Augmented Generation paper

## Architecture

The RAG pipeline consists of the following components:

1. **Document Loader**: Loads PDFs and extracts text content
2. **Text Splitter**: Chunks documents with overlap for context preservation
3. **Embeddings**: Converts text chunks to vector representations
4. **Vector Store**: FAISS index for efficient similarity search
5. **Retriever**: Fetches top-k relevant documents
6. **LLM Chain**: Generates answers using retrieved context

## Installation

```bash
pip install langchain langchain-community langchain-openai langgraph
pip install faiss-cpu openai python-dotenv pydantic
pip install youtube-transcript-api pymupdf
```

## Configuration

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

## Usage

### Basic RAG Query

```python
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate

# Initialize embeddings and vector store
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = FAISS.from_documents(chunks, embeddings)

# Create retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

# Query the system
response = main_chain.invoke("Explain the transformer architecture")
```

### Chain Architecture

The system uses a parallel processing chain:

```python
parallel_chain = RunnableParallel({
    'context': retriever | RunnableLambda(format_docs),
    'question': RunnablePassthrough()
})

main_chain = parallel_chain | prompt | llm | parser
```

## Components

### Document Processing

- **RecursiveCharacterTextSplitter**: Splits documents into 1000-character chunks with 200-character overlap
- **FAISS Vector Store**: Fast similarity search for document retrieval
- **OpenAI Embeddings**: text-embedding-3-small model for semantic representations

### Language Model

- **Model**: GPT-4o-mini
- **Temperature**: 0.2 (for more deterministic outputs)
- **Context Window**: Handles multiple document chunks

### Prompt Template

The system uses a carefully crafted prompt that:
- Restricts answers to provided context
- Admits when information is insufficient
- Maintains factual accuracy

## Key Features

### Semantic Search

The retriever fetches the 4 most relevant document chunks using cosine similarity in the vector space.

### Context Formatting

Retrieved documents are concatenated and formatted for optimal LLM processing.

### Output Parsing

Responses are parsed into clean string format for easy consumption.

## Project Structure

```
.
├── 1.Traditional_RAG_LG.ipynb    # Main notebook implementation
├── ALBERT.pdf                     # Research paper
├── BERT.pdf                       # Research paper
├── RoBerta.pdf                    # Research paper
├── DistilBERT.pdf                 # Research paper
├── Attentoion_is_all_you_need.pdf # Research paper
├── RAG.pdf                        # Research paper
├── .env                           # Environment variables
└── README.md                      # This file
```

## Technical Stack

- **LangChain**: Framework for LLM applications
- **LangGraph**: Stateful orchestration for LLM workflows
- **FAISS**: Vector similarity search
- **OpenAI**: GPT-4 and embeddings
- **PyMuPDF**: PDF processing
- **YouTube Transcript API**: Video transcript extraction

## Limitations

- Requires OpenAI API key
- YouTube transcript extraction may be blocked by IP restrictions
- Vector store is in-memory (not persistent by default)
- Context limited to top-4 retrieved chunks

## Future Enhancements

- Persistent vector store with Supabase pgvector
- Multi-turn conversation support
- Query refinement and expansion
- Advanced retrieval strategies (hybrid search, re-ranking)
- Support for additional document formats
- Web-based user interface

## License

This project is for educational and research purposes.

## Acknowledgments

Built with the powerful LangChain and LangGraph frameworks, leveraging state-of-the-art language models and retrieval techniques.
