# Agentic AI Playground

A modular, container-based environment for experimenting with multi-agent orchestration, A2A communication, MCP, and local LLMs.

This repository provides a fully containerized playground for building and testing AI agent orchestration architectures, including:
	•	Front-end UI (Streamlit)
	•	Chat server (FastAPI)
	•	Orchestrator layer (Semantic Kernel, LangGraph, custom loops)
	•	Local LLM (Ollama)
	•	Vector database (Qdrant)

The goal is to explore agent routing, memory, function-calling, A2A, MCP, and multi-layer orchestration in a reproducible local environment.


## Architecture Overview

```mermaid
flowchart LR
    subgraph FrontendLayer["Frontend Layer"]
        A["Streamlit App\nPort 8501"]
    end

    subgraph ChatLayer["Chat Server Layer"]
        B["FastAPI Chat Server\nPort 8000"]
    end

    subgraph OrchestratorLayer["Orchestrator Layer"]
        C["Python Orchestrator\nSK / LangGraph / Custom\nPort 8100"]
    end

    subgraph LLMLayer["LLM Layer"]
        D["Ollama Server\nPort 11434"]
    end

    subgraph VectorDBLayer["Vector DB Layer"]
        E["Qdrant\nPort 6333"]
    end

    A --> B
    B --> C
    C --> D
    C --> E

    style A fill:#E3F2FD,stroke:#64B5F6
    style B fill:#E8F5E9,stroke:#81C784
    style C fill:#FFF3E0,stroke:#FFB74D
    style D fill:#F3E5F5,stroke:#BA68C8
    style E fill:#FBE9E7,stroke:#FF8A65
```


## Getting Started
### 1. Clone the repo
```
git clone https://github.com/<your-username>/agenticai-playground.git
cd agenticai-playground
```

### 2. Build containers
```
docker compose build
```

### 3. Start services
```
docker compose up -d
```

### 4. Install an LLM model via Ollama
```
docker exec -it ai_ollama bash
ollama pull llama3.1
exit
```

### 5. Access the playground
•	Streamlit app:
    http://localhost:8501
•	Chat server health:
    http://localhost:8000/health
•	Orchestrator health:
    http://localhost:8100/health