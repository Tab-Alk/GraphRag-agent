# GraphRAG Agent

## Overview
GraphRAG Agent is a powerful tool that combines the structured querying capabilities of Neo4j with the semantic understanding of large language models. It creates a knowledge graph of technical skills, enabling intelligent skill recommendations and analysis.

## Key Features

- **Graph-Powered Search**: Leverage Neo4j's graph database for complex relationship queries
- **Semantic Understanding**: Utilizes Google's Generative AI for skill embeddings and semantic search
- **Interactive Exploration**: Jupyter notebooks for hands-on analysis and visualization
- **RAG Implementation**: Combines retrieval-augmented generation with graph-based knowledge

## The GraphRAG Concept

GraphRAG enhances traditional RAG (Retrieval-Augmented Generation) by:
1. Storing data in a graph structure that captures rich relationships
2. Using vector embeddings for semantic search
3. Combining graph traversal with vector similarity for more contextual responses
4. Enabling multi-hop reasoning across connected concepts

## Project Structure

```
.
├── module_01_graph_basics.ipynb    # Introduction to Neo4j and graph queries
├── module_03_graphrag_agent.ipynb  # Advanced RAG implementation
├── module_03_graphrag_agent.py     # RAG agent module
├── generate_embeddings.py          # Script to generate skill embeddings
├── requirements.txt                # Python dependencies
├── expanded_skills.csv            # Source skill data
└── skills_embeddings.csv          # Generated vector embeddings
```

## Quick Start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set up your `.env` file with API keys and Neo4j credentials

3. Generate embeddings:
   ```bash
   python generate_embeddings.py
   ```

4. Explore the notebooks to see GraphRAG in action!