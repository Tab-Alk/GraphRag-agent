# GraphRAG Agent

## Overview: Graph-Powered Knowledge Management

GraphRAG Agent is an advanced knowledge management system built on Neo4j, the industry-leading graph database. By leveraging Neo4j's native graph architecture, it creates a dynamic, interconnected web of technical skills where relationships are first-class citizens of the data model.

### Why Neo4j?
Neo4j's graph database is the perfect foundation for this project because it:
- **Models Real-World Relationships**: Stores data as nodes (skills, people, roles) and relationships (knows, works_with, requires)
- **Enables Relationship-Centric Queries**: Finds connections between skills, even through multiple degrees of separation
- **Delivers Real-Time Insights**: Performs complex traversals to discover patterns in milliseconds
- **Supports Flexible Schema**: Adapts the data model as your understanding of skill relationships evolves
- **Facilitates Visual Exploration**: Intuitively explore connections using Neo4j's graph visualization tools

This powerful combination of graph technology and semantic understanding transforms traditional skill mapping into an intelligent system that understands relationships between technologies, identifies skill gaps, and provides context-aware recommendations.

## Key Features & Semantic Processing

### 1. Graph-Enhanced Semantic Search
- **Vector Embeddings**: Converts skills into high-dimensional vectors using Google's Generative AI
- **Hybrid Queries**: Combines traditional graph pattern matching with vector similarity search
- **Context-Aware Results**: Understands that "React" can refer to both the JavaScript library and the programming concept

### 2. Intelligent Skill Mapping
- **Skill Clustering**: Automatically groups related technologies (e.g., React, Vue, Angular as frontend frameworks)
- **Proximity Analysis**: Identifies skills that frequently appear together in job roles or projects
- **Gap Identification**: Highlights missing but relevant skills based on existing skill sets

### 3. Real-World Semantic Example
When analyzing a skill like "Python":
1. **Graph Context**: Identifies Python's relationships with frameworks (Django, Flask), libraries (Pandas, NumPy), and use cases (Data Science, Web Development)
2. **Semantic Understanding**: Recognizes that "ML" is closely related to "Machine Learning" and "AI"
3. **Multi-hop Reasoning**: Can suggest that someone who knows Python and Data Analysis might want to learn Pandas, even if they've never used it
4. **Contextual Recommendations**: Suggests complementary skills based on industry trends and job market demands

## The GraphRAG Advantage

Traditional RAG systems use simple document retrieval, but GraphRAG goes further by:
1. **Structured Knowledge**: Organizing information in a graph for efficient traversal and relationship mapping
2. **Semantic Enrichment**: Enhancing raw data with AI-generated context and relationships
3. **Multi-dimensional Analysis**: Combining graph algorithms with vector search for deeper insights
4. **Explainable AI**: Making the reasoning behind recommendations transparent through graph visualizations

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