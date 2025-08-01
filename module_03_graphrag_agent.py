#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module 3 - GraphRAG and Agents

This module has the following objectives:
- Experiment with queries for an Agent
- Define Tooling
- Create an agents with the available tools
- Chatbot for an Agent
- Text2Cypher
"""

import os
import pandas as pd
from dotenv import load_dotenv
from neo4j import Query, GraphDatabase, RoutingControl, Result
from langchain.schema import HumanMessage
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.prebuilt import create_react_agent
import google.generativeai as genai
from typing import List, Optional
from pydantic import BaseModel, Field, validator
import functools
from langchain_core.tools import tool
import gradio as gr
import time

# Load environment variables
env_file = '.env'

if os.path.exists(env_file):
    load_dotenv(env_file, override=True)

    # Neo4j
    HOST = os.getenv('NEO4J_URI')
    USERNAME = os.getenv('NEO4J_USERNAME')
    PASSWORD = os.getenv('NEO4J_PASSWORD')
    DATABASE = os.getenv('NEO4J_DATABASE')

    # AI - Google Gemini
    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
    genai.configure(api_key=GOOGLE_API_KEY)
    # Using gemini-1.5-flash which is optimized for speed and efficiency
    LLM = os.getenv('LLM', 'gemini-1.5-flash')
    EMBEDDINGS_MODEL = os.getenv('EMBEDDINGS_MODEL', 'models/embedding-001')
else:
    print(f"File {env_file} not found.")

# Connect to Neo4j
driver = GraphDatabase.driver(
    HOST,
    auth=(USERNAME, PASSWORD)
)

# Initialize embeddings
embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDINGS_MODEL)

# Define models
class Skill(BaseModel):
    """Represents a professional skill or knowledge of a person."""
    name: str = Field(..., description="Shortened name of the skill")

# Tool 1: Retrieve skills of a person
def retrieve_skills_of_person(person_name: str) -> pd.DataFrame:
    """Retrieve the skills of a person. Person is provided with its name"""
    return driver.execute_query(
        """
        MATCH (p:Person{name: $person_name})-[:KNOWS]->(s:Skill)
        RETURN p.name as name, COLLECT(s.name) as skills
        """,
        database_=DATABASE,
        routing_=RoutingControl.READ,
        result_transformer_=lambda r: r.to_df(),
        person_name=person_name
    )

# Tool 2: Find similar skills
def find_similar_skills(skills: List[Skill]) -> pd.DataFrame:
    """Find similar skills to list of skills specified. Skills are specified by a list of their names"""
    skills = [s.name for s in skills]
    skills_vectors = embeddings.embed_documents(skills)
    return driver.execute_query(
        """
        UNWIND $skills_vectors AS v
        CALL db.index.vector.queryNodes('skill-embeddings', 3, TOFLOATLIST(v)) YIELD node, score
        WHERE score > 0.89
        OPTIONAL MATCH (node)-[:SIMILAR_SEMANTIC]-(s:Skill)
        WITH COLLECT(node) AS nodes, COLLECT(DISTINCT s) AS skills
        WITH nodes + skills AS all_skills
        UNWIND all_skills AS skill
        RETURN DISTINCT skill.name as skill_name
        """,
        database_=DATABASE,
        routing_=RoutingControl.READ,
        result_transformer_=lambda r: r.to_df(),
        skills_vectors=skills_vectors
    )

# Tool 3: Find similar persons
def person_similarity(person_name: str) -> pd.DataFrame:
    """Find a similar person to the one specified based on their skill similarity."""
    return driver.execute_query(
        """
        MATCH (p1:Person {name: $person_name})-[:KNOWS]->(s:Skill)
        WITH p1, COLLECT(s.name) as skills_1
        CALL (p1){
          MATCH (p1)-[:KNOWS]->(s1:Skill)-[r:SIMILAR_SEMANTIC]-(s2:Skill)<-[:KNOWS]-(p2:Person)
          RETURN p1 as person_1, p2 as person_2, SUM(r.score) AS score
          UNION 
          MATCH (p1)-[r:SIMILAR_SKILLSET]-(p2:Person)
          RETURN p1 as person_1, p2 AS person_2, SUM(r.overlap) AS score
        }
        WITH person_1.name as person_1, skills_1, person_2, SUM(score) as score
        WHERE score >= 1
        MATCH (person_2)-[:KNOWS]->(s:Skill)
        RETURN person_1, skills_1, person_2.name as person_2, COLLECT(s.name) as skills_2, score
        ORDER BY score DESC LIMIT 5
        """,
        database_=DATABASE,
        routing_=RoutingControl.READ,
        result_transformer_=lambda r: r.to_df(),
        person_name=person_name
    )

# Tool 4: Find persons based on skills
def find_person_based_on_skills(skills: List[Skill]) -> pd.DataFrame:
    """
    Find persons based on skills they have. Skills are specified by their names. 
    Note that similar skills can be found. These are considered similar. 
    """
    skills = [s.name for s in skills]
    skills_vectors = embeddings.embed_documents(skills)
    return driver.execute_query(
        """
        UNWIND $skills_vectors AS v
        CALL db.index.vector.queryNodes('skill-embeddings', 3, TOFLOATLIST(v)) YIELD node, score
        WHERE score > 0.89
        OPTIONAL MATCH (node)-[:SIMILAR_SEMANTIC]-(s:Skill)
        WITH COLLECT(node) AS nodes, COLLECT(DISTINCT s) AS skills
        WITH nodes + skills AS all_skills
        UNWIND all_skills AS skill
        MATCH (p:Person)-[:KNOWS]->(skill)
        RETURN p.name AS person, COUNT(DISTINCT(skill)) AS score, COLLECT(DISTINCT(skill.name)) as similar_skills
        ORDER BY score DESC LIMIT 10
        """,
        database_=DATABASE,
        routing_=RoutingControl.READ,
        result_transformer_=lambda r: r.to_df(),
        skills_vectors=skills_vectors
    )

# Set up the agent
llm = ChatGoogleGenerativeAI(model=LLM, temperature=0, convert_system_message_to_human=True)
tools = [
    retrieve_skills_of_person, 
    find_similar_skills,
    person_similarity,
    find_person_based_on_skills,
]

llm_with_tools = llm.bind_tools(tools)
agent_executor = create_react_agent(llm, tools)

def ask_to_agent(question):
    for step in agent_executor.stream(
        {"messages": [HumanMessage(content=question)]},
        stream_mode="values",
    ):
        step["messages"][-1].pretty_print()

# Chatbot functions
def user(user_message, history):
    if history is None:
        history = []
    history.append({"role": "user", "content": user_message})
    return "", history

def get_answer(history):
    steps = []
    full_prompt = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in history])
    
    for step in agent_executor.stream(
            {"messages": [HumanMessage(content=full_prompt)]},
            stream_mode="values",
    ):
        step["messages"][-1].pretty_print()
        steps.append(step["messages"][-1].content)
    
    return steps[-1]

def bot(history):
    bot_message = get_answer(history)
    history.append({"role": "assistant", "content": ""})

    for character in bot_message:
        history[-1]["content"] += character
        time.sleep(0.01)
        yield history

def launch_chatbot():
    with gr.Blocks() as demo:
        chatbot = gr.Chatbot(
            label="Chatbot on a Graph",
            avatar_images=[
                "https://png.pngtree.com/png-vector/20220525/ourmid/pngtree-concept-of-facial-animal-avatar-chatbot-dog-chat-machine-illustration-vector-png-image_46652864.jpg",
                "https://d-cb.jc-cdn.com/sites/crackberry.com/files/styles/larger/public/article_images/2023/08/openai-logo.jpg"
            ],
            type="messages", 
        )
        msg = gr.Textbox(label="Message")
        clear = gr.Button("Clear")

        msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
            bot, [chatbot], chatbot
        )

        clear.click(lambda: [], None, chatbot, queue=False)

    demo.queue()
    demo.launch(share=True)

# Text2Cypher functionality
text2cypher_prompt = PromptTemplate.from_template(
    """
    Task: Generate a Cypher statement for querying a Neo4j graph database from a user input. 
    - Do not include triple backticks ``` or ```cypher or any additional text except the generated Cypher statement in your response.
    - Do not use any properties or relationships not included in the schema.
    
    Schema:
    {schema}
    
    #User Input
    {question}
    
    Cypher query:
    """
)

annotated_schema = """
    Nodes:
      Person:
        description: "A person in our talent pool."
        properties:
          name:
            type: "string"
            description: "The full name of the person. serves as a unique identifier."
          email:
            type: "string"
            description: "The email address of the person."
          leiden_community:
            type: "integer"
            description: "The talent community for the person. People in the same talent segment share similar skills."
      Skill:
        description: "A professional skill."
        properties:
          name:
            type: "string"
            description: "The unique name of the skill."
    Relationships:
        KNOWS:
            description: "A person knowing a skill."
            query_pattern: "(:Person)-[:KNOWS]->(:Skill)"
    """

text2cypher_llm = ChatGoogleGenerativeAI(model=LLM, temperature=0, convert_system_message_to_human=True)

@tool
def perform_aggregation_query(question: str) -> pd.DataFrame:
    """Perform an aggregation query on the Neo4j graph database and obtain the results."""
    prompt = text2cypher_prompt.invoke({'schema': annotated_schema, 'question': question})
    query = text2cypher_llm.invoke(prompt).content
    print(f"executing Cypher query:\n{query}")
    return driver.execute_query(
        query,
        database_=DATABASE,
        routing_=RoutingControl.READ,
        result_transformer_=lambda r: r.to_df()
    )

if __name__ == "__main__":
    # Example usage
    question = "What skills does John Garcia have?"
    ask_to_agent(question)
    
    # Uncomment to launch the chatbot
    launch_chatbot()
