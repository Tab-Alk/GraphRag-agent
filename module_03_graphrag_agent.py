#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GraphRAG and Agents

This module implements a GraphRAG agent with the following capabilities:
1. Retrieve skills of a person
2. Find similar skills based on semantic similarity
3. Find similar persons based on skill sets
4. Find persons with specific skills
5. Interactive chatbot interface for querying the knowledge graph
"""

import os
import json
import pandas as pd
from dotenv import load_dotenv
from neo4j import Query, GraphDatabase, RoutingControl, Result

# LangChain imports
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import google.generativeai as genai

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.prebuilt import create_react_agent

# Other imports
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, validator
import functools
from langchain_core.tools import tool
import gradio as gr
import time
import numpy as np

def setup_environment():
    """Load environment variables and set up the application."""
    env_file = 'ws.env'
    
    if not os.path.exists(env_file):
        print(f"File {env_file} not found. Looking for .env...")
        env_file = '.env'
    
    if os.path.exists(env_file):
        load_dotenv(env_file, override=True)

        # Neo4j configuration
        HOST = os.getenv('NEO4J_URI')
        USERNAME = os.getenv('NEO4J_USERNAME')
        PASSWORD = os.getenv('NEO4J_PASSWORD')
        DATABASE = os.getenv('NEO4J_DATABASE', 'neo4j')

        # AI configuration
        GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
        if not GOOGLE_API_KEY:
            print("Warning: GOOGLE_API_KEY not found in environment variables")
        
        # Get model names from environment variables with defaults
        LLM = os.getenv('MODEL', 'gemini-pro')
        EMBEDDINGS_MODEL = os.getenv('EMBEDDING_MODEL', 'models/embedding-001')
        
        return {
            'HOST': HOST,
            'USERNAME': USERNAME,
            'PASSWORD': PASSWORD,
            'DATABASE': DATABASE,
            'LLM': LLM,
            'EMBEDDINGS_MODEL': EMBEDDINGS_MODEL,
            'GOOGLE_API_KEY': GOOGLE_API_KEY
        }
    else:
        print(f"No environment file found. Please create {env_file} or .env")
        return None

class Neo4jConnection:
    """Handles connection to Neo4j database and provides query methods for the GraphRAG agent."""
    
    def __init__(self, uri: str, user: str, password: str, database: str = None):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database
        self.embeddings = None
        self.llm = None
    
    def close(self):
        """Close the database connection."""
        if self.driver is not None:
            self.driver.close()
    
    def test_connection(self) -> bool:
        """Test the database connection."""
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run("RETURN 1 AS test")
                return result.single() is not None
        except Exception as e:
            print(f"Connection test failed: {e}")
            return False
    
    def execute_query(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """Execute a Cypher query and return the result as a list of dictionaries."""
        with self.driver.session(database=self.database) as session:
            result = session.run(query, **kwargs)
            return [dict(record) for record in result]
    
    def execute_query_df(self, query: str, **kwargs) -> pd.DataFrame:
        """Execute a Cypher query and return the result as a pandas DataFrame."""
        return pd.DataFrame(self.execute_query(query, **kwargs))
    
    def get_person_skills(self, person_name: str) -> List[Dict[str, Any]]:
        """Get all skills for a given person."""
        return self.execute_query(
            """
            MATCH (p:Person{name: $person_name})-[:KNOWS]->(s:Skill)
            RETURN p.name as name, COLLECT(s.name) as skills
            """,
            person_name=person_name
        )
    
    def find_similar_skills(self, skills: List[str], threshold: float = 0.89, top_n: int = 3) -> List[Dict[str, Any]]:
        """Find skills similar to the provided list of skills."""
        if not self.embeddings:
            raise ValueError("Embeddings not initialized. Call initialize_embeddings() first.")
        
        # Get embeddings for input skills
        skill_vectors = self.embeddings.embed_documents(skills)
        
        # Convert to list of lists for Neo4j
        vectors_list = [list(map(float, vec)) for vec in skill_vectors]
        
        # Query for similar skills
        results = self.execute_query(
            """
            UNWIND $skill_vectors AS v
            CALL db.index.vector.queryNodes('skill-embeddings', $top_n, v) 
            YIELD node, score
            WHERE score > $threshold
            RETURN node.name as skill_name, score
            ORDER BY score DESC
            """,
            skill_vectors=vectors_list,
            top_n=top_n,
            threshold=threshold
        )
        
        return results
    
    def find_similar_persons(self, person_name: str, community: bool = True) -> List[Dict[str, Any]]:
        """Find persons with similar skills to the given person."""
        if community:
            # Using community detection
            return self.execute_query(
                """
                MATCH (p:Person {name: $person_name})-[:IN_COMMUNITY]->(c:Community)<-[:IN_COMMUNITY]-(other:Person)
                WHERE p <> other
                RETURN other.name as similar_person, c.communityId as community_id
                LIMIT 10
                """,
                person_name=person_name
            )
        else:
            # Using skill overlap
            return self.execute_query(
                """
                MATCH (p:Person {name: $person_name})-[:KNOWS]->(s:Skill)<-[:KNOWS]-(other:Person)
                WHERE p <> other
                WITH other, COUNT(s) as common_skills
                RETURN other.name as similar_person, common_skills
                ORDER BY common_skills DESC
                LIMIT 10
                """,
                person_name=person_name
            )
    
    def find_persons_by_skills(self, skills: List[str], min_matches: int = 1) -> List[Dict[str, Any]]:
        """Find persons who have at least min_matches of the specified skills."""
        return self.execute_query(
            """
            MATCH (p:Person)
            WITH p, [skill IN $skills WHERE (p)-[:KNOWS]->(:Skill {name: skill}) | skill] as matching_skills
            WHERE size(matching_skills) >= $min_matches
            RETURN p.name as person, matching_skills, size(matching_skills) as match_count
            ORDER BY match_count DESC
            """,
            skills=skills,
            min_matches=min_matches
        )
    
    def initialize_embeddings(self, embeddings_model: str = 'models/embedding-001'):
        """Initialize the embeddings model."""
        self.embeddings = GoogleGenerativeAIEmbeddings(model=embeddings_model)
    
    def initialize_llm(self, model_name: str = 'gemini-pro'):
        """Initialize the language model."""
        self.llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.7, convert_system_message_to_human=True)

# Initialize the application
def create_agent_tools(db: Neo4jConnection) -> list:
    """Create tools for the agent to interact with the Neo4j database."""
    
    @tool
    def get_person_skills(person_name: str) -> str:
        """Get all skills for a given person."""
        try:
            result = db.get_person_skills(person_name)
            if not result:
                return f"No skills found for {person_name}"
            skills = result[0].get('skills', [])
            return f"{person_name} has the following skills: {', '.join(skills)}"
        except Exception as e:
            return f"Error retrieving skills: {str(e)}"
    
    @tool
    def find_similar_skills(skills: str) -> str:
        """Find skills similar to the provided comma-separated list of skills.
        
        Args:
            skills: A comma-separated list of skills to find similar skills for
            
        Returns:
            A formatted string with similar skills and their similarity scores
        """
        try:
            if not db.embeddings:
                return "Error: Embeddings not initialized. Please check if embeddings are properly initialized."
                
            if not skills or not skills.strip():
                return "Please provide at least one skill to find similar skills for."
                
            skill_list = [s.strip() for s in skills.split(',') if s.strip()]
            if not skill_list:
                return "No valid skills provided. Please check your input."
                
            results = db.find_similar_skills(skill_list)
            
            if not results:
                return "No similar skills found in the knowledge base."
                
            # Group by skill and collect scores
            skill_scores = {}
            for item in results:
                skill = item.get('skill_name', 'Unknown')
                score = item.get('score', 0.0)
                if skill not in skill_scores:
                    skill_scores[skill] = []
                skill_scores[skill].append(score)
            
            # Calculate average scores
            avg_scores = {k: sum(v)/len(v) for k, v in skill_scores.items()}
            
            # Sort by average score in descending order
            sorted_skills = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)
            
            # Format the output
            if len(sorted_skills) == 0:
                return "No similar skills found."
                
            formatted_skills = []
            for skill, score in sorted_skills[:10]:  # Limit to top 10 similar skills
                formatted_skills.append(f"{skill} ({score:.2f})")
                
            return f"Similar skills found: {', '.join(formatted_skills)}"
            
        except Exception as e:
            return f"Error finding similar skills: {str(e)}"
    
    @tool
    def find_similar_persons(person_name: str, method: str = 'community') -> str:
        """Find persons with similar skills to the given person.
        
        Args:
            person_name: The name of the person to find similar people for
            method: The method to determine similarity. 
                   - 'community': Find people in the same community (faster, less precise)
                   - 'skills': Find people with similar skills (slower, more precise)
                   
        Returns:
            A formatted string with similar persons and their similarity details
        """
        try:
            if not person_name or not person_name.strip():
                return "Please provide a valid person's name to find similar people."
                
            # Normalize method parameter
            method = method.lower() if method else 'community'
            use_community = method == 'community'
            
            if method not in ['community', 'skills']:
                return "Invalid method. Please use 'community' or 'skills' as the method."
            
            results = db.find_similar_persons(person_name.strip(), community=use_community)
            
            if not results:
                return f"No similar persons found for '{person_name}' using {method} method."
            
            # Format the results based on the method used
            if use_community:
                similar_people = []
                for r in results[:10]:  # Limit to top 10 results
                    person = r.get('similar_person', 'Unknown')
                    community_id = r.get('community_id', 'N/A')
                    similar_people.append(f"{person} (community {community_id})")
                
                if not similar_people:
                    return f"No community information available for {person_name}."
                    
                return f"People in the same community as {person_name}: {', '.join(similar_people)}"
            else:
                similar_people = []
                for r in results[:10]:  # Limit to top 10 results
                    person = r.get('similar_person', 'Unknown')
                    common_skills = r.get('common_skills', 0)
                    if common_skills > 0:  # Only include if they share at least one skill
                        similar_people.append(f"{person} ({common_skills} shared skills)")
                
                if not similar_people:
                    return f"No people found with similar skills to {person_name}."
                    
                return f"People with similar skills to {person_name}: {', '.join(similar_people)}"
                
        except Exception as e:
            return f"Error finding similar persons: {str(e)}"
    
    @tool
    def find_persons_by_skills(skills: str, min_matches: int = 1) -> str:
        """Find persons who have at least min_matches of the specified skills.
        
        Args:
            skills: A comma-separated list of skills to search for
            min_matches: Minimum number of skills that must match (default: 1)
            
        Returns:
            A formatted string with matching persons and their matching skills
        """
        try:
            # Validate inputs
            if not skills or not skills.strip():
                return "Please provide at least one skill to search for."
                
            if not isinstance(min_matches, int) or min_matches < 1:
                return "min_matches must be a positive integer."
                
            # Process skills list
            skill_list = [s.strip() for s in skills.split(',') if s.strip()]
            if not skill_list:
                return "No valid skills provided. Please check your input."
                
            # Adjust min_matches if it's greater than the number of skills
            if min_matches > len(skill_list):
                min_matches = len(skill_list)
                
            # Search for persons with matching skills
            results = db.find_persons_by_skills(skill_list, min_matches)
            
            if not results:
                skill_text = "', '".join(skill_list[:3])
                if len(skill_list) > 3:
                    skill_text += f"' and {len(skill_list) - 3} more"
                return f"No persons found with at least {min_matches} of these skills: '{skill_text}'"
            
            # Format the results
            response = []
            for item in results[:20]:  # Limit to top 20 results
                person = item.get('person', 'Unknown')
                match_count = item.get('match_count', 0)
                matching_skills = item.get('matching_skills', [])
                
                # Format the matching skills for display
                if len(matching_skills) > 3:
                    skills_display = f"{', '.join(matching_skills[:3])}, and {len(matching_skills) - 3} more"
                else:
                    skills_display = ', '.join(matching_skills)
                
                response.append(f"• {person} ({match_count} matching skills: {skills_display})")
            
            # Add a summary line
            total_found = len(results)
            if total_found > 20:
                response.append(f"\n... and {total_found - 20} more results not shown.")
            
            return "\n".join(response)
            
        except Exception as e:
            return f"Error searching for persons by skills: {str(e)}"
    
    @tool
    def perform_cypher_query(question: str) -> str:
        """Execute a complex query on the Neo4j database using natural language.
        
        Use this tool when you need to answer complex questions that can't be answered
        by the other tools. The tool will translate natural language to a Neo4j Cypher query.
        
        Args:
            question: A natural language question about the data
            
        Returns:
            The results of the query in a human-readable format
        """
        try:
            # Text2Cypher prompt template
            text2cypher_prompt = PromptTemplate.from_template(
                """
                Task: Generate a Cypher statement for querying a Neo4j graph database from a user input. 
                - Do not include triple backticks ``` or ```cypher or any additional text except the generated Cypher statement in your response.
                - Do not use any properties or relationships not included in the schema.
                
                Schema:
                {schema}
                
                # User Input
                {question}
                
                Cypher query:
                """
            )
            
            # Annotated schema for the LLM
            annotated_schema = """
            Nodes:
              Person:
                description: "A person in our talent pool."
                properties:
                  name:
                    type: "string"
                    description: "The full name of the person. Serves as a unique identifier."
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
            
            # Generate the Cypher query using the LLM
            prompt = text2cypher_prompt.invoke({
                'schema': annotated_schema, 
                'question': question
            })
            
            # Use the configured LLM to generate the query
            query = db.llm.invoke(prompt).content
            
            # Clean up the query (remove any markdown code blocks)
            if '```' in query:
                query = query.split('```')[1].replace('cypher', '').strip()
            
            print(f"Generated Cypher query:\n{query}\n")
            
            # Execute the query
            results = db.execute_query(query)
            
            if not results:
                return "No results found for your query."
                
            # Format the results as a string
            if isinstance(results, list):
                if len(results) == 1 and isinstance(results[0], str):
                    return results[0]
                elif len(results) == 1 and isinstance(results[0], dict):
                    return "\n".join([f"{k}: {v}" for k, v in results[0].items()])
                else:
                    return "\n---\n".join([str(r) for r in results[:10]]) + \
                           ("\n... (more results not shown)" if len(results) > 10 else "")
            return str(results)
            
        except Exception as e:
            return f"Error executing query: {str(e)}"
    
    return [
        get_person_skills,
        find_similar_skills,
        find_similar_persons,
        find_persons_by_skills,
        perform_cypher_query
    ]

def create_chatbot_interface(llm, tools):
    """Create a simple Gradio interface for the chatbot."""
    # Create the agent
    agent = create_react_agent(llm, tools)
    
    def respond(message, chat_history):
        """Generate a response to the user message."""
        try:
            # Format the input for the agent
            input_data = {
                "messages": [
                    {
                        "role": "user",
                        "content": message
                    }
                ]
            }
            
            # Process the message with the agent
            response = agent.invoke(input_data)
            
            # Get the response text
            if isinstance(response, dict) and "output" in response:
                bot_message = response["output"]
            elif isinstance(response, str):
                bot_message = response
            else:
                bot_message = str(response)
            
            # Add to chat history
            chat_history.append((message, bot_message))
            
            return "", chat_history
            
        except Exception as e:
            error_msg = f"An error occurred: {str(e)}"
            chat_history.append((message, error_msg))
            return "", chat_history
    
    # Create a simple interface
    with gr.Blocks(title="GraphRAG Agent") as demo:
        gr.Markdown("# GraphRAG Agent")
        gr.Markdown("Ask me about people and their skills in the knowledge graph.")
        
        # Chat interface
        chatbot = gr.Chatbot(height=500)
        
        with gr.Row():
            msg = gr.Textbox(
                label="Your message",
                placeholder="Type your message here...",
                scale=4
            )
            submit_btn = gr.Button("Send", variant="primary")
        
        clear_btn = gr.Button("Clear Chat")
        
        # Event handlers
        msg.submit(respond, [msg, chatbot], [msg, chatbot])
        submit_btn.click(respond, [msg, chatbot], [msg, chatbot])
        clear_btn.click(lambda: None, None, chatbot, queue=False)
        
        # Example questions
        gr.Examples(
            examples=[
                ["What skills does John Garcia have?"],
                ["Find people with Python and AWS experience"],
                ["What skills are similar to Data Science?"],
                ["Who has similar skills to Jane Smith?"]
            ],
            inputs=msg
        )
        
        return demo

import socket

def find_available_port(start_port=7860, max_attempts=10):
    """Find an available port starting from start_port."""
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', port))
                return port
        except OSError:
            continue
    return None

def main():
    """Main function to run the application."""
    # Load configuration
    config = setup_environment()
    if not config:
        print("Failed to load configuration. Exiting...")
        return
    
    # Initialize database connection
    db = Neo4jConnection(
        uri=config['HOST'],
        user=config['USERNAME'],
        password=config['PASSWORD'],
        database=config['DATABASE']
    )
    
    # Test the connection
    if not db.test_connection():
        print("Failed to connect to Neo4j. Please check your connection settings.")
        return
    
    print("Successfully connected to Neo4j!")
    
    # Initialize AI models
    try:
        # Initialize embeddings and LLM
        print("Initializing AI models...")
        db.initialize_embeddings(config['EMBEDDINGS_MODEL'])
        db.initialize_llm(config['LLM'])
        
        if config.get('GOOGLE_API_KEY'):
            genai.configure(api_key=config['GOOGLE_API_KEY'])
        
        print("Successfully initialized AI models!")
        
        # Create tools for the agent
        print("Creating agent tools...")
        tools = create_agent_tools(db)
        
        # Create the chatbot interface
        print("Creating chatbot interface... (this may take a moment)")
        demo = create_chatbot_interface(llm=db.llm, tools=tools)
        
        # Find an available port
        port = find_available_port()
        if port is None:
            print("Error: Could not find an available port. Please close other applications using ports 7860-7869.")
            return
        
        # Launch the interface with appropriate settings
        print("\n" + "="*80)
        print(f"Starting GraphRAG Agent on port {port}...")
        print("The web interface should open in your default browser.")
        print(f"If it doesn't, you can access it at: http://localhost:{port}")
        print("="*80 + "\n")
        
        # Try to launch with local settings first
        try:
            print(f"Attempting to launch on port {port}...")
            demo.launch(
                server_name="127.0.0.1",  # Use localhost
                server_port=port,         # Use the found available port
                share=False,              # Don't create a public link
                show_error=True,          # Show detailed errors
                debug=False,              # Disable debug mode for cleaner output
                show_api=False,           # Hide API documentation
                inbrowser=True            # Try to open in default browser
            )
        except Exception as e:
            print("\n" + "-"*80)
            print("Local launch failed, attempting to create a shareable link...")
            print("-"*80 + "\n")
            
            # If local launch fails, try with a shareable link
            demo.launch(
                share=True,       # Create a public link
                server_port=port,  # Use the found available port
                show_error=True,  # Show detailed errors
                debug=False,      # Disable debug mode
                inbrowser=True    # Try to open in default browser
            )
        
    except Exception as e:
        import traceback
        print("\n" + "!"*80)
        print("Error starting the application:")
        print(str(e))
        print("\nDetailed traceback:")
        traceback.print_exc()
        print("!"*80 + "\n")
    finally:
        # Close the database connection when done
        print("Closing database connections...")
        db.close()

if __name__ == "__main__":
    main()
