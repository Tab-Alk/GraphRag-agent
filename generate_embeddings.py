import os
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure the Google AI with your API key
google_api_key = os.getenv('GOOGLE_API_KEY')
if not google_api_key:
    raise ValueError("GOOGLE_API_KEY not found in environment variables")

genai.configure(api_key=google_api_key)

# Get the model and embedding model from environment variables
MODEL = os.getenv('MODEL', 'gemini-1.5-pro-latest')
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'models/embedding-001')

def generate_embeddings(texts):
    """Generate embeddings for a list of text strings."""
    try:
        response = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=texts,
            task_type="retrieval_document"
        )
        return response['embedding']
    except Exception as e:
        print(f"Error generating embeddings: {str(e)}")
        return None

def generate_description(skill_name):
    """Generate a description for a single skill using the Gemini model."""
    try:
        # Use the chat completion API directly
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(
            f"Write a brief and engaging description of the IT skill: {skill_name}. Focus on what the skill is used for and its importance in the tech industry."
        )
        return response.text
    except Exception as e:
        print(f"Error generating description for '{skill_name}': {str(e)}")
        # Return a default description if the API call fails
        return f"{skill_name} is a valuable skill in the technology industry with various applications."

def main():
    print("Embedding generation script initialized.")
    print(f"Using model: {MODEL}")
    print(f"Using embedding model: {EMBEDDING_MODEL}")

    # Read the expanded_skills.csv file
    try:
        df = pd.read_csv('expanded_skills.csv')
    except FileNotFoundError:
        print("Error: 'expanded_skills.csv' not found. Make sure it's in the same directory as the script.")
        return

    print("Extracting unique skills from the CSV file")
    # Extract, split, and deduplicate skills from the 'skills' column
    df['Skill'] = df['skills'].str.split(', ')
    skills_df = df[['Skill']].explode('Skill')
    skills_df = skills_df.drop_duplicates().sort_values(by='Skill').reset_index(drop=True)

    print(f"Found {len(skills_df)} unique skills.")

    print("Generating descriptions for skills")
    skills_df['Description'] = skills_df['Skill'].apply(generate_description)

    print("Generating embeddings for skill descriptions")
    embedding_list = generate_embeddings(list(skills_df['Description']))
    if embedding_list:
        skills_df['Embedding'] = embedding_list
    else:
        print("Failed to generate embeddings. Exiting.")
        return

    print("Successfully generated embeddings. Saving to CSV")
    skills_df.to_csv('skills_embeddings.csv', index=False)

    print("skills_embeddings.csv has been successfully created.")
    print(f"File saved to: {os.path.join(os.getcwd(), 'skills_embeddings.csv')}")

if __name__ == "__main__":
    main()