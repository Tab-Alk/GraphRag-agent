import os
import google.generativeai as genai
from dotenv import load_dotenv

def test_api():
    # Load environment variables
    load_dotenv()
    
    # Get API key
    google_api_key = os.getenv('GOOGLE_API_KEY')
    if not google_api_key:
        print("Error: GOOGLE_API_KEY not found in environment variables")
        print("Please make sure your .env file exists and contains GOOGLE_API_KEY=your_api_key_here")
        return
    
    print("API Key found. Testing connection to Google's Generative AI...")
    
    # Try to configure the API
    try:
        genai.configure(api_key=google_api_key)
        print("Successfully configured the API!")
        
        # Test model listing
        print("\nAttempting to list available models...")
        models = genai.list_models()
        print("Successfully connected to Google's API. Available models:")
        for model in models:
            print(f"- {model.name}")
            
        # Test a simple generation
        print("\nTesting text generation...")
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content("Say hello to confirm the API is working!")
        print("\nAPI Response:")
        print(response.text)
        
    except Exception as e:
        print(f"\nError occurred while testing the API:")
        print(f"Type: {type(e).__name__}")
        print(f"Error: {str(e)}")
        print("\nPossible issues:")
        print("1. The API key might be invalid or expired")
        print("2. There might be network connectivity issues")
        print("3. The API might be experiencing downtime")
        print("4. Your account might not have access to the Gemini models")

if __name__ == "__main__":
    test_api()
