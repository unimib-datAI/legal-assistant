import os

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Neo4j configuration
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# OpenAI API Key
OPENAI_API_KEY = os.getenv("OPEN_API_KEY")
OPENAI_BASE_URL = os.getenv('OPENAI_BASE_URL', '')
OPENAI_ENDPOINT = f"{OPENAI_BASE_URL}/embeddings" if OPENAI_BASE_URL else None