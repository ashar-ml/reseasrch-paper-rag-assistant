import os
import requests
from loguru import logger
from dotenv import load_dotenv

# Resolve workspace directory and load .env file
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
env_path = os.path.join(base_dir, '.env')
if os.path.exists(env_path):
    load_dotenv(dotenv_path=env_path)
else:
    load_dotenv()

BACKEND_API_URL = os.getenv("BACKEND_API_URL", "http://localhost:8000")

class RAGApiClient:
    def __init__(self, base_url: str = BACKEND_API_URL):
        self.base_url = base_url.rstrip('/')
        self.api_v1_url = f"{self.base_url}/api/v1"

    def login(self, username, password) -> dict:
        """Authenticates user with backend."""
        try:
            payload = {"username": username, "password": password}
            response = requests.post(f"{self.api_v1_url}/auth/login", json=payload, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Login request failed: {e}")
            detail = "Connection failed."
            if hasattr(e, 'response') and e.response is not None:
                try:
                    detail = e.response.json().get("detail", detail)
                except Exception:
                    detail = e.response.text
            return {"status": "error", "message": detail}

    def signup(self, username, password) -> dict:
        """Registers a new user account."""
        try:
            payload = {"username": username, "password": password}
            response = requests.post(f"{self.api_v1_url}/auth/signup", json=payload, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Signup request failed: {e}")
            detail = "Connection failed."
            if hasattr(e, 'response') and e.response is not None:
                try:
                    detail = e.response.json().get("detail", detail)
                except Exception:
                    detail = e.response.text
            return {"status": "error", "message": detail}

    def check_health(self) -> dict:
        """Checks connection and settings of the FastAPI backend."""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {"status": "unreachable", "error": str(e)}

    def upload_pdf(self, file_name: str, file_bytes: bytes) -> dict:
        """Uploads a PDF paper to the processing endpoint."""
        try:
            files = {"file": (file_name, file_bytes, "application/pdf")}
            response = requests.post(f"{self.base_url}/upload-pdf", files=files, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to upload PDF: {e}")
            return {"status": "error", "message": str(e)}

    def list_papers(self) -> list:
        """Retrieves a list of uploaded research papers from the backend."""
        try:
            response = requests.get(f"{self.api_v1_url}/upload", timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to list papers: {e}")
            return []

    def query_assistant(self, query_text: str) -> dict:
        """Queries the Agentic RAG assistant."""
        try:
            payload = {"query": query_text}
            response = requests.post(f"{self.base_url}/ask", json=payload, timeout=60)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to query assistant: {e}")
            return {"status": "error", "message": str(e)}


    def run_evaluation(self) -> dict:
        """Runs the RAGAS evaluation suite."""
        try:
            response = requests.post(f"{self.api_v1_url}/evaluate", timeout=120)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to run evaluation: {e}")
            return {"status": "error", "message": str(e)}

    def evaluate_embeddings(self) -> dict:
        """Runs comparative embedding model benchmarks."""
        try:
            response = requests.get(f"{self.api_v1_url}/evaluate/embeddings", timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to run embedding evaluation: {e}")
            return {"status": "error", "message": str(e)}

    def fetch_history(self, username: str) -> dict:
        """Retrieves persistent chat history for the given user."""
        try:
            response = requests.get(f"{self.api_v1_url}/auth/history", params={"username": username}, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to fetch chat history for {username}: {e}")
            return {"status": "error", "message": str(e), "history": []}

    def save_history(self, username: str, role: str, content: str, citations: list = None) -> dict:
        """Saves a single query or answer message to user's chat history."""
        try:
            payload = {
                "username": username,
                "role": role,
                "content": content,
                "citations": citations if citations is not None else []
            }
            response = requests.post(f"{self.api_v1_url}/auth/history", json=payload, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to save chat history item for {username}: {e}")
            return {"status": "error", "message": str(e)}

    def clear_history(self, username: str) -> dict:
        """Deletes all persistent chat history for the user."""
        try:
            response = requests.delete(f"{self.api_v1_url}/auth/history", params={"username": username}, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to clear chat history for {username}: {e}")
            return {"status": "error", "message": str(e)}

