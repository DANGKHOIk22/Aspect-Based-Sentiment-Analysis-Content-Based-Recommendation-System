import requests
import os
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("API_URL", "http://backend:8000")

def predict(book_id: int, comment: str) -> dict:
    # Ensure book_id is a native Python int
    book_id = int(book_id)

    url = f"{API_URL}/api/st/predict"

    json_data = {
        "book_id": book_id,
        "comment": comment
    }

    try:
        response = requests.post(url, json=json_data, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"Request failed: {str(e)}"}
