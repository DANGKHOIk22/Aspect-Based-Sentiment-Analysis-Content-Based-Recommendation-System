import requests
import os
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("API_URL", "http://backend:8000")

def predict(input_data: str) -> dict:
    
    url = f"{API_URL}/api/st/predict"  

    json_data = {
        "text": input_data  
    }
    
    try:
        response = requests.post(url, json=json_data, timeout=10)
        response.raise_for_status()  # Raise an error for bad status codes
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"Request failed: {str(e)}"}