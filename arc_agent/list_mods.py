import os
import dotenv
from openai import OpenAI

dotenv.load_dotenv(r"d:\Documents\董耀择\dyz电子资料\2.2计科\01.励行导师\第二周\llm_benchmark\llm_benchmark\.env")

client = OpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL")
)

try:
    print("Listing models...")
    models = client.models.list()
    for m in models.data:
        print(m.id)
except Exception as e:
    print(f"Error: {e}")
