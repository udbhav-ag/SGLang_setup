import asyncio
from openai import AsyncOpenAI
import concurrent.futures
import time
from utils import count_tokens
# Load prompts from files
def load_prompt(file_path):
    with open(file_path, 'r') as f:
        return f.read().strip()

prompt1 = load_prompt("prompts/1.txt")
prompt1 = prompt1[:120000]
prompt2 = load_prompt("prompts/2.txt")
prompt2 = prompt2[:134000]

# print(f"Prompt 1 tokens: {count_tokens(prompt1)}")
# print(f"Prompt 2 tokens: {count_tokens(prompt2)}")

async def query_llm(client, prompt, idx):
    start = time.time()
    resp = await client.chat.completions.create(
        model="sglang",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=128,
        temperature=0.0
    )
    elapsed = time.time() - start
    print(f"Request {idx} done in {elapsed:.2f}s: {resp.choices[0].message.content}...")

async def main():
    client = AsyncOpenAI(base_url="http://localhost:30000/v1", api_key="dummy")
    
    # Send 2 concurrent requests
    await asyncio.gather(
        query_llm(client, prompt1, 1),
        query_llm(client, prompt2, 2)
    )

if __name__ == "__main__":
    asyncio.run(main())
