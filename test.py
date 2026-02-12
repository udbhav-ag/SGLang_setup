from openai import OpenAI
import time
import random

# ---------- Utils ----------

def load_prompt(file_path):
    with open(file_path, 'r') as f:
        return f.read().strip()

def make_unique_deterministic_prompts(prompt1, prompt2, n=20, prompt_size=5000):
    """
    Generates 'n' distinct prompts.
    - No shared prefix.
    - Deterministic (same prompts every run via random.seed).
    """
    # 1. Fix the seed for reproducibility
    random.seed(42)

    # Combine text
    base_text = (prompt1 + "\n" + prompt2)
    
    # Repeat text if it's too small for the requested slice size
    while len(base_text) < (prompt_size + 1000):
        base_text += "\n" + base_text

    prompts = []
    max_start_index = len(base_text) - prompt_size

    for i in range(n):
        # 2. Pick a random start index
        start_idx = random.randint(0, max_start_index)
        
        # 3. Slice the text
        this_prompt = base_text[start_idx : start_idx + prompt_size]
        
        # Add header for verification
        header = f"### PROMPT ID {i:02d} (Deterministic Random) ###\n"
        final_prompt = header + this_prompt[:-(len(header))]

        prompts.append(final_prompt)

    return prompts


# ---------- LLM Query (Sync) ----------

def query_llm(client, prompt, idx):
    start = time.time()
    
    # Extract ID for logging
    prompt_id_line = prompt.split('\n')[0]

    try:
        # Standard blocking call
        resp = client.chat.completions.create(
            model="sglang",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=128,
            temperature=0.0
        )
    except Exception as e:
        print(f"Request {idx} failed: {e}")
        return

    elapsed = time.time() - start
    print(f"Request {idx:02d} finished in {elapsed:.2f}s | {prompt_id_line}")


# ---------- Main ----------

def main():
    # Load source texts
    try:
        p1 = load_prompt("prompts/1.txt")
        p2 = load_prompt("prompts/2.txt")
    except FileNotFoundError:
        print("Warning: Files not found. Using dummy text.")
        p1 = "Dummy text A. " * 5000
        p2 = "Dummy text B. " * 5000

    # Generate deterministic prompts
    all_prompts = make_unique_deterministic_prompts(p1, p2, n=10, prompt_size=20000)

    # Connect to local SGLang (Sync Client)
    client = OpenAI(
        base_url="http://localhost:30000/v1",
        api_key="dummy"
    )

    print(f"Starting synchronous execution of {len(all_prompts)} requests...")

    # Simple sequential loop
    for i, p in enumerate(all_prompts, start=1):
        query_llm(client, p, i)

if __name__ == "__main__":
    main()