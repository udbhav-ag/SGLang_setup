import tiktoken
import sys

def count_tokens(text, model="cl100k_base"):  # or "cl100k_base" for Llama-3.1
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "gpt-4o"
    
    # From file
    with open("prompts/2.txt", "r") as f:
        text = f.read()
        
    text = text[:134000]    
    print(f"Tokens in prompt1.txt ({model}): {count_tokens(text, model)}")
    

