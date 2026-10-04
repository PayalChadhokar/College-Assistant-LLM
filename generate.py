import torch
import torch.nn.functional as F

from model.model import TinyLanguageModel
from tokenizer.tokenizer import CharacterTokenizer


# --------------------------------
# Load training text
# --------------------------------

with open(
    "dataset/college_data.txt",
    "r",
    encoding="utf-8"
) as f:
    text = f.read()


# --------------------------------
# Create tokenizer
# --------------------------------

tokenizer = CharacterTokenizer(text)


# --------------------------------
# Model configuration
# --------------------------------

block_size = 64
embedding_size = 128
num_heads = 4
num_layers = 2


# --------------------------------
# Device
# --------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# --------------------------------
# Create model
# --------------------------------

model = TinyLanguageModel(
    vocab_size=tokenizer.vocab_size,
    embedding_size=embedding_size,
    block_size=block_size,
    num_heads=num_heads,
    num_layers=num_layers
)


# --------------------------------
# Load trained weights
# --------------------------------

model.load_state_dict(
    torch.load(
        "college_assistant_model.pth",
        map_location=device
    )
)

model = model.to(device)

model.eval()


# --------------------------------
# Generate text
# --------------------------------

def generate_text(prompt, max_new_tokens=200):

    # Convert prompt to token IDs
    tokens = tokenizer.encode(prompt)

    # Convert to tensor
    x = torch.tensor(
        [tokens],
        dtype=torch.long,
        device=device
    )

    for _ in range(max_new_tokens):

        # Keep only the last block_size tokens
        x_context = x[:, -block_size:]

        # Model prediction
        with torch.no_grad():

            logits = model(x_context)

        # Get prediction for last position
        logits = logits[:, -1, :]

        # Convert to probabilities
        probabilities = F.softmax(
            logits,
            dim=-1
        )

        # Choose next token
        next_token = torch.multinomial(
            probabilities,
            num_samples=1
        )

        # Add token to sequence
        x = torch.cat(
            (x, next_token),
            dim=1
        )

    # Convert tokens back to text
    generated_tokens = x[0].tolist()

    return tokenizer.decode(generated_tokens)


# --------------------------------
# User interface
# --------------------------------

print("===================================")
print("       COLLEGE ASSISTANT LLM")
print("===================================")

prompt = input("\nEnter your prompt: ")

result = generate_text(
    prompt,
    max_new_tokens=200
)

print("\nGenerated text:")
print(result)