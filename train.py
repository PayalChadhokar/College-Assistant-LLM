import torch  
import torch.nn as nn
from model.model import TinyLanguageModel
from data import get_batch, tokenizer


# -----------------------------
# Configuration
# -----------------------------

batch_size = 16
learning_rate = 0.001
max_iterations = 3000

block_size = 64
embedding_size = 128
num_heads = 4
num_layers = 2


# -----------------------------
# Select device
# -----------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# -----------------------------
# Create model
# -----------------------------

model = TinyLanguageModel(
    vocab_size=tokenizer.vocab_size,
    embedding_size=embedding_size,
    block_size=block_size,
    num_heads=num_heads,
    num_layers=num_layers
)

model = model.to(device)


# -----------------------------
# Optimizer
# -----------------------------

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate
)


# -----------------------------
# Training
# -----------------------------

print("Starting training...")

for iteration in range(max_iterations):

    # Get training batch
    x, y = get_batch(batch_size)

    x = x.to(device)
    y = y.to(device)

    # Forward pass
    logits = model(x)

    # Calculate loss
    loss = nn.functional.cross_entropy(
        logits.view(-1, tokenizer.vocab_size),
        y.view(-1)
    )

    # Clear previous gradients
    optimizer.zero_grad()

    # Backpropagation
    loss.backward()

    # Update model parameters
    optimizer.step()

    # Display progress
    if iteration % 100 == 0:

        print(
            f"Iteration {iteration}/{max_iterations} "
            f"| Loss: {loss.item():.4f}"
        )


# -----------------------------
# Save trained model
# -----------------------------

torch.save(
    model.state_dict(),
    "college_assistant_model.pth"
)

print("\nTraining complete!")
print("Model saved as college_assistant_model.pth")