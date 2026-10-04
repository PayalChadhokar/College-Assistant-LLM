import torch

from model.model import TinyLanguageModel


# Our vocabulary size
vocab_size = 46

# Create model
model = TinyLanguageModel(vocab_size)


# Create fake input
x = torch.randint(
    0,
    vocab_size,
    (16, 64)
)


# Run the model
output = model(x)


print("Input shape:", x.shape)
print("Output shape:", output.shape)