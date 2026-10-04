import torch
from tokenizer.tokenizer import CharacterTokenizer


# Load the dataset
with open("dataset/college_data.txt", "r", encoding="utf-8") as f:
    text = f.read()


# Create tokenizer
tokenizer = CharacterTokenizer(text)


# Convert complete text into token IDs
data = torch.tensor(
    tokenizer.encode(text),
    dtype=torch.long
)


# Context length
block_size = 64


def get_batch(batch_size=16):
    """
    Create a batch of input and target sequences.
    """

    # Random starting positions
    ix = torch.randint(
        len(data) - block_size,
        (batch_size,)
    )

    # Input sequences
    x = torch.stack([
        data[i:i + block_size]
        for i in ix
    ])

    # Target sequences
    y = torch.stack([
        data[i + 1:i + block_size + 1]
        for i in ix
    ])

    return x, y


if __name__ == "__main__":

    print("Dataset characters:", len(text))
    print("Vocabulary size:", tokenizer.vocab_size)
    print("Total tokens:", len(data))

    x, y = get_batch()

    print("Input shape:", x.shape)
    print("Target shape:", y.shape)

    print("\nFirst input:")
    print(x[0])

    print("\nFirst target:")
    print(y[0])