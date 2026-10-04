import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------
# Single Attention Head
# --------------------------------------------------

class AttentionHead(nn.Module):

    def __init__(self, embedding_size, head_size, block_size):
        super().__init__()

        self.key = nn.Linear(
            embedding_size,
            head_size,
            bias=False
        )

        self.query = nn.Linear(
            embedding_size,
            head_size,
            bias=False
        )

        self.value = nn.Linear(
            embedding_size,
            head_size,
            bias=False
        )

        # Causal mask
        self.register_buffer(
            "mask",
            torch.tril(
                torch.ones(block_size, block_size)
            )
        )

        self.head_size = head_size

    def forward(self, x):

        batch_size, sequence_length, embedding_size = x.shape

        # Create Key, Query and Value
        K = self.key(x)
        Q = self.query(x)
        V = self.value(x)

        # Attention scores
        scores = Q @ K.transpose(-2, -1)

        # Scale scores
        scores = scores / (self.head_size ** 0.5)

        # Causal masking
        scores = scores.masked_fill(
            self.mask[:sequence_length, :sequence_length] == 0,
            float("-inf")
        )

        # Convert scores to probabilities
        weights = F.softmax(
            scores,
            dim=-1
        )

        # Weighted values
        output = weights @ V

        return output


# --------------------------------------------------
# Multi-Head Attention
# --------------------------------------------------

class MultiHeadAttention(nn.Module):

    def __init__(
        self,
        embedding_size,
        num_heads,
        block_size
    ):
        super().__init__()

        head_size = embedding_size // num_heads

        self.heads = nn.ModuleList([
            AttentionHead(
                embedding_size,
                head_size,
                block_size
            )
            for _ in range(num_heads)
        ])

        self.projection = nn.Linear(
            embedding_size,
            embedding_size
        )

    def forward(self, x):

        # Run all attention heads
        outputs = [
            head(x)
            for head in self.heads
        ]

        # Combine heads
        x = torch.cat(
            outputs,
            dim=-1
        )

        # Final projection
        x = self.projection(x)

        return x


# --------------------------------------------------
# Feed Forward Network
# --------------------------------------------------

class FeedForward(nn.Module):

    def __init__(self, embedding_size):
        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                embedding_size,
                4 * embedding_size
            ),

            nn.ReLU(),

            nn.Linear(
                4 * embedding_size,
                embedding_size
            )
        )

    def forward(self, x):

        return self.network(x)


# --------------------------------------------------
# Transformer Block
# --------------------------------------------------

class TransformerBlock(nn.Module):

    def __init__(
        self,
        embedding_size,
        num_heads,
        block_size
    ):
        super().__init__()

        self.layer_norm_1 = nn.LayerNorm(
            embedding_size
        )

        self.attention = MultiHeadAttention(
            embedding_size,
            num_heads,
            block_size
        )

        self.layer_norm_2 = nn.LayerNorm(
            embedding_size
        )

        self.feed_forward = FeedForward(
            embedding_size
        )

    def forward(self, x):

        # Attention + residual connection
        x = x + self.attention(
            self.layer_norm_1(x)
        )

        # Feed forward + residual connection
        x = x + self.feed_forward(
            self.layer_norm_2(x)
        )

        return x


# --------------------------------------------------
# College Assistant Language Model
# --------------------------------------------------

class TinyLanguageModel(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_size=128,
        block_size=64,
        num_heads=4,
        num_layers=2
    ):
        super().__init__()

        self.block_size = block_size

        # Token embeddings
        self.token_embedding = nn.Embedding(
            vocab_size,
            embedding_size
        )

        # Position embeddings
        self.position_embedding = nn.Embedding(
            block_size,
            embedding_size
        )

        # Transformer blocks
        self.blocks = nn.Sequential(*[
            TransformerBlock(
                embedding_size,
                num_heads,
                block_size
            )
            for _ in range(num_layers)
        ])

        # Final normalization
        self.final_layer_norm = nn.LayerNorm(
            embedding_size
        )

        # Output layer
        self.output_layer = nn.Linear(
            embedding_size,
            vocab_size
        )

    def forward(self, x):

        batch_size, sequence_length = x.shape

        # Token embeddings
        token_embeddings = self.token_embedding(x)

        # Position numbers
        positions = torch.arange(
            sequence_length,
            device=x.device
        )

        # Position embeddings
        position_embeddings = self.position_embedding(
            positions
        )

        # Combine token and position information
        x = token_embeddings + position_embeddings

        # Transformer blocks
        x = self.blocks(x)

        # Final normalization
        x = self.final_layer_norm(x)

        # Predict next token
        logits = self.output_layer(x)

        return logits