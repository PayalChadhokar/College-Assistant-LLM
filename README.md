# College Assistant LLM

A small Transformer-based College Assistant Language Model built from scratch using Python and PyTorch. The model is trained on a custom college-related Question and Answer dataset and can generate responses to educational questions.

## Project Overview

The College Assistant LLM is an educational language model developed to understand and generate college-related text. Unlike projects that use pretrained models such as GPT or LLaMA, this project implements the main components of a Transformer language model from scratch.

The model learns patterns from a custom dataset containing questions and answers related to Artificial Intelligence, Machine Learning, Python, C++, DBMS, Data Structures, Operating Systems, Computer Networks, Software Engineering, NLP, and other computer science topics.

## Features

- Custom college-related Q&A dataset
- Character-level tokenizer built from scratch
- Token embedding
- Positional embedding
- Self-attention mechanism
- Multi-head attention
- Feed-forward neural network
- Transformer blocks
- Language model training using PyTorch
- Text generation
- Trained model saved as a `.pth` file

## Technologies Used

- Python
- PyTorch
- NumPy
- Transformer Architecture
- Neural Networks
- Character-level Tokenization

## Model Architecture

The model follows a simplified Transformer architecture:

```text
Input Text
    ↓
Character Tokenization
    ↓
Token Embeddings
    +
Positional Embeddings
    ↓
Multi-Head Self-Attention
    ↓
Feed-Forward Network
    ↓
Transformer Blocks
    ↓
Layer Normalization
    ↓
Output Layer
    ↓
Next Character Prediction
    ↓
Generated Text