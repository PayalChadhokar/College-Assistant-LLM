from flask import Flask, render_template, request, jsonify
import torch
import torch.nn.functional as F
import re
from difflib import SequenceMatcher

from tokenizer.tokenizer import CharacterTokenizer
from model.model import TinyLanguageModel


app = Flask(__name__)

# ==========================================================
# LOAD DATASET
# ==========================================================

with open("dataset/college_data.txt", "r", encoding="utf-8") as f:
    text = f.read()


# ==========================================================
# TOKENIZER
# ==========================================================

tokenizer = CharacterTokenizer(text)


# ==========================================================
# EXTRACT QUESTION-ANSWER PAIRS
# ==========================================================

def load_qa_pairs(text):
    pairs = []

    pattern = r"Question:\s*(.*?)\s*Answer:\s*(.*?)(?=\s*Question:|$)"

    matches = re.findall(
        pattern,
        text,
        re.DOTALL | re.IGNORECASE
    )

    for question, answer in matches:

        question = question.strip()
        answer = answer.strip()

        if question and answer:
            pairs.append({
                "question": question,
                "answer": answer
            })

    return pairs


qa_pairs = load_qa_pairs(text)

print(f"Loaded {len(qa_pairs)} question-answer pairs.")


# ==========================================================
# QUESTION NORMALIZATION
# ==========================================================

def normalize_question(question):

    question = question.lower()

    # Common variations
    replacements = {
        "artificial intelligence": "ai",
        "machine learning": "ml",
        "deep learning": "dl",
        "natural language processing": "nlp",
        "what's": "what is",
        "whats": "what is",
        "pls": "please",
        "plz": "please"
    }

    for old, new in replacements.items():
        question = question.replace(old, new)

    # Remove punctuation
    question = re.sub(r"[^a-z0-9\s]", " ", question)

    # Remove extra spaces
    question = re.sub(r"\s+", " ", question)

    return question.strip()


# ==========================================================
# QUESTION SIMILARITY
# ==========================================================

def question_similarity(user_question, stored_question):

    user = normalize_question(user_question)
    stored = normalize_question(stored_question)

    # Exact match
    if user == stored:
        return 1.0

    # Sequence similarity
    sequence_score = SequenceMatcher(
        None,
        user,
        stored
    ).ratio()

    # Word overlap
    user_words = set(user.split())
    stored_words = set(stored.split())

    if user_words and stored_words:

        intersection = user_words.intersection(stored_words)

        word_score = len(intersection) / len(
            user_words.union(stored_words)
        )

    else:
        word_score = 0.0

    # Combined score
    score = (
        0.65 * sequence_score +
        0.35 * word_score
    )

    return score


# ==========================================================
# FIND BEST DATASET ANSWER
# ==========================================================

def find_dataset_answer(question):

    best_score = 0
    best_answer = None
    best_question = None

    for pair in qa_pairs:

        score = question_similarity(
            question,
            pair["question"]
        )

        if score > best_score:
            best_score = score
            best_answer = pair["answer"]
            best_question = pair["question"]

    # Threshold
    if best_score >= 0.45:

        print(
            f"Dataset match: {best_question} "
            f"(score: {best_score:.2f})"
        )

        return best_answer

    return None


# ==========================================================
# DEVICE
# ==========================================================

device = "cuda" if torch.cuda.is_available() else "cpu"


# ==========================================================
# LOAD TRANSFORMER MODEL
# ==========================================================

model = TinyLanguageModel(
    tokenizer.vocab_size
).to(device)


model.load_state_dict(
    torch.load(
        "college_assistant_model.pth",
        map_location=device
    )
)

model.eval()

print(f"Transformer model loaded on {device}")


# ==========================================================
# TRANSFORMER FALLBACK
# ==========================================================

def generate_answer(question, max_new_tokens=250, temperature=0.6):

    prompt = f"Question: {question}\nAnswer:"

    try:
        encoded = tokenizer.encode(prompt)

    except KeyError:
        return (
            "Sorry, I cannot process some characters "
            "in that question."
        )

    x = torch.tensor(
        [encoded],
        dtype=torch.long,
        device=device
    )

    with torch.no_grad():

        for _ in range(max_new_tokens):

            x_cond = x[:, -64:]

            logits = model(x_cond)

            logits = logits[:, -1, :]

            logits = logits / temperature

            probabilities = F.softmax(
                logits,
                dim=-1
            )

            next_token = torch.multinomial(
                probabilities,
                num_samples=1
            )

            x = torch.cat(
                (x, next_token),
                dim=1
            )

            generated = tokenizer.decode(
                x[0].tolist()
            )

            if "Question:" in generated[len(prompt):]:
                break

    generated = tokenizer.decode(
        x[0].tolist()
    )

    if "Answer:" in generated:

        answer = generated.split(
            "Answer:",
            1
        )[1]

    else:
        answer = generated

    if "Question:" in answer:

        answer = answer.split(
            "Question:",
            1
        )[0]

    answer = answer.strip()

    if not answer:

        answer = (
            "I could not generate a suitable answer."
        )

    return answer


# ==========================================================
# MAIN ANSWER FUNCTION
# ==========================================================

def answer_question(question):

    # First search the custom knowledge base
    dataset_answer = find_dataset_answer(question)

    if dataset_answer:

        return dataset_answer

    # Otherwise use Transformer
    return generate_answer(question)


# ==========================================================
# HOME PAGE
# ==========================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ==========================================================
# ASK API
# ==========================================================

@app.route("/ask", methods=["POST"])
def ask():

    data = request.get_json()

    if not data or "question" not in data:

        return jsonify({
            "answer": "Please enter a question."
        }), 400

    question = data["question"].strip()

    if not question:

        return jsonify({
            "answer": "Please enter a question."
        }), 400

    answer = answer_question(question)

    return jsonify({
        "answer": answer
    })


# ==========================================================
# START FLASK
# ==========================================================

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )
