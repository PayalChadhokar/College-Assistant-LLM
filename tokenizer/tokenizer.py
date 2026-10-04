class CharacterTokenizer:

    def __init__(self, text):
        # Create a sorted list of all unique characters
        self.chars = sorted(list(set(text)))

        # Vocabulary size
        self.vocab_size = len(self.chars)

        # Character → number
        self.char_to_id = {
            ch: i for i, ch in enumerate(self.chars)
        }

        # Number → character
        self.id_to_char = {
            i: ch for i, ch in enumerate(self.chars)
        }

    def encode(self, text):
        """Convert text into numbers."""
        return [self.char_to_id[ch] for ch in text]

    def decode(self, ids):
        """Convert numbers back into text."""
        return ''.join(self.id_to_char[i] for i in ids)