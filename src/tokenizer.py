import re
from collections import Counter
import random
import os
import json


class Tokenizer:
    def __init__(self, threshold=1e-5, min_frequency=10, load_vocab=True):
        self.threshold = threshold
        self.min_frequency = min_frequency
        self.word2idx = {}
        self.idx2word = {}
        if load_vocab:
            self.load_vocab()

    def preprocess_text(self, text: str) -> list[str]:
        text = text.lower()
        text = text.replace(".", " <PERIOD> ")
        text = text.replace(",", " <COMMA> ")
        text = text.replace('"', " <QUOTATION_MARK> ")
        text = text.replace(";", " <SEMICOLON> ")
        text = text.replace("!", " <EXCLAMATION_MARK> ")
        text = text.replace("?", " <QUESTION_MARK> ")
        text = text.replace("(", " <LEFT_PAREN> ")
        text = text.replace(")", " <RIGHT_PAREN> ")
        text = text.replace("[", " <LEFT_BRACKET> ")
        text = text.replace("]", " <RIGHT_BRACKET> ")
        text = text.replace("{", " <LEFT_BRACE> ")
        text = text.replace("}", " <RIGHT_BRACE> ")
        text = text.replace("/", " <SLASH> ")
        text = text.replace("\\", " <BACKSLASH> ")
        text = text.replace("-", " <HYPHEN> ")
        text = text.replace("--", " <DOUBLE_HYPHEN> ")
        text = text.replace(":", " <COLON> ")
        text = text.replace("+", " <PLUS> ")
        text = text.replace("*", " <ASTERISK> ")
        text = text.replace("&", " <AMPERSAND> ")
        text = re.sub(r"\d+", " <DIGIT> ", text)
        text = re.sub(r"[^\x00-\x7F]+", "<UNICODE> ", text)

        return text.split()

    def preprocess(self, input_filepath, output_filepath):
        """
        Process the input file through all tokenization steps and save to output file.

        Args:
            input_filepath (str): Path to input file
            output_filepath (str): Path to save processed output

        Raises:
            FileNotFoundError: If input file doesn't exist
            PermissionError: If can't read input or write output
            Exception: For other processing errors
        """
        try:
            # Check if input file exists
            if not os.path.exists(input_filepath):
                raise FileNotFoundError(f"Input file not found: {input_filepath}")

            # Read input file
            with open(input_filepath, "r", encoding="utf-8") as file:
                text = file.read()

            tokens = self.preprocess_text(text)

            # Remove single character tokens
            tokens = [token for token in tokens if len(token) > 1]

            # Remove rare words
            filtered_tokens = self._remove_rare_words(tokens)

            # add <unknown> to the tokens
            filtered_tokens.append("<UNKNOWN>")

            # Apply stemming
            stemmed_tokens = [self._simple_stem(token) for token in filtered_tokens]

            # Apply subsampling
            final_tokens = self._subsample_words(stemmed_tokens, self.threshold)

            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_filepath), exist_ok=True)

            # Save processed tokens as JSON to preserve them exactly
            with open(output_filepath, "w", encoding="utf-8") as file:
                json.dump(final_tokens, file)

            return final_tokens

        except FileNotFoundError:
            raise
        except PermissionError:
            raise PermissionError(
                "Permission denied when accessing "
                f"{input_filepath} or {output_filepath}"
            )
        except Exception as e:
            raise Exception(f"Error processing file: {str(e)}")

    def build_vocab(
        self,
        tokens,
        word2idxPath="data/word2idx.json",
        idx2wordPath="data/idx2word.json",
    ):
        """
        Build vocabulary from tokens and save to files.

        Args:
            tokens: List of tokens to build vocabulary from
            word2idxPath: Path to save word to index mapping
            idx2wordPath: Path to save index to word mapping

        Returns:
            Tuple of (word2idx, idx2word) dictionaries
        """
        try:
            # Create vocabulary with sorted unique tokens for consistency
            unique_tokens = sorted(set(tokens))
            self.word2idx = {word: idx for idx, word in enumerate(unique_tokens)}
            self.idx2word = {idx: word for idx, word in enumerate(unique_tokens)}

            # Save vocabularies
            os.makedirs(os.path.dirname(word2idxPath), exist_ok=True)
            with open(word2idxPath, "w", encoding="utf-8") as f:
                json.dump(self.word2idx, f, indent=2)

            os.makedirs(os.path.dirname(idx2wordPath), exist_ok=True)
            with open(idx2wordPath, "w", encoding="utf-8") as f:
                json.dump(self.idx2word, f, indent=2)

        except PermissionError:
            raise PermissionError("Permission denied when writing vocabulary files")
        except Exception as e:
            raise Exception(f"Error saving vocabulary: {str(e)}")

        return self.word2idx, self.idx2word

    def load_vocab(self, word2idx="data/word2idx.json", idx2word="data/idx2word.json"):
        """
        Load a vocabulary from a JSON file.

        Args:
            filepath (str): Path to the vocabulary file

        Raises:
            FileNotFoundError: If the vocabulary file doesn't exist
            PermissionError: If can't read the file
        """
        try:
            if not os.path.exists(word2idx):
                raise FileNotFoundError(f"Vocabulary file not found: {word2idx}")
            with open(word2idx, "r", encoding="utf-8") as f:
                self.word2idx = json.load(f)

            if not os.path.exists(idx2word):
                raise FileNotFoundError(f"Vocabulary file not found: {idx2word}")
            with open(idx2word, "r", encoding="utf-8") as f:
                self.idx2word = json.load(f)
        except FileNotFoundError:
            raise
        except PermissionError:
            raise PermissionError("Permission denied when reading")
        except Exception as e:
            raise Exception(f"Error loading vocabulary: {str(e)}")

    def _remove_rare_words(self, tokens):
        # Count word frequencies
        word_counts = Counter(tokens)

        # Keep only words that appear at least min_frequency times
        return [word for word in tokens if word_counts[word] >= self.min_frequency]

    def _simple_stem(self, word):
        # Don't stem if word is too short
        if len(word) <= 3:
            return word

        # Handle special cases first
        if word.endswith("ies"):
            # flies -> fly, ties -> tie
            if len(word) <= 4:  # ties -> tie
                return word[:-1]
            return word[:-3] + "y"  # flies -> fly

        if word.endswith("ing"):
            stem = word[:-3]
            # caring -> care
            if len(stem) > 0 and not any(c in stem for c in "aeiou"):
                return word  # Keep original if stem has no vowels
            if len(stem) >= 2:
                if stem[-1] in "aeiou" and stem[-2] not in "aeiou":
                    return stem + "e"  # caring -> care
                return stem
            return word

        if word.endswith("ed"):
            stem = word[:-2]
            if len(stem) > 0 and not any(c in stem for c in "aeiou"):
                return word  # Keep original if stem has no vowels
            if len(stem) >= 2:
                if stem[-1] in "aeiou" and stem[-2] not in "aeiou":
                    return stem + "e"  # cared -> care
                return stem
            return word

        # Handle simple plural 's' and 'es'
        if word.endswith("s"):
            # Don't stem special cases
            if word.endswith(("ss", "us", "is", "ous", "as", "yes")):
                return word

            if word.endswith("es"):
                # boxes -> box
                if len(word) > 4 and word[-3] in "xsz":
                    return word[:-2]
                return word

            # Regular plural
            if len(word[:-1]) > 2:  # Ensure we don't create too short words
                return word[:-1]

        return word

    def _subsample_words(self, words, threshold=1e-5):
        # Count the frequency of each word
        word_counts = Counter(words)
        total_words = len(words)

        # Calculate the probability of keeping each word
        word_probabilities = {
            word: count / total_words for word, count in word_counts.items()
        }

        # Calculate the subsampling threshold
        subsampling_threshold = {
            word: (1 - (threshold / prob) ** 0.5)
            for word, prob in word_probabilities.items()
            if prob > threshold
        }

        # Subsample the words
        kept_words = []
        for word in words:
            if word in subsampling_threshold:
                # Randomly decide whether to KEEP the word based on discard probability
                if (
                    random.random() >= subsampling_threshold[word]
                ):  # Keep if random number is >= discard probability
                    kept_words.append(word)
            else:
                kept_words.append(word)

        return kept_words
