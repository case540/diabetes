import pickle
import numpy as np

class CGMTokenizer:
    """
    A simple tokenizer for CGM data. It rounds glucose values to the nearest integer
    and maps them to token IDs based on a pre-trained vocabulary file.
    """
    def __init__(self, token2id_path, seq_len=288):
        try:
            with open(token2id_path, 'rb') as f:
                self.token2id = pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Tokenizer vocabulary file not found at {token2id_path}")

        self.seq_len = seq_len
        self.pad_token_id = self.token2id.get('<pad>', self.token2id.get('<PAD>', 0))

    def tokenize(self, glucose_values):
        """
        Takes a list of glucose float values, rounds them, and converts to token IDs.
        """
        # Round to nearest integer and handle potential NaNs from resampling
        rounded_values = np.round(glucose_values).astype('int64')
        
        # Convert glucose integers to token IDs, using pad_token for unknown values
        token_ids = [self.token2id.get(val, self.pad_token_id) for val in rounded_values]
        
        return token_ids 