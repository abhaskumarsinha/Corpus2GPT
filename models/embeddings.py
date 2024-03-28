import math
import keras
import numpy as np
from utils import C2GModelBase

class SharedEmbedding(keras.layers.Layer, C2GModelBase):
    """
    SharedEmbedding layer for providing embeddings of tokens.

    This layer implements the embedding functionality for tokens. 
    It uses a single shared embedding matrix for all tokens. 
    The size of the embedding matrix is determined by the 
    vocabulary size and the embedding dimension.

    Parameters:
        vocab (int): Vocabulary size, determining the number of unique tokens.
        embedding_size (int): Dimensionality of the embedding space.

    Attributes:
        embedding (keras.layers.Embedding): Embedding layer 
        instance with shared parameters.
    """

    def __init__(self, vocab, embedding_size):
        """
        Initializes the SharedEmbedding layer.

        Args:
            vocab (int): Vocabulary size, determining the number of unique tokens.
            embedding_size (int): Dimensionality of the embedding space.
        """
        super().__init__()
        self.embedding = keras.layers.Embedding(vocab, embedding_size)

        self._config = {'vocab': vocab, 'embedding_size': embedding_size}

    def call(self, inputs):
        """
        Generates token embeddings.

        Args:
            inputs: Input tensor representing the token indices.

        Returns:
            Tensor: Embedded representation of input tokens.
        """
        return self.embedding(inputs)


class PositionalEmbedding(keras.layers.Layer):
    """
    PositionalEmbedding layer for adding positional embeddings to 
    token embeddings.

    This layer adds positional embeddings to the token embeddings. 
    It generates positional embeddings based on the position of 
    tokens in the sequence.

    Parameters:
        seq_len (int): Length of the sequence.
        embedding_size (int): Dimensionality of the embedding space.

    Attributes:
        embedding (keras.layers.Embedding): Embedding layer instance 
        for positional embeddings.
    """

    def __init__(self, seq_len, embedding_size):
        """
        Initializes the PositionalEmbedding layer.

        Args:
            seq_len (int): Length of the sequence.
            embedding_size (int): Dimensionality of the embedding space.
        """
        super().__init__()
        self.embedding = keras.layers.Embedding(seq_len, embedding_size)

        self._config = {'seq_len': seq_len, 'embedding_size': embedding_size}

    def call(self, inputs, length=0):
        """
        Adds positional embeddings to token embeddings.

        Args:
            inputs: Input tensor representing the token indices.
            length (int): Starting position for positional 
                          encoding. Defaults to 0.

        Returns:
            Tensor: Embedded representation of input tokens 
                    with positional encodings added.
        """
        bsz, seq_len = keras.backend.shape(inputs)[:2]

        ones = keras.backend.ones((bsz, seq_len))
        seq = keras.backend.reshape(keras.backend.arange(length, length + seq_len), (1, seq_len))

        y = self.embedding(keras.backend.einsum('bi, xi -> bi', ones, seq))

        return y
