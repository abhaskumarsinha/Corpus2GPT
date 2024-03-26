import tensorflow as tf
import math
import keras
import numpy as np

class SharedEmbedding(keras.layers.Layer):
    """
    SharedEmbedding layer for providing embeddings of tokens.

    This layer implements the embedding functionality for tokens. 
    It uses a single shared embedding matrix for all tokens. 
    The size of the embedding matrix is determined by the 
    vocabulary size and the embedding dimension.

    Attributes:
        embedding (keras.layers.Embedding): Embedding layer 
        instance with shared parameters.
    """

    def __init__(self):
        """Initializes the SharedEmbedding layer."""
        super().__init__()
        self.embedding = keras.layers.Embedding(8008, 1280)

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

    Attributes:
        embedding (keras.layers.Embedding): Embedding layer instance 
        for positional embeddings.
    """

    def __init__(self):
        """Initializes the PositionalEmbedding layer."""
        super().__init__()
        self.embedding = keras.layers.Embedding(128, 1280)

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
        bsz, seq_len = inputs[:2]

        ones = keras.ops.ones((bsz, seq_len))
        seq = keras.ops.reshape(keras.ops.arange(length, length + seq_len), (1, seq_len))

        y = self.embedding(keras.ops.einsum('bi, xi -> bi', ones, seq))

        return y
