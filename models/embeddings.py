import math
import keras
import numpy as np


class TokenAndPositionEmbedding(keras.layers.Layer):
    """
    Token and Position Embedding layer for Transformer models.

    This layer combines token embeddings and positional embeddings
    to provide input embeddings for Transformer models.

    Parameters:
        maxlen (int): Maximum length of the input sequence.
        vocab_size (int): Size of the vocabulary.
        embed_dim (int): Dimensionality of the embedding vectors.

    Example:
        ```
        >>> embed = TokenAndPositionEmbedding(64, 8008, 1280)
        >>> input_sentence = keras.ops.ones((1, 10))

        >>> output = embed(input_sentence)
        ```
    """

    def __init__(self, maxlen, vocab_size, embed_dim):
        """
        Initializes the TokenAndPositionEmbedding layer.

        Args:
            maxlen (int): Maximum length of the input sequence.
            vocab_size (int): Size of the vocabulary.
            embed_dim (int): Dimensionality of the embedding vectors.
        """
        super().__init__()
        self.token_emb = keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = keras.layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        """
        Executes the forward pass of the TokenAndPositionEmbedding layer.

        Args:
            x: Input tensor representing token indices.

        Returns:
            tf.Tensor: Output tensor representing the combined embeddings.
        """
        maxlen = keras.ops.shape(x)[-1]
        positions = keras.ops.arange(0, maxlen, 1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

