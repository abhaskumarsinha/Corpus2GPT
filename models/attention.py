import keras
import numpy as np


class Attention(keras.layers.Layer):
    """
    Multihead attention layer.

    This layer performs multihead attention on input sequences
    `(key, query, value)`. It splits the input into multiple heads,
    applies attention mechanism independently to each head,
    and concatenates the outputs.

    Parameters:
        head_dims (int): Dimensionality of each head.
        num_heads (int): Number of attention heads.
        dropout (float): Dropout rate to apply within the
                         attention mechanism.

    Usage:
    ```
        attention = Attention(head_dims=40, num_heads=32, dropout=0.2)
        output, cache = attention([key, query, value])
    ```
    """

    def __init__(self, head_dims=40, num_heads=32, dropout=0.2):
        """
        Initializes the multihead attention layer.

        Args:
            head_dims (int): Dimensionality of each head.
            num_heads (int): Number of attention heads.
            dropout (float): Dropout rate to apply within
                             the attention mechanism.
        """
        super().__init__()

        self.head_dims = head_dims
        self.num_heads = num_heads
        self.dropout = dropout
        self.dense_units = self.head_dims * self.num_heads

        self.key = keras.layers.Dense(self.dense_units)
        self.query = keras.layers.Dense(self.dense_units)
        self.value = keras.layers.Dense(self.dense_units)
        self.out = keras.layers.Dense(self.dense_units)
        self.norm = keras.layers.Normalization(-1)
        self.dropout = keras.layers.Dropout(self.dropout)

    def generate_mask(self, num_words):
        """
        Generates a triangular mask to be applied 
        to attention scores to prevent attending to 
        future positions.

        Args:
            num_words (int): Number of words in the 
            sequence.

        Returns:
            tf.Tensor: Triangular mask tensor.
        """
        tensor = np.full((num_words, num_words), np.inf)  # Initialize tensor with infinity
        for i in range(num_words):
            tensor[i, :i + 1] = 0
        return keras.ops.convert_to_tensor(tensor, dtype="float32")

    def _shape(self, tensor):
        """
        Reshapes the input tensor for multihead attention
        computations.

        Args:
            tensor (tf.Tensor): Input tensor.

        Returns:
            tf.Tensor: Reshaped tensor.
        """
        bsz = keras.ops.shape(tensor)[0]
        tensor = keras.ops.reshape(tensor, (bsz, -1, self.num_heads, self.head_dims))
        tensor = keras.ops.transpose(tensor, (0, 2, 1, 3))
        return tensor

    def call(self, inputs, use_cache=None):
        """
        Forward pass of the multihead attention layer.

        Args:
            inputs (list): List containing key, query, 
              	           and value tensors.
            use_cache (tuple): Cache from previous 
                                attention operation (unsupported).

        Returns:
            tf.Tensor: Output tensor.
            tuple: Cache for subsequent attention operations.
        """
        k, q, v = inputs

        if use_cache is None:
            k = self.key(k)
            q = self.query(q)
            v = self.value(v)

            k, q, v = self._shape(k), self._shape(q), self._shape(v)

        else:
            raise NotImplementedError("`use_cache` argument is not supported yet!")

        cache = (k, q, v)

        kq = keras.ops.einsum('bijk, bilk -> bij', k, q)

        num_words = keras.ops.shape(kq)[-1]
        bsz = keras.ops.shape(kq)[0]

        kq = keras.ops.reshape(kq, (bsz, 1, -1, num_words))

        kq_copy = keras.ops.copy(kq)

        for counter in range(num_words - 1):
            kq = keras.ops.append(kq, kq_copy, 1)

        mask = self.generate_mask(num_words)
        kq = keras.ops.transpose(kq, (0, 2, 1, 3))
        kq = kq - mask
        kq = keras.ops.transpose(kq, (0, 2, 1, 3))
        kq = keras.ops.softmax(kq, -1)
        kqv = keras.ops.einsum('bijk, bjkl -> bijl', kq, v)
        kqv = keras.ops.reshape(kqv, (bsz, num_words, -1))
        try:
            kqv = self.norm(kqv)
            kqv = self.dropout(kqv)
        except:
            pass
        kqv = self.out(kqv)

        return kqv, cache
