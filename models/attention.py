import keras
import numpy as np


import keras
import numpy as np

class AttentionTrain(keras.layers.Layer):
    """
    Custom attention layer for training Transformer models.

    This layer implements the attention mechanism used in the
    Transformer model during training. It computes attention
    scores between query, key, and value tensors, applies
    masking to prevent attending to future positions, and
    performs dropout.

    Attributes:
        num_heads (int): Number of attention heads.
        head_dims (int): Dimensionality of each attention head.
        k (keras.layers.Dense): Dense layer for computing the keys.
        q (keras.layers.Dense): Dense layer for computing the queries.
        v (keras.layers.Dense): Dense layer for computing the values.
        out (keras.layers.Dense): Dense layer for the output.
        q_norm (float): Normalization factor for queries.
        mask (tf.Tensor): Triangular mask tensor.
        dropout (keras.layers.Dropout): Dropout layer.

    Formula:
        $Attention(K, Q, V)_{\text{head}} = softmax \left ( \dfrac{QK^T}{\sqrt{d_k}} \right ) V$
        for each head

    Example:
        ```
        >>> attn = AttentionTrain(32, 40)
        >>> print(attn(keras.ops.ones((1, 1, 1280))
        ```

    References:
        - Vaswani, Ashish, et al. "Attention is all you need." Advances in neural information processing systems 30 (2017).
    """

    def _shape(self, tensor):
        """
        Reshapes the input tensor for multihead attention computations.

        Args:
            tensor (tf.Tensor): Input tensor.

        Returns:
            tf.Tensor: Reshaped tensor.
        """
        bsz = keras.ops.shape(tensor)[0]
        tensor = keras.ops.reshape(tensor, (bsz, -1, self.num_heads, self.head_dims))
        tensor = keras.ops.transpose(tensor, (0, 2, 1, 3))
        return tensor

    def generate_mask(self, num_words):
        """
        Generates a triangular mask to be applied to attention scores
        to prevent attending to future positions.

        Args:
            num_words (int): Number of words in the sequence.

        Returns:
            tf.Tensor: Triangular mask tensor.
        """
        tensor = np.full((num_words, num_words), np.inf)  # Initialize tensor with infinity
        for i in range(num_words):
            tensor[i, :i + 1] = 0
        return keras.ops.convert_to_tensor(tensor, dtype="float32")

    def __init__(self, num_heads, head_dims, dropout=0.2, input_len=64):
        """
        Initializes the AttentionTrain layer.

        Args:
            num_heads (int): Number of attention heads.
            head_dims (int): Dimensionality of each attention head.
            dropout (float): Dropout rate. Default is 0.2.
            input_len (int): Length of the input sequence. Default is 64.
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dims = head_dims

        self.k = keras.layers.Dense(self.num_heads * self.head_dims)
        self.q = keras.layers.Dense(self.num_heads * self.head_dims)
        self.v = keras.layers.Dense(self.num_heads * self.head_dims)
        self.out = keras.layers.Dense(self.num_heads * self.head_dims)

        self.q_norm = 1 / keras.ops.sqrt(self.num_heads * self.head_dims)
        self.mask = self.generate_mask(input_len)

        self.dropout = keras.layers.Dropout(dropout)

    def call(self, inputs):
        """
        Executes the forward pass of the AttentionTrain layer.

        Args:
            inputs: Input tensor.

        Returns:
            tf.Tensor: Output tensor.
        """
        k = self.k(inputs)
        q = self.q(inputs)
        v = self.v(inputs)

        k, q, v = self._shape(k), self._shape(q), self._shape(v)

        # (b, head, k_token, dims), (b, head, q_token, dims) -> (b, head, q_token, k_token)
        kq = keras.ops.einsum('bijk, bilk -> bilj', k, q)
        kq *= self.q_norm
        kq -= self.mask
        kq = self.dropout(kq)
        kq = keras.ops.softmax(kq, -1)

        # (b, head, q_token, k_token), (b, head, k_token, dims) -> (b, head, q_token, dims)
        kqv = keras.ops.einsum('bilj, bijk -> bilk', kq, v)

        kqv = keras.ops.transpose(kqv, (0, 2, 1, 3))

        bsz = keras.ops.shape(v)[0]
        kqv = keras.ops.reshape(kqv, (bsz, -1, self.num_heads * self.head_dims))
        kqv = self.out(kqv)

        return kqv
