import keras
from Corpus2GPT.models.attention import AttentionTrain

class Decoder(keras.layers.Layer):
    """
    Decoder layer in a Transformer model architecture.

    This layer implements the decoder component of the Transformer model, which is responsible for generating
    the output sequence based on the encoded input sequence and previously generated output tokens.

    Parameters:
    - dropout_rate (float): Dropout rate applied to the outputs of each sub-layer. Default is 0.2.
    - num_heads (int): Number of attention heads. Default is 32.
    - head_dims (int): Dimensionality of each attention head. Default is 40.
    - fc_dim_factor (int): Factor controlling the dimensionality of the fully connected layers. Default is 5.
    - input_len (int): Length of the input sequence. Default is 64.
    
    References:
        - Vaswani, Ashish, et al. "Attention is all you need." Advances in neural information processing systems 30 (2017).

    Example:
        ```python
        >>> decoder = Decoder()
        >>> output = decoder(keras.ops.ones((1, 10, 1280)
        >>> print(output)
        ```
    """

    def __init__(self, dropout_rate=0.2, num_heads=32, head_dims=40, fc_dim_factor=5, input_len=64, epsilon=1e-9):
        """
        Initializes the Decoder layer.

        Args:
        - dropout_rate (float): Dropout rate applied to the outputs of each sub-layer. Default is 0.2.
        - num_heads (int): Number of attention heads. Default is 32.
        - head_dims (int): Dimensionality of each attention head. Default is 40.
        - fc_dim_factor (int): Factor controlling the dimensionality of the fully connected layers. Default is 5.
        - input_len (int): Length of the input sequence. Default is 64.
        """
        super().__init__()

        # Layer Normalization for the first sub-layer
        self.norm1 = keras.layers.LayerNormalization(epsilon=epsilon)

        # Layer Normalization for the second sub-layer
        self.norm2 = keras.layers.LayerNormalization(epsilon=epsilon)

        # Attention mechanism
        self.attn = AttentionTrain(head_dims=head_dims, num_heads=num_heads, dropout=dropout_rate, input_len=input_len)

        # Dense layer for the first feed-forward sub-layer
        self.fc1 = keras.layers.Dense(num_heads * head_dims * fc_dim_factor, activation='gelu')

        # Dense layer for the second feed-forward sub-layer
        self.fc2 = keras.layers.Dense(num_heads * head_dims, activation='gelu')

        # Dropout layers
        self.dropout1 = keras.layers.Dropout(dropout_rate)
        self.dropout2 = keras.layers.Dropout(dropout_rate)
        
        self._config = {'dropout_rate': dropout_rate}

    def call(self, inputs):
        """
        Executes the forward pass of the Decoder layer.

        Args:
        - inputs: Input tensor.

        Returns:
        - keras.Tensor: Output tensor.
        """
        x = inputs
        x = self.attn(x)
        x = self.dropout1(x)
        out1 = self.norm1(x + inputs)

        x = out1
        out1 = self.fc2(self.fc1(out1))
        out1 = self.dropout2(out1)
        return self.norm2(out1 + x)
