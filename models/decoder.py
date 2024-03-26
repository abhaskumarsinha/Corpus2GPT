import keras

class Decoder(keras.layers.Layer):
    """
    Decoder layer in a Transformer model architecture.

    This layer implements the decoder component of the Transformer model, which is responsible for generating
    the output sequence based on the encoded input sequence and previously generated output tokens.

    Parameters:
        dropout_rate (float): Dropout rate applied to the outputs of each sub-layer. Default is 0.2.
    """

    def __init__(self, attention, dropout_rate=0.2):
        """
        Initializes the Decoder layer.

        Args:
            attention (keras.layers.Layer): Attention layer for attention.
            dropout_rate (float): Dropout rate applied to the outputs of each sub-layer. Default is 0.2.
        """
        super().__init__()

        # Layer Normalization for the first sub-layer
        self.norm1 = keras.layers.LayerNormalization(-1)

        # Layer Normalization for the second sub-layer
        self.norm2 = keras.layers.LayerNormalization(-1)

        # Attention mechanism
        self.attn = Attention()

        # Dense layer for the first feed-forward sub-layer
        self.fc1 = keras.layers.Dense(self.attn.num_heads * self.attn.head_dims * 5)

        # Dense layer for the second feed-forward sub-layer
        self.fc2 = keras.layers.Dense(self.attn.num_heads * self.attn.head_dims)

        # Dropout layers
        self.dropout1 = keras.layers.Dropout(dropout_rate)
        self.dropout2 = keras.layers.Dropout(dropout_rate)

    def call(self, inputs):
        """
        Executes the forward pass of the Decoder layer.

        Args:
            inputs: Input tensor representing the sequence to be decoded.

        Returns:
            Tensor: Output tensor representing the decoded sequence.
        """
        x = inputs
        residual = x

        # First sub-layer: Multi-head Self-Attention
        x = self.norm1(x)
        x = self.dropout1(x)
        x, _ = self.attn((x, x, x))  # Multi-head self-attention mechanism
        x += residual  # Add residual connection

        residual = x

        # Second sub-layer: Feed-forward Neural Network
        x = self.norm2(x)
        x = self.dropout2(x)
        x = self.fc1(x)  # First feed-forward layer
        x = self.fc2(x)  # Second feed-forward layer

        x += residual  # Add residual connection

        return x
