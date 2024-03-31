import keras
from utils import C2GModelBase
from models.attention import Attention, AttentionTrain

class Decoder(keras.layers.Layer, C2GModelBase):
    """
    Decoder layer in a Transformer model architecture.

    This layer implements the decoder component of the Transformer model, which is responsible for generating
    the output sequence based on the encoded input sequence and previously generated output tokens.

    Parameters:
        attention (keras.layers.Layer): Attention layer for the decoder.
        dropout_rate (float): Dropout rate applied to the outputs of each sub-layer. Default is 0.2.
    """

    def __init__(self, dropout_rate=0.2, num_heads = 32, head_dims = 40, fc_dim_factor = 5, input_len = 64):
        """
        Initializes the Decoder layer.

        Args:
            attention (keras.layers.Layer): Attention layer for the decoder.
            dropout_rate (float): Dropout rate applied to the outputs of each sub-layer. Default is 0.2.
        """
        super().__init__()

        # Layer Normalization for the first sub-layer
        self.norm1 = keras.layers.LayerNormalization(epsilon=1e-6)

        # Layer Normalization for the second sub-layer
        self.norm2 = keras.layers.LayerNormalization(epsilon=1e-6)

        # Attention mechanism
        self.attn = AttentionTrain(head_dims=head_dims, num_heads=num_heads, dropout=dropout_rate, input_len = input_len)

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
            inputs: Input tensor representing the sequence to be decoded.

        Returns:
            Tensor: Output tensor representing the decoded sequence.
        """
        x = inputs
        residual = x

        # First sub-layer: Multi-head Self-Attention
        x = self.norm1(x)
        x = self.dropout1(x)
        x = self.attn(x)
        x += residual  # Add residual connection

        residual = x

        # Second sub-layer: Feed-forward Neural Network
        x = self.norm2(x)
        x = self.dropout2(x)
        x = self.fc1(x)  # First feed-forward layer
        x = self.fc2(x)  # Second feed-forward layer

        x += residual  # Add residual connection

        return x
