import keras
from utils import C2GModelBase

class GPT(keras.layers.Layer, C2GModelBase):
    """
    GPT (Generative Pre-trained Transformer) layer.

    This layer implements the architecture of the GPT model, which consists of multiple decoder layers followed
    by a linear mapping head for language modeling.

    Parameters:
        decoder (class): Class representing the decoder layer of the Transformer model.
        attention (class): Class representing the attention mechanism used in the decoder layer.
        embeddings (class): Class representing the token embeddings.
        pos_embeddings (class): Class representing the positional embeddings.
        embedding_size (int): Size of the token embeddings. Default is 1280.
        vocab_size (int): Size of the vocabulary. Default is 8008.
        input_len (int): Length of the input sequence. Default is 64.
        num_decoders (int): Number of decoder layers in the GPT model. Default is 10.

    Attributes:
        num_decoders (int): Number of decoder layers in the GPT model.
        decoders (list): List of decoder layer instances.
        embeddings (keras.layers.Layer): Token embeddings layer instance.
        pos_embeddings (keras.layers.Layer): Positional embeddings layer instance.
        lm_head (keras.layers.Dense): Dense layer for language modeling.
    """

    def __init__(self, decoder, embeddings, pos_embeddings, embedding_size=1280, vocab_size=8008, input_len=64, num_decoders=10, dropout_rate=0.2, num_heads = 32, head_dims = 40, fc_dim_factor = 5):
        """
        Initializes the GPT layer.

        Args:
            decoder (class): Class representing the decoder layer of the Transformer model.
            attention (class): Class representing the attention mechanism used in the decoder layer.
            embeddings (class): Class representing the token embeddings.
            pos_embeddings (class): Class representing the positional embeddings.
            embedding_size (int): Size of the token embeddings. Default is 1280.
            vocab_size (int): Size of the vocabulary. Default is 8008.
            input_len (int): Length of the input sequence. Default is 64.
            num_decoders (int): Number of decoder layers in the GPT model. Default is 10.
        """
        super().__init__()

        self.num_decoders = num_decoders
        self.decoders = []
        for _ in range(self.num_decoders):
            self.decoders.append(decoder(dropout_rate, num_heads, head_dims, fc_dim_factor, input_len = input_len))
        
        self.embeddings = embeddings(vocab_size, embedding_size)
        self.pos_embeddings = pos_embeddings(input_len, embedding_size)

        self.lm_head = keras.layers.Dense(vocab_size)

        self._config = {'decoder' : decoder, 'embeddings': embeddings, 'pos_embeddings': pos_embeddings, 'embedding_size': embedding_size, 'vocab_size': vocab_size, 'input_len': input_len, 'num_decoders': num_decoders}


    def call(self, inputs):
        """
        Executes the forward pass of the GPT layer.

        Args:
            inputs: Input tensor representing the token indices.

        Returns:
            Tensor: Output tensor representing the logits for language modeling.
        """
        x = inputs

        x = self.embeddings(x)
        x = self.pos_embeddings(x)

        for decoder in self.decoders:
            x = decoder(x)
        
        x = self.lm_head(x)

        return x
