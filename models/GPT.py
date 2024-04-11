import keras
from tokenizer.tokenizer import *
from models.attention import *
from models.decoder import *
from models.embeddings import *

def FLOP(input_len, vocab_size, embed_dim, num_heads, num_decoders, fc_dim_factor):
    """ 
    Calculate total number of FLOPs, see Chinchilla 
    paper Appendix F as reference: https://arxiv.org/pdf/2203.15556.pdf

    Copied from: https://github.com/karpathy/nanoGPT/blob/master/scaling_laws.ipynb
    """ 
    key_size = embed_dim // num_heads

    # embeddings
    embeddings = 2 * input_len * vocab_size * embed_dim

    # attention
    # key, query, value projections
    attention = 2 * 3 * input_len * embed_dim * (key_size * num_heads)
    # key @ query logits
    attlogits = 2 * input_len * input_len * (key_size * num_heads)
    # softmax
    attsoftmax = 3 * num_heads * input_len * input_len # 3* is for subtract (max), exp, divide (?)
    # softmax @ value reductions
    attvalue = 2 * input_len * input_len * (key_size * num_heads)
    # final linear
    attlinear = 2 * input_len * (key_size * num_heads) * embed_dim
    att = attention + attlogits + attsoftmax + attvalue + attlinear
    # feed forward
    dense = 2 * input_len * (embed_dim * embed_dim*fc_dim_factor + embed_dim * embed_dim*fc_dim_factor)

    # logits
    logits = 2 * input_len * embed_dim * vocab_size
    
    # this is what you'd expect:
    # forward_flops = embeddings + num_decoders * (att + dense) + logits
    # but:
    # per author correspondence apparently there is typo in the paper,
    # they do not count embeddings and logits to repro table 4. So instead:
    forward_flops = num_decoders * (att + dense)
    backward_flops = 2 * forward_flops # as in Kaplan et al. 2020
    total_flops = forward_flops + backward_flops

    return total_flops

class GPT(keras.layers.Layer):
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

    def __init__(self, decoder, embeddings, pos_embeddings = None, embedding_size=1280, vocab_size=8008, input_len=64, num_decoders=5, dropout_rate=0.1, num_heads = 32, head_dims = 40, fc_dim_factor = 5):
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
        
        self.embeddings = embeddings(input_len, vocab_size+1, embed_dim=embedding_size)

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

        for decoder in self.decoders:
            x = decoder(x)
        
        x = self.lm_head(x)

        return x



def build_GPT(input_len,
              vocab_size,
              embed_dim,
              num_decoders,
              dropout_rate,
              num_heads,
              head_dims,
              fc_dim_factor,
              optimizer='adam'
              ):
    """
    Builds a GPT (Generative Pre-trained Transformer) model.

    Parameters:
        input_len (int): The length of the input sequence.
        vocab_size (int): The size of the vocabulary.
        embed_dim (int): The dimensionality of the token embeddings.
        num_decoders (int): The number of decoder layers.
        dropout_rate (float): The dropout rate to apply within the model.
        num_heads (int): The number of attention heads in each decoder layer.
        head_dims (int): The dimensionality of each attention head.
        fc_dim_factor (int): The factor to determine the dimensionality
                             of the feedforward network within each decoder layer.
        optimizer (str, optional): The optimizer to use for training. 
                                   Defaults to 'adam'.

    Returns:
        tuple: A tuple containing the GPT model and the total number of floating-point operations (FLOPs).

    """
    GPT = keras.Sequential()
    GPT.add(keras.Input(shape=(input_len,)))
    GPT.add(TokenAndPositionEmbedding(input_len, vocab_size, embed_dim))
    for _ in range(num_decoders):
        GPT.add(Decoder(dropout_rate, num_heads, head_dims, fc_dim_factor, input_len))
    GPT.add(keras.layers.Dense(vocab_size+1))

    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    GPT.compile(optimizer=optimizer, loss=[loss_fn])

    # Calculate the total number of floating-point operations              
    flops = FLOP(input_len, vocab_size, embed_dim, num_heads, num_decoders, fc_dim_factor)
    return GPT, flops

